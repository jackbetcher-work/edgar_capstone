"""Official Google ADK `BasePlugin` integration (`ModelArmorSafetyPlugin`)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from edgar_capstone.common.domain import sha256_text
from edgar_capstone.security.model_armor.service import (
    ModelArmorGuardrailService,
    get_model_armor_guardrail,
    schema_safe_block_text,
)
from edgar_capstone.security.rbac import CUSTOM_RUNTIME_SA_ACCOUNT_ID

_PLUGIN_BLOCK_STATE_KEY = "_model_armor_plugin_pending_block_message"
_PENDING_STATE_DELTA_KEY = "_model_armor_pending_state_delta"
_VERIFIED_PROMPT_HASH_KEY = "_model_armor_verified_prompt_sha256"
_VERIFIED_RESPONSE_HASH_KEY = "_model_armor_verified_response_sha256"


class ModelArmorSafetyPlugin(BasePlugin):
    """ADK `BasePlugin` enforcing Google Cloud Model Armor, SDP de-identification, and Projection Filters.

    Follows the Google ADK `safety-plugins` architecture updated for ADK 2.x keyword-only
    plugin callbacks, offloading synchronous HTTP guardrail calls via ``asyncio.to_thread``
    so callbacks never block the shared ADK event loop, and preventing multi-turn session
    poisoning on ``on_user_message_callback`` before short-circuiting in ``before_run_callback``.
    """

    def __init__(
        self,
        *,
        name: str = "ModelArmorSafetyPlugin",
        guardrail: ModelArmorGuardrailService | None = None,
        inspect_tool_outputs: bool | None = None,
        pre_verified_prompt_hashes: set[str] | frozenset[str] | None = None,
    ) -> None:
        super().__init__(name=name)
        self._guardrail = guardrail
        self._inspect_tool_outputs = inspect_tool_outputs
        self._pre_verified_prompt_hashes = frozenset(pre_verified_prompt_hashes or ())

    @property
    def guardrail(self) -> ModelArmorGuardrailService:
        return self._guardrail if self._guardrail is not None else get_model_armor_guardrail()

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: genai_types.Content,
    ) -> genai_types.Content | None:
        parts = list(user_message.parts or [])
        text = "\n".join(p.text for p in parts if getattr(p, "text", None))
        if not text.strip():
            return None

        state = invocation_context.session.state
        text_hash = sha256_text(text)
        if state.get(_VERIFIED_PROMPT_HASH_KEY) == text_hash or (
            text_hash in self._pre_verified_prompt_hashes
        ):
            state[_VERIFIED_PROMPT_HASH_KEY] = text_hash
            return None

        decision = await asyncio.to_thread(
            self.guardrail.inspect_user_prompt,
            text,
            caller_agent=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
        )
        delta: dict[str, Any] = {
            "model_armor_last_decision": decision.to_audit_dict(),
            "model_armor_blocked": bool(decision.blocked),
            "model_armor_response_blocked": False,
            "model_armor_response_block_message": "",
        }

        if decision.blocked:
            delta["model_armor_findings"] = list(decision.matched_categories)
            state.update(delta)
            state[_PLUGIN_BLOCK_STATE_KEY] = decision.block_message
            state[_PENDING_STATE_DELTA_KEY] = delta
            # Replace unsafe prompt in session history so subsequent turns are never poisoned.
            return genai_types.Content(
                role=user_message.role or "user",
                parts=[
                    genai_types.Part(
                        text=(
                            "[User message withheld by Model Armor Guardrail: "
                            f"{', '.join(decision.matched_categories)}]"
                        )
                    )
                ],
            )

        if decision.sdp_redacted:
            delta["sdp_redacted"] = True
            delta["sdp_info_types"] = list(decision.sdp_info_types)
            delta[_VERIFIED_PROMPT_HASH_KEY] = sha256_text(decision.sanitized_text)
            state.update(delta)
            state[_PENDING_STATE_DELTA_KEY] = delta
            return genai_types.Content(
                role=user_message.role or "user",
                parts=[genai_types.Part(text=decision.sanitized_text)],
            )

        delta[_VERIFIED_PROMPT_HASH_KEY] = text_hash
        state.update(delta)
        state[_PENDING_STATE_DELTA_KEY] = delta
        return None

    async def before_run_callback(
        self,
        *,
        invocation_context: InvocationContext,
    ) -> genai_types.Content | None:
        state = invocation_context.session.state
        pending_msg = state.get(_PLUGIN_BLOCK_STATE_KEY)
        if pending_msg:
            state[_PLUGIN_BLOCK_STATE_KEY] = ""
            return genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=str(pending_msg))],
            )
        return None

    async def on_event_callback(
        self,
        *,
        invocation_context: InvocationContext,
        event: Event,
    ) -> Event | None:
        if getattr(event, "partial", False):
            return None
        pending = invocation_context.session.state.pop(_PENDING_STATE_DELTA_KEY, None)
        if isinstance(pending, dict) and pending:
            event.actions.state_delta.update(pending)
            return event
        return None

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        latest_user_content: genai_types.Content | None = None
        latest_text_parts: list[genai_types.Part] = []
        for content in llm_request.contents or []:
            if content.role == "user":
                text_parts = [p for p in (content.parts or []) if getattr(p, "text", None)]
                if text_parts:
                    latest_user_content = content
                    latest_text_parts = text_parts

        if not latest_text_parts or latest_user_content is None:
            return None

        # Join ALL text parts of the latest user turn (matching `on_user_message_callback`)
        # so multi-part prompts cannot smuggle injections in earlier parts.
        latest_user_text = "\n".join(p.text for p in latest_text_parts if p.text)
        if not latest_user_text.strip():
            return None

        state = callback_context.state
        prompt_hash = sha256_text(latest_user_text)
        if state.get(_VERIFIED_PROMPT_HASH_KEY) == prompt_hash:
            return None

        caller = str(getattr(callback_context, "agent_name", "") or CUSTOM_RUNTIME_SA_ACCOUNT_ID)
        decision = await asyncio.to_thread(
            self.guardrail.inspect_user_prompt,
            latest_user_text,
            caller_agent=caller,
        )
        state["model_armor_last_decision"] = decision.to_audit_dict()
        if decision.blocked:
            state["model_armor_blocked"] = True
            state["model_armor_findings"] = list(decision.matched_categories)
            return LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=decision.block_message)],
                )
            )

        if decision.sdp_redacted:
            if len(latest_text_parts) == 1:
                latest_text_parts[0].text = decision.sanitized_text
            else:
                non_text_parts = [
                    p for p in (latest_user_content.parts or []) if not getattr(p, "text", None)
                ]
                latest_user_content.parts = [
                    *non_text_parts,
                    genai_types.Part(text=decision.sanitized_text),
                ]
            state["sdp_redacted"] = True
            state["sdp_info_types"] = list(decision.sdp_info_types)
        state[_VERIFIED_PROMPT_HASH_KEY] = sha256_text(decision.sanitized_text)
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        _ = tool_args
        should_inspect = (
            self._inspect_tool_outputs
            if self._inspect_tool_outputs is not None
            else self.guardrail.config.inspect_tool_outputs
        )
        if not should_inspect or not isinstance(result, dict):
            return None

        serialized = json.dumps(result, default=str)
        caller = str(getattr(tool_context, "agent_name", "") or CUSTOM_RUNTIME_SA_ACCOUNT_ID)
        decision = await asyncio.to_thread(
            self.guardrail.inspect_user_prompt,
            serialized,
            caller_agent=caller,
        )
        if decision.blocked:
            tool_context.state["model_armor_blocked"] = True
            tool_context.state["model_armor_tool_blocked"] = getattr(tool, "name", "tool")
            return {
                "status": "blocked_by_model_armor",
                "error": (
                    f"Tool output from '{getattr(tool, 'name', 'tool')}' withheld by Model Armor: "
                    f"{', '.join(decision.matched_categories)}"
                ),
            }
        if decision.sdp_redacted:
            tool_context.state["sdp_redacted"] = True
            tool_context.state["sdp_info_types"] = list(decision.sdp_info_types)
            try:
                redacted_obj = json.loads(decision.sanitized_text)
                if isinstance(redacted_obj, dict):
                    return redacted_obj
            except (json.JSONDecodeError, ValueError):
                pass
            return {
                "status": "sanitized_by_model_armor",
                "sdp_redacted": True,
                "sdp_info_types": list(decision.sdp_info_types),
                "content": decision.sanitized_text,
            }
        return None

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        # Never block or add network hops to partial SSE streaming chunks or intermediate tool calls
        if getattr(llm_response, "partial", False):
            return None
        if not llm_response.content or not llm_response.content.parts:
            return None

        text_parts: list[genai_types.Part] = []
        has_fn_call = False
        for part in llm_response.content.parts:
            if getattr(part, "function_call", None) is not None:
                has_fn_call = True
            if getattr(part, "text", None) and not getattr(part, "thought", False):
                text_parts.append(part)

        if has_fn_call or not text_parts:
            return None

        combined_text = "\n".join(p.text for p in text_parts if p.text)
        if not combined_text.strip():
            return None

        state = callback_context.state
        response_hash = sha256_text(combined_text)
        if state.get(_VERIFIED_RESPONSE_HASH_KEY) == response_hash:
            return None

        caller = str(getattr(callback_context, "agent_name", "") or "FinancialAnalystAgent")
        decision = await asyncio.to_thread(
            self.guardrail.inspect_model_response,
            combined_text,
            caller_agent=caller,
        )
        state["model_armor_last_response_decision"] = decision.to_audit_dict()

        if decision.blocked:
            state["model_armor_blocked"] = True
            state["model_armor_response_blocked"] = True
            state["model_armor_response_block_message"] = decision.block_message
            state["model_armor_findings"] = list(decision.matched_categories)
            return LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            text=schema_safe_block_text(combined_text, decision.block_message)
                        )
                    ],
                )
            )

        state["model_armor_response_blocked"] = False
        state["model_armor_response_block_message"] = ""

        if decision.sdp_redacted:
            state["sdp_redacted"] = True
            state["sdp_info_types"] = list(decision.sdp_info_types)
            state[_VERIFIED_RESPONSE_HASH_KEY] = sha256_text(decision.sanitized_text)
            return LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=decision.sanitized_text)],
                ),
                usage_metadata=llm_response.usage_metadata,
                model_version=llm_response.model_version,
            )

        state[_VERIFIED_RESPONSE_HASH_KEY] = response_hash
        return None
