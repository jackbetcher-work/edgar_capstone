"""ADK lifecycle callbacks for latency tracking, Google Cloud Model Armor, SDP, and fact-checking."""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any, cast

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from edgar_capstone.agents.context_memory import (
    TASK_TO_MODEL_TIER,
    extract_llm_request_context_breakdown,
    get_context_latency_tracker,
    reconcile_breakdown_with_usage_metadata,
    resolve_task_and_model_for_agent,
)
from edgar_capstone.agents.schemas import (
    ContextBreakdown,
    FactCheckVerification,
    ModelContextLatencyRecord,
    ModelTaskCategory,
)
from edgar_capstone.agents.structured_report import (
    parse_analyst_report,
    render_analyst_report_markdown,
    verify_analyst_report,
)
from edgar_capstone.common.domain import RETRIEVAL_TOOL_NAMES, sha256_text
from edgar_capstone.common.env import env_positive_int
from edgar_capstone.security.agent_identity import (
    AgentAuthorizationError,
    authorize_agent_tool_invocation,
    resolve_agent_identity,
)
from edgar_capstone.security.auth import record_security_audit_event
from edgar_capstone.security.model_armor import get_model_armor_guardrail, schema_safe_block_text
from edgar_capstone.security.rbac import CUSTOM_RUNTIME_SA_ACCOUNT_ID

logger = logging.getLogger(__name__)

_CITATION_TAG_RE = re.compile(
    r"(?:\b[A-Z]{1,5}_\d{4}_item_(?:1a|7)_\d{4}\b|\[[A-Z.\-]{1,6}\s+FY\d{4}[^\]]*\]|Item\s+(?:1A|7)\b)",
    re.IGNORECASE,
)
_PERCENT_NUMBER_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")

_VARIANCE_TOOLS: frozenset[str] = frozenset(
    {
        "calculate_period_variances",
        "fetch_financial_metrics",
        "query_looker_financial_explore",
        "flag_material_variances",
    }
)

DEFAULT_ANALYST_RETRIEVAL_BUDGET = 6
_RETRIEVAL_CALLS_STATE_KEY = "temp:retrieval_calls_used"  # temp: = scoped to one invocation
_DEFER_STRUCTURED_VERIFICATION_KEY = "temp:defer_structured_verification"
_ACTIVE_CALL_TOKEN_KEY = "_active_model_call_token"
_RECORDED_CALL_TOKEN_KEY = "_recorded_model_call_token"
_RECORDED_CALL_LATENCY_MS_KEY = "_recorded_model_call_latency_ms"


def before_agent_telemetry_callback(
    callback_context: CallbackContext,
) -> genai_types.Content | None:
    """Initialize session state keys, bind per-agent identity, and track S2S attestations."""
    state = callback_context.state
    state.setdefault("calculated_variances", {})
    state.setdefault("fetched_metrics", [])
    state.setdefault("retrieved_citations", [])
    state.setdefault(
        "context_cache_stats",
        {"hits": 0, "misses": 0, "cached_tokens_saved": 0},
    )
    state.setdefault("tool_telemetry", [])
    state.setdefault("latency_metrics", [])
    state.setdefault("context_latency_summary", {})
    state.setdefault("react_trace", [])
    state.setdefault("memory_context", {})
    state.setdefault("agent_identity_attestations", [])
    state.setdefault("s2s_audit_trail", [])
    state["_agent_start_monotonic"] = time.perf_counter()

    agent_name = getattr(callback_context, "agent_name", "") or ""
    if isinstance(agent_name, str) and agent_name.strip():
        try:
            spec = resolve_agent_identity(agent_name.strip())
            identity_record = spec.to_public_dict()
            state["active_agent_identity"] = identity_record
            attestations = list(state.get("agent_identity_attestations", []))
            attestations.append(
                {
                    "agent_name": spec.agent_name,
                    "service_account_email": identity_record["service_account_email"],
                    "spiffe_id": identity_record["spiffe_id"],
                    "allowed_tools": identity_record["allowed_tools"],
                }
            )
            state["agent_identity_attestations"] = attestations
        except AgentAuthorizationError:
            logger.warning(
                "Agent '%s' does not have a registered per-agent identity specification.",
                agent_name,
            )

    return None


def retrieval_budget_before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext | Any,
) -> dict[str, Any] | None:
    """Cap 10-K retrieval calls per turn so the analyst answers instead of hitting max_llm_calls.

    Once the budget (`EDGAR_ANALYST_RETRIEVAL_BUDGET`, default 6) is spent, retrieval tools
    return an instruction to write the AnalystReport from the evidence already gathered.
    """
    _ = args
    if getattr(tool, "name", "") not in RETRIEVAL_TOOL_NAMES:
        return None
    budget = env_positive_int(
        "EDGAR_ANALYST_RETRIEVAL_BUDGET",
        DEFAULT_ANALYST_RETRIEVAL_BUDGET,
        minimum=0,
    )
    state = tool_context.state
    used = int(state.get(_RETRIEVAL_CALLS_STATE_KEY, 0) or 0)
    if used >= budget:
        return {
            "status": "retrieval_budget_exhausted",
            "message": (
                f"Retrieval budget of {budget} calls reached for this question. Write the "
                "AnalystReport now using only the passages already retrieved."
            ),
        }
    state[_RETRIEVAL_CALLS_STATE_KEY] = used + 1
    return None


def before_tool_agent_identity_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext | Any,
) -> dict[str, Any] | None:
    """Enforce per-agent least-privilege tool & S2S service boundaries before tool execution."""
    _ = args
    state = getattr(tool_context, "state", None)
    state_dict = state if isinstance(state, dict) or hasattr(state, "get") else {}

    raw_agent_name = getattr(tool_context, "agent_name", None)
    if not isinstance(raw_agent_name, str) or not raw_agent_name.strip():
        active_id = state_dict.get("active_agent_identity") if state_dict else None
        if isinstance(active_id, dict):
            raw_agent_name = active_id.get("agent_name")

    if not isinstance(raw_agent_name, str) or not raw_agent_name.strip():
        return None

    agent_name = raw_agent_name.strip()
    tool_name = getattr(tool, "name", "") or str(tool)

    try:
        s2s_claims = authorize_agent_tool_invocation(
            agent_name=agent_name,
            tool_name=tool_name,
            caller=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
        )
    except AgentAuthorizationError as exc:
        record_security_audit_event(
            event_type="agent_s2s_denied",
            user=None,
            resource=f"tool://{tool_name}",
            status_outcome="denied_403",
            detail=f"agent={agent_name} target={exc.target}: {exc.message}",
        )
        return {
            "status": "forbidden",
            "error": "agent_least_privilege_violation",
            "agent_name": agent_name,
            "tool": tool_name,
            "message": exc.message,
        }

    if state is not None and hasattr(state, "__setitem__"):
        s2s_log = list(state_dict.get("s2s_audit_trail", []))
        s2s_log.append(
            {
                "agent_name": s2s_claims.agent_name,
                "service_account_email": s2s_claims.service_account_email,
                "spiffe_id": s2s_claims.spiffe_id,
                "tool_name": tool_name,
                "target_service": s2s_claims.target_service,
                "audience": s2s_claims.audience,
                "jti": s2s_claims.jti,
            }
        )
        state["s2s_audit_trail"] = s2s_log

    return None


def _seed_before_model_telemetry(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> tuple[str, float]:
    prep_start = time.perf_counter()
    state = callback_context.state
    breakdown = extract_llm_request_context_breakdown(llm_request, state=state)
    agent_name = str(getattr(callback_context, "agent_name", "") or "UnknownAgent")
    req_model = getattr(llm_request, "model", None)
    task_category, resolved_model = resolve_task_and_model_for_agent(
        agent_name=agent_name,
        explicit_model=str(req_model) if req_model else None,
    )
    state["last_model_context_tokens"] = breakdown.total_context_tokens
    state["last_model_context_breakdown"] = breakdown.model_dump(mode="json")
    state["last_model_name"] = resolved_model
    state["last_model_task_category"] = task_category
    return agent_name, prep_start


def _finalize_before_model_timing(state: Any, prep_start: float) -> None:
    context_prep_ms = (time.perf_counter() - prep_start) * 1000.0
    state["_context_prep_latency_ms"] = round(context_prep_ms, 4)
    state["_model_call_start_monotonic"] = time.perf_counter()
    state["_observed_ttft_ms"] = None
    state[_ACTIVE_CALL_TOKEN_KEY] = f"call-{uuid.uuid4().hex[:12]}"


def _extract_latest_user_turn(
    llm_request: LlmRequest,
) -> tuple[genai_types.Content | None, list[genai_types.Part], str]:
    """Return `(content, text_parts, joined_text)` matching `ModelArmorSafetyPlugin` exactly."""
    latest_content: genai_types.Content | None = None
    latest_text_parts: list[genai_types.Part] = []
    for content in llm_request.contents or []:
        if content.role == "user":
            text_parts = [p for p in (content.parts or []) if getattr(p, "text", None)]
            if text_parts:
                latest_content = content
                latest_text_parts = text_parts
    joined = "\n".join(p.text for p in latest_text_parts if p.text)
    return latest_content, latest_text_parts, joined


def before_model_telemetry_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Record input context breakdown & start timer without invoking Model Armor HTTP guardrails.

    Intended for internal utility agents (e.g. `NarrativeEvaluatorAgent`) whose inputs are already
    guardrail-verified analyst reports.
    """
    _, prep_start = _seed_before_model_telemetry(callback_context, llm_request)
    _finalize_before_model_timing(callback_context.state, prep_start)
    return None


def before_model_safety_and_latency_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Enforce Model Armor guardrails and quantify input context components for latency tracking."""
    state = callback_context.state
    agent_name, prep_start = _seed_before_model_telemetry(callback_context, llm_request)

    latest_user_content, latest_text_parts, latest_user_text = _extract_latest_user_turn(
        llm_request
    )

    # Managed Google Cloud Model Armor + SDP + Financial Projection guardrail check.
    # Uses `"\n".join` matching `ModelArmorSafetyPlugin` so pre-verified prompts never trigger
    # duplicate Model Armor HTTP calls.
    if latest_user_text.strip():
        prompt_hash = sha256_text(latest_user_text)
        if state.get("_model_armor_verified_prompt_sha256") != prompt_hash:
            decision = get_model_armor_guardrail().inspect_user_prompt(
                latest_user_text,
                caller_agent=agent_name,
            )
            state["model_armor_last_decision"] = decision.to_audit_dict()
            if decision.blocked:
                logger.warning(
                    "Model Armor blocked user prompt in agent '%s' (categories=%s)",
                    callback_context.agent_name,
                    list(decision.matched_categories),
                )
                state["model_armor_blocked"] = True
                state["model_armor_findings"] = list(decision.matched_categories)
                return LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text=decision.block_message)],
                    )
                )
            if decision.sdp_redacted and latest_user_content is not None:
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
            state["_model_armor_verified_prompt_sha256"] = sha256_text(decision.sanitized_text)

    _finalize_before_model_timing(state, prep_start)
    return None


def verify_narrative_against_calculations(
    response_text: str,
    calculated_variances: dict[str, Any],
    retrieved_citations: list[dict[str, Any]],
    model_latency_ms: float = 0.0,
) -> FactCheckVerification:
    """Deterministically verify that narrative claims cite 10-K sources and align with calculated percentages."""
    citation_matches = _CITATION_TAG_RE.findall(response_text)
    has_citations = len(citation_matches) > 0 if retrieved_citations else True

    expected_pcts: list[float] = []
    for rep in calculated_variances.values():
        for metric_key in (
            "revenue_variance",
            "operating_income_variance",
            "net_income_variance",
            "research_and_development_variance",
        ):
            m_obj = rep.get(metric_key) or {}
            pct = m_obj.get("percentage_delta")
            if pct is not None:
                expected_pcts.append(float(pct))
        for step in rep.get("annual_series") or []:
            for step_key in (
                "revenue_yoy_pct",
                "operating_income_yoy_pct",
                "net_income_yoy_pct",
                "research_and_development_yoy_pct",
            ):
                val = step.get(step_key)
                if val is not None:
                    expected_pcts.append(float(val))

    discrepancies: list[str] = []
    if retrieved_citations and not has_citations and len(response_text.strip()) > 120:
        discrepancies.append(
            "Narrative response omitted explicit 10-K section or chunk_id citations."
        )

    # Check if expected calculated variance percentages are represented when calculations exist
    mentioned_pcts = [float(m) for m in _PERCENT_NUMBER_RE.findall(response_text)]
    if expected_pcts and mentioned_pcts:
        matched_any = any(
            any(abs(abs(m) - abs(exp)) <= 0.6 for m in mentioned_pcts) for exp in expected_pcts
        )
        if not matched_any:
            discrepancies.append(
                f"Narrative percentages {mentioned_pcts[:5]} did not match any deterministic "
                f"calculated variance percentages {[round(x, 2) for x in expected_pcts[:5]]}."
            )

    return FactCheckVerification(
        passed=len(discrepancies) == 0,
        checked_metrics_count=len(expected_pcts),
        matched_citations_count=len(citation_matches),
        has_required_citations=has_citations,
        discrepancies=discrepancies,
        model_latency_ms=round(model_latency_ms, 2),
    )


def _capture_stream_ttft(state: Any, llm_response: LlmResponse) -> bool:
    """Capture wall-clock TTFT on the first non-empty SSE partial chunk; return True if partial."""
    if not getattr(llm_response, "partial", False):
        return False
    start_ts = state.get("_model_call_start_monotonic")
    if state.get("_observed_ttft_ms") is None and start_ts is not None:
        has_signal = False
        if llm_response.content and llm_response.content.parts:
            has_signal = any(
                bool(getattr(p, "text", None)) or getattr(p, "function_call", None) is not None
                for p in llm_response.content.parts
            )
        if has_signal:
            state["_observed_ttft_ms"] = round((time.perf_counter() - float(start_ts)) * 1000.0, 4)
    return True


def _extract_response_parts(llm_response: LlmResponse) -> tuple[str, str, list[str], bool]:
    """Extract `(response_text, thought_text, called_tools, has_function_call)` using `"\n".join`."""
    text_parts: list[str] = []
    thought_parts: list[str] = []
    called_tools: list[str] = []
    has_function_call = False
    if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                has_function_call = True
                called_tools.append(str(getattr(fc, "name", "") or "tool"))
            part_text = getattr(part, "text", None)
            if part_text:
                if getattr(part, "thought", False):
                    thought_parts.append(part_text)
                else:
                    text_parts.append(part_text)
    return "\n".join(text_parts), "\n".join(thought_parts), called_tools, has_function_call


def _infer_react_phase(called_tools: list[str], has_function_call: bool) -> str:
    if not has_function_call:
        return "format_grounded_output"
    if any(tool in _VARIANCE_TOOLS for tool in called_tools):
        return "evaluate_variances"
    if any(tool in RETRIEVAL_TOOL_NAMES for tool in called_tools):
        return "query_cached_10k_context"
    return "scope_and_intent_routing"


def _resolve_session_and_user_ids(
    callback_context: CallbackContext,
    state: Any,
) -> tuple[str | None, str | None]:
    """Read session_id and user_id from public ADK 2.x `CallbackContext` properties."""
    adk_session = getattr(callback_context, "session", None)
    cb_session_id = getattr(adk_session, "id", None) or state.get("session_id")
    cb_user_id = (
        getattr(callback_context, "user_id", None)
        or getattr(adk_session, "user_id", None)
        or state.get("user_id")
    )
    return (
        str(cb_session_id) if cb_session_id else None,
        str(cb_user_id) if cb_user_id else None,
    )


def _resolve_task_category(raw: Any, fallback: ModelTaskCategory) -> ModelTaskCategory:
    if isinstance(raw, str) and raw in TASK_TO_MODEL_TIER:
        return cast(ModelTaskCategory, raw)
    return fallback


def _record_model_telemetry(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
    *,
    thought_text: str,
    called_tools: list[str],
    has_function_call: bool,
) -> tuple[float, ModelContextLatencyRecord | None]:
    """Record latency, usage, cost, and `react_trace` exactly once per model call."""
    state = callback_context.state
    call_token = state.get(_ACTIVE_CALL_TOKEN_KEY)
    if call_token and state.get(_RECORDED_CALL_TOKEN_KEY) == call_token:
        return float(state.get(_RECORDED_CALL_LATENCY_MS_KEY, 0.0) or 0.0), None

    start_ts = state.get("_model_call_start_monotonic")
    raw_inference_ms = (
        (time.perf_counter() - float(start_ts)) * 1000.0 if start_ts is not None else 0.0
    )
    raw_observed_ttft = state.get("_observed_ttft_ms")
    state["_observed_ttft_ms"] = None
    observed_ttft_ms = float(raw_observed_ttft) if raw_observed_ttft is not None else None
    prep_ms = float(state.get("_context_prep_latency_ms", 0.0) or 0.0)

    agent_name = str(getattr(callback_context, "agent_name", "") or "UnknownAgent")
    resp_model = getattr(llm_response, "model_version", None)
    task_category, default_model = resolve_task_and_model_for_agent(agent_name)
    model_name = str(resp_model or state.get("last_model_name") or default_model)
    active_task = _resolve_task_category(state.get("last_model_task_category"), task_category)

    raw_breakdown = state.get("last_model_context_breakdown")
    if isinstance(raw_breakdown, dict):
        base_breakdown = ContextBreakdown.model_validate(raw_breakdown)
    else:
        fallback_toks = int(state.get("last_model_context_tokens", 0) or 0)
        base_breakdown = ContextBreakdown(
            user_prompt_tokens=fallback_toks,
            total_context_tokens=fallback_toks,
            uncached_context_tokens=fallback_toks,
        )

    reconciled_breakdown, output_tokens, thoughts_tokens = reconcile_breakdown_with_usage_metadata(
        base_breakdown, llm_response
    )
    react_phase = _infer_react_phase(called_tools, has_function_call)
    cb_session_id, cb_user_id = _resolve_session_and_user_ids(callback_context, state)

    tracker = get_context_latency_tracker()
    latency_rec = tracker.record_call(
        agent_name=agent_name,
        model_name=model_name,
        task_category=active_task,
        react_phase=react_phase,
        context_breakdown=reconciled_breakdown,
        output_tokens=output_tokens,
        thoughts_tokens=thoughts_tokens,
        raw_inference_latency_ms=raw_inference_ms,
        context_prep_latency_ms=prep_ms,
        observed_ttft_ms=observed_ttft_ms,
        session_id=cb_session_id,
        user_id=cb_user_id,
    )

    latency_log = list(state.get("latency_metrics", []))
    latency_log.append(
        {
            "call_id": latency_rec.call_id,
            "agent_name": agent_name,
            "model_name": latency_rec.model_name,
            "task_category": latency_rec.task_category,
            "react_phase": latency_rec.react_phase,
            "model_latency_ms": round(latency_rec.total_model_latency_ms, 2),
            "context_tokens": latency_rec.context_breakdown.total_context_tokens,
            "cached_context_tokens": latency_rec.context_breakdown.cached_context_tokens,
            "uncached_context_tokens": latency_rec.context_breakdown.uncached_context_tokens,
            "context_prep_latency_ms": latency_rec.context_prep_latency_ms,
            "context_prefill_latency_ms": latency_rec.context_prefill_latency_ms,
            "context_latency_ms": latency_rec.context_latency_ms,
            "generation_latency_ms": latency_rec.generation_latency_ms,
            "ttft_ms": latency_rec.ttft_ms,
            "observed_ttft_ms": (
                round(observed_ttft_ms, 3) if observed_ttft_ms is not None else None
            ),
            "streaming_ttft_captured": observed_ttft_ms is not None,
            "context_latency_ratio": latency_rec.context_latency_ratio,
            "latency_per_1k_context_tokens_ms": latency_rec.latency_per_1k_context_tokens_ms,
            "estimated_cache_latency_saved_ms": latency_rec.estimated_cache_latency_saved_ms,
            "within_ttft_slo": latency_rec.within_ttft_slo,
        }
    )
    state["latency_metrics"] = latency_log
    state["context_latency_summary"] = tracker.summarize().model_dump(mode="json")

    react_log = list(state.get("react_trace", []))
    react_log.append(
        {
            "iteration": len(react_log) + 1,
            "phase": react_phase,
            "agent": agent_name,
            "model_used": model_name,
            "thought": thought_text.strip()
            or (
                f"Selected tool(s): {', '.join(called_tools)}"
                if called_tools
                else "Synthesized final grounded response."
            ),
            "action": ", ".join(called_tools) if called_tools else "final_answer",
            "context_tokens": latency_rec.context_breakdown.total_context_tokens,
            "context_latency_ms": latency_rec.context_latency_ms,
            "duration_ms": round(latency_rec.total_model_latency_ms, 2),
        }
    )
    state["react_trace"] = react_log

    if call_token:
        state[_RECORDED_CALL_TOKEN_KEY] = call_token
        state[_RECORDED_CALL_LATENCY_MS_KEY] = latency_rec.total_model_latency_ms

    return latency_rec.total_model_latency_ms, latency_rec


def _run_response_guardrail_and_fact_check(
    state: Any,
    *,
    agent_name: str,
    response_text: str,
    llm_response: LlmResponse,
    model_latency_ms: float,
) -> LlmResponse | None:
    """Inspect final model output via Model Armor and run optional structured report fact-check."""
    sanitized_return_resp: LlmResponse | None = None
    response_hash = sha256_text(response_text)
    if state.get("_model_armor_verified_response_sha256") != response_hash:
        resp_decision = get_model_armor_guardrail().inspect_model_response(
            response_text,
            caller_agent=agent_name,
        )
        state["model_armor_last_response_decision"] = resp_decision.to_audit_dict()
        if resp_decision.blocked:
            logger.warning(
                "Model Armor blocked model response in agent '%s' (categories=%s)",
                agent_name,
                list(resp_decision.matched_categories),
            )
            state["model_armor_blocked"] = True
            state["model_armor_findings"] = list(resp_decision.matched_categories)
            state["model_armor_response_blocked"] = True
            state["model_armor_response_block_message"] = resp_decision.block_message
            return LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            text=schema_safe_block_text(response_text, resp_decision.block_message)
                        )
                    ],
                )
            )
        if resp_decision.sdp_redacted:
            state["sdp_redacted"] = True
            state["sdp_info_types"] = list(resp_decision.sdp_info_types)
            response_text = resp_decision.sanitized_text
            sanitized_return_resp = LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=resp_decision.sanitized_text)],
                ),
                usage_metadata=llm_response.usage_metadata,
                model_version=llm_response.model_version,
            )
        state["_model_armor_verified_response_sha256"] = sha256_text(response_text)

    if state.get(_DEFER_STRUCTURED_VERIFICATION_KEY):
        return sanitized_return_resp

    calc_vars = dict(state.get("calculated_variances", {}))
    citations = list(state.get("retrieved_citations", []))
    checked_text = response_text
    # Structured AnalystReport answers (JSON) are verified claim by claim and rendered
    # first, so the numeric/citation check runs on what the analyst will actually see.
    structured = (
        parse_analyst_report(response_text.strip())
        if response_text.strip().startswith("{")
        else None
    )
    if structured is not None:
        reports = [v for v in calc_vars.values() if isinstance(v, dict)]
        verified = verify_analyst_report(
            structured,
            variance_reports=reports,
            citations=citations,
            material_flags=state.get("material_variance_flags"),
        )
        checked_text = render_analyst_report_markdown(
            verified, variance_reports=reports, citations=citations
        )
        state["structured_report_markdown"] = checked_text
        state["structured_report_verification"] = verified.verification.model_dump(mode="json")

    verification = verify_narrative_against_calculations(
        response_text=checked_text,
        calculated_variances=calc_vars,
        retrieved_citations=citations,
        model_latency_ms=model_latency_ms,
    )
    state["fact_check_report"] = verification.model_dump(mode="json")
    return sanitized_return_resp


def after_model_telemetry_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """Record TTFT, token/cost telemetry, and `react_trace` without Model Armor or fact-checking.

    Wired onto `NarrativeEvaluatorAgent` so the judge records accurate latency/token/cost
    metrics without triggering redundant Model Armor HTTP calls or `AnalystReport` parsing.
    """
    if _capture_stream_ttft(callback_context.state, llm_response):
        return None
    _, thought_text, called_tools, has_function_call = _extract_response_parts(llm_response)
    _record_model_telemetry(
        callback_context,
        llm_response,
        thought_text=thought_text,
        called_tools=called_tools,
        has_function_call=has_function_call,
    )
    return None


def after_model_fact_check_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """Track context-induced model call latency (including SSE streaming TTFT), record ReAct steps, and fact-check."""
    state = callback_context.state
    if _capture_stream_ttft(state, llm_response):
        return None

    response_text, thought_text, called_tools, has_function_call = _extract_response_parts(
        llm_response
    )
    model_latency_ms, _ = _record_model_telemetry(
        callback_context,
        llm_response,
        thought_text=thought_text,
        called_tools=called_tools,
        has_function_call=has_function_call,
    )

    if response_text.strip() and not has_function_call:
        agent_name = str(getattr(callback_context, "agent_name", "") or "UnknownAgent")
        return _run_response_guardrail_and_fact_check(
            state,
            agent_name=agent_name,
            response_text=response_text,
            llm_response=llm_response,
            model_latency_ms=model_latency_ms,
        )

    return None


class ModelTelemetryPlugin(BasePlugin):
    """ADK `BasePlugin` capturing TTFT & token/cost telemetry ahead of safety plugins.

    In ADK 2.x, if `ModelArmorSafetyPlugin.after_model_callback` blocks or SDP-redacts a response
    and returns a replacement `LlmResponse`, agent-level `after_model_callback` is short-circuited.
    Placing `ModelTelemetryPlugin` before `ModelArmorSafetyPlugin` guarantees every real Gemini
    invocation records TTFT, token usage, and USD cost without double-counting allowed turns.
    """

    def __init__(self, *, name: str = "ModelTelemetryPlugin") -> None:
        super().__init__(name=name)

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        after_model_telemetry_callback(callback_context, llm_response)
        return None
