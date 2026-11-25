"""Helpers for test LLMs that answer FinancialAnalystAgent requests with a structured AnalystReport."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from edgar_capstone.agents.core.schemas import AnalystReport
from edgar_capstone.agents.runtime.model_config import (
    FINANCIAL_REASONING_MODEL,
    FLASH_CODING_AND_EVAL_MODEL,
)

try:
    from google.adk.models.base_llm import LlmCapabilities
except ImportError:  # pragma: no cover - older google-adk fallback
    LlmCapabilities = None  # type: ignore[assignment,misc]

_LEADS_BLOCK_RE = re.compile(r"## 10-K Evidence Leads[^\n]*\n```json\n(.*?)\n```", re.S)


def request_text(llm_request: LlmRequest) -> str:
    return "\n".join(
        p.text for c in (llm_request.contents or []) for p in (c.parts or []) if p.text
    )


def evidence_leads(llm_request: LlmRequest) -> list[dict[str, Any]]:
    """Evidence leads (chunk_id + excerpt) from the latest synthesis prompt in `llm_request`.

    Uses the last block so multi-turn persistent sessions with prior turns in history
    inspect the current turn's evidence leads rather than Turn 1's.
    """
    matches = _LEADS_BLOCK_RE.findall(request_text(llm_request))
    if not matches:
        return []
    return [lead for lead in json.loads(matches[-1]) if lead.get("chunk_id")]


_CHUNK_HEADER_RE = re.compile(r"^\s*\[[^\]\n]*\]\s*")


def verbatim_quote(text: str, words: int = 14) -> str:
    """First `words` words of a chunk's body (skipping its `[TICKER | FY | Section]` header)."""
    body = _CHUNK_HEADER_RE.sub("", str(text or ""), count=1)
    return " ".join(body.split()[:words])


def finding_from_citation(
    citation: dict[str, Any], statement: str, *, words: int = 14
) -> dict[str, Any]:
    return {
        "statement": statement,
        "ticker": str(citation.get("ticker") or "NVDA"),
        "fiscal_year": int(citation.get("fiscal_year") or 2026),
        "chunk_ids": [str(citation["chunk_id"])],
        "evidence_quote": verbatim_quote(str(citation.get("excerpt") or ""), words),
    }


def report_json(
    *,
    headline: str,
    intent: str | None = None,
    findings: list[dict[str, Any]] | None = None,
    key_metrics: list[tuple[str, str]] | None = None,
) -> str:
    """Serialize a schema-valid AnalystReport, as constrained decoding would return it."""
    kwargs: dict[str, Any] = {
        "headline": headline,
        "key_metrics": [{"ticker": t, "metric": m} for t, m in (key_metrics or [])],
        "findings": findings or [],
    }
    if intent is not None:
        kwargs["intent"] = intent
    return AnalystReport(**kwargs).model_dump_json()


class ScriptedBaseLlm(BaseLlm):
    """BaseLlm subclass declaring explicit ADK capabilities to silence name-detection warnings."""

    if LlmCapabilities is not None:

        @property
        def capabilities(self) -> Any:
            return LlmCapabilities(output_schema_and_tools=True)


class ScriptedAnalystLlm(ScriptedBaseLlm):
    """Deterministic `FinancialAnalystAgent` LLM that optionally runs tool calls then emits a report."""

    model: str = FINANCIAL_REASONING_MODEL
    tool_calls: list[tuple[str, dict[str, Any]]] = []
    headline: str = "Grounded financial analysis from 10-K evidence."
    intent: str | None = "longitudinal_single_company"
    key_metrics: list[tuple[str, str]] = []
    statement: str = "Management commentary discusses core drivers in the Form 10-K."
    cached_prompt_tokens: int = 2400
    prompt_tokens: int = 3600
    output_tokens: int = 220
    thoughts_tokens: int = 40
    call_count: int = 0

    def _has_tool_responses(self, llm_request: LlmRequest) -> bool:
        return any(
            getattr(p, "function_response", None) is not None
            for c in (llm_request.contents or [])
            for p in (c.parts or [])
        )

    def _last_user_prompt_has_tool_responses(self, llm_request: LlmRequest) -> bool:
        # Multi-turn sessions keep prior turns in history; only check parts since the latest user prompt.
        contents = list(llm_request.contents or [])
        last_lead_idx = -1
        for i, c in enumerate(contents):
            txt = " ".join(p.text for p in (c.parts or []) if p.text)
            if "## 10-K Evidence Leads" in txt:
                last_lead_idx = i
        tail = contents[last_lead_idx + 1 :] if last_lead_idx >= 0 else contents
        return any(
            getattr(p, "function_response", None) is not None for c in tail for p in (c.parts or [])
        )

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        self.call_count += 1
        if self.tool_calls and not self._last_user_prompt_has_tool_responses(llm_request):
            parts = [
                genai_types.Part(function_call=genai_types.FunctionCall(name=name, args=args))
                for name, args in self.tool_calls
            ]
            yield LlmResponse(
                model_version=self.model,
                partial=False,
                content=genai_types.Content(role="model", parts=parts),
                usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=self.prompt_tokens,
                    cached_content_token_count=self.cached_prompt_tokens,
                    candidates_token_count=64,
                    thoughts_token_count=self.thoughts_tokens,
                ),
            )
            return

        leads = evidence_leads(llm_request)
        findings = [finding_from_citation(leads[0], self.statement)] if leads else []
        payload = report_json(
            headline=self.headline,
            intent=self.intent,
            findings=findings,
            key_metrics=self.key_metrics,
        )
        yield LlmResponse(
            model_version=self.model,
            partial=False,
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(text="Synthesizing grounded 10-K answer.", thought=True),
                    genai_types.Part(text=payload),
                ],
            ),
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=self.prompt_tokens,
                cached_content_token_count=self.cached_prompt_tokens,
                candidates_token_count=self.output_tokens,
                thoughts_token_count=self.thoughts_tokens,
            ),
        )


class ScriptedJudgeLlm(ScriptedBaseLlm):
    """Deterministic `NarrativeEvaluatorAgent` LLM returning structured judge scores."""

    model: str = FLASH_CODING_AND_EVAL_MODEL
    overall_score: float = 0.94
    numeric_fidelity_score: float = 0.98
    citation_grounding_score: float = 0.96
    explanation_quality_score: float = 0.90
    grade: str = "pass"
    rationale: str = "All metrics and 10-K citations verified against retrieved passages."
    call_count: int = 0

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        self.call_count += 1
        body = json.dumps(
            {
                "overall_score": self.overall_score,
                "numeric_fidelity_score": self.numeric_fidelity_score,
                "citation_grounding_score": self.citation_grounding_score,
                "explanation_quality_score": self.explanation_quality_score,
                "grade": self.grade,
                "rationale": self.rationale,
            }
        )
        yield LlmResponse(
            model_version=self.model,
            partial=False,
            content=genai_types.Content(role="model", parts=[genai_types.Part(text=body)]),
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=1200,
                cached_content_token_count=0,
                candidates_token_count=90,
                thoughts_token_count=0,
            ),
        )


def patch_http_chat_llms(
    monkeypatch: Any,
    *,
    reasoning_llm: BaseLlm,
    judge_llm: BaseLlm | None = None,
    skip_background_judge: bool = True,
) -> None:
    """Inject scripted ADK LLM(s) into `POST /api/v1/chat` & `/chat/stream` without mocking the pipeline."""
    from edgar_capstone.api.routers import chat as chat_router

    real_fn = chat_router.run_conversational_rag_turn

    def _wrapped(**kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("reasoning_llm", reasoning_llm)
        if judge_llm is not None:
            kwargs.setdefault("judge_llm", judge_llm)
        kwargs.setdefault("skip_background_judge", skip_background_judge)
        return real_fn(**kwargs)

    monkeypatch.setattr(chat_router, "run_conversational_rag_turn", _wrapped)
