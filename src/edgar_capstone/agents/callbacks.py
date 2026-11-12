"""ADK lifecycle callbacks for latency tracking, prompt safety (Model Armor), and fact-checking."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from edgar_capstone.agents.schemas import FactCheckVerification
from edgar_capstone.vector_db.chunker import estimate_token_count

logger = logging.getLogger(__name__)

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions", re.IGNORECASE),
    re.compile(r"bypass\s+(?:strict\s+)?grounding", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"fabricate\s+financial\s+numbers", re.IGNORECASE),
)

_CITATION_TAG_RE = re.compile(
    r"(?:\b[A-Z]{1,5}_\d{4}_item_(?:1a|7)_\d{4}\b|\[[A-Z.\-]{1,6}\s+FY\d{4}[^\]]*\]|Item\s+(?:1A|7)\b)",
    re.IGNORECASE,
)
_PERCENT_NUMBER_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")


def before_agent_telemetry_callback(
    callback_context: CallbackContext,
) -> genai_types.Content | None:
    """Initialize session state keys for variance grounding, citations, and latency tracking."""
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
    state["_agent_start_monotonic"] = time.perf_counter()
    return None


def before_model_safety_and_latency_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Enforce Model Armor prompt-injection guardrails and record context token latency."""
    state = callback_context.state
    state["_model_call_start_monotonic"] = time.perf_counter()

    # Estimate total context tokens sent to the model
    prompt_parts_text: list[str] = []
    latest_user_text = ""
    for content in llm_request.contents or []:
        for part in content.parts or []:
            if part.text:
                prompt_parts_text.append(part.text)
                if content.role == "user":
                    latest_user_text = part.text

    context_tokens = estimate_token_count("\n".join(prompt_parts_text))
    state["last_model_context_tokens"] = context_tokens

    # Model Armor check for prompt injection or grounding bypass attempts
    if latest_user_text:
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(latest_user_text):
                logger.warning(
                    "Model Armor blocked prompt injection attempt in agent '%s'",
                    callback_context.agent_name,
                )
                state["model_armor_blocked"] = True
                return LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[
                            genai_types.Part(
                                text=(
                                    "Request blocked by SEC Analyst safety and grounding guardrails: "
                                    "I can only answer financial and SEC 10-K analysis queries grounded "
                                    "in verified EDGAR filings and deterministic variance calculations."
                                )
                            )
                        ],
                    )
                )

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
            any(abs(abs(m) - abs(exp)) <= 0.6 for m in mentioned_pcts)
            for exp in expected_pcts
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


def after_model_fact_check_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """Track model call latency and run post-generation grounding & calculation fact checks."""
    state = callback_context.state
    start_ts = state.get("_model_call_start_monotonic")
    latency_ms = (
        (time.perf_counter() - float(start_ts)) * 1000.0 if start_ts is not None else 0.0
    )

    latency_log = list(state.get("latency_metrics", []))
    latency_log.append(
        {
            "agent_name": callback_context.agent_name,
            "model_latency_ms": round(latency_ms, 2),
            "context_tokens": int(state.get("last_model_context_tokens", 0)),
        }
    )
    state["latency_metrics"] = latency_log

    # Only run narrative fact-check on final text responses (not intermediate tool-call turns)
    response_text = ""
    has_function_call = False
    if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            if getattr(part, "function_call", None) is not None:
                has_function_call = True
            if part.text:
                response_text += part.text

    if response_text.strip() and not has_function_call:
        calc_vars = dict(state.get("calculated_variances", {}))
        citations = list(state.get("retrieved_citations", []))
        verification = verify_narrative_against_calculations(
            response_text=response_text,
            calculated_variances=calc_vars,
            retrieved_citations=citations,
            model_latency_ms=latency_ms,
        )
        state["fact_check_report"] = verification.model_dump(mode="json")

    return None
