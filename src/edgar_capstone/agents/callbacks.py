"""ADK lifecycle callbacks for latency tracking, prompt safety (Model Armor), and fact-checking."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.genai import types as genai_types

from edgar_capstone.agents.context_memory import (
    extract_llm_request_context_breakdown,
    get_context_latency_tracker,
    reconcile_breakdown_with_usage_metadata,
    resolve_task_and_model_for_agent,
)
from edgar_capstone.agents.schemas import ContextBreakdown, FactCheckVerification
from edgar_capstone.security.agent_identity import (
    AgentAuthorizationError,
    authorize_agent_tool_invocation,
    resolve_agent_identity,
)
from edgar_capstone.security.auth import record_security_audit_event
from edgar_capstone.security.rbac import CUSTOM_RUNTIME_SA_ACCOUNT_ID

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


def before_tool_agent_identity_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: Any,
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


def before_model_safety_and_latency_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Enforce Model Armor guardrails and quantify input context components for latency tracking."""
    prep_start = time.perf_counter()
    state = callback_context.state

    latest_user_text = ""
    for content in llm_request.contents or []:
        for part in content.parts or []:
            if part.text and content.role == "user":
                latest_user_text = part.text

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

    context_prep_ms = (time.perf_counter() - prep_start) * 1000.0
    state["_context_prep_latency_ms"] = round(context_prep_ms, 4)
    state["_model_call_start_monotonic"] = time.perf_counter()
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
    """Track context-induced model call latency, record ReAct trace steps, and run grounding fact checks."""
    state = callback_context.state
    start_ts = state.get("_model_call_start_monotonic")
    raw_inference_ms = (
        (time.perf_counter() - float(start_ts)) * 1000.0 if start_ts is not None else 0.0
    )
    prep_ms = float(state.get("_context_prep_latency_ms", 0.0) or 0.0)

    agent_name = str(getattr(callback_context, "agent_name", "") or "UnknownAgent")
    resp_model = getattr(llm_response, "model_version", None)
    task_category, default_model = resolve_task_and_model_for_agent(agent_name)
    model_name = str(resp_model or state.get("last_model_name") or default_model)
    active_task = state.get("last_model_task_category") or task_category

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

    reconciled_breakdown, output_tokens, thoughts_tokens = (
        reconcile_breakdown_with_usage_metadata(base_breakdown, llm_response)
    )

    response_text = ""
    thought_text = ""
    called_tools: list[str] = []
    has_function_call = False
    if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                has_function_call = True
                called_tools.append(str(getattr(fc, "name", "") or "tool"))
            if part.text:
                if getattr(part, "thought", False):
                    thought_text += part.text
                else:
                    response_text += part.text

    # Infer ReAct phase from tool calls or final answer synthesis
    if has_function_call:
        if any(
            t in ("calculate_period_variances", "fetch_financial_metrics", "query_looker_financial_explore")
            for t in called_tools
        ):
            react_phase = "evaluate_variances"
        elif any(
            t in ("load_cached_10k_section", "query_cached_10k_context", "search_10k_chunks_rag")
            for t in called_tools
        ):
            react_phase = "query_cached_10k_context"
        else:
            react_phase = "scope_and_intent_routing"
    else:
        react_phase = "format_grounded_output"

    tracker = get_context_latency_tracker()
    latency_rec = tracker.record_call(
        agent_name=agent_name,
        model_name=model_name,
        task_category=active_task,  # type: ignore[arg-type]
        react_phase=react_phase,
        context_breakdown=reconciled_breakdown,
        output_tokens=output_tokens,
        thoughts_tokens=thoughts_tokens,
        raw_inference_latency_ms=raw_inference_ms,
        context_prep_latency_ms=prep_ms,
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
            "thought": thought_text.strip() or (
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

    # Only run narrative fact-check on final text responses (not intermediate tool-call turns)
    if response_text.strip() and not has_function_call:
        calc_vars = dict(state.get("calculated_variances", {}))
        citations = list(state.get("retrieved_citations", []))
        verification = verify_narrative_against_calculations(
            response_text=response_text,
            calculated_variances=calc_vars,
            retrieved_citations=citations,
            model_latency_ms=latency_rec.total_model_latency_ms,
        )
        state["fact_check_report"] = verification.model_dump(mode="json")

    return None
