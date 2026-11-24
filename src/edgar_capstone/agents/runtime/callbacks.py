"""ADK lifecycle callbacks for latency tracking, retrieval budgets, and narrative fact-checking."""

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

from edgar_capstone.agents.core.schemas import (
    ContextBreakdown,
    FactCheckVerification,
    ModelContextLatencyRecord,
    ModelTaskCategory,
)
from edgar_capstone.agents.runtime.context_memory import logical_session_id
from edgar_capstone.agents.runtime.latency_tracker import (
    extract_llm_request_context_breakdown,
    get_context_latency_tracker,
    reconcile_breakdown_with_usage_metadata,
)
from edgar_capstone.agents.runtime.model_config import (
    TASK_TO_MODEL_TIER,
    resolve_task_and_model_for_agent,
)
from edgar_capstone.agents.tools.runtime import current_tool_turn_context, emit_tool_progress
from edgar_capstone.common.domain import (
    CORE_METRICS,
    RETRIEVAL_TOOL_NAMES,
    VARIANCE_TOOL_NAMES,
)
from edgar_capstone.common.env import env_positive_int

logger = logging.getLogger(__name__)

_CITATION_TAG_RE = re.compile(
    r"(?:\b[A-Z]{1,5}_\d{4}_item_(?:1a|7)_\d{4}\b|\[[A-Z.\-]{1,6}\s+FY\d{4}[^\]]*\]|Item\s+(?:1A|7)\b)",
    re.IGNORECASE,
)
_PERCENT_NUMBER_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")

DEFAULT_ANALYST_RETRIEVAL_BUDGET = 6
_RETRIEVAL_CALLS_STATE_KEY = "temp:retrieval_calls_used"  # temp: = scoped to one invocation
_ACTIVE_CALL_TOKEN_KEY = "_active_model_call_token"
_RECORDED_CALL_TOKEN_KEY = "_recorded_model_call_token"
_RECORDED_CALL_LATENCY_MS_KEY = "_recorded_model_call_latency_ms"


def before_agent_telemetry_callback(
    callback_context: CallbackContext,
) -> genai_types.Content | None:
    """Initialize per-invocation telemetry and tool session state keys."""
    state = callback_context.state
    state.setdefault("calculated_variances", {})
    state.setdefault("fetched_metrics", [])
    state.setdefault("retrieved_citations", [])
    state.setdefault(
        "context_cache_stats",
        {"hits": 0, "misses": 0},
    )
    state.setdefault("tool_telemetry", [])
    state.setdefault("latency_metrics", [])
    state.setdefault("react_trace", [])
    state.setdefault("memory_context", {})
    state["_agent_start_monotonic"] = time.perf_counter()
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


_AGENT_TOOL_PROGRESS_LABELS: dict[str, str] = {
    "calculate_period_variances": "Computing deterministic period-over-period variances",
    "flag_material_variances": "Flagging material variances and attaching MD&A explanations",
    "compare_disclosures_across_years": "Comparing 10-K disclosure wording across fiscal years",
    "sector_thematic_map_reduce": "Running full-sector thematic map-reduce",
    "query_analytics_view": "Querying BigQuery analytics view",
    "fetch_financial_metrics": "Fetching structured XBRL financial metrics",
    "search_10k_chunks_rag": "Searching 10-K passages",
    "query_cached_10k_context": "Reading cached 10-K section passages",
}


def agent_tool_progress_before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext | Any,
) -> dict[str, Any] | None:
    """Stream an SSE `agent_tool` progress event whenever the analyst invokes a tool.

    Runs last in the before-tool chain, so it only fires for calls that passed the identity and
    retrieval-budget checks. Never short-circuits the tool.
    """
    _ = tool_context
    name = str(getattr(tool, "name", "") or "")
    label = _AGENT_TOOL_PROGRESS_LABELS.get(name, f"Running {name}")
    scope_bits = [
        str(args.get(k))
        for k in ("tickers", "gics_sector", "explore_name")
        if isinstance(args, dict) and args.get(k) not in (None, "", "ALL")
    ]
    suffix = f" ({', '.join(scope_bits)})" if scope_bits else ""
    emit_tool_progress("agent_tool", f"{label}{suffix}...", {"tool": name})
    turn_ctx = current_tool_turn_context()
    if turn_ctx is not None:
        call_id = str(getattr(tool_context, "function_call_id", "") or name)
        turn_ctx.tool_started_at[call_id] = time.perf_counter()
    return None


def agent_tool_timing_after_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext | Any,
    tool_response: Any,
) -> dict[str, Any] | None:
    """Record wall-clock duration of each agent-invoked tool on the active turn context."""
    _ = args, tool_response
    turn_ctx = current_tool_turn_context()
    if turn_ctx is None:
        return None
    name = str(getattr(tool, "name", "") or "")
    call_id = str(getattr(tool_context, "function_call_id", "") or name)
    started = turn_ctx.tool_started_at.pop(call_id, None)
    if started is not None:
        turn_ctx.tool_durations_ms.append(
            (name, round((time.perf_counter() - started) * 1000.0, 2))
        )
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


def before_model_telemetry_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Quantify input context components and start timing timers before a model call.

    Prompt Model Armor / SDP guardrails are enforced at the ADK plugin layer by
    `ModelArmorSafetyPlugin`, keeping this callback focused on context telemetry.
    """
    _, prep_start = _seed_before_model_telemetry(callback_context, llm_request)
    _finalize_before_model_timing(callback_context.state, prep_start)
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
        for metric in CORE_METRICS:
            m_obj = rep.get(f"{metric}_variance") or {}
            pct = m_obj.get("percentage_delta")
            if pct is not None:
                expected_pcts.append(float(pct))
        for step in rep.get("annual_series") or []:
            for metric in CORE_METRICS:
                val = step.get(f"{metric}_yoy_pct")
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
    if any(tool in VARIANCE_TOOL_NAMES for tool in called_tools):
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
        logical_session_id(str(cb_session_id)) if cb_session_id else None,
        str(cb_user_id) if cb_user_id else None,
    )


def _resolve_task_category(raw: Any, fallback: ModelTaskCategory) -> ModelTaskCategory:
    if isinstance(raw, str) and raw in TASK_TO_MODEL_TIER:
        return cast(ModelTaskCategory, raw)
    return fallback


def _build_latency_log_entry(
    latency_rec: ModelContextLatencyRecord,
    *,
    agent_name: str,
    observed_ttft_ms: float | None,
) -> dict[str, Any]:
    return {
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
        "observed_ttft_ms": (round(observed_ttft_ms, 3) if observed_ttft_ms is not None else None),
        "streaming_ttft_captured": observed_ttft_ms is not None,
        "context_latency_ratio": latency_rec.context_latency_ratio,
        "latency_per_1k_context_tokens_ms": latency_rec.latency_per_1k_context_tokens_ms,
        "estimated_cache_latency_saved_ms": latency_rec.estimated_cache_latency_saved_ms,
        "within_ttft_slo": latency_rec.within_ttft_slo,
    }


def _build_react_trace_entry(
    *,
    iteration: int,
    react_phase: str,
    agent_name: str,
    model_name: str,
    thought_text: str,
    called_tools: list[str],
    latency_rec: ModelContextLatencyRecord,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
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
        _build_latency_log_entry(
            latency_rec, agent_name=agent_name, observed_ttft_ms=observed_ttft_ms
        )
    )
    state["latency_metrics"] = latency_log

    react_log = list(state.get("react_trace", []))
    react_log.append(
        _build_react_trace_entry(
            iteration=len(react_log) + 1,
            react_phase=react_phase,
            agent_name=agent_name,
            model_name=model_name,
            thought_text=thought_text,
            called_tools=called_tools,
            latency_rec=latency_rec,
        )
    )
    state["react_trace"] = react_log

    if call_token:
        state[_RECORDED_CALL_TOKEN_KEY] = call_token
        state[_RECORDED_CALL_LATENCY_MS_KEY] = latency_rec.total_model_latency_ms

    return latency_rec.total_model_latency_ms, latency_rec


def after_model_telemetry_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """Record TTFT, token/cost telemetry, and `react_trace` for every ADK model call."""
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


class ModelTelemetryPlugin(BasePlugin):
    """ADK `BasePlugin` capturing TTFT & token/cost telemetry for agents without per-agent callbacks."""

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
