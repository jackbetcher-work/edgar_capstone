"""Context & Memory Strategy, Model Tier Routing, and Context-Induced Latency Tracking.

Implements:
1. Dual-Model Specialization Strategy:
   - `gemini-2.5-pro`: Complex financial reasoning, multi-period variance synthesis, and ReAct orchestration.
   - `gemini-3.5-flash`: Coding assistance, LLM-as-a-Judge evaluation tasks, and context compaction summarization.
2. Context-Induced Latency Tracking:
   - Decomposes every model call into context preparation latency, context prefill latency (cached vs. uncached),
     and output token generation latency, tracking TTFT SLO compliance (<= 3.0s p95) and KV-cache savings.
3. Session State, Context Caching, Compaction & Long-Term Memory:
   - Configures ADK `ContextCacheConfig`, token-triggered `EventsCompactionConfig` (`gemini-3.5-flash`),
     `InMemorySessionService` / `InMemoryMemoryService`, and multi-turn conversational entity/variance recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
import threading
import time
from typing import Any
import uuid

from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.memory import InMemoryMemoryService
from google.adk.models import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from edgar_capstone.agents.schemas import (
    ContextBreakdown,
    ContextLatencySummary,
    ModelContextLatencyRecord,
    ModelTaskCategory,
)
from edgar_capstone.vector_db.chunker import estimate_token_count

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Model Tier Constants & Task Specialization Strategy
# ---------------------------------------------------------------------------

FINANCIAL_REASONING_MODEL: str = os.environ.get("EDGAR_ANALYST_MODEL", "gemini-2.5-pro")
FLASH_CODING_AND_EVAL_MODEL: str = os.environ.get("EDGAR_FLASH_MODEL", "gemini-3.5-flash")

TTFT_SLO_TARGET_MS: float = float(os.environ.get("EDGAR_TTFT_SLO_MS", "3000.0"))
DEFAULT_CONTEXT_CACHE_MIN_TOKENS: int = 2048
DEFAULT_CONTEXT_CACHE_TTL_SECONDS: int = 1800
DEFAULT_CONTEXT_CACHE_INTERVALS: int = 10
DEFAULT_COMPACTION_TOKEN_THRESHOLD: int = 32000
DEFAULT_COMPACTION_RETENTION_SIZE: int = 5


@dataclass(frozen=True)
class ModelPerformanceProfile:
    """Calibrated token-throughput characteristics for attributing prefill vs decode latency."""

    model_name: str
    tier: str
    primary_use_cases: tuple[str, ...]
    uncached_prefill_ms_per_token: float
    cached_prefill_discount_factor: float  # Fraction of uncached prefill cost when cached (e.g. 0.20 = 5x faster)
    decode_ms_per_token: float


MODEL_PROFILES: dict[str, ModelPerformanceProfile] = {
    "gemini-2.5-pro": ModelPerformanceProfile(
        model_name="gemini-2.5-pro",
        tier="complex_financial_reasoning",
        primary_use_cases=(
            "financial_reasoning",
            "variance_explanation",
            "peer_comparison",
            "thematic_synthesis",
            "orchestration",
        ),
        uncached_prefill_ms_per_token=0.14,
        cached_prefill_discount_factor=0.20,
        decode_ms_per_token=1.85,
    ),
    "gemini-3.5-flash": ModelPerformanceProfile(
        model_name="gemini-3.5-flash",
        tier="coding_assistance_and_evaluation",
        primary_use_cases=(
            "evaluation",
            "llm_as_judge",
            "coding_assistance",
            "context_compaction",
            "context_cache_warmup",
        ),
        uncached_prefill_ms_per_token=0.045,
        cached_prefill_discount_factor=0.18,
        decode_ms_per_token=0.55,
    ),
}

TASK_TO_MODEL_TIER: dict[ModelTaskCategory, str] = {
    "financial_reasoning": "complex_financial_reasoning",
    "variance_explanation": "complex_financial_reasoning",
    "peer_comparison": "complex_financial_reasoning",
    "thematic_synthesis": "complex_financial_reasoning",
    "orchestration": "complex_financial_reasoning",
    "evaluation": "coding_assistance_and_evaluation",
    "llm_as_judge": "coding_assistance_and_evaluation",
    "coding_assistance": "coding_assistance_and_evaluation",
    "context_compaction": "coding_assistance_and_evaluation",
    "context_cache_warmup": "coding_assistance_and_evaluation",
}

AGENT_DEFAULT_TASK_CATEGORY: dict[str, ModelTaskCategory] = {
    "RootOrchestrator": "orchestration",
    "FinancialAnalystAgent": "financial_reasoning",
    "ContextCacheAgent": "context_cache_warmup",
    "ContextCacheManager": "context_cache_warmup",
    "NarrativeEvaluatorAgent": "evaluation",
    "CodingAssistantAgent": "coding_assistance",
}


def resolve_model_for_task(task_category: ModelTaskCategory) -> str:
    """Resolve the active Gemini model identifier for a given workload category."""
    tier = TASK_TO_MODEL_TIER.get(task_category, "complex_financial_reasoning")
    if tier == "coding_assistance_and_evaluation":
        return os.environ.get("EDGAR_FLASH_MODEL", FLASH_CODING_AND_EVAL_MODEL)
    return os.environ.get("EDGAR_ANALYST_MODEL", FINANCIAL_REASONING_MODEL)


def resolve_task_and_model_for_agent(
    agent_name: str,
    explicit_model: str | None = None,
) -> tuple[ModelTaskCategory, str]:
    """Return `(task_category, model_name)` for an ADK agent."""
    task_cat = AGENT_DEFAULT_TASK_CATEGORY.get(agent_name, "financial_reasoning")
    if explicit_model and explicit_model.strip():
        return task_cat, explicit_model.strip()
    return task_cat, resolve_model_for_task(task_cat)


def _get_profile_for_model(model_name: str) -> ModelPerformanceProfile:
    norm = model_name.strip().lower()
    if "flash" in norm:
        return MODEL_PROFILES["gemini-3.5-flash"]
    return MODEL_PROFILES["gemini-2.5-pro"]


def get_model_strategy_metadata() -> dict[str, Any]:
    """Return public metadata describing the dual-model specialization & context strategy."""
    fin_model = resolve_model_for_task("financial_reasoning")
    flash_model = resolve_model_for_task("evaluation")
    return {
        "financial_reasoning_model": fin_model,
        "coding_and_evaluation_model": flash_model,
        "task_routing": {
            task: resolve_model_for_task(task)  # type: ignore[arg-type]
            for task in TASK_TO_MODEL_TIER
        },
        "context_cache_config": {
            "min_tokens": DEFAULT_CONTEXT_CACHE_MIN_TOKENS,
            "ttl_seconds": DEFAULT_CONTEXT_CACHE_TTL_SECONDS,
            "cache_intervals": DEFAULT_CONTEXT_CACHE_INTERVALS,
        },
        "events_compaction_config": {
            "token_threshold": DEFAULT_COMPACTION_TOKEN_THRESHOLD,
            "event_retention_size": DEFAULT_COMPACTION_RETENTION_SIZE,
            "summarizer_model": flash_model,
        },
        "ttft_slo_target_ms": TTFT_SLO_TARGET_MS,
    }


# ---------------------------------------------------------------------------
# 2. Context Breakdown & Context-Induced Latency Tracking
# ---------------------------------------------------------------------------


def extract_llm_request_context_breakdown(
    llm_request: LlmRequest,
    state: Any | None = None,
) -> ContextBreakdown:
    """Inspect an ADK `LlmRequest` and session state to quantify input context components."""
    state_dict = state if isinstance(state, dict) or hasattr(state, "get") else {}

    # 1. System instruction tokens
    sys_tokens = 0
    cfg = getattr(llm_request, "config", None)
    sys_inst = getattr(cfg, "system_instruction", None) if cfg is not None else None
    if isinstance(sys_inst, str) and sys_inst.strip():
        sys_tokens = estimate_token_count(sys_inst)
    elif sys_inst is not None and hasattr(sys_inst, "parts"):
        sys_text = "\n".join(
            p.text for p in (getattr(sys_inst, "parts", None) or []) if getattr(p, "text", None)
        )
        sys_tokens = estimate_token_count(sys_text)

    # 2. Classify conversation contents into active user prompt, history/memory, variance JSON, and 10-K context
    contents = list(getattr(llm_request, "contents", None) or [])
    last_user_idx = -1
    for idx in range(len(contents) - 1, -1, -1):
        if getattr(contents[idx], "role", None) == "user":
            has_text = any(
                bool(getattr(p, "text", None))
                for p in (getattr(contents[idx], "parts", None) or [])
            )
            if has_text:
                last_user_idx = idx
                break

    user_prompt_tokens = 0
    memory_and_history_tokens = 0
    variance_payload_tokens = 0
    cached_10k_tokens = 0
    uncached_10k_and_tool_tokens = 0

    for idx, content in enumerate(contents):
        role = getattr(content, "role", "") or ""
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            fn_resp = getattr(part, "function_response", None)
            fn_call = getattr(part, "function_call", None)

            if part_text:
                t_count = estimate_token_count(part_text)
                if idx == last_user_idx and role == "user":
                    user_prompt_tokens += t_count
                else:
                    memory_and_history_tokens += t_count

            if fn_call is not None:
                fc_str = f"{getattr(fn_call, 'name', '')}({json.dumps(getattr(fn_call, 'args', {}) or {}, default=str)})"
                memory_and_history_tokens += estimate_token_count(fc_str)

            if fn_resp is not None:
                resp_name = str(getattr(fn_resp, "name", "") or "")
                resp_payload = getattr(fn_resp, "response", None) or {}
                serialized = json.dumps(resp_payload, default=str)
                resp_tokens = estimate_token_count(serialized)
                if resp_name in (
                    "calculate_period_variances",
                    "fetch_financial_metrics",
                    "query_looker_financial_explore",
                ):
                    variance_payload_tokens += resp_tokens
                elif resp_name in ("load_cached_10k_section", "query_cached_10k_context"):
                    if isinstance(resp_payload, dict) and resp_payload.get("cache_hit"):
                        cached_10k_tokens += resp_tokens
                    else:
                        uncached_10k_and_tool_tokens += resp_tokens
                else:
                    uncached_10k_and_tool_tokens += resp_tokens

    # Also account for session memory injected via state if not already in contents
    mem_ctx = state_dict.get("memory_context") if state_dict else None
    if isinstance(mem_ctx, dict):
        injected_mem_tokens = int(mem_ctx.get("memory_tokens", 0) or 0)
        if injected_mem_tokens > 0 and memory_and_history_tokens == 0:
            memory_and_history_tokens += injected_mem_tokens

    # Check ADK-level cache metadata on LlmRequest
    adk_cacheable_tokens = int(getattr(llm_request, "cacheable_contents_token_count", 0) or 0)
    has_cache_metadata = getattr(llm_request, "cache_metadata", None) is not None

    total_tokens = (
        sys_tokens
        + memory_and_history_tokens
        + variance_payload_tokens
        + cached_10k_tokens
        + uncached_10k_and_tool_tokens
        + user_prompt_tokens
    )
    cached_total = cached_10k_tokens
    if has_cache_metadata and adk_cacheable_tokens > cached_total:
        cached_total = min(total_tokens, adk_cacheable_tokens)

    cache_stats = state_dict.get("context_cache_stats") if state_dict else None
    cache_hit = cached_total > 0 or (
        isinstance(cache_stats, dict) and int(cache_stats.get("hits", 0) or 0) > 0 and cached_10k_tokens > 0
    )
    uncached_total = max(0, total_tokens - cached_total)

    return ContextBreakdown(
        system_instruction_tokens=sys_tokens,
        memory_and_history_tokens=memory_and_history_tokens,
        variance_payload_tokens=variance_payload_tokens,
        cached_10k_tokens=cached_10k_tokens,
        uncached_10k_and_tool_tokens=uncached_10k_and_tool_tokens,
        user_prompt_tokens=user_prompt_tokens,
        total_context_tokens=total_tokens,
        cached_context_tokens=cached_total,
        uncached_context_tokens=uncached_total,
        cache_hit=cache_hit,
    )


def reconcile_breakdown_with_usage_metadata(
    breakdown: ContextBreakdown,
    llm_response: LlmResponse | None,
) -> tuple[ContextBreakdown, int, int]:
    """Reconcile estimated context breakdown with authoritative Gemini `usage_metadata` when available."""
    output_tokens = 0
    thoughts_tokens = 0

    if llm_response is not None:
        if llm_response.content and llm_response.content.parts:
            out_text_parts: list[str] = []
            thought_text_parts: list[str] = []
            for part in llm_response.content.parts:
                txt = getattr(part, "text", None)
                if txt:
                    if getattr(part, "thought", False):
                        thought_text_parts.append(txt)
                    else:
                        out_text_parts.append(txt)
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    out_text_parts.append(
                        f"{getattr(fc, 'name', '')}({json.dumps(getattr(fc, 'args', {}) or {}, default=str)})"
                    )
            output_tokens = estimate_token_count("\n".join(out_text_parts))
            thoughts_tokens = estimate_token_count("\n".join(thought_text_parts))

        usage = getattr(llm_response, "usage_metadata", None)
        if usage is not None:
            prompt_cnt = getattr(usage, "prompt_token_count", None)
            cached_cnt = getattr(usage, "cached_content_token_count", None)
            cand_cnt = getattr(usage, "candidates_token_count", None)
            th_cnt = getattr(usage, "thoughts_token_count", None)

            if cand_cnt is not None and int(cand_cnt) >= 0:
                output_tokens = int(cand_cnt)
            if th_cnt is not None and int(th_cnt) >= 0:
                thoughts_tokens = int(th_cnt)

            if prompt_cnt is not None and int(prompt_cnt) > 0:
                authoritative_total = int(prompt_cnt)
                authoritative_cached = (
                    min(authoritative_total, int(cached_cnt))
                    if cached_cnt is not None and int(cached_cnt) >= 0
                    else min(authoritative_total, breakdown.cached_context_tokens)
                )
                breakdown = breakdown.model_copy(
                    update={
                        "total_context_tokens": authoritative_total,
                        "cached_context_tokens": authoritative_cached,
                        "uncached_context_tokens": max(
                            0, authoritative_total - authoritative_cached
                        ),
                        "cache_hit": authoritative_cached > 0 or breakdown.cache_hit,
                    }
                )

    return breakdown, output_tokens, thoughts_tokens


def attribute_context_latency(
    *,
    agent_name: str,
    model_name: str,
    task_category: ModelTaskCategory,
    react_phase: str,
    context_breakdown: ContextBreakdown,
    output_tokens: int,
    thoughts_tokens: int = 0,
    raw_inference_latency_ms: float,
    context_prep_latency_ms: float = 0.0,
    observed_ttft_ms: float | None = None,
) -> ModelContextLatencyRecord:
    """Compute a `ModelContextLatencyRecord` isolating latency due to context vs generation."""
    profile = _get_profile_for_model(model_name)
    prep_ms = max(0.0, float(context_prep_latency_ms))
    inf_ms = max(0.0, float(raw_inference_latency_ms))
    total_ms = prep_ms + inf_ms

    uncached_toks = context_breakdown.uncached_context_tokens
    cached_toks = context_breakdown.cached_context_tokens
    total_ctx_toks = context_breakdown.total_context_tokens
    gen_toks = max(0, int(output_tokens) + int(thoughts_tokens))

    # Calibrated prefill vs decode work units
    uncached_prefill_work = uncached_toks * profile.uncached_prefill_ms_per_token
    cached_prefill_work = (
        cached_toks
        * profile.uncached_prefill_ms_per_token
        * profile.cached_prefill_discount_factor
    )
    total_prefill_work = uncached_prefill_work + cached_prefill_work
    decode_work = max(1, gen_toks) * profile.decode_ms_per_token if gen_toks > 0 else 0.0
    combined_work = total_prefill_work + decode_work

    if observed_ttft_ms is not None and observed_ttft_ms >= 0.0:
        prefill_ms = min(inf_ms, float(observed_ttft_ms))
    elif combined_work > 0.0 and inf_ms > 0.0:
        prefill_share = total_prefill_work / combined_work
        prefill_ms = inf_ms * prefill_share
    elif total_ctx_toks > 0 and gen_toks == 0:
        prefill_ms = inf_ms
    else:
        prefill_ms = 0.0

    gen_ms = max(0.0, inf_ms - prefill_ms)
    ctx_latency_ms = prep_ms + prefill_ms
    ttft_ms = (
        float(observed_ttft_ms) + prep_ms
        if observed_ttft_ms is not None and observed_ttft_ms >= 0.0
        else ctx_latency_ms
    )

    ctx_ratio = (ctx_latency_ms / total_ms) if total_ms > 0.0 else (
        (total_prefill_work / combined_work) if combined_work > 0.0 else 0.0
    )
    ctx_ratio = max(0.0, min(1.0, ctx_ratio))

    per_1k_ms = (
        (ctx_latency_ms / (total_ctx_toks / 1000.0))
        if total_ctx_toks > 0
        else 0.0
    )

    # Prefill latency saved because cached_toks avoided full uncached prefill
    cache_saved_ms = (
        cached_toks
        * profile.uncached_prefill_ms_per_token
        * (1.0 - profile.cached_prefill_discount_factor)
    )

    return ModelContextLatencyRecord(
        call_id=f"llm-ctx-{uuid.uuid4().hex[:12]}",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        agent_name=agent_name,
        model_name=model_name,
        task_category=task_category,
        react_phase=react_phase,
        context_breakdown=context_breakdown,
        output_tokens=max(0, int(output_tokens)),
        thoughts_tokens=max(0, int(thoughts_tokens)),
        total_model_latency_ms=round(total_ms, 3),
        context_prep_latency_ms=round(prep_ms, 3),
        context_prefill_latency_ms=round(prefill_ms, 3),
        context_latency_ms=round(ctx_latency_ms, 3),
        generation_latency_ms=round(gen_ms, 3),
        ttft_ms=round(ttft_ms, 3),
        context_latency_ratio=round(ctx_ratio, 4),
        latency_per_1k_context_tokens_ms=round(per_1k_ms, 3),
        estimated_cache_latency_saved_ms=round(cache_saved_ms, 3),
        within_ttft_slo=ttft_ms <= TTFT_SLO_TARGET_MS,
    )


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


class ContextLatencyTracker:
    """Thread-safe telemetry collector tracking context-attributable latency across model calls."""

    def __init__(self, max_records: int = 500) -> None:
        self._lock = threading.Lock()
        self._max_records = max_records
        self._records: list[ModelContextLatencyRecord] = []

    def record_call(
        self,
        *,
        agent_name: str,
        model_name: str,
        task_category: ModelTaskCategory,
        react_phase: str = "general",
        context_breakdown: ContextBreakdown,
        output_tokens: int = 0,
        thoughts_tokens: int = 0,
        raw_inference_latency_ms: float = 0.0,
        context_prep_latency_ms: float = 0.0,
        observed_ttft_ms: float | None = None,
    ) -> ModelContextLatencyRecord:
        """Attribute and store a model call's context latency telemetry record."""
        rec = attribute_context_latency(
            agent_name=agent_name,
            model_name=model_name,
            task_category=task_category,
            react_phase=react_phase,
            context_breakdown=context_breakdown,
            output_tokens=output_tokens,
            thoughts_tokens=thoughts_tokens,
            raw_inference_latency_ms=raw_inference_latency_ms,
            context_prep_latency_ms=context_prep_latency_ms,
            observed_ttft_ms=observed_ttft_ms,
        )
        with self._lock:
            self._records.append(rec)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records :]
        return rec

    def recent_records(
        self,
        *,
        limit: int = 50,
        model_name: str | None = None,
    ) -> list[ModelContextLatencyRecord]:
        with self._lock:
            items = list(self._records)
        if model_name:
            items = [r for r in items if r.model_name == model_name]
        return items[-max(1, limit) :]

    def summarize(
        self,
        records: list[ModelContextLatencyRecord] | None = None,
    ) -> ContextLatencySummary:
        """Aggregate p50/p95 context latency, TTFT SLO compliance, and Pro vs Flash breakdowns."""
        if records is None:
            with self._lock:
                items = list(self._records)
        else:
            items = list(records)

        if not items:
            return ContextLatencySummary(ttft_slo_target_ms=TTFT_SLO_TARGET_MS)

        ctx_latencies = sorted(r.context_latency_ms for r in items)
        total_latencies = [r.total_model_latency_ms for r in items]
        ttfts = sorted(r.ttft_ms for r in items)
        ratios = [r.context_latency_ratio for r in items]
        per_1k = [
            r.latency_per_1k_context_tokens_ms
            for r in items
            if r.context_breakdown.total_context_tokens > 0
        ]
        within_slo = sum(1 for r in items if r.within_ttft_slo)

        by_model_groups: dict[str, list[ModelContextLatencyRecord]] = {}
        for rec in items:
            by_model_groups.setdefault(rec.model_name, []).append(rec)

        by_model_summary: dict[str, dict[str, Any]] = {}
        for m_name, m_recs in sorted(by_model_groups.items()):
            m_ctx = sorted(x.context_latency_ms for x in m_recs)
            m_tot = [x.total_model_latency_ms for x in m_recs]
            by_model_summary[m_name] = {
                "call_count": len(m_recs),
                "task_categories": sorted({x.task_category for x in m_recs}),
                "total_context_tokens": sum(
                    x.context_breakdown.total_context_tokens for x in m_recs
                ),
                "cached_context_tokens": sum(
                    x.context_breakdown.cached_context_tokens for x in m_recs
                ),
                "mean_context_latency_ms": round(sum(m_ctx) / len(m_ctx), 3),
                "p95_context_latency_ms": round(_percentile(m_ctx, 95.0), 3),
                "mean_total_model_latency_ms": round(sum(m_tot) / len(m_tot), 3),
                "estimated_cache_latency_saved_ms": round(
                    sum(x.estimated_cache_latency_saved_ms for x in m_recs), 3
                ),
            }

        return ContextLatencySummary(
            total_calls=len(items),
            total_context_tokens=sum(
                r.context_breakdown.total_context_tokens for r in items
            ),
            total_cached_context_tokens=sum(
                r.context_breakdown.cached_context_tokens for r in items
            ),
            total_uncached_context_tokens=sum(
                r.context_breakdown.uncached_context_tokens for r in items
            ),
            mean_context_latency_ms=round(sum(ctx_latencies) / len(ctx_latencies), 3),
            p50_context_latency_ms=round(_percentile(ctx_latencies, 50.0), 3),
            p95_context_latency_ms=round(_percentile(ctx_latencies, 95.0), 3),
            mean_total_model_latency_ms=round(
                sum(total_latencies) / len(total_latencies), 3
            ),
            p95_ttft_ms=round(_percentile(ttfts, 95.0), 3),
            ttft_slo_target_ms=TTFT_SLO_TARGET_MS,
            ttft_slo_compliance_ratio=round(within_slo / len(items), 4),
            mean_context_latency_ratio=round(sum(ratios) / len(ratios), 4),
            mean_latency_per_1k_context_tokens_ms=round(
                (sum(per_1k) / len(per_1k)) if per_1k else 0.0, 3
            ),
            total_estimated_cache_latency_saved_ms=round(
                sum(r.estimated_cache_latency_saved_ms for r in items), 3
            ),
            by_model=by_model_summary,
        )

    def reset(self) -> None:
        with self._lock:
            self._records.clear()


_GLOBAL_LATENCY_TRACKER = ContextLatencyTracker()


def get_context_latency_tracker() -> ContextLatencyTracker:
    """Return the process-wide `ContextLatencyTracker` singleton."""
    return _GLOBAL_LATENCY_TRACKER


# ---------------------------------------------------------------------------
# 3. Conversational Session State, Compaction & Cross-Session Memory Bank
# ---------------------------------------------------------------------------


@dataclass
class TurnMemoryEntry:
    """Structured memory entry stored per conversational turn."""

    turn_index: int
    timestamp: str
    query: str
    intent: str
    tickers: list[str]
    gics_sector: str
    start_year: int
    end_year: int
    variance_highlights: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    cached_10k_keys: list[str] = field(default_factory=list)
    token_estimate: int = 0


@dataclass
class SessionMemoryState:
    """Active session memory state with automatic token-threshold compaction."""

    session_id: str
    user_id: str = "analyst"
    turns: list[TurnMemoryEntry] = field(default_factory=list)
    compacted_summary: str = ""
    compaction_count: int = 0
    remembered_tickers: list[str] = field(default_factory=list)
    last_start_year: int = 2024
    last_end_year: int = 2026
    last_gics_sector: str = "ALL"


class ConversationMemoryManager:
    """Manages conversational session state, follow-up entity recall, and ADK Memory/Compaction."""

    def __init__(
        self,
        *,
        token_threshold: int = DEFAULT_COMPACTION_TOKEN_THRESHOLD,
        retention_turns: int = DEFAULT_COMPACTION_RETENTION_SIZE,
    ) -> None:
        self.token_threshold = token_threshold
        self.retention_turns = retention_turns
        self.adk_session_service = InMemorySessionService()
        self.adk_memory_service = InMemoryMemoryService()
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionMemoryState] = {}

    def get_or_create_session(
        self,
        session_id: str,
        user_id: str = "analyst",
    ) -> SessionMemoryState:
        norm_id = session_id.strip()
        with self._lock:
            if norm_id not in self._sessions:
                self._sessions[norm_id] = SessionMemoryState(
                    session_id=norm_id,
                    user_id=user_id,
                )
            return self._sessions[norm_id]

    def recall_session_context(
        self,
        session_id: str | None,
    ) -> dict[str, Any]:
        """Retrieve active session memory and measure context recall latency."""
        t0 = time.perf_counter()
        if not session_id or not session_id.strip():
            return {
                "session_id": None,
                "has_prior_memory": False,
                "turn_count": 0,
                "remembered_tickers": [],
                "last_start_year": None,
                "last_end_year": None,
                "last_gics_sector": "ALL",
                "compaction_count": 0,
                "memory_summary_text": "",
                "memory_tokens": 0,
                "memory_lookup_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }

        with self._lock:
            sess = self._sessions.get(session_id.strip())
            if sess is None or (not sess.turns and not sess.compacted_summary):
                return {
                    "session_id": session_id.strip(),
                    "has_prior_memory": False,
                    "turn_count": 0,
                    "remembered_tickers": [],
                    "last_start_year": None,
                    "last_end_year": None,
                    "last_gics_sector": "ALL",
                    "compaction_count": 0,
                    "memory_summary_text": "",
                    "memory_tokens": 0,
                    "memory_lookup_latency_ms": round(
                        (time.perf_counter() - t0) * 1000.0, 3
                    ),
                }

            lines: list[str] = []
            if sess.compacted_summary:
                lines.append(f"[Compacted Prior Context]: {sess.compacted_summary}")
            for t in sess.turns[-self.retention_turns :]:
                highlights = "; ".join(t.variance_highlights[:2]) if t.variance_highlights else "N/A"
                lines.append(
                    f"[Turn {t.turn_index} | {','.join(t.tickers) or t.gics_sector} FY{t.start_year}-FY{t.end_year}]: "
                    f"Query='{t.query}' -> {highlights}"
                )
            summary_text = "\n".join(lines)
            mem_tokens = estimate_token_count(summary_text)
            return {
                "session_id": sess.session_id,
                "has_prior_memory": True,
                "turn_count": len(sess.turns),
                "remembered_tickers": list(sess.remembered_tickers),
                "last_start_year": sess.last_start_year,
                "last_end_year": sess.last_end_year,
                "last_gics_sector": sess.last_gics_sector,
                "compaction_count": sess.compaction_count,
                "memory_summary_text": summary_text,
                "memory_tokens": mem_tokens,
                "memory_lookup_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }

    def record_turn(
        self,
        *,
        session_id: str | None,
        user_id: str = "analyst",
        query: str,
        intent: str,
        tickers: list[str],
        gics_sector: str,
        start_year: int,
        end_year: int,
        variance_highlights: list[str] | None = None,
        cited_chunk_ids: list[str] | None = None,
        cached_10k_keys: list[str] | None = None,
        answer_markdown: str = "",
    ) -> dict[str, Any]:
        """Persist a completed turn into session memory and compact older turns if over token threshold."""
        if not session_id or not session_id.strip():
            return {"persisted": False, "compaction_triggered": False, "turn_count": 0}

        compaction_start = time.perf_counter()
        compaction_triggered = False
        compacted_tokens_before = 0

        with self._lock:
            sess = self._sessions.get(session_id.strip())
            if sess is None:
                sess = SessionMemoryState(session_id=session_id.strip(), user_id=user_id)
                self._sessions[sess.session_id] = sess

            turn_text = f"{query}\n{answer_markdown}"
            entry = TurnMemoryEntry(
                turn_index=len(sess.turns) + 1,
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                query=query,
                intent=intent,
                tickers=list(tickers),
                gics_sector=gics_sector,
                start_year=start_year,
                end_year=end_year,
                variance_highlights=list(variance_highlights or []),
                cited_chunk_ids=list(cited_chunk_ids or []),
                cached_10k_keys=list(cached_10k_keys or []),
                token_estimate=estimate_token_count(turn_text),
            )
            sess.turns.append(entry)
            if tickers:
                sess.remembered_tickers = list(tickers)
            sess.last_start_year = start_year
            sess.last_end_year = end_year
            if gics_sector and gics_sector != "ALL":
                sess.last_gics_sector = gics_sector

            total_mem_tokens = sum(t.token_estimate for t in sess.turns) + estimate_token_count(
                sess.compacted_summary
            )
            if (
                total_mem_tokens >= self.token_threshold
                and len(sess.turns) > self.retention_turns
            ):
                compaction_triggered = True
                compacted_tokens_before = total_mem_tokens
                older_turns = sess.turns[: -self.retention_turns]
                sess.turns = sess.turns[-self.retention_turns :]
                digest_parts: list[str] = []
                if sess.compacted_summary:
                    digest_parts.append(sess.compacted_summary)
                for old in older_turns:
                    tk_str = ",".join(old.tickers) if old.tickers else old.gics_sector
                    hl_str = "; ".join(old.variance_highlights[:1]) if old.variance_highlights else old.intent
                    digest_parts.append(
                        f"T{old.turn_index}({tk_str} FY{old.start_year}-{old.end_year}): {hl_str}"
                    )
                sess.compacted_summary = " | ".join(digest_parts)[-2000:]
                sess.compaction_count += 1

            turn_count = len(sess.turns)
            compaction_count = sess.compaction_count
            summary_after = sess.compacted_summary

        if compaction_triggered:
            compaction_ms = (time.perf_counter() - compaction_start) * 1000.0
            flash_model = resolve_model_for_task("context_compaction")
            get_context_latency_tracker().record_call(
                agent_name="ContextCompactionSummarizer",
                model_name=flash_model,
                task_category="context_compaction",
                react_phase="context_compaction",
                context_breakdown=ContextBreakdown(
                    memory_and_history_tokens=compacted_tokens_before,
                    total_context_tokens=compacted_tokens_before,
                    uncached_context_tokens=compacted_tokens_before,
                ),
                output_tokens=estimate_token_count(summary_after),
                raw_inference_latency_ms=compaction_ms,
            )

        return {
            "persisted": True,
            "session_id": session_id.strip(),
            "turn_count": turn_count,
            "compaction_triggered": compaction_triggered,
            "compaction_count": compaction_count,
        }

    def reset(self) -> None:
        with self._lock:
            self._sessions.clear()


_GLOBAL_MEMORY_MANAGER = ConversationMemoryManager()


def get_conversation_memory_manager() -> ConversationMemoryManager:
    """Return the process-wide `ConversationMemoryManager` singleton."""
    return _GLOBAL_MEMORY_MANAGER


async def after_agent_memory_callback(
    callback_context: CallbackContext,
) -> genai_types.Content | None:
    """ADK `after_agent_callback` persisting session events to Memory Bank & tracking agent duration."""
    state = callback_context.state
    start_ts = state.get("_agent_start_monotonic")
    if start_ts is not None:
        state["last_agent_total_duration_ms"] = round(
            (time.perf_counter() - float(start_ts)) * 1000.0, 3
        )
    add_mem = getattr(callback_context, "add_session_to_memory", None)
    if callable(add_mem):
        try:
            await add_mem()
        except Exception as exc:
            logger.debug("Optional Memory Bank sync skipped in local mode: %s", exc)
    return None


def create_edgar_adk_app(
    root_agent: BaseAgent,
    *,
    app_name: str = "agents",
    min_cache_tokens: int = DEFAULT_CONTEXT_CACHE_MIN_TOKENS,
    cache_ttl_seconds: int = DEFAULT_CONTEXT_CACHE_TTL_SECONDS,
    compaction_token_threshold: int = DEFAULT_COMPACTION_TOKEN_THRESHOLD,
    compaction_retention_size: int = DEFAULT_COMPACTION_RETENTION_SIZE,
) -> App:
    """Create the production ADK `App` with Vertex AI Context Caching & Flash-powered Event Compaction."""
    flash_model = resolve_model_for_task("context_compaction")
    return App(
        name=app_name,
        root_agent=root_agent,
        context_cache_config=ContextCacheConfig(
            min_tokens=min_cache_tokens,
            ttl_seconds=cache_ttl_seconds,
            cache_intervals=DEFAULT_CONTEXT_CACHE_INTERVALS,
        ),
        events_compaction_config=EventsCompactionConfig(
            token_threshold=compaction_token_threshold,
            event_retention_size=compaction_retention_size,
            summarizer=LlmEventSummarizer(llm=Gemini(model=flash_model)),
        ),
    )
