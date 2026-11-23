"""Context-induced latency attribution, token breakdown reconciliation, and cost telemetry.

Decomposes every model call into context preparation latency, context prefill latency (cached vs.
uncached), and output token generation latency, tracking TTFT SLO compliance (<= 3.0s p95) and
KV-cache USD savings.
"""

from __future__ import annotations

import json
import threading
import uuid
from typing import Any

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from edgar_capstone.agents.core.schemas import (
    ContextBreakdown,
    ContextLatencySummary,
    ModelContextLatencyRecord,
    ModelTaskCategory,
)
from edgar_capstone.agents.runtime.model_config import _get_profile_for_model, get_ttft_slo_ms
from edgar_capstone.common.domain import RETRIEVAL_TOOL_NAMES, VARIANCE_TOOL_NAMES
from edgar_capstone.common.formatting import percentile, utc_now_iso
from edgar_capstone.telemetry import get_telemetry_sink
from edgar_capstone.vector_db.chunker import estimate_token_count


def _count_system_instruction_tokens(llm_request: LlmRequest) -> int:
    cfg = getattr(llm_request, "config", None)
    sys_inst = getattr(cfg, "system_instruction", None) if cfg is not None else None
    if isinstance(sys_inst, str) and sys_inst.strip():
        return estimate_token_count(sys_inst)
    if sys_inst is not None and hasattr(sys_inst, "parts"):
        sys_text = "\n".join(
            p.text for p in (getattr(sys_inst, "parts", None) or []) if getattr(p, "text", None)
        )
        return estimate_token_count(sys_text)
    return 0


def _find_last_user_text_index(contents: list[Any]) -> int:
    for idx in range(len(contents) - 1, -1, -1):
        if getattr(contents[idx], "role", None) == "user":
            if any(
                bool(getattr(p, "text", None))
                for p in (getattr(contents[idx], "parts", None) or [])
            ):
                return idx
    return -1


def _classify_function_response_tokens(resp_name: str, resp_payload: Any) -> tuple[str, int]:
    serialized = json.dumps(resp_payload, default=str)
    resp_tokens = estimate_token_count(serialized)
    if resp_name in VARIANCE_TOOL_NAMES:
        return "variance", resp_tokens
    if (
        resp_name in RETRIEVAL_TOOL_NAMES
        and isinstance(resp_payload, dict)
        and resp_payload.get("cache_hit")
    ):
        return "cached_10k", resp_tokens
    return "uncached_10k", resp_tokens


def extract_llm_request_context_breakdown(
    llm_request: LlmRequest,
    state: Any | None = None,
) -> ContextBreakdown:
    """Inspect an ADK `LlmRequest` and session state to quantify input context components."""
    state_dict = state if isinstance(state, dict) or hasattr(state, "get") else {}
    sys_tokens = _count_system_instruction_tokens(llm_request)

    contents = list(getattr(llm_request, "contents", None) or [])
    last_user_idx = _find_last_user_text_index(contents)

    user_prompt_tokens = 0
    memory_and_history_tokens = 0
    variance_payload_tokens = 0
    cached_10k_tokens = 0
    uncached_10k_and_tool_tokens = 0

    for idx, content in enumerate(contents):
        role = getattr(content, "role", "") or ""
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                t_count = estimate_token_count(part_text)
                if idx == last_user_idx and role == "user":
                    user_prompt_tokens += t_count
                else:
                    memory_and_history_tokens += t_count

            fn_call = getattr(part, "function_call", None)
            if fn_call is not None:
                fc_str = (
                    f"{getattr(fn_call, 'name', '')}"
                    f"({json.dumps(getattr(fn_call, 'args', {}) or {}, default=str)})"
                )
                memory_and_history_tokens += estimate_token_count(fc_str)

            fn_resp = getattr(part, "function_response", None)
            if fn_resp is not None:
                resp_name = str(getattr(fn_resp, "name", "") or "")
                resp_payload = getattr(fn_resp, "response", None) or {}
                bucket, resp_tokens = _classify_function_response_tokens(resp_name, resp_payload)
                if bucket == "variance":
                    variance_payload_tokens += resp_tokens
                elif bucket == "cached_10k":
                    cached_10k_tokens += resp_tokens
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

    cache_hit = cached_total > 0
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


def _estimate_response_output_tokens(llm_response: LlmResponse) -> tuple[int, int]:
    if not llm_response.content or not llm_response.content.parts:
        return 0, 0
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
    return estimate_token_count("\n".join(out_text_parts)), estimate_token_count(
        "\n".join(thought_text_parts)
    )


def reconcile_breakdown_with_usage_metadata(
    breakdown: ContextBreakdown,
    llm_response: LlmResponse | None,
) -> tuple[ContextBreakdown, int, int]:
    """Reconcile estimated context breakdown with authoritative Gemini `usage_metadata` when available."""
    if llm_response is None:
        return breakdown, 0, 0

    output_tokens, thoughts_tokens = _estimate_response_output_tokens(llm_response)
    usage = getattr(llm_response, "usage_metadata", None)
    if usage is None:
        return breakdown, output_tokens, thoughts_tokens

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
                "uncached_context_tokens": max(0, authoritative_total - authoritative_cached),
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
    session_id: str | None = None,
    user_id: str | None = None,
) -> ModelContextLatencyRecord:
    """Compute a `ModelContextLatencyRecord` isolating latency due to context vs generation and USD cost."""
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
        cached_toks * profile.uncached_prefill_ms_per_token * profile.cached_prefill_discount_factor
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

    ctx_ratio = (
        (ctx_latency_ms / total_ms)
        if total_ms > 0.0
        else ((total_prefill_work / combined_work) if combined_work > 0.0 else 0.0)
    )
    ctx_ratio = max(0.0, min(1.0, ctx_ratio))

    per_1k_ms = (ctx_latency_ms / (total_ctx_toks / 1000.0)) if total_ctx_toks > 0 else 0.0

    # Prefill latency saved because cached_toks avoided full uncached prefill
    cache_saved_ms = (
        cached_toks
        * profile.uncached_prefill_ms_per_token
        * (1.0 - profile.cached_prefill_discount_factor)
    )

    # USD token cost & cached-token discount savings
    input_cost_usd = (uncached_toks / 1_000_000.0) * profile.uncached_input_usd_per_1m + (
        cached_toks / 1_000_000.0
    ) * profile.cached_input_usd_per_1m
    uncached_input_baseline_usd = (
        (uncached_toks + cached_toks) / 1_000_000.0
    ) * profile.uncached_input_usd_per_1m
    output_cost_usd = (gen_toks / 1_000_000.0) * profile.output_usd_per_1m
    total_cost_usd = input_cost_usd + output_cost_usd
    uncached_baseline_cost_usd = uncached_input_baseline_usd + output_cost_usd
    cached_token_savings_usd = max(0.0, uncached_input_baseline_usd - input_cost_usd)
    slo_ms = get_ttft_slo_ms()

    return ModelContextLatencyRecord(
        call_id=f"llm-ctx-{uuid.uuid4().hex[:12]}",
        timestamp=utc_now_iso(),
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
        within_ttft_slo=ttft_ms <= slo_ms,
        input_cost_usd=round(input_cost_usd, 8),
        output_cost_usd=round(output_cost_usd, 8),
        total_cost_usd=round(total_cost_usd, 8),
        uncached_baseline_cost_usd=round(uncached_baseline_cost_usd, 8),
        cached_token_savings_usd=round(cached_token_savings_usd, 8),
        session_id=session_id,
        user_id=user_id,
    )


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
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> ModelContextLatencyRecord:
        """Attribute, store, and stream a model call's context latency & cost telemetry record."""
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
            session_id=session_id,
            user_id=user_id,
        )
        with self._lock:
            self._records.append(rec)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records :]
        # Called outside `self._lock`; in live production mode `BigQueryTelemetrySink` dispatches
        # `tabledata.insertAll` asynchronously on its background pool so `_ADK_LOOP` is never blocked.
        get_telemetry_sink().record_llm_telemetry(rec)
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

    def get_records_by_ids(
        self,
        call_ids: list[str],
    ) -> list[ModelContextLatencyRecord]:
        """Return latency records matching `call_ids` in their original recording order."""
        if not call_ids:
            return []
        wanted = set(call_ids)
        with self._lock:
            return [r for r in self._records if r.call_id in wanted]

    def summarize(
        self,
        records: list[ModelContextLatencyRecord] | None = None,
    ) -> ContextLatencySummary:
        """Aggregate p50/p95 context latency, TTFT SLO compliance, and model-tier breakdowns."""
        slo_ms = get_ttft_slo_ms()
        if records is None:
            with self._lock:
                items = list(self._records)
        else:
            items = list(records)

        if not items:
            return ContextLatencySummary(ttft_slo_target_ms=slo_ms)

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
                "output_tokens": sum(x.output_tokens for x in m_recs),
                "mean_context_latency_ms": round(sum(m_ctx) / len(m_ctx), 3),
                "p95_context_latency_ms": round(percentile(m_ctx, 95.0), 3),
                "mean_total_model_latency_ms": round(sum(m_tot) / len(m_tot), 3),
                "estimated_cache_latency_saved_ms": round(
                    sum(x.estimated_cache_latency_saved_ms for x in m_recs), 3
                ),
                "total_cost_usd": round(sum(x.total_cost_usd for x in m_recs), 8),
                "cached_token_savings_usd": round(
                    sum(x.cached_token_savings_usd for x in m_recs), 8
                ),
            }

        total_cost = sum(r.total_cost_usd for r in items)
        total_baseline_cost = sum(r.uncached_baseline_cost_usd for r in items)
        total_savings = sum(r.cached_token_savings_usd for r in items)
        savings_pct = (
            round((total_savings / total_baseline_cost) * 100.0, 2)
            if total_baseline_cost > 0.0
            else 0.0
        )

        return ContextLatencySummary(
            total_calls=len(items),
            total_context_tokens=sum(r.context_breakdown.total_context_tokens for r in items),
            total_cached_context_tokens=sum(
                r.context_breakdown.cached_context_tokens for r in items
            ),
            total_uncached_context_tokens=sum(
                r.context_breakdown.uncached_context_tokens for r in items
            ),
            total_output_tokens=sum(r.output_tokens for r in items),
            total_thoughts_tokens=sum(r.thoughts_tokens for r in items),
            mean_context_latency_ms=round(sum(ctx_latencies) / len(ctx_latencies), 3),
            p50_context_latency_ms=round(percentile(ctx_latencies, 50.0), 3),
            p95_context_latency_ms=round(percentile(ctx_latencies, 95.0), 3),
            mean_total_model_latency_ms=round(sum(total_latencies) / len(total_latencies), 3),
            p95_ttft_ms=round(percentile(ttfts, 95.0), 3),
            ttft_slo_target_ms=slo_ms,
            ttft_slo_compliance_ratio=round(within_slo / len(items), 4),
            mean_context_latency_ratio=round(sum(ratios) / len(ratios), 4),
            mean_latency_per_1k_context_tokens_ms=round(
                (sum(per_1k) / len(per_1k)) if per_1k else 0.0, 3
            ),
            total_estimated_cache_latency_saved_ms=round(
                sum(r.estimated_cache_latency_saved_ms for r in items), 3
            ),
            total_cost_usd=round(total_cost, 8),
            total_uncached_baseline_cost_usd=round(total_baseline_cost, 8),
            total_cached_token_savings_usd=round(total_savings, 8),
            cache_cost_savings_pct=savings_pct,
            by_model=by_model_summary,
        )

    def reset(self) -> None:
        with self._lock:
            self._records.clear()


_GLOBAL_LATENCY_TRACKER = ContextLatencyTracker()


def get_context_latency_tracker() -> ContextLatencyTracker:
    """Return the process-wide `ContextLatencyTracker` singleton."""
    return _GLOBAL_LATENCY_TRACKER
