"""Context & Memory Strategy, Model Tier Routing, and Context-Induced Latency Tracking.

Implements:
1. Dual-Model Specialization Strategy:
   - `gemini-3.8-flash` (`EDGAR_ANALYST_MODEL`): Complex financial reasoning, multi-period variance
     synthesis, and ReAct orchestration.
   - `gemini-3.5-flash` (`EDGAR_FLASH_MODEL`): LLM-as-a-Judge evaluation tasks and ADK app event
     compaction summarization.
2. Context-Induced Latency Tracking:
   - Decomposes every model call into context preparation latency, context prefill latency (cached
     vs. uncached), and output token generation latency, tracking TTFT SLO compliance (<= 3.0s p95)
     and KV-cache savings.
3. Session State, Context Caching, Compaction & Long-Term Memory:
   - Configures ADK `ContextCacheConfig`, token-triggered `EventsCompactionConfig`, bounded
     `InMemorySessionService` / `BoundedInMemoryMemoryService`, and tenant-isolated multi-turn
     conversational memory backed by `SessionStoreProtocol` (Firestore REST v1 in production).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.memory import InMemoryMemoryService
from google.adk.models import Gemini
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from edgar_capstone.agents.schemas import (
    ContextBreakdown,
    ContextLatencySummary,
    ModelContextLatencyRecord,
    ModelTaskCategory,
)
from edgar_capstone.common.domain import RETRIEVAL_TOOL_NAMES
from edgar_capstone.common.env import env_positive_float
from edgar_capstone.common.formatting import percentile, utc_now_iso
from edgar_capstone.security.model_armor import ModelArmorSafetyPlugin
from edgar_capstone.telemetry import FirestoreSessionStore, get_telemetry_sink
from edgar_capstone.vector_db.chunker import estimate_token_count

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Model Tier Constants & Task Specialization Strategy
# ---------------------------------------------------------------------------

FINANCIAL_REASONING_MODEL: str = os.environ.get("EDGAR_ANALYST_MODEL", "gemini-3.8-flash")
FLASH_CODING_AND_EVAL_MODEL: str = os.environ.get("EDGAR_FLASH_MODEL", "gemini-3.5-flash")

DEFAULT_TTFT_SLO_MS: float = 3000.0
TTFT_SLO_TARGET_MS: float = env_positive_float(
    "EDGAR_TTFT_SLO_MS", DEFAULT_TTFT_SLO_MS, minimum=1.0
)
DEFAULT_CONTEXT_CACHE_MIN_TOKENS: int = 2048
DEFAULT_CONTEXT_CACHE_TTL_SECONDS: int = 1800
DEFAULT_CONTEXT_CACHE_INTERVALS: int = 10
DEFAULT_COMPACTION_TOKEN_THRESHOLD: int = 32000
DEFAULT_COMPACTION_RETENTION_SIZE: int = 5
DEFAULT_MAX_LOCAL_SESSIONS: int = 500
DEFAULT_MAX_ADK_MEMORY_SESSIONS: int = 200

EPHEMERAL_AGENT_NAMES: frozenset[str] = frozenset(
    {
        "NarrativeEvaluatorAgent",
        "SectorBatchMapper",
    }
)
EPHEMERAL_SESSION_PREFIXES: tuple[str, ...] = (
    "eval-",
    "sector-map-",
    "adk-sess-",
)

_VARIANCE_PAYLOAD_TOOLS: frozenset[str] = frozenset(
    {
        "calculate_period_variances",
        "fetch_financial_metrics",
        "query_looker_financial_explore",
        "flag_material_variances",
    }
)


def get_ttft_slo_ms() -> float:
    """Resolve the active TTFT SLO target in milliseconds dynamically from environment."""
    return env_positive_float("EDGAR_TTFT_SLO_MS", DEFAULT_TTFT_SLO_MS, minimum=1.0)


def is_ephemeral_session_id(session_id: str | None) -> bool:
    """Return True when `session_id` belongs to a short-lived utility/retry/eval invocation."""
    if not session_id:
        return True
    cleaned = session_id.strip()
    if not cleaned:
        return True
    return cleaned.startswith(EPHEMERAL_SESSION_PREFIXES) or cleaned.endswith("-retry")


@dataclass(frozen=True)
class ModelPerformanceProfile:
    """Calibrated token-throughput and USD token pricing characteristics for prefill/decode & cost telemetry."""

    model_name: str
    tier: str
    primary_use_cases: tuple[str, ...]
    uncached_prefill_ms_per_token: float
    cached_prefill_discount_factor: (
        float  # Fraction of uncached prefill cost when cached (e.g. 0.20 = 5x faster)
    )
    decode_ms_per_token: float
    uncached_input_usd_per_1m: float = 1.25
    cached_input_usd_per_1m: float = 0.3125
    output_usd_per_1m: float = 10.00


MODEL_PROFILES: dict[str, ModelPerformanceProfile] = {
    # Vertex AI introductory pricing (list price from 2027-01-01: $1.50 input / $7.50 output).
    "gemini-3.8-flash": ModelPerformanceProfile(
        model_name="gemini-3.8-flash",
        tier="complex_financial_reasoning",
        primary_use_cases=(
            "financial_reasoning",
            "variance_explanation",
            "peer_comparison",
            "thematic_synthesis",
            "orchestration",
        ),
        uncached_prefill_ms_per_token=0.05,
        cached_prefill_discount_factor=0.18,
        decode_ms_per_token=0.60,
        uncached_input_usd_per_1m=0.75,
        cached_input_usd_per_1m=0.075,
        output_usd_per_1m=3.75,
    ),
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
        uncached_input_usd_per_1m=1.25,
        cached_input_usd_per_1m=0.3125,
        output_usd_per_1m=10.00,
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
        uncached_input_usd_per_1m=0.30,
        cached_input_usd_per_1m=0.075,
        output_usd_per_1m=2.50,
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
    "ContextCacheManager": "context_cache_warmup",
    "NarrativeEvaluatorAgent": "evaluation",
    "SectorBatchMapper": "context_compaction",
}


def low_thinking_config_for_model(model: str | BaseLlm) -> genai_types.ThinkingConfig | None:
    """Low-effort thinking for structured Gemini tasks (judge, map-reduce, utility calls).

    Empirically A/B tested on the 43-item live golden suite: Gemini 3.x native
    `ThinkingLevel.LOW` outperformed legacy `thinking_budget=512` on both speed and quality
    (p50 TTFT 1,981 ms vs 2,251 ms, p95 4,713 ms vs 7,750 ms, grounding recall 0.9444 vs
    0.9028, citation coverage 1.0 vs 0.9677). Gemini 2.5 uses `thinking_budget=1024`.
    """
    name = (model.model if isinstance(model, BaseLlm) else str(model or "")).lower()
    short = name.rsplit("/", maxsplit=1)[-1]
    if short.startswith("gemini-3"):
        return genai_types.ThinkingConfig(thinking_level=genai_types.ThinkingLevel.LOW)
    if short.startswith("gemini-2.5"):
        return genai_types.ThinkingConfig(thinking_budget=1024)
    return None


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
    norm = model_name.strip().lower().rsplit("/", maxsplit=1)[-1]
    if norm in MODEL_PROFILES:
        return MODEL_PROFILES[norm]
    for known in MODEL_PROFILES:  # versioned ids, e.g. "gemini-3.8-flash-001"
        if norm.startswith(known):
            return MODEL_PROFILES[known]
    # No silent default: pricing and latency telemetry for an unlisted model would be wrong.
    raise ValueError(
        f"No performance/pricing profile for model {model_name!r}. Add it to MODEL_PROFILES "
        f"in context_memory.py (known models: {', '.join(sorted(MODEL_PROFILES))})."
    )


def get_model_strategy_metadata() -> dict[str, Any]:
    """Return public metadata describing the dual-model specialization & context strategy."""
    fin_model = resolve_model_for_task("financial_reasoning")
    flash_model = resolve_model_for_task("evaluation")
    return {
        "financial_reasoning_model": fin_model,
        "coding_and_evaluation_model": flash_model,
        "task_routing": {task: resolve_model_for_task(task) for task in TASK_TO_MODEL_TIER},
        "context_cache_config": {
            "min_tokens": DEFAULT_CONTEXT_CACHE_MIN_TOKENS,
            "ttl_seconds": DEFAULT_CONTEXT_CACHE_TTL_SECONDS,
            "cache_intervals": DEFAULT_CONTEXT_CACHE_INTERVALS,
        },
        "events_compaction_config": {
            "token_threshold": DEFAULT_COMPACTION_TOKEN_THRESHOLD,
            "event_retention_size": DEFAULT_COMPACTION_RETENTION_SIZE,
            "compaction_method": "deterministic_digest",
            "adk_app_summarizer_model": flash_model,
        },
        "pricing_per_1m_tokens_usd": {
            name: {
                "uncached_input_usd_per_1m": prof.uncached_input_usd_per_1m,
                "cached_input_usd_per_1m": prof.cached_input_usd_per_1m,
                "output_usd_per_1m": prof.output_usd_per_1m,
            }
            for name, prof in MODEL_PROFILES.items()
        },
        "ttft_slo_target_ms": get_ttft_slo_ms(),
    }


# ---------------------------------------------------------------------------
# 2. Context Breakdown & Context-Induced Latency Tracking
# ---------------------------------------------------------------------------


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
    if resp_name in _VARIANCE_PAYLOAD_TOOLS:
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


# ---------------------------------------------------------------------------
# 3. Conversational Session State, Compaction & Cross-Session Memory Bank
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionStoreProtocol(Protocol):
    """Minimal persistence contract for durable conversational session stores."""

    def is_live_enabled(self) -> bool: ...
    def load_session(self, user_id: str, session_id: str) -> dict[str, Any] | None: ...
    def save_session(self, user_id: str, session_id: str, payload: dict[str, Any]) -> None: ...
    def delete_session(self, user_id: str, session_id: str) -> bool: ...
    def list_sessions(self, user_id: str, *, limit: int = 25) -> list[dict[str, Any]]: ...
    def status_metadata(self) -> dict[str, Any]: ...


class BoundedInMemoryMemoryService(InMemoryMemoryService):
    """LRU-bounded ADK `InMemoryMemoryService` that ignores ephemeral utility sessions."""

    def __init__(self, *, max_sessions: int = DEFAULT_MAX_ADK_MEMORY_SESSIONS) -> None:
        super().__init__()
        self._max_sessions = max(1, int(max_sessions))
        self._insertion_order: OrderedDict[tuple[tuple[str, str], str | None], None] = OrderedDict()

    async def add_session_to_memory(self, session: Any) -> None:
        session_id = getattr(session, "id", None)
        if is_ephemeral_session_id(session_id):
            return
        await super().add_session_to_memory(session)
        user_key = (str(getattr(session, "app_name", "")), str(getattr(session, "user_id", "")))
        with self._lock:
            entry_key = (user_key, session_id)
            self._insertion_order.pop(entry_key, None)
            self._insertion_order[entry_key] = None
            while len(self._insertion_order) > self._max_sessions:
                (old_user_key, old_sid), _ = self._insertion_order.popitem(last=False)
                user_map = self._session_events.get(old_user_key)
                if user_map is not None:
                    user_map.pop(old_sid, None)
                    if not user_map:
                        self._session_events.pop(old_user_key, None)

    def drop_session(self, session_id: str, candidate_users: set[str]) -> bool:
        """Drop all cached memory events for `session_id` owned by any user in `candidate_users`."""
        removed = False
        with self._lock:
            for user_key in list(self._session_events.keys()):
                if user_key[1] in candidate_users:
                    user_map = self._session_events.get(user_key)
                    if user_map is not None and session_id in user_map:
                        user_map.pop(session_id, None)
                        self._insertion_order.pop((user_key, session_id), None)
                        removed = True
                        if not user_map:
                            self._session_events.pop(user_key, None)
        return removed

    def tracked_session_count(self) -> int:
        with self._lock:
            return len(self._insertion_order)


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
    updated_at: str = field(default_factory=utc_now_iso)

    def to_firestore_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id.strip().lower(),
            "compacted_summary": self.compacted_summary,
            "compaction_count": int(self.compaction_count),
            "remembered_tickers": list(self.remembered_tickers),
            "last_start_year": int(self.last_start_year),
            "last_end_year": int(self.last_end_year),
            "last_gics_sector": self.last_gics_sector,
            "turn_count": len(self.turns),
            "turns_json": json.dumps([asdict(t) for t in self.turns]),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_firestore_dict(cls, doc: dict[str, Any]) -> SessionMemoryState:
        raw_turns = doc.get("turns_json") or "[]"
        try:
            parsed_turns = json.loads(raw_turns) if isinstance(raw_turns, str) else list(raw_turns)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to deserialize turns_json for session '%s': %s",
                doc.get("session_id", ""),
                exc,
            )
            parsed_turns = []
        turns: list[TurnMemoryEntry] = []
        for t in parsed_turns:
            if not isinstance(t, dict):
                continue
            turns.append(
                TurnMemoryEntry(
                    turn_index=int(t.get("turn_index") or len(turns) + 1),
                    timestamp=str(t.get("timestamp") or ""),
                    query=str(t.get("query") or ""),
                    intent=str(t.get("intent") or ""),
                    tickers=[str(x) for x in (t.get("tickers") or [])],
                    gics_sector=str(t.get("gics_sector") or "ALL"),
                    start_year=int(t.get("start_year") or 2024),
                    end_year=int(t.get("end_year") or 2026),
                    variance_highlights=[str(x) for x in (t.get("variance_highlights") or [])],
                    cited_chunk_ids=[str(x) for x in (t.get("cited_chunk_ids") or [])],
                    cached_10k_keys=[str(x) for x in (t.get("cached_10k_keys") or [])],
                    token_estimate=int(t.get("token_estimate") or 0),
                )
            )
        return cls(
            session_id=str(doc.get("session_id") or ""),
            user_id=str(doc.get("user_id") or "analyst"),
            turns=turns,
            compacted_summary=str(doc.get("compacted_summary") or ""),
            compaction_count=int(doc.get("compaction_count") or 0),
            remembered_tickers=[str(x) for x in (doc.get("remembered_tickers") or [])],
            last_start_year=int(doc.get("last_start_year") or 2024),
            last_end_year=int(doc.get("last_end_year") or 2026),
            last_gics_sector=str(doc.get("last_gics_sector") or "ALL"),
            updated_at=str(doc.get("updated_at") or utc_now_iso()),
        )


class ConversationMemoryManager:
    """Manages durable, tenant-isolated conversational session state and follow-up entity recall.

    Sessions are keyed strictly by `(user_id, session_id)` so two authenticated users supplying
    the same `session_id` never collide. Per-session locking serializes concurrent `record_turn`
    load-mutate-save cycles to prevent lost updates across threads while allowing distinct
    sessions to proceed concurrently.
    """

    def __init__(
        self,
        *,
        token_threshold: int = DEFAULT_COMPACTION_TOKEN_THRESHOLD,
        retention_turns: int = DEFAULT_COMPACTION_RETENTION_SIZE,
        session_store: SessionStoreProtocol | None = None,
        refresh_ttl_seconds: float = 5.0,
        max_local_sessions: int = DEFAULT_MAX_LOCAL_SESSIONS,
        max_adk_memory_sessions: int = DEFAULT_MAX_ADK_MEMORY_SESSIONS,
    ) -> None:
        self.token_threshold = token_threshold
        self.retention_turns = retention_turns
        self.refresh_ttl_seconds = float(refresh_ttl_seconds)
        self._max_local_sessions = max(1, int(max_local_sessions))
        self._max_adk_memory_sessions = max(1, int(max_adk_memory_sessions))
        self._explicit_store = session_store
        self._default_store: SessionStoreProtocol | None = None
        self.adk_session_service = InMemorySessionService()
        self.adk_memory_service = BoundedInMemoryMemoryService(
            max_sessions=self._max_adk_memory_sessions
        )
        self._lock = threading.Lock()
        self._session_locks: dict[tuple[str, str], threading.Lock] = {}
        # LRU-bounded map keyed by (norm_user_id, norm_session_id) -> (loaded_monotonic, SessionMemoryState)
        self._sessions: OrderedDict[tuple[str, str], tuple[float, SessionMemoryState]] = (
            OrderedDict()
        )

    def _get_store(self) -> SessionStoreProtocol:
        if self._explicit_store is not None:
            return self._explicit_store
        if self._default_store is None:
            self._default_store = FirestoreSessionStore()
        return self._default_store

    @staticmethod
    def _session_key(session_id: str, user_id: str = "analyst") -> tuple[str, str]:
        norm_user = (user_id or "analyst").strip().lower() or "analyst"
        return (norm_user, session_id.strip())

    def _get_session_lock(self, key: tuple[str, str]) -> threading.Lock:
        with self._lock:
            lock = self._session_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[key] = lock
            return lock

    def _store_local_session_locked(
        self,
        key: tuple[str, str],
        state: SessionMemoryState,
    ) -> None:
        self._sessions[key] = (time.monotonic(), state)
        self._sessions.move_to_end(key)
        while len(self._sessions) > self._max_local_sessions:
            old_key, _ = self._sessions.popitem(last=False)
            self._session_locks.pop(old_key, None)

    def _evict_adk_session_state(self, session_id: str, user_id: str) -> bool:
        """Delete matching `(user_id, session_id)` entries from in-memory ADK session & memory services."""
        norm_user = (user_id or "analyst").strip().lower() or "analyst"
        candidate_users = {user_id.strip(), norm_user}
        removed_session = False
        sessions_by_app = getattr(self.adk_session_service, "sessions", None)
        if isinstance(sessions_by_app, dict):
            for app_sessions in list(sessions_by_app.values()):
                if not isinstance(app_sessions, dict):
                    continue
                for uid in candidate_users:
                    user_sessions = app_sessions.get(uid)
                    if isinstance(user_sessions, dict) and session_id in user_sessions:
                        del user_sessions[session_id]
                        removed_session = True
        removed_memory = self.adk_memory_service.drop_session(session_id, candidate_users)
        return removed_session or removed_memory

    def _load_or_refresh_session(
        self,
        session_id: str,
        user_id: str = "analyst",
        *,
        force_remote: bool = False,
    ) -> SessionMemoryState | None:
        key = self._session_key(session_id, user_id)
        now = time.monotonic()
        with self._lock:
            cached = self._sessions.get(key)
        store = self._get_store()
        live = store.is_live_enabled()
        if cached is not None:
            loaded_at, state = cached
            if (not live) or (not force_remote and (now - loaded_at) <= self.refresh_ttl_seconds):
                return state
        if not live:
            return cached[1] if cached is not None else None
        doc = store.load_session(key[0], key[1])
        if doc is None:
            return cached[1] if cached is not None else None
        hydrated = SessionMemoryState.from_firestore_dict(doc)
        with self._lock:
            self._store_local_session_locked(key, hydrated)
        return hydrated

    def recall_session_context(
        self,
        session_id: str | None,
        user_id: str = "analyst",
    ) -> dict[str, Any]:
        """Retrieve active tenant-isolated session memory and measure context recall latency."""
        t0 = time.perf_counter()
        backend_label = "firestore" if self._get_store().is_live_enabled() else "in_memory"
        norm_user = (user_id or "analyst").strip().lower()
        if not session_id or not session_id.strip():
            return {
                "session_id": None,
                "user_id": norm_user,
                "has_prior_memory": False,
                "turn_count": 0,
                "remembered_tickers": [],
                "last_start_year": None,
                "last_end_year": None,
                "last_gics_sector": "ALL",
                "compaction_count": 0,
                "memory_summary_text": "",
                "memory_tokens": 0,
                "persistence_backend": backend_label,
                "memory_lookup_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }

        sess = self._load_or_refresh_session(session_id, user_id)
        if sess is None or (not sess.turns and not sess.compacted_summary):
            return {
                "session_id": session_id.strip(),
                "user_id": norm_user,
                "has_prior_memory": False,
                "turn_count": 0,
                "remembered_tickers": [],
                "last_start_year": None,
                "last_end_year": None,
                "last_gics_sector": "ALL",
                "compaction_count": 0,
                "memory_summary_text": "",
                "memory_tokens": 0,
                "persistence_backend": backend_label,
                "memory_lookup_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
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
            "user_id": sess.user_id,
            "has_prior_memory": True,
            "turn_count": len(sess.turns),
            "remembered_tickers": list(sess.remembered_tickers),
            "last_start_year": sess.last_start_year,
            "last_end_year": sess.last_end_year,
            "last_gics_sector": sess.last_gics_sector,
            "compaction_count": sess.compaction_count,
            "memory_summary_text": summary_text,
            "memory_tokens": mem_tokens,
            "persistence_backend": backend_label,
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
        """Persist a completed turn into durable session memory and compact if over token threshold."""
        if not session_id or not session_id.strip():
            return {"persisted": False, "compaction_triggered": False, "turn_count": 0}

        key = self._session_key(session_id, user_id)
        session_lock = self._get_session_lock(key)
        compaction_triggered = False

        # Serialize load -> mutate -> remote save per `(user_id, session_id)` to prevent lost updates.
        with session_lock:
            existing = self._load_or_refresh_session(session_id, user_id)
            with self._lock:
                sess = existing
                if sess is None:
                    cached = self._sessions.get(key)
                    sess = cached[1] if cached is not None else None
                if sess is None:
                    sess = SessionMemoryState(session_id=key[1], user_id=key[0])

                now_iso = utc_now_iso()
                turn_text = f"{query}\n{answer_markdown}"
                entry = TurnMemoryEntry(
                    turn_index=len(sess.turns) + 1,
                    timestamp=now_iso,
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
                sess.updated_at = now_iso

                total_mem_tokens = sum(t.token_estimate for t in sess.turns) + estimate_token_count(
                    sess.compacted_summary
                )
                if (
                    total_mem_tokens >= self.token_threshold
                    and len(sess.turns) > self.retention_turns
                ):
                    compaction_triggered = True
                    older_turns = sess.turns[: -self.retention_turns]
                    sess.turns = sess.turns[-self.retention_turns :]
                    digest_parts: list[str] = []
                    if sess.compacted_summary:
                        digest_parts.append(sess.compacted_summary)
                    for old in older_turns:
                        tk_str = ",".join(old.tickers) if old.tickers else old.gics_sector
                        hl_str = (
                            "; ".join(old.variance_highlights[:1])
                            if old.variance_highlights
                            else old.intent
                        )
                        digest_parts.append(
                            f"T{old.turn_index}({tk_str} FY{old.start_year}-{old.end_year}): {hl_str}"
                        )
                    sess.compacted_summary = " | ".join(digest_parts)[-2000:]
                    sess.compaction_count += 1

                self._store_local_session_locked(key, sess)
                turn_count = len(sess.turns)
                compaction_count = sess.compaction_count
                firestore_payload = sess.to_firestore_dict()

            store = self._get_store()
            backend_label = "in_memory"
            if store.is_live_enabled():
                store.save_session(key[0], key[1], firestore_payload)
                backend_label = "firestore"

        return {
            "persisted": True,
            "persistence_backend": backend_label,
            "session_id": key[1],
            "user_id": key[0],
            "turn_count": turn_count,
            "compaction_triggered": compaction_triggered,
            "compaction_method": "deterministic_digest" if compaction_triggered else None,
            "compaction_count": compaction_count,
        }

    def list_user_sessions(self, user_id: str, *, limit: int = 25) -> list[dict[str, Any]]:
        """List recent sessions owned by `user_id`."""
        norm_user = (user_id or "analyst").strip().lower() or "analyst"
        store = self._get_store()
        if store.is_live_enabled():
            docs = store.list_sessions(norm_user, limit=limit)
            return [
                {
                    "session_id": str(d.get("session_id") or ""),
                    "user_id": str(d.get("user_id") or norm_user),
                    "turn_count": int(d.get("turn_count") or 0),
                    "remembered_tickers": list(d.get("remembered_tickers") or []),
                    "last_gics_sector": str(d.get("last_gics_sector") or "ALL"),
                    "last_start_year": int(d.get("last_start_year") or 2024),
                    "last_end_year": int(d.get("last_end_year") or 2026),
                    "updated_at": str(d.get("updated_at") or ""),
                }
                for d in docs
            ]
        with self._lock:
            items = [
                {
                    "session_id": s.session_id,
                    "user_id": s.user_id,
                    "turn_count": len(s.turns),
                    "remembered_tickers": list(s.remembered_tickers),
                    "last_gics_sector": s.last_gics_sector,
                    "last_start_year": s.last_start_year,
                    "last_end_year": s.last_end_year,
                    "updated_at": s.updated_at,
                }
                for (u, _), (_, s) in self._sessions.items()
                if u == norm_user
            ]
        items.sort(key=lambda d: str(d.get("updated_at") or ""), reverse=True)
        return items[:limit]

    def delete_session(self, session_id: str, user_id: str = "analyst") -> bool:
        """Delete a session belonging to `(user_id, session_id)` from L1 cache, ADK services, and Firestore."""
        key = self._session_key(session_id, user_id)
        session_lock = self._get_session_lock(key)
        with session_lock:
            with self._lock:
                removed_local = self._sessions.pop(key, None) is not None
                self._session_locks.pop(key, None)
            removed_adk = self._evict_adk_session_state(key[1], user_id)
            store = self._get_store()
            removed_remote = (
                store.delete_session(key[0], key[1]) if store.is_live_enabled() else False
            )
        return removed_local or removed_adk or removed_remote

    def persistence_status(self) -> dict[str, Any]:
        """Return health metadata for session memory storage."""
        return self._get_store().status_metadata()

    def reset(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._session_locks.clear()
            self.adk_session_service = InMemorySessionService()
            self.adk_memory_service = BoundedInMemoryMemoryService(
                max_sessions=self._max_adk_memory_sessions
            )


_GLOBAL_MEMORY_MANAGER = ConversationMemoryManager()


def get_conversation_memory_manager() -> ConversationMemoryManager:
    """Return the process-wide `ConversationMemoryManager` singleton."""
    return _GLOBAL_MEMORY_MANAGER


async def after_agent_memory_callback(
    callback_context: CallbackContext,
) -> genai_types.Content | None:
    """ADK `after_agent_callback` persisting non-ephemeral sessions to Memory Bank & tracking duration."""
    state = callback_context.state
    start_ts = state.get("_agent_start_monotonic")
    if start_ts is not None:
        state["last_agent_total_duration_ms"] = round(
            (time.perf_counter() - float(start_ts)) * 1000.0, 3
        )
    agent_name = str(getattr(callback_context, "agent_name", "") or "")
    adk_session = getattr(callback_context, "session", None)
    sess_id = getattr(adk_session, "id", None) or state.get("session_id")
    if agent_name in EPHEMERAL_AGENT_NAMES or (
        sess_id is not None and is_ephemeral_session_id(str(sess_id))
    ):
        return None
    add_mem = getattr(callback_context, "add_session_to_memory", None)
    if callable(add_mem):
        try:
            await add_mem()
        except Exception as exc:
            logger.debug("Optional ADK Memory Bank sync skipped: %s", exc)
    return None


def create_edgar_adk_app(
    root_agent: BaseAgent,
    *,
    app_name: str = "agents",
    min_cache_tokens: int = DEFAULT_CONTEXT_CACHE_MIN_TOKENS,
    cache_ttl_seconds: int = DEFAULT_CONTEXT_CACHE_TTL_SECONDS,
    compaction_token_threshold: int = DEFAULT_COMPACTION_TOKEN_THRESHOLD,
    compaction_retention_size: int = DEFAULT_COMPACTION_RETENTION_SIZE,
    summarizer_llm: BaseLlm | None = None,
    enable_context_cache: bool = True,
    plugins: list[BasePlugin] | None = None,
) -> App:
    """Create the production ADK `App` with Model Armor safety plugin, Context Caching & Compaction."""
    flash_model = resolve_model_for_task("context_compaction")
    cache_cfg = (
        ContextCacheConfig(
            min_tokens=min_cache_tokens,
            ttl_seconds=cache_ttl_seconds,
            cache_intervals=DEFAULT_CONTEXT_CACHE_INTERVALS,
        )
        if enable_context_cache
        else None
    )
    resolved_plugins: list[BasePlugin] = (
        list(plugins) if plugins is not None else [ModelArmorSafetyPlugin()]
    )
    return App(
        name=app_name,
        root_agent=root_agent,
        plugins=resolved_plugins,
        context_cache_config=cache_cfg,
        events_compaction_config=EventsCompactionConfig(
            token_threshold=compaction_token_threshold,
            event_retention_size=compaction_retention_size,
            summarizer=LlmEventSummarizer(
                llm=summarizer_llm if summarizer_llm is not None else Gemini(model=flash_model)
            ),
        ),
    )
