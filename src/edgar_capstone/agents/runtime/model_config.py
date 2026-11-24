"""Model tier specialization, token pricing/throughput profiles, and thinking configuration.

Implements the model routing strategy:
- `gemini-3.8-flash` (`EDGAR_ANALYST_MODEL`): Complex financial reasoning, multi-period variance
  synthesis, and ReAct orchestration (`FinancialAnalystAgent`, native default thinking).
- `gemini-3.8-flash` (`EDGAR_FLASH_MODEL`, `ThinkingLevel.LOW`): LLM-as-a-Judge evaluation tasks
  (`NarrativeEvaluatorAgent`).
- `gemini-3.5-flash-lite` (`EDGAR_LITE_MODEL`): High-throughput sector map-reduce batch extraction
  (`SectorMapSummarizerAgent`) and ADK app event compaction (`LlmEventSummarizer`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.genai import types as genai_types

from edgar_capstone.agents.core.schemas import ModelTaskCategory
from edgar_capstone.common.env import env_positive_float

FINANCIAL_REASONING_MODEL: str = os.environ.get("EDGAR_ANALYST_MODEL", "gemini-3.8-flash")
FLASH_CODING_AND_EVAL_MODEL: str = os.environ.get("EDGAR_FLASH_MODEL", "gemini-3.8-flash")
FLASH_LITE_MODEL: str = os.environ.get("EDGAR_LITE_MODEL", "gemini-3.5-flash-lite")

DEFAULT_TTFT_SLO_MS: float = 3000.0
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
        "SectorMapSummarizerAgent",
        "ScopeContinuityAgent",
    }
)
EPHEMERAL_SESSION_PREFIXES: tuple[str, ...] = (
    "eval-",
    "sector-map-",
    "scope-resolve-",
    "adk-sess-",
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
    cached_prefill_discount_factor: float  # Fraction of uncached prefill cost when cached
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
            "evaluation",
            "llm_as_judge",
            "coding_assistance",
        ),
        uncached_prefill_ms_per_token=0.05,
        cached_prefill_discount_factor=0.18,
        decode_ms_per_token=0.60,
        uncached_input_usd_per_1m=0.75,
        cached_input_usd_per_1m=0.075,
        output_usd_per_1m=3.75,
    ),
    "gemini-3.5-flash-lite": ModelPerformanceProfile(
        model_name="gemini-3.5-flash-lite",
        tier="fast_compaction_and_mapping",
        primary_use_cases=(
            "context_compaction",
            "context_cache_warmup",
        ),
        uncached_prefill_ms_per_token=0.03,
        cached_prefill_discount_factor=0.18,
        decode_ms_per_token=0.35,
        uncached_input_usd_per_1m=0.30,
        cached_input_usd_per_1m=0.03,
        output_usd_per_1m=2.50,
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
        uncached_input_usd_per_1m=1.50,
        cached_input_usd_per_1m=0.15,
        output_usd_per_1m=9.00,
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
    "context_compaction": "fast_compaction_and_mapping",
    "context_cache_warmup": "fast_compaction_and_mapping",
}

AGENT_DEFAULT_TASK_CATEGORY: dict[str, ModelTaskCategory] = {
    "RootOrchestrator": "orchestration",
    "FinancialAnalystAgent": "financial_reasoning",
    "ContextCacheManager": "context_cache_warmup",
    "NarrativeEvaluatorAgent": "evaluation",
    "SectorMapSummarizerAgent": "context_compaction",
    "ScopeContinuityAgent": "context_compaction",
}


def minimal_thinking_config_for_model(
    model: str | BaseLlm,
) -> genai_types.ThinkingConfig | None:
    """Minimal/zero thinking for ultra-low-latency structured classification (scope/continuity)."""
    name = (model.model if isinstance(model, BaseLlm) else str(model or "")).lower()
    short = name.rsplit("/", maxsplit=1)[-1]
    if short.startswith("gemini-3"):
        return genai_types.ThinkingConfig(thinking_level=genai_types.ThinkingLevel.MINIMAL)
    if short.startswith("gemini-2.5"):
        return genai_types.ThinkingConfig(thinking_budget=0)
    return None


def low_thinking_config_for_model(
    model: str | BaseLlm,
    *,
    include_thoughts: bool | None = None,
) -> genai_types.ThinkingConfig | None:
    """Low-effort thinking for structured Gemini tasks (judge, map-reduce, repair retries).

    Empirically A/B tested on the 43-item live golden suite: Gemini 3.x native
    `ThinkingLevel.LOW` outperformed legacy `thinking_budget=512` on both speed and quality
    (p50 TTFT 1,981 ms vs 2,251 ms, p95 4,713 ms vs 7,750 ms, grounding recall 0.9444 vs
    0.9028, citation coverage 1.0 vs 0.9677). Gemini 2.5 uses `thinking_budget=1024`.
    """
    name = (model.model if isinstance(model, BaseLlm) else str(model or "")).lower()
    short = name.rsplit("/", maxsplit=1)[-1]
    extra: dict[str, Any] = {}
    if include_thoughts is not None:
        extra["include_thoughts"] = include_thoughts
    if short.startswith("gemini-3"):
        return genai_types.ThinkingConfig(
            thinking_level=genai_types.ThinkingLevel.LOW,
            **extra,
        )
    if short.startswith("gemini-2.5"):
        return genai_types.ThinkingConfig(thinking_budget=1024, **extra)
    return None


def resolve_model_for_task(task_category: ModelTaskCategory) -> str:
    """Resolve the active Gemini model identifier for a given workload category."""
    tier = TASK_TO_MODEL_TIER.get(task_category, "complex_financial_reasoning")
    if tier == "fast_compaction_and_mapping":
        return os.environ.get("EDGAR_LITE_MODEL", FLASH_LITE_MODEL)
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
    # Match longest prefix first so versioned "gemini-3.5-flash-lite-001" beats "gemini-3.5-flash".
    for known in sorted(MODEL_PROFILES, key=len, reverse=True):
        if norm.startswith(known):
            return MODEL_PROFILES[known]
    # No silent default: pricing and latency telemetry for an unlisted model would be wrong.
    raise ValueError(
        f"No performance/pricing profile for model {model_name!r}. Add it to MODEL_PROFILES "
        f"in model_config.py (known models: {', '.join(sorted(MODEL_PROFILES))})."
    )


def get_model_strategy_metadata() -> dict[str, Any]:
    """Return public metadata describing the model tier specialization & context strategy."""
    fin_model = resolve_model_for_task("financial_reasoning")
    flash_model = resolve_model_for_task("evaluation")
    lite_model = resolve_model_for_task("context_compaction")
    return {
        "financial_reasoning_model": fin_model,
        "coding_and_evaluation_model": flash_model,
        "compaction_and_mapping_model": lite_model,
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
            "adk_app_summarizer_model": lite_model,
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
