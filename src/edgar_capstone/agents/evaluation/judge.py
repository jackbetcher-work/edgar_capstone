"""Gemini 3.8 Flash (low-thinking) LLM-as-a-Judge evaluation specialist for SEC 10-K narratives.

Model Strategy:
- `gemini-3.8-flash` with `ThinkingLevel.LOW` (`EDGAR_FLASH_MODEL`):
  Dedicated to LLM-as-a-Judge evaluation tasks (`NarrativeEvaluatorAgent`,
  `evaluate_variance_explanation`), while `FinancialAnalystAgent` runs `gemini-3.8-flash` with
  default native thinking and `SectorMapSummarizerAgent` / `LlmEventSummarizer` run
  `gemini-3.5-flash-lite` (`EDGAR_LITE_MODEL`).
"""

from __future__ import annotations

from typing import Any, Literal

from google.adk.models.base_llm import BaseLlm

from edgar_capstone.agents.core.schemas import (
    FactCheckVerification,
    ModelContextLatencyRecord,
    NarrativeEvaluationReport,
)
from edgar_capstone.agents.runtime.callbacks import verify_narrative_against_calculations
from edgar_capstone.agents.runtime.live_runtime import (
    resolve_live_adk_runtime,
    run_live_adk_narrative_evaluation,
)
from edgar_capstone.agents.runtime.model_config import resolve_model_for_task
from edgar_capstone.agents.tools.runtime import variance_map
from edgar_capstone.observability.tracing import traced


def _clamp(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


@traced("evaluation.narrative_judge")
def evaluate_variance_explanation_with_latency(
    *,
    query: str,
    response_markdown: str,
    variance_reports: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    llm: BaseLlm | None = None,
    reference_explanation: str | None = None,
    fact_check: FactCheckVerification | None = None,
) -> tuple[NarrativeEvaluationReport, list[ModelContextLatencyRecord]]:
    """Score a synthesized report and return `(report, latency_records)` for THIS invocation.

    Returning `stream_res.latency_records` directly avoids cross-request concurrency races
    when callers previously read `ContextLatencyTracker.recent_records(limit=1)` globally.
    """
    eval_model = llm.model if llm is not None else resolve_model_for_task("evaluation")
    fc = (
        fact_check
        if fact_check is not None
        else verify_narrative_against_calculations(
            response_text=response_markdown,
            calculated_variances=variance_map(variance_reports),
            retrieved_citations=citations,
        )
    )
    runtime_res = resolve_live_adk_runtime(custom_llm=llm)
    raw_eval, stream_res = run_live_adk_narrative_evaluation(
        query=query,
        response_markdown=response_markdown,
        variance_reports=variance_reports,
        citations=citations,
        resolution=runtime_res,
        custom_llm=llm,
        reference_explanation=reference_explanation,
    )

    num_score = _clamp(raw_eval.get("numeric_fidelity_score"), 1.0 if fc.passed else 0.45)
    if not fc.passed:
        num_score = min(num_score, 0.45)
    cit_score = _clamp(raw_eval.get("citation_grounding_score"), 1.0)
    if citations and not fc.has_required_citations:
        cit_score = min(cit_score, 0.25)
    raw_prec = raw_eval.get("citation_precision_score")
    cit_precision = _clamp(raw_prec, cit_score) if raw_prec is not None else cit_score
    exp_score = _clamp(raw_eval.get("explanation_quality_score"), 1.0)
    thm_score = _clamp(raw_eval.get("thematic_consistency_score"), 1.0)
    try:
        raw_unsup = int(raw_eval.get("unsupported_claims_count", 0) or 0)
    except (TypeError, ValueError):
        raw_unsup = 0
    unsupported_claims = max(0, raw_unsup)
    rationale = str(
        raw_eval.get("rationale")
        or f"Evaluated via live ADK NarrativeEvaluatorAgent ({eval_model})."
    )
    overall = round(0.35 * num_score + 0.30 * cit_score + 0.20 * exp_score + 0.15 * thm_score, 4)
    if overall >= 0.80 and num_score >= 0.90:
        grade: Literal["pass", "needs_revision", "fail"] = "pass"
    elif overall >= 0.55:
        grade = "needs_revision"
    else:
        grade = "fail"

    adk_lat = stream_res.latency_records[-1] if stream_res.latency_records else None
    report = NarrativeEvaluationReport(
        evaluator_model=adk_lat.model_name if adk_lat else eval_model,
        task_category="evaluation",
        grade=grade,
        overall_score=overall,
        numeric_fidelity_score=round(num_score, 4),
        citation_grounding_score=round(cit_score, 4),
        citation_precision_score=round(cit_precision, 4),
        explanation_quality_score=round(exp_score, 4),
        thematic_consistency_score=round(thm_score, 4),
        unsupported_claims_count=unsupported_claims,
        rationale=rationale,
        context_tokens=adk_lat.context_breakdown.total_context_tokens if adk_lat else 0,
        context_latency_ms=adk_lat.context_latency_ms if adk_lat else 0.0,
        total_latency_ms=(
            adk_lat.total_model_latency_ms if adk_lat else stream_res.total_stream_latency_ms
        ),
    )
    return report, list(stream_res.latency_records)


def evaluate_variance_explanation(
    *,
    query: str,
    response_markdown: str,
    variance_reports: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    llm: BaseLlm | None = None,
    reference_explanation: str | None = None,
    fact_check: FactCheckVerification | None = None,
) -> NarrativeEvaluationReport:
    """Score a synthesized report with the live `NarrativeEvaluatorAgent` (`gemini-3.8-flash`, low thinking).

    The deterministic numeric and citation fact-check caps the judge's scores, so a report with
    mismatched figures or missing citations can't pass on the judge's opinion alone.
    `reference_explanation` (golden eval runs only) is forwarded to the judge as ground truth.
    """
    report, _ = evaluate_variance_explanation_with_latency(
        query=query,
        response_markdown=response_markdown,
        variance_reports=variance_reports,
        citations=citations,
        llm=llm,
        reference_explanation=reference_explanation,
        fact_check=fact_check,
    )
    return report


__all__ = [
    "evaluate_variance_explanation",
    "evaluate_variance_explanation_with_latency",
]
