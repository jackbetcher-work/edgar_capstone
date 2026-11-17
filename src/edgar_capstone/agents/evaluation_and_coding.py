"""Gemini 3.5 Flash specialist agents and execution functions for Evaluation and Coding Assistance.

Model Strategy:
- `gemini-3.5-flash` (`DEFAULT_FLASH_MODEL`): Dedicated to LLM-as-a-Judge evaluation tasks
  (`NarrativeEvaluatorAgent`, `evaluate_variance_explanation`) and analytical coding assistance
  (`CodingAssistantAgent`, `generate_financial_coding_assistance`), while the financial reasoning model (`gemini-3.8-flash`)
  handles complex financial reasoning in `FinancialAnalystAgent` and `RootOrchestrator`.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.genai import types as genai_types

from edgar_capstone.agents.callbacks import (
    after_model_fact_check_callback,
    before_agent_telemetry_callback,
    before_model_safety_and_latency_callback,
    verify_narrative_against_calculations,
)
from edgar_capstone.agents.context_memory import (
    FLASH_CODING_AND_EVAL_MODEL,
    after_agent_memory_callback,
    low_thinking_config_for_model,
    resolve_model_for_task,
)
from edgar_capstone.agents.live_runtime import (
    resolve_live_adk_runtime,
    run_live_adk_coding_assistance,
    run_live_adk_narrative_evaluation,
)
from edgar_capstone.agents.prompt_store import (
    load_agent_prompt,
    make_adk_instruction_provider,
)
from edgar_capstone.agents.schemas import (
    CodingAssistanceReport,
    NarrativeEvaluationReport,
)
from edgar_capstone.agents.tools import _parse_ticker_list
from edgar_capstone.observability.tracing import traced

logger = logging.getLogger(__name__)

DEFAULT_FLASH_MODEL: str = FLASH_CODING_AND_EVAL_MODEL

NARRATIVE_EVALUATOR_INSTRUCTION: str = load_agent_prompt("narrative_evaluator")
CODING_ASSISTANT_INSTRUCTION: str = load_agent_prompt("coding_assistant")


def create_narrative_evaluator_agent(
    *,
    name: str = "NarrativeEvaluatorAgent",
    model: str | BaseLlm | None = None,
    prompt_version: str | None = None,
) -> Agent:
    """Create the `NarrativeEvaluatorAgent` ADK `LlmAgent` pinned to `gemini-3.5-flash`."""
    resolved_model = model if model is not None else resolve_model_for_task("evaluation")
    return Agent(
        name=name,
        model=resolved_model,
        description=(
            "Gemini 3.5 Flash LLM-as-a-Judge evaluation specialist scoring numeric fidelity, "
            "10-K citation grounding, variance explanation quality, and thematic coherence."
        ),
        instruction=make_adk_instruction_provider("narrative_evaluator", version=prompt_version),
        output_schema=NarrativeEvaluationReport,
        output_key="narrative_evaluation_report",
        # Thinking tokens count against max_output_tokens; keep headroom for the JSON report.
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8192,
            thinking_config=low_thinking_config_for_model(resolved_model),
        ),
        before_agent_callback=before_agent_telemetry_callback,
        after_agent_callback=after_agent_memory_callback,
        before_model_callback=before_model_safety_and_latency_callback,
        after_model_callback=after_model_fact_check_callback,
    )


def create_coding_assistant_agent(
    *,
    name: str = "CodingAssistantAgent",
    model: str | BaseLlm | None = None,
    prompt_version: str | None = None,
) -> Agent:
    """Create the `CodingAssistantAgent` ADK `LlmAgent` pinned to `gemini-3.5-flash`."""
    resolved_model = model if model is not None else resolve_model_for_task("coding_assistance")
    return Agent(
        name=name,
        model=resolved_model,
        description=(
            "Gemini 3.5 Flash coding assistance specialist generating BigQuery SQL, "
            "deterministic Python variance verification scripts, and LookML explores."
        ),
        instruction=make_adk_instruction_provider("coding_assistant", version=prompt_version),
        output_schema=CodingAssistanceReport,
        output_key="coding_assistance_report",
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=8192,
            thinking_config=low_thinking_config_for_model(resolved_model),
        ),
        before_agent_callback=before_agent_telemetry_callback,
        after_agent_callback=after_agent_memory_callback,
        before_model_callback=before_model_safety_and_latency_callback,
        after_model_callback=after_model_fact_check_callback,
    )


def _clamp(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


@traced("evaluation.narrative_judge")
def evaluate_variance_explanation(
    *,
    query: str,
    response_markdown: str,
    variance_reports: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    cached_explanations_count: int = 0,
    llm: BaseLlm | None = None,
    reference_explanation: str | None = None,
) -> NarrativeEvaluationReport:
    """Score a synthesized report with the live `NarrativeEvaluatorAgent` (`gemini-3.5-flash`).

    The deterministic numeric and citation fact-check caps the judge's scores, so a report with
    mismatched figures or missing citations can't pass on the judge's opinion alone.
    `reference_explanation` (golden eval runs only) is forwarded to the judge as ground truth.
    """
    eval_model = llm.model if llm is not None else resolve_model_for_task("evaluation")
    calc_map = {
        f"{r.get('ticker', 'UNK')}_{r.get('start_year', 0)}_{r.get('end_year', 0)}": r
        for r in variance_reports
    }
    fc = verify_narrative_against_calculations(
        response_text=response_markdown,
        calculated_variances=calc_map,
        retrieved_citations=citations,
    )
    runtime_res = resolve_live_adk_runtime(custom_llm=llm)
    raw_eval, stream_res = run_live_adk_narrative_evaluation(
        query=query,
        response_markdown=response_markdown,
        variance_reports=variance_reports,
        citations=citations,
        cached_explanations_count=cached_explanations_count,
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
    exp_score = _clamp(raw_eval.get("explanation_quality_score"), 1.0)
    thm_score = _clamp(raw_eval.get("thematic_consistency_score"), 1.0)
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
    return NarrativeEvaluationReport(
        evaluator_model=adk_lat.model_name if adk_lat else eval_model,
        task_category="evaluation",
        grade=grade,
        overall_score=overall,
        numeric_fidelity_score=round(num_score, 4),
        citation_grounding_score=round(cit_score, 4),
        explanation_quality_score=round(exp_score, 4),
        thematic_consistency_score=round(thm_score, 4),
        rationale=rationale,
        context_tokens=adk_lat.context_breakdown.total_context_tokens if adk_lat else 0,
        context_latency_ms=adk_lat.context_latency_ms if adk_lat else 0.0,
        total_latency_ms=(
            adk_lat.total_model_latency_ms if adk_lat else stream_res.total_stream_latency_ms
        ),
    )


@traced("coding.assist", record_args=("language",))
def generate_financial_coding_assistance(
    *,
    prompt: str,
    language: Literal["sql", "python", "lookml"] = "sql",
    tickers: str = "NVDA,AAPL",
    start_year: int = 2024,
    end_year: int = 2026,
    dataset_id: str = "sec_edgar",
    llm: BaseLlm | None = None,
) -> CodingAssistanceReport:
    """Generate analytical SQL, Python verification code, or LookML via the live `CodingAssistantAgent`."""
    coder_model = llm.model if llm is not None else resolve_model_for_task("coding_assistance")
    parsed_tickers = sorted(_parse_ticker_list(tickers)) or ["NVDA", "AAPL"]
    runtime_res = resolve_live_adk_runtime(custom_llm=llm)
    raw_live_code, stream_res = run_live_adk_coding_assistance(
        prompt=prompt,
        language=language,
        parsed_tickers=parsed_tickers,
        start_year=start_year,
        end_year=end_year,
        dataset_id=dataset_id,
        resolution=runtime_res,
        custom_llm=llm,
    )
    live_code_str = str(raw_live_code.get("code") or "").strip()
    if not live_code_str:
        raise ValueError("CodingAssistantAgent returned an empty code payload.")
    live_title = str(
        raw_live_code.get("artifact_title")
        or f"{language.upper()} Analysis ({', '.join(parsed_tickers)})"
    )
    live_explanation = str(
        raw_live_code.get("explanation")
        or f"Generated via live ADK CodingAssistantAgent ({coder_model})."
    )
    live_tickers = [
        str(t) for t in (raw_live_code.get("tickers_referenced") or parsed_tickers)
    ] or parsed_tickers
    adk_lat = stream_res.latency_records[-1] if stream_res.latency_records else None
    return CodingAssistanceReport(
        assistant_model=adk_lat.model_name if adk_lat else coder_model,
        task_category="coding_assistance",
        language=language,
        artifact_title=live_title,
        code=live_code_str,
        explanation=live_explanation,
        tickers_referenced=live_tickers,
        context_tokens=adk_lat.context_breakdown.total_context_tokens if adk_lat else 0,
        context_latency_ms=adk_lat.context_latency_ms if adk_lat else 0.0,
        total_latency_ms=(
            adk_lat.total_model_latency_ms if adk_lat else stream_res.total_stream_latency_ms
        ),
    )


narrative_evaluator_agent: Agent = create_narrative_evaluator_agent()
NarrativeEvaluatorAgent: Agent = narrative_evaluator_agent

coding_assistant_agent: Agent = create_coding_assistant_agent()
CodingAssistantAgent: Agent = coding_assistant_agent
