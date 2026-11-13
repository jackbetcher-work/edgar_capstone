"""Gemini 3.5 Flash specialist agents and execution functions for Evaluation and Coding Assistance.

Model Strategy:
- `gemini-3.5-flash` (`DEFAULT_FLASH_MODEL`): Dedicated to LLM-as-a-Judge evaluation tasks
  (`NarrativeEvaluatorAgent`, `evaluate_variance_explanation`) and analytical coding assistance
  (`CodingAssistantAgent`, `generate_financial_coding_assistance`), while `gemini-2.5-pro`
  handles complex financial reasoning in `FinancialAnalystAgent` and `RootOrchestrator`.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Literal

from google.adk.agents import Agent
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
    get_context_latency_tracker,
    resolve_model_for_task,
)
from edgar_capstone.agents.schemas import (
    CodingAssistanceReport,
    ContextBreakdown,
    NarrativeEvaluationReport,
)
from edgar_capstone.agents.tools import _parse_ticker_list
from edgar_capstone.vector_db.chunker import estimate_token_count

DEFAULT_FLASH_MODEL: str = FLASH_CODING_AND_EVAL_MODEL

NARRATIVE_EVALUATOR_INSTRUCTION = """You are the NarrativeEvaluatorAgent powered by Gemini 3.5 Flash (`gemini-3.5-flash`).
Your responsibility is fast, rigorous LLM-as-a-Judge evaluation of SEC 10-K financial variance reports across four axes:
1. `numeric_fidelity_score`: 100% agreement between narrative USD/percentage figures and deterministic calculation engine outputs.
2. `citation_grounding_score`: Every narrative business driver or risk claim is backed by an inline `[TICKER FYxxxx | Section | chunk_id](sec_filing_url)` citation.
3. `explanation_quality_score`: Material variances (Revenue, Operating Income, Net Income, margins) are causally explained using cached Item 7 MD&A and Item 1A Risk Factor passages.
4. `thematic_consistency_score`: Coherent multi-period or multi-company synthesis free of ungrounded speculation.
"""

CODING_ASSISTANT_INSTRUCTION = """You are the CodingAssistantAgent powered by Gemini 3.5 Flash (`gemini-3.5-flash`).
You assist financial engineers and analysts with analytical code generation and validation:
- BigQuery Standard SQL queries against `sec_edgar.v_financial_variances_yoy`, `sec_edgar.financial_metrics`, and `sec_edgar.filing_chunks`.
- Deterministic Python verification scripts (`pydantic` / `pandas`) for period-over-period variance, margin bps, and CAGR audits.
- Looker / LookML semantic explore snippets for variance and context-latency telemetry dashboards.
Never hardcode credentials or fabricate financial figures in generated code.
"""


def create_narrative_evaluator_agent(
    *,
    name: str = "NarrativeEvaluatorAgent",
    model: str | None = None,
) -> Agent:
    """Create the `NarrativeEvaluatorAgent` ADK `LlmAgent` pinned to `gemini-3.5-flash`."""
    resolved_model = model or resolve_model_for_task("evaluation")
    return Agent(
        name=name,
        model=resolved_model,
        description=(
            "Gemini 3.5 Flash LLM-as-a-Judge evaluation specialist scoring numeric fidelity, "
            "10-K citation grounding, variance explanation quality, and thematic coherence."
        ),
        instruction=NARRATIVE_EVALUATOR_INSTRUCTION,
        output_schema=NarrativeEvaluationReport,
        output_key="narrative_evaluation_report",
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=1024,
        ),
        before_agent_callback=before_agent_telemetry_callback,
        after_agent_callback=after_agent_memory_callback,
        before_model_callback=before_model_safety_and_latency_callback,
        after_model_callback=after_model_fact_check_callback,
    )


def create_coding_assistant_agent(
    *,
    name: str = "CodingAssistantAgent",
    model: str | None = None,
) -> Agent:
    """Create the `CodingAssistantAgent` ADK `LlmAgent` pinned to `gemini-3.5-flash`."""
    resolved_model = model or resolve_model_for_task("coding_assistance")
    return Agent(
        name=name,
        model=resolved_model,
        description=(
            "Gemini 3.5 Flash coding assistance specialist generating BigQuery SQL, "
            "deterministic Python variance verification scripts, and LookML explores."
        ),
        instruction=CODING_ASSISTANT_INSTRUCTION,
        output_schema=CodingAssistanceReport,
        output_key="coding_assistance_report",
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2048,
        ),
        before_agent_callback=before_agent_telemetry_callback,
        after_agent_callback=after_agent_memory_callback,
        before_model_callback=before_model_safety_and_latency_callback,
        after_model_callback=after_model_fact_check_callback,
    )


def evaluate_variance_explanation(
    *,
    query: str,
    response_markdown: str,
    variance_reports: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    cached_explanations_count: int = 0,
    llm_judge_fn: Callable[[str, str], dict[str, Any]] | None = None,
) -> NarrativeEvaluationReport:
    """Evaluate a synthesized financial variance report using `gemini-3.5-flash` and track context latency."""
    prep_start = time.perf_counter()
    eval_model = resolve_model_for_task("evaluation")

    calc_map = {
        f"{r.get('ticker', 'UNK')}_{r.get('start_year', 0)}_{r.get('end_year', 0)}": r
        for r in variance_reports
    }
    sys_tokens = estimate_token_count(NARRATIVE_EVALUATOR_INSTRUCTION)
    var_tokens = estimate_token_count(json.dumps(variance_reports, default=str))
    cit_tokens = estimate_token_count(json.dumps(citations[:8], default=str))
    prompt_tokens = estimate_token_count(f"{query}\n{response_markdown}")
    prep_ms = (time.perf_counter() - prep_start) * 1000.0

    inf_start = time.perf_counter()
    if llm_judge_fn is not None:
        raw_eval = llm_judge_fn(eval_model, response_markdown)
        num_score = float(raw_eval.get("numeric_fidelity_score", 1.0))
        cit_score = float(raw_eval.get("citation_grounding_score", 1.0))
        exp_score = float(raw_eval.get("explanation_quality_score", 1.0))
        thm_score = float(raw_eval.get("thematic_consistency_score", 1.0))
        rationale = str(raw_eval.get("rationale") or "Evaluated via Gemini 3.5 Flash judge.")
    else:
        fc = verify_narrative_against_calculations(
            response_text=response_markdown,
            calculated_variances=calc_map,
            retrieved_citations=citations,
        )
        num_score = 1.0 if fc.passed else 0.45
        if citations:
            cit_score = min(1.0, max(0.3, fc.matched_citations_count / max(1, min(len(citations), 3))))
            if not fc.has_required_citations:
                cit_score = 0.2
        else:
            cit_score = 1.0 if len(response_markdown.strip()) <= 240 else 0.75

        has_variance_section = bool(
            re.search(r"variance|revenue|operating\s+income|benchmark|diagnostic", response_markdown, re.IGNORECASE)
        )
        has_explanation_section = (
            cached_explanations_count > 0
            or bool(re.search(r"item\s*7|md&a|item\s*1a|risk\s+factor|disclosure", response_markdown, re.IGNORECASE))
        )
        exp_score = (
            1.0
            if (has_variance_section and has_explanation_section)
            else (0.8 if has_variance_section or has_explanation_section else 0.5)
        )
        thm_score = 1.0 if len(response_markdown.strip()) >= 80 else 0.65

        if fc.discrepancies:
            rationale = f"Detected grounding discrepancies: {'; '.join(fc.discrepancies)}"
        else:
            rationale = (
                f"Verified {fc.checked_metrics_count} deterministic variance metric(s), "
                f"{fc.matched_citations_count} inline 10-K citation tag(s), and "
                f"{cached_explanations_count} cached 10-K section explanation(s)."
            )

    overall = round(0.35 * num_score + 0.30 * cit_score + 0.20 * exp_score + 0.15 * thm_score, 4)
    if overall >= 0.80 and num_score >= 0.90:
        grade: Literal["pass", "needs_revision", "fail"] = "pass"
    elif overall >= 0.55:
        grade = "needs_revision"
    else:
        grade = "fail"

    inf_ms = (time.perf_counter() - inf_start) * 1000.0
    total_ctx_tokens = sys_tokens + var_tokens + cit_tokens + prompt_tokens
    out_tokens = estimate_token_count(rationale) + 24

    lat_rec = get_context_latency_tracker().record_call(
        agent_name="NarrativeEvaluatorAgent",
        model_name=eval_model,
        task_category="evaluation",
        react_phase="fact_check_and_eval",
        context_breakdown=ContextBreakdown(
            system_instruction_tokens=sys_tokens,
            variance_payload_tokens=var_tokens,
            uncached_10k_and_tool_tokens=cit_tokens,
            user_prompt_tokens=prompt_tokens,
            total_context_tokens=total_ctx_tokens,
            cached_context_tokens=0,
            uncached_context_tokens=total_ctx_tokens,
            cache_hit=False,
        ),
        output_tokens=out_tokens,
        raw_inference_latency_ms=inf_ms,
        context_prep_latency_ms=prep_ms,
    )

    return NarrativeEvaluationReport(
        evaluator_model=eval_model,
        task_category="evaluation",
        grade=grade,
        overall_score=overall,
        numeric_fidelity_score=round(num_score, 4),
        citation_grounding_score=round(cit_score, 4),
        explanation_quality_score=round(exp_score, 4),
        thematic_consistency_score=round(thm_score, 4),
        rationale=rationale,
        context_tokens=lat_rec.context_breakdown.total_context_tokens,
        context_latency_ms=lat_rec.context_latency_ms,
        total_latency_ms=lat_rec.total_model_latency_ms,
    )


def generate_financial_coding_assistance(
    *,
    prompt: str,
    language: Literal["sql", "python", "lookml"] = "sql",
    tickers: str = "NVDA,AAPL",
    start_year: int = 2024,
    end_year: int = 2026,
    dataset_id: str = "sec_edgar",
    llm_coder_fn: Callable[[str, str], dict[str, Any]] | None = None,
) -> CodingAssistanceReport:
    """Generate analytical SQL, Python verification code, or LookML explores via `gemini-3.5-flash`."""
    prep_start = time.perf_counter()
    coder_model = resolve_model_for_task("coding_assistance")
    parsed_tickers = sorted(_parse_ticker_list(tickers)) or ["NVDA", "AAPL"]
    tickers_sql = ", ".join(f"'{t}'" for t in parsed_tickers)

    sys_tokens = estimate_token_count(CODING_ASSISTANT_INSTRUCTION)
    prompt_tokens = estimate_token_count(
        f"{prompt} language={language} tickers={parsed_tickers} FY{start_year}-{end_year}"
    )
    prep_ms = (time.perf_counter() - prep_start) * 1000.0

    inf_start = time.perf_counter()
    if llm_coder_fn is not None:
        custom = llm_coder_fn(coder_model, prompt)
        title = str(custom.get("artifact_title") or "Custom Financial Analysis Snippet")
        code = str(custom.get("code") or "")
        explanation = str(custom.get("explanation") or "Generated via Gemini 3.5 Flash.")
    elif language == "python":
        title = f"Deterministic SEC 10-K Variance Verification ({', '.join(parsed_tickers)} FY{start_year}–FY{end_year})"
        code = (
            "from edgar_capstone.agents.tools import calculate_period_variances\n\n"
            f"result = calculate_period_variances(\n"
            f"    tickers={','.join(parsed_tickers)!r},\n"
            f"    start_year={start_year},\n"
            f"    end_year={end_year},\n"
            ")\n"
            "assert result['status'] == 'success'\n"
            "for report in result['variance_reports']:\n"
            "    rev = report['revenue_variance']\n"
            "    print(\n"
            "        f\"{report['ticker']} FY{report['start_year']}->FY{report['end_year']}: \"\n"
            "        f\"Revenue delta={rev['percentage_delta']:.2f}% (CAGR={rev['cagr_pct']}%)\"\n"
            "    )\n"
        )
        explanation = (
            f"Invokes `calculate_period_variances` deterministically for {', '.join(parsed_tickers)} "
            f"across FY{start_year}–FY{end_year} to audit exact USD deltas, percentage changes, and CAGRs."
        )
    elif language == "lookml":
        title = f"LookML Semantic Explore: SEC 10-K YoY Variances ({dataset_id})"
        code = (
            f"explore: sec_10k_yoy_variances {{\n"
            f"  view_name: v_financial_variances_yoy\n"
            f"  sql_always_where: ${{fiscal_year}} BETWEEN {start_year} AND {end_year} ;;\n"
            f"  always_filter: {{\n"
            f"    filters: [v_financial_variances_yoy.ticker: \"{','.join(parsed_tickers)}\"]\n"
            f"  }}\n"
            f"}}\n"
        )
        explanation = (
            "Defines a LookML semantic explore bound to the deterministic BigQuery view "
            f"`{dataset_id}.v_financial_variances_yoy` scoped to {', '.join(parsed_tickers)}."
        )
    else:
        title = f"BigQuery YoY Variance & Grounded 10-K Query ({', '.join(parsed_tickers)} FY{start_year}–FY{end_year})"
        code = (
            "SELECT\n"
            "  ticker,\n"
            "  company_name,\n"
            "  fiscal_year,\n"
            "  revenue,\n"
            "  ROUND(revenue_yoy_pct, 2) AS revenue_yoy_pct,\n"
            "  operating_income,\n"
            "  ROUND(operating_income_yoy_pct, 2) AS operating_income_yoy_pct,\n"
            "  net_income,\n"
            "  ROUND(net_income_yoy_pct, 2) AS net_income_yoy_pct,\n"
            "  sec_filing_url\n"
            f"FROM `{dataset_id}.v_financial_variances_yoy`\n"
            f"WHERE ticker IN ({tickers_sql})\n"
            f"  AND fiscal_year BETWEEN {start_year} AND {end_year}\n"
            "ORDER BY ticker, fiscal_year;\n"
        )
        explanation = (
            f"Queries authoritative BigQuery view `{dataset_id}.v_financial_variances_yoy` "
            f"for {', '.join(parsed_tickers)} across FY{start_year}–FY{end_year} with SEC EDGAR filing provenance."
        )

    inf_ms = (time.perf_counter() - inf_start) * 1000.0
    total_ctx_tokens = sys_tokens + prompt_tokens
    out_tokens = estimate_token_count(f"{code}\n{explanation}")

    lat_rec = get_context_latency_tracker().record_call(
        agent_name="CodingAssistantAgent",
        model_name=coder_model,
        task_category="coding_assistance",
        react_phase="coding_assistance",
        context_breakdown=ContextBreakdown(
            system_instruction_tokens=sys_tokens,
            user_prompt_tokens=prompt_tokens,
            total_context_tokens=total_ctx_tokens,
            uncached_context_tokens=total_ctx_tokens,
        ),
        output_tokens=out_tokens,
        raw_inference_latency_ms=inf_ms,
        context_prep_latency_ms=prep_ms,
    )

    return CodingAssistanceReport(
        assistant_model=coder_model,
        task_category="coding_assistance",
        language=language,
        artifact_title=title,
        code=code,
        explanation=explanation,
        tickers_referenced=parsed_tickers,
        context_tokens=lat_rec.context_breakdown.total_context_tokens,
        context_latency_ms=lat_rec.context_latency_ms,
        total_latency_ms=lat_rec.total_model_latency_ms,
    )


narrative_evaluator_agent: Agent = create_narrative_evaluator_agent()
NarrativeEvaluatorAgent: Agent = narrative_evaluator_agent

coding_assistant_agent: Agent = create_coding_assistant_agent()
CodingAssistantAgent: Agent = coding_assistant_agent
