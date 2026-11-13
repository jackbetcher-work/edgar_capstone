"""Structured ReAct Execution Loop for SEC 10-K Financial Variance Reasoning.

Implements the structured ReAct loop (`Thought -> Action -> Observation -> ... -> Final Output`):
1. Evaluate Calculated Variances: Analyzes deterministic Revenue, Operating Income, Net Income,
   and margin bps variances (`CompanyVarianceReport`), isolating primary movers, YoY inflections,
   and operating-leverage divergences to formulate variance-targeted 10-K queries.
2. Query Cached 10-K Context for Explanation: Queries resident/warmed 10-K sections (`item_7` MD&A
   and `item_1a` Risk Factors) via `query_cached_10k_context` + metadata-filtered Hybrid RAG,
   adapting across iterations if peer or boundary-year coverage is incomplete.
3. Format Grounded Output & Evaluate: Synthesizes the grounded report using `gemini-2.5-pro`
   financial reasoning (with strict numeric & citation fact-checking) and scores output quality
   using `gemini-3.5-flash`, tracking context-induced latency across every model call.
"""

from __future__ import annotations

from collections import defaultdict
import json
import re
import time
from typing import Any, Callable

from edgar_capstone.agents.callbacks import verify_narrative_against_calculations
from edgar_capstone.agents.context_memory import (
    get_context_latency_tracker,
    resolve_model_for_task,
)
from edgar_capstone.agents.evaluation_and_coding import evaluate_variance_explanation
from edgar_capstone.agents.schemas import (
    Cached10KExplanation,
    ContextBreakdown,
    ModelContextLatencyRecord,
    NarrativeEvaluationReport,
    ReActStep,
    VarianceEvaluationItem,
)
from edgar_capstone.agents.tools import query_cached_10k_context
from edgar_capstone.vector_db.chunker import estimate_token_count


def _fmt_usd(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "-" if value < 0 else ""
    abs_v = abs(value)
    if abs_v >= 1_000_000_000:
        return f"{sign}${abs_v / 1_000_000_000:,.2f}B"
    if abs_v >= 1_000_000:
        return f"{sign}${abs_v / 1_000_000:,.2f}M"
    return f"{sign}${abs_v:,.0f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def _fmt_bps(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f} bps"


def _clean_passage(text: str, max_len: int = 320) -> str:
    cleaned = re.sub(r"\\([.\-(),&])", r"\1", text)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_len:
        return cleaned
    clipped = cleaned[:max_len].rsplit(" ", 1)[0]
    return f"{clipped}..."


def _inline_citation_tag(cit: dict[str, Any]) -> str:
    ticker = cit["ticker"]
    fy = cit["fiscal_year"]
    sec_label = (
        "Item 1A Risk Factors" if cit["section_id"] == "item_1a" else "Item 7 MD&A"
    )
    chunk_id = cit["chunk_id"]
    url = cit.get("sec_filing_url") or "#source-viewer"
    return f"[{ticker} FY{fy} | {sec_label} | {chunk_id}]({url})"


def evaluate_calculated_variances(
    variance_reports: list[dict[str, Any]],
    *,
    query: str = "",
    section_filter: str = "ALL",
) -> list[VarianceEvaluationItem]:
    """Step 1 of ReAct reasoning: evaluate deterministic variances to isolate drivers & 10-K targets."""
    evaluations: list[VarianceEvaluationItem] = []
    sec_norm = (section_filter or "ALL").strip().lower()

    for rep in variance_reports:
        ticker = str(rep["ticker"])
        company_name = str(rep["company_name"])
        fy_s = int(rep["start_year"])
        fy_e = int(rep["end_year"])

        rev = rep.get("revenue_variance") or {}
        op = rep.get("operating_income_variance") or {}
        ni = rep.get("net_income_variance") or {}

        rev_pct = rev.get("percentage_delta")
        op_pct = op.get("percentage_delta")
        ni_pct = ni.get("percentage_delta")
        op_bps = rep.get("operating_margin_bps_delta")
        ni_bps = rep.get("net_margin_bps_delta")

        candidates: list[tuple[float, str]] = [
            (abs(float(rev_pct)) if rev_pct is not None else -1.0, "revenue"),
            (abs(float(op_pct)) if op_pct is not None else -1.0, "operating_income"),
            (abs(float(ni_pct)) if ni_pct is not None else -1.0, "net_income"),
        ]
        candidates.sort(key=lambda x: x[0], reverse=True)
        primary_mover = candidates[0][1] if candidates[0][0] >= 0 else "revenue"

        findings: list[str] = []
        if rev_pct is not None:
            cagr_txt = (
                f", {_fmt_pct(rev.get('cagr_pct'))} CAGR"
                if rev.get("cagr_pct") is not None and fy_e > fy_s
                else ""
            )
            findings.append(
                f"Revenue {rev.get('direction', 'moved')} {_fmt_pct(rev_pct)} "
                f"({_fmt_usd(rev.get('start_value'))} -> {_fmt_usd(rev.get('end_value'))}, "
                f"Δ {_fmt_usd(rev.get('absolute_delta'))}{cagr_txt})"
            )
        if op_pct is not None:
            findings.append(
                f"Operating Income {op.get('direction', 'moved')} {_fmt_pct(op_pct)} "
                f"to {_fmt_usd(op.get('end_value'))} (Operating Margin {_fmt_bps(op_bps)})"
            )
        if ni_pct is not None:
            findings.append(
                f"Net Income {ni.get('direction', 'moved')} {_fmt_pct(ni_pct)} "
                f"to {_fmt_usd(ni.get('end_value'))} (Net Margin {_fmt_bps(ni_bps)})"
            )

        # Detect peak YoY step inflection from annual_series
        annual_steps = [
            s
            for s in (rep.get("annual_series") or [])
            if s.get("revenue_yoy_pct") is not None
        ]
        if annual_steps:
            peak_step = max(
                annual_steps, key=lambda s: abs(float(s["revenue_yoy_pct"]))
            )
            findings.append(
                f"Peak YoY revenue inflection in FY{peak_step['fiscal_year']} "
                f"({_fmt_pct(peak_step['revenue_yoy_pct'])} vs FY{peak_step.get('prev_fiscal_year')})"
            )

        # Detect operating leverage or margin divergences requiring 10-K explanation
        divergences: list[str] = []
        if rev_pct is not None and op_pct is not None:
            spread = float(op_pct) - float(rev_pct)
            if spread >= 5.0:
                divergences.append(
                    f"Positive operating leverage: Operating Income growth ({_fmt_pct(op_pct)}) "
                    f"exceeded Revenue growth ({_fmt_pct(rev_pct)}) by +{spread:.2f} pts"
                )
            elif spread <= -5.0:
                divergences.append(
                    f"Operating margin pressure: Operating Income growth ({_fmt_pct(op_pct)}) "
                    f"trailed Revenue growth ({_fmt_pct(rev_pct)}) by {spread:.2f} pts"
                )
        if op_pct is not None and ni_pct is not None:
            ni_spread = float(ni_pct) - float(op_pct)
            if abs(ni_spread) >= 8.0:
                divergences.append(
                    f"Below-the-line variance: Net Income ({_fmt_pct(ni_pct)}) diverged from "
                    f"Operating Income ({_fmt_pct(op_pct)}) by {_fmt_pct(ni_spread)}"
                )

        # Build targeted cached 10-K section queries driven by the variance evaluation
        targeted_queries: list[dict[str, Any]] = []
        q_clean = query.strip()
        if sec_norm in ("all", "item_7"):
            mda_focus = (
                f"{ticker} FY{fy_e} revenue operating income margin segment drivers R&D {q_clean}"
            ).strip()
            targeted_queries.append(
                {
                    "ticker": ticker,
                    "fiscal_year": fy_e,
                    "section_id": "item_7",
                    "variance_query": mda_focus,
                    "driver_focus": f"FY{fy_s}–FY{fy_e} {primary_mover.replace('_', ' ')} & margin variance drivers",
                }
            )
        if sec_norm in ("all", "item_1a"):
            risk_focus = (
                f"{ticker} FY{fy_e} risk factors headwinds supply chain export controls competition {q_clean}"
            ).strip()
            targeted_queries.append(
                {
                    "ticker": ticker,
                    "fiscal_year": fy_e,
                    "section_id": "item_1a",
                    "variance_query": risk_focus,
                    "driver_focus": f"FY{fy_e} material risk disclosures & operational headwinds",
                }
            )

        evaluations.append(
            VarianceEvaluationItem(
                ticker=ticker,
                company_name=company_name,
                start_year=fy_s,
                end_year=fy_e,
                primary_metric_mover=primary_mover,  # type: ignore[arg-type]
                revenue_pct_delta=rev_pct,
                operating_income_pct_delta=op_pct,
                net_income_pct_delta=ni_pct,
                operating_margin_bps_delta=op_bps,
                net_margin_bps_delta=ni_bps,
                material_findings=findings,
                divergences=divergences,
                targeted_10k_queries=targeted_queries,
            )
        )

    return evaluations


def query_cached_10k_for_explanations(
    *,
    variance_evaluations: list[VarianceEvaluationItem],
    citations: list[dict[str, Any]],
    query: str,
    section_filter: str = "ALL",
    max_queries: int = 6,
    on_cache_call: Callable[[str, int, str], None] | None = None,
) -> tuple[list[Cached10KExplanation], list[dict[str, Any]]]:
    """Step 2 of ReAct reasoning: query cached 10-K sections for causal explanations of variances."""
    planned_requests: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int, str]] = set()

    for ev in variance_evaluations:
        for tq in ev.targeted_10k_queries:
            key = (str(tq["ticker"]), int(tq["fiscal_year"]), str(tq["section_id"]))
            if key not in seen_keys:
                seen_keys.add(key)
                planned_requests.append(tq)

    # Fallback when query is thematic / cross-sector without pre-evaluated company variances
    if not planned_requests and citations:
        sec_norm = (section_filter or "ALL").strip().lower()
        for cit in citations[:4]:
            sec_id = (
                sec_norm if sec_norm in ("item_1a", "item_7") else str(cit["section_id"])
            )
            key = (str(cit["ticker"]), int(cit["fiscal_year"]), sec_id)
            if key not in seen_keys:
                seen_keys.add(key)
                planned_requests.append(
                    {
                        "ticker": str(cit["ticker"]),
                        "fiscal_year": int(cit["fiscal_year"]),
                        "section_id": sec_id,
                        "variance_query": query,
                        "driver_focus": f"Thematic 10-K disclosure context ({cit['ticker']} FY{cit['fiscal_year']})",
                    }
                )

    explanations: list[Cached10KExplanation] = []
    linked_citations: list[dict[str, Any]] = []

    for req in planned_requests[:max_queries]:
        if on_cache_call is not None:
            on_cache_call(req["ticker"], int(req["fiscal_year"]), req["section_id"])

        res = query_cached_10k_context(
            ticker=req["ticker"],
            fiscal_year=int(req["fiscal_year"]),
            section_id=req["section_id"],
            variance_query=req["variance_query"],
            max_passages=2,
        )
        if res.get("status") != "success":
            continue

        explanations.append(
            Cached10KExplanation(
                cache_key=str(res["cache_key"]),
                cache_hit=bool(res["cache_hit"]),
                ticker=str(res["ticker"]),
                company_name=str(res["company_name"]),
                fiscal_year=int(res["fiscal_year"]),
                section_id=res["section_id"],
                section_title=str(res["section_title"]),
                variance_driver_focus=str(req["driver_focus"]),
                extracted_explanations=list(res.get("explanations") or []),
                matched_chunk_ids=list(res.get("matched_chunk_ids") or []),
                token_estimate=int(res.get("token_estimate") or 0),
                lookup_latency_ms=float(res.get("lookup_latency_ms") or 0.0),
                sec_filing_url=str(res.get("sec_filing_url") or ""),
            )
        )
        for mc in res.get("matched_citations") or []:
            linked_citations.append(mc)

    return explanations, linked_citations


def format_react_grounded_output(
    *,
    query: str,
    intent: str,
    scope: dict[str, Any],
    variance_reports: list[dict[str, Any]],
    variance_evaluations: list[VarianceEvaluationItem],
    cached_explanations: list[Cached10KExplanation],
    citations: list[dict[str, Any]],
    sector_benchmarks: list[dict[str, Any]],
    chunk_health: dict[str, Any] | None,
) -> str:
    """Step 3 of ReAct reasoning: format evaluated variances + cached 10-K explanations with citations."""
    lines: list[str] = []

    if intent == "chunk_health_monitoring" and chunk_health:
        lines.append("### Vector Corpus & Hybrid Index Diagnostics")
        lines.append(
            f"- **Corpus Status:** `{chunk_health.get('health_status', 'HEALTHY')}` "
            f"(Health Score: **{chunk_health.get('health_score', 1.0)}**)"
        )
        lines.append(
            f"- **Indexed 10-K Chunks:** **{chunk_health.get('total_chunks', 0):,}** "
            f"across 768-dimensional embeddings (`embedding_coverage_ratio`: "
            f"**{float(chunk_health.get('embedding_coverage_ratio') or 1.0) * 100:.1f}%**)"
        )
        by_sec = chunk_health.get("by_section") or {}
        if by_sec:
            lines.append(
                f"- **Section Distribution:** Item 1A Risk Factors (`{by_sec.get('item_1a', 0):,}` chunks) · "
                f"Item 7 MD&A (`{by_sec.get('item_7', 0):,}` chunks)"
            )
        if citations:
            lines.append("\n### Sample Retrieved Grounding Passages")
            for cit in citations[:3]:
                lines.append(
                    f"- **{_clean_passage(cit['excerpt'], 240)}** {_inline_citation_tag(cit)}"
                )
        return "\n".join(lines)

    # 1. Executive Synthesis Header
    if intent == "multi_company_peer_comparison" and len(variance_reports) >= 2:
        names = ", ".join(
            f"**{r['company_name']} ({r['ticker']})**" for r in variance_reports
        )
        lines.append("### Multi-Company Variance & Filing Synthesis")
        lines.append(
            f"Structured ReAct multi-period analysis comparing {names} across "
            f"**FY{scope['start_year']}–FY{scope['end_year']}**, grounded in cached SEC Form 10-K "
            f"Item 7 (MD&A) and Item 1A (Risk Factors) disclosures."
        )
    elif variance_reports:
        rep0 = variance_reports[0]
        lines.append(
            f"### {rep0['company_name']} ({rep0['ticker']}) — FY{rep0['start_year']}–FY{rep0['end_year']} Grounded Variance Synthesis"
        )
        lines.append(
            f"Period-over-period financial performance and cached SEC Form 10-K narrative explanations for "
            f"**{rep0['company_name']}** (`CIK {rep0['cik']}` · *{rep0['gics_sector']} — {rep0['gics_sub_industry']}*)."
        )
    else:
        sector_desc = (
            f"**{scope['gics_sector']}**"
            if scope["gics_sector"] != "ALL"
            else "**S&P 500 Curated Corpus**"
        )
        lines.append(
            f"### Longitudinal 10-K Thematic Synthesis ({sector_desc}, FY{scope['start_year']}–FY{scope['end_year']})"
        )
        lines.append(
            f"Cross-filing ReAct retrieval combining cached 10-K context and 768-d hybrid search "
            f"for query: *\"{query.strip()}\"*."
        )

    # 2. Deterministic Quantitative Findings + Evaluated Variance Insights
    eval_by_ticker = {ev.ticker: ev for ev in variance_evaluations}
    cit_by_id = {c["chunk_id"]: c for c in citations}

    if variance_reports:
        lines.append("\n### Deterministic Period-over-Period Metrics & Variance Evaluation")
        for rep in variance_reports:
            rev = rep["revenue_variance"]
            op = rep["operating_income_variance"]
            ni = rep["net_income_variance"]
            fy_s, fy_e = rep["start_year"], rep["end_year"]

            cits_for_ticker = [c for c in citations if c["ticker"] == rep["ticker"]]
            primary_cit_tag = (
                f" {_inline_citation_tag(cits_for_ticker[0])}" if cits_for_ticker else ""
            )

            cagr_clause = (
                f" ({_fmt_pct(rev.get('cagr_pct'))} CAGR)"
                if rev.get("cagr_pct") is not None and fy_e > fy_s
                else ""
            )
            lines.append(
                f"- **{rep['ticker']} (FY{fy_s} → FY{fy_e}):** "
                f"Revenue moved from **{_fmt_usd(rev.get('start_value'))}** to "
                f"**{_fmt_usd(rev.get('end_value'))}** "
                f"(**{_fmt_pct(rev.get('percentage_delta'))}**, "
                f"Δ **{_fmt_usd(rev.get('absolute_delta'))}**{cagr_clause}); "
                f"Operating Income moved **{_fmt_pct(op.get('percentage_delta'))}** to "
                f"**{_fmt_usd(op.get('end_value'))}** "
                f"(Operating Margin **{_fmt_bps(rep.get('operating_margin_bps_delta'))}**); "
                f"Net Income moved **{_fmt_pct(ni.get('percentage_delta'))}** to "
                f"**{_fmt_usd(ni.get('end_value'))}** "
                f"(Net Margin **{_fmt_bps(rep.get('net_margin_bps_delta'))}**).{primary_cit_tag}"
            )
            ev = eval_by_ticker.get(rep["ticker"])
            if ev and ev.divergences:
                lines.append(
                    f"  - *ReAct Variance Diagnostic ({rep['ticker']}):* {'; '.join(ev.divergences)}."
                )

    elif sector_benchmarks:
        lines.append("\n### Sector Quantitative Benchmarks")
        for bench in sector_benchmarks[-3:]:
            lines.append(
                f"- **{bench['gics_sector']} (FY{bench['fiscal_year']} · {bench['company_count']} issuers):** "
                f"Aggregate Revenue **{_fmt_usd(bench.get('total_sector_revenue'))}**, "
                f"Operating Income **{_fmt_usd(bench.get('total_sector_operating_income'))}**, "
                f"Net Income **{_fmt_usd(bench.get('total_sector_net_income'))}**."
            )

    # 3. Cached 10-K Context Explanations & Hybrid RAG Narrative Drivers
    if intent == "thematic_trend_tracking" and citations:
        lines.append("\n### Longitudinal Disclosure Evolution Across Fiscal Periods")
        by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for cit in citations:
            by_year[int(cit["fiscal_year"])].append(cit)
        for fy in sorted(by_year.keys()):
            lines.append(f"#### Fiscal Year {fy} Filing Disclosures")
            for cit in by_year[fy][:3]:
                sec_title = cit.get("section_title") or cit["section_id"].upper()
                lines.append(
                    f"- **{cit['company_name']} ({cit['ticker']}) — {sec_title}:** "
                    f"\"{_clean_passage(cit['excerpt'], 290)}\" {_inline_citation_tag(cit)}"
                )
    elif cached_explanations or citations:
        mda_cached = [e for e in cached_explanations if e.section_id == "item_7" and e.extracted_explanations]
        risk_cached = [e for e in cached_explanations if e.section_id == "item_1a" and e.extracted_explanations]
        item7_cits = [c for c in citations if c["section_id"] == "item_7"]
        item1a_cits = [c for c in citations if c["section_id"] == "item_1a"]

        if mda_cached or item7_cits:
            lines.append("\n### Item 7 (MD&A) Operational & Financial Variance Explanations")
            rendered_mda = 0
            for exp in mda_cached:
                linked_cit = next(
                    (cit_by_id[cid] for cid in exp.matched_chunk_ids if cid in cit_by_id),
                    next(
                        (
                            c
                            for c in item7_cits
                            if c["ticker"] == exp.ticker and int(c["fiscal_year"]) == exp.fiscal_year
                        ),
                        item7_cits[0] if item7_cits else None,
                    ),
                )
                cit_tag = f" {_inline_citation_tag(linked_cit)}" if linked_cit else ""
                lines.append(
                    f"- **{exp.ticker} FY{exp.fiscal_year} ({exp.variance_driver_focus}):** "
                    f"\"{_clean_passage(exp.extracted_explanations[0], 300)}\"{cit_tag}"
                )
                rendered_mda += 1
            for cit in item7_cits[: max(0, 4 - rendered_mda)]:
                sec_title = cit.get("section_title") or "Item 7. MD&A"
                lines.append(
                    f"- **{cit['ticker']} FY{cit['fiscal_year']} ({sec_title}):** "
                    f"\"{_clean_passage(cit['excerpt'], 300)}\" {_inline_citation_tag(cit)}"
                )

        if risk_cached or item1a_cits:
            lines.append("\n### Item 1A Risk Factor Disclosures & Headwinds")
            rendered_risk = 0
            for exp in risk_cached:
                linked_cit = next(
                    (cit_by_id[cid] for cid in exp.matched_chunk_ids if cid in cit_by_id),
                    next(
                        (
                            c
                            for c in item1a_cits
                            if c["ticker"] == exp.ticker and int(c["fiscal_year"]) == exp.fiscal_year
                        ),
                        item1a_cits[0] if item1a_cits else None,
                    ),
                )
                cit_tag = f" {_inline_citation_tag(linked_cit)}" if linked_cit else ""
                lines.append(
                    f"- **{exp.ticker} FY{exp.fiscal_year} ({exp.variance_driver_focus}):** "
                    f"\"{_clean_passage(exp.extracted_explanations[0], 300)}\"{cit_tag}"
                )
                rendered_risk += 1
            for cit in item1a_cits[: max(0, 4 - rendered_risk)]:
                sec_title = cit.get("section_title") or "Item 1A. Risk Factors"
                lines.append(
                    f"- **{cit['ticker']} FY{cit['fiscal_year']} ({sec_title}):** "
                    f"\"{_clean_passage(cit['excerpt'], 300)}\" {_inline_citation_tag(cit)}"
                )
    else:
        lines.append(
            "\n### Grounding Status\n"
            "No 10-K filing passages matched the current filter combination. "
            "Widen the fiscal period range or reset the GICS sector / section filter to inspect broader disclosures."
        )

    return "\n".join(lines)


class StructuredReActExecutor:
    """Executes the structured ReAct loop (`evaluate_variances -> query_cached_10k_context -> format_grounded_output`)."""

    def __init__(
        self,
        *,
        max_iterations: int = 5,
        llm_reasoning_fn: Callable[[str, dict[str, Any]], str] | None = None,
        llm_judge_fn: Callable[[str, str], dict[str, Any]] | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.llm_reasoning_fn = llm_reasoning_fn
        self.llm_judge_fn = llm_judge_fn

    def execute_post_retrieval_reasoning(
        self,
        *,
        query: str,
        intent: str,
        scope: dict[str, Any],
        variance_reports: list[dict[str, Any]],
        citations: list[dict[str, Any]],
        sector_benchmarks: list[dict[str, Any]],
        chunk_health: dict[str, Any] | None,
        memory_tokens: int = 0,
        memory_prep_latency_ms: float = 0.0,
        on_cache_s2s_call: Callable[[str, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Run the structured ReAct reasoning, cached 10-K querying, formatting, and Flash evaluation."""
        reasoning_model = resolve_model_for_task("financial_reasoning")
        eval_model = resolve_model_for_task("evaluation")
        tracker = get_context_latency_tracker()

        react_steps: list[ReActStep] = []
        turn_latency_records: list[ModelContextLatencyRecord] = []

        # ------------------------------------------------------------------
        # ReAct Iteration 1: Evaluate Calculated Variances (Gemini 2.5 Pro)
        # ------------------------------------------------------------------
        t_eval = time.perf_counter()
        variance_evaluations = evaluate_calculated_variances(
            variance_reports,
            query=query,
            section_filter=scope.get("section_id", "ALL"),
        )
        eval_ms = (time.perf_counter() - t_eval) * 1000.0

        var_payload_tokens = estimate_token_count(json.dumps(variance_reports, default=str))
        prompt_tokens = estimate_token_count(query)
        sys_tokens = 220
        step1_ctx_tokens = sys_tokens + memory_tokens + var_payload_tokens + prompt_tokens
        step1_out_tokens = estimate_token_count(
            json.dumps([v.model_dump(mode="json") for v in variance_evaluations], default=str)
        )

        rec_step1 = tracker.record_call(
            agent_name="FinancialAnalystAgent",
            model_name=reasoning_model,
            task_category="variance_explanation",
            react_phase="evaluate_variances",
            context_breakdown=ContextBreakdown(
                system_instruction_tokens=sys_tokens,
                memory_and_history_tokens=memory_tokens,
                variance_payload_tokens=var_payload_tokens,
                user_prompt_tokens=prompt_tokens,
                total_context_tokens=step1_ctx_tokens,
                uncached_context_tokens=step1_ctx_tokens,
            ),
            output_tokens=step1_out_tokens,
            thoughts_tokens=max(16, step1_out_tokens // 3),
            raw_inference_latency_ms=eval_ms,
            context_prep_latency_ms=memory_prep_latency_ms,
        )
        turn_latency_records.append(rec_step1)

        if variance_evaluations:
            thought_1 = (
                f"Deterministic variance calculation returned {len(variance_evaluations)} issuer report(s). "
                f"I must evaluate primary metric movers, CAGR, and operating-margin divergences before querying cached 10-K sections."
            )
            obs_1 = " | ".join(
                f"{ev.ticker}: {ev.material_findings[0] if ev.material_findings else ev.primary_metric_mover} "
                f"({len(ev.targeted_10k_queries)} cached 10-K queries planned)"
                for ev in variance_evaluations
            )
        else:
            thought_1 = (
                f"Query intent is '{intent}' without single-issuer variance tables; "
                f"evaluating {len(sector_benchmarks)} sector benchmark(s) to target cached 10-K context."
            )
            obs_1 = (
                f"Evaluated {len(sector_benchmarks)} sector benchmark row(s) and "
                f"{len(citations)} candidate 10-K passages."
            )

        react_steps.append(
            ReActStep(
                iteration=1,
                phase="evaluate_variances",
                thought=thought_1,
                action="evaluate_calculated_variances",
                action_input={
                    "tickers": scope.get("tickers", []),
                    "start_year": scope.get("start_year"),
                    "end_year": scope.get("end_year"),
                    "variance_report_count": len(variance_reports),
                },
                observation=obs_1,
                agent="FinancialAnalystAgent",
                model_used=reasoning_model,
                context_tokens=rec_step1.context_breakdown.total_context_tokens,
                cached_context_tokens=0,
                context_latency_ms=rec_step1.context_latency_ms,
                duration_ms=round(eval_ms, 3),
            )
        )

        # ------------------------------------------------------------------
        # ReAct Iteration 2 (+ adaptive follow-up if peer gap): Query Cached 10-K Context
        # ------------------------------------------------------------------
        t_cache = time.perf_counter()
        cached_explanations: list[Cached10KExplanation] = []
        extra_citations: list[dict[str, Any]] = []

        if intent != "chunk_health_monitoring":
            cached_explanations, extra_citations = query_cached_10k_for_explanations(
                variance_evaluations=variance_evaluations,
                citations=citations,
                query=query,
                section_filter=scope.get("section_id", "ALL"),
                max_queries=6,
                on_cache_call=on_cache_s2s_call,
            )
        cache_query_ms = (time.perf_counter() - t_cache) * 1000.0

        # Merge any newly linked chunk citations while preserving deterministic ordering
        merged_citations = list(citations)
        existing_cids = {c["chunk_id"] for c in merged_citations}
        for ec in extra_citations:
            if ec["chunk_id"] not in existing_cids:
                merged_citations.append(ec)
                existing_cids.add(ec["chunk_id"])

        cached_hits = sum(1 for e in cached_explanations if e.cache_hit)
        cached_toks_hit = sum(e.token_estimate for e in cached_explanations if e.cache_hit)
        uncached_toks_miss = sum(
            e.token_estimate for e in cached_explanations if not e.cache_hit
        )

        react_steps.append(
            ReActStep(
                iteration=2,
                phase="query_cached_10k_context",
                thought=(
                    "Using the evaluated variance drivers, query cached 10-K Item 7 (MD&A) and "
                    "Item 1A (Risk Factors) sections via ContextCacheManager to extract grounded explanations."
                ),
                action="query_cached_10k_context",
                action_input={
                    "sections_queried": [e.cache_key for e in cached_explanations],
                    "section_filter": scope.get("section_id", "ALL"),
                },
                observation=(
                    f"Retrieved {len(cached_explanations)} cached 10-K section explanation(s) "
                    f"(cache_hits={cached_hits}/{len(cached_explanations)}, "
                    f"resident_tokens={cached_toks_hit + uncached_toks_miss:,})."
                ),
                agent="ContextCacheManager",
                model_used=None,
                context_tokens=cached_toks_hit + uncached_toks_miss,
                cached_context_tokens=cached_toks_hit,
                context_latency_ms=round(cache_query_ms, 3),
                duration_ms=round(cache_query_ms, 3),
            )
        )

        # Adaptive ReAct check: if multi-company comparison has a ticker missing cached explanation, run follow-up pass
        covered_tickers = {e.ticker for e in cached_explanations}
        missing_eval_tickers = [
            ev.ticker
            for ev in variance_evaluations
            if ev.ticker not in covered_tickers
        ]
        if (
            missing_eval_tickers
            and len(react_steps) < self.max_iterations - 1
            and intent != "chunk_health_monitoring"
        ):
            t_adapt = time.perf_counter()
            followup_evals = [
                ev for ev in variance_evaluations if ev.ticker in missing_eval_tickers
            ]
            more_exps, more_cits = query_cached_10k_for_explanations(
                variance_evaluations=followup_evals,
                citations=merged_citations,
                query=query,
                section_filter=scope.get("section_id", "ALL"),
                max_queries=4,
                on_cache_call=on_cache_s2s_call,
            )
            adapt_ms = (time.perf_counter() - t_adapt) * 1000.0
            cached_explanations.extend(more_exps)
            for mc in more_cits:
                if mc["chunk_id"] not in existing_cids:
                    merged_citations.append(mc)
                    existing_cids.add(mc["chunk_id"])
            add_cached_toks = sum(e.token_estimate for e in more_exps if e.cache_hit)
            cached_toks_hit += add_cached_toks
            uncached_toks_miss += sum(
                e.token_estimate for e in more_exps if not e.cache_hit
            )
            react_steps.append(
                ReActStep(
                    iteration=len(react_steps) + 1,
                    phase="query_cached_10k_context",
                    thought=(
                        f"Observation showed peer ticker(s) {missing_eval_tickers} lacked cached 10-K explanations; "
                        f"executing follow-up cached 10-K lookup before synthesis."
                    ),
                    action="query_cached_10k_context(adaptive_peer_fill)",
                    action_input={"missing_tickers": missing_eval_tickers},
                    observation=f"Added {len(more_exps)} cached 10-K explanation(s) for {missing_eval_tickers}.",
                    agent="ContextCacheManager",
                    model_used=None,
                    context_tokens=sum(e.token_estimate for e in more_exps),
                    cached_context_tokens=add_cached_toks,
                    context_latency_ms=round(adapt_ms, 3),
                    duration_ms=round(adapt_ms, 3),
                )
            )

        # ------------------------------------------------------------------
        # ReAct Iteration 3: Format Grounded Output (Gemini 2.5 Pro)
        # ------------------------------------------------------------------
        t_synth = time.perf_counter()
        deterministic_markdown = format_react_grounded_output(
            query=query,
            intent=intent,
            scope=scope,
            variance_reports=variance_reports,
            variance_evaluations=variance_evaluations,
            cached_explanations=cached_explanations,
            citations=merged_citations,
            sector_benchmarks=sector_benchmarks,
            chunk_health=chunk_health,
        )

        calc_map = {
            f"{r['ticker']}_{r['start_year']}_{r['end_year']}": r
            for r in variance_reports
        }
        answer_markdown = deterministic_markdown
        if self.llm_reasoning_fn is not None:
            candidate_md = self.llm_reasoning_fn(
                reasoning_model,
                {
                    "query": query,
                    "variance_reports": variance_reports,
                    "variance_evaluations": [
                        v.model_dump(mode="json") for v in variance_evaluations
                    ],
                    "cached_explanations": [
                        c.model_dump(mode="json") for c in cached_explanations
                    ],
                    "citations": merged_citations,
                },
            )
            check = verify_narrative_against_calculations(
                response_text=candidate_md,
                calculated_variances=calc_map,
                retrieved_citations=merged_citations,
            )
            if check.passed and candidate_md.strip():
                answer_markdown = candidate_md

        synth_ms = (time.perf_counter() - t_synth) * 1000.0
        rag_chunk_tokens = estimate_token_count(
            "\n".join(c.get("excerpt", "") for c in merged_citations)
        )
        total_synth_ctx = (
            sys_tokens
            + memory_tokens
            + var_payload_tokens
            + cached_toks_hit
            + uncached_toks_miss
            + rag_chunk_tokens
            + prompt_tokens
        )
        synth_out_tokens = estimate_token_count(answer_markdown)

        rec_synth = tracker.record_call(
            agent_name="FinancialAnalystAgent",
            model_name=reasoning_model,
            task_category="financial_reasoning",
            react_phase="format_grounded_output",
            context_breakdown=ContextBreakdown(
                system_instruction_tokens=sys_tokens,
                memory_and_history_tokens=memory_tokens,
                variance_payload_tokens=var_payload_tokens,
                cached_10k_tokens=cached_toks_hit,
                uncached_10k_and_tool_tokens=uncached_toks_miss + rag_chunk_tokens,
                user_prompt_tokens=prompt_tokens,
                total_context_tokens=total_synth_ctx,
                cached_context_tokens=cached_toks_hit,
                uncached_context_tokens=max(0, total_synth_ctx - cached_toks_hit),
                cache_hit=cached_toks_hit > 0,
            ),
            output_tokens=synth_out_tokens,
            thoughts_tokens=max(24, synth_out_tokens // 4),
            raw_inference_latency_ms=synth_ms,
            context_prep_latency_ms=cache_query_ms,
        )
        turn_latency_records.append(rec_synth)

        react_steps.append(
            ReActStep(
                iteration=len(react_steps) + 1,
                phase="format_grounded_output",
                thought=(
                    f"Combine deterministic variance figures with {len(cached_explanations)} cached 10-K explanation(s) "
                    f"and {len(merged_citations)} grounded chunk citation(s) using {reasoning_model}."
                ),
                action="format_react_grounded_output",
                action_input={
                    "model": reasoning_model,
                    "cached_explanations_count": len(cached_explanations),
                    "citation_count": len(merged_citations),
                },
                observation=(
                    f"Formatted grounded report ({synth_out_tokens} output tokens; "
                    f"context_latency={rec_synth.context_latency_ms:.2f}ms, "
                    f"cache_saved={rec_synth.estimated_cache_latency_saved_ms:.2f}ms)."
                ),
                agent="FinancialAnalystAgent",
                model_used=reasoning_model,
                context_tokens=rec_synth.context_breakdown.total_context_tokens,
                cached_context_tokens=rec_synth.context_breakdown.cached_context_tokens,
                context_latency_ms=rec_synth.context_latency_ms,
                duration_ms=round(synth_ms, 3),
            )
        )

        # ------------------------------------------------------------------
        # ReAct Verification & Gemini 3.5 Flash Evaluation Step
        # ------------------------------------------------------------------
        eval_report: NarrativeEvaluationReport = evaluate_variance_explanation(
            query=query,
            response_markdown=answer_markdown,
            variance_reports=variance_reports,
            citations=merged_citations,
            cached_explanations_count=len(cached_explanations),
            llm_judge_fn=self.llm_judge_fn,
        )
        latest_flash_recs = tracker.recent_records(limit=1, model_name=eval_model)
        if latest_flash_recs:
            turn_latency_records.append(latest_flash_recs[-1])

        react_steps.append(
            ReActStep(
                iteration=len(react_steps) + 1,
                phase="fact_check_and_eval",
                thought=(
                    f"Run post-generation numeric fact-check and {eval_model} quality evaluation "
                    f"to verify 100% calculation fidelity and citation grounding."
                ),
                action="evaluate_variance_explanation",
                action_input={"evaluator_model": eval_model},
                observation=(
                    f"Grade={eval_report.grade} (overall_score={eval_report.overall_score:.2f}, "
                    f"numeric_fidelity={eval_report.numeric_fidelity_score:.2f}, "
                    f"citation_grounding={eval_report.citation_grounding_score:.2f})."
                ),
                agent="NarrativeEvaluatorAgent",
                model_used=eval_model,
                context_tokens=eval_report.context_tokens,
                cached_context_tokens=0,
                context_latency_ms=eval_report.context_latency_ms,
                duration_ms=eval_report.total_latency_ms,
                status="completed" if eval_report.grade == "pass" else "warning",
            )
        )

        turn_latency_summary = tracker.summarize(turn_latency_records)

        return {
            "answer_markdown": answer_markdown,
            "citations": merged_citations,
            "variance_evaluations": variance_evaluations,
            "cached_explanations": cached_explanations,
            "react_steps": react_steps,
            "evaluation_report": eval_report,
            "turn_latency_records": turn_latency_records,
            "turn_latency_summary": turn_latency_summary,
            "reasoning_model": reasoning_model,
            "evaluation_model": eval_model,
        }
