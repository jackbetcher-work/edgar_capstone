"""Structured ReAct Execution Loop for SEC 10-K Financial Variance Reasoning.

Implements the structured ReAct loop (`Thought -> Action -> Observation -> ... -> Final Output`):
1. Evaluate Calculated Variances: Analyzes deterministic Revenue, Operating Income, Net Income,
   and margin bps variances (`CompanyVarianceReport`), isolating primary movers, YoY inflections,
   and operating-leverage divergences to formulate variance-targeted 10-K queries.
2. Query Cached 10-K Context for Explanation: Queries resident/warmed 10-K sections (`item_7` MD&A
   and `item_1a` Risk Factors) via `query_cached_10k_context` + metadata-filtered Hybrid RAG,
   adapting across iterations if peer or boundary-year coverage is incomplete.
3. Live Synthesis & Evaluate: the live `FinancialAnalystAgent` (`EDGAR_ANALYST_MODEL`) writes a
   structured report that is verified claim by claim (numeric and citation fact-checks); answers
   that fail verification are withheld, never replaced by template text. Output quality is scored
   by the live `gemini-3.5-flash` judge, tracking context-induced latency across every model call.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from google.adk.models.base_llm import BaseLlm
from pydantic import ValidationError

from edgar_capstone.agents.callbacks import verify_narrative_against_calculations
from edgar_capstone.agents.context_memory import (
    get_context_latency_tracker,
    resolve_model_for_task,
)
from edgar_capstone.agents.evaluation_and_coding import evaluate_variance_explanation
from edgar_capstone.agents.live_runtime import (
    AdkStreamResult,
    resolve_live_adk_runtime,
    run_live_adk_financial_reasoning,
)
from edgar_capstone.agents.schemas import (
    Cached10KExplanation,
    ModelContextLatencyRecord,
    NarrativeEvaluationReport,
    ReActStep,
    VarianceEvaluationItem,
)
from edgar_capstone.agents.structured_report import (
    ReportVerification,
    parse_analyst_report,
    render_analyst_report_markdown,
    verify_analyst_report,
)
from edgar_capstone.agents.tools import query_cached_10k_context
from edgar_capstone.observability.tracing import traced
from edgar_capstone.vector_db.chunker import estimate_token_count

logger = logging.getLogger(__name__)


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


@traced("react.evaluate_variances")
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
        rd = rep.get("research_and_development_variance") or {}
        rd_pct = rd.get("percentage_delta")
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
        if rd_pct is not None:
            findings.append(
                f"R&D expense {rd.get('direction', 'moved')} {_fmt_pct(rd_pct)} "
                f"({_fmt_usd(rd.get('start_value'))} -> {_fmt_usd(rd.get('end_value'))}; "
                f"R&D intensity {_fmt_bps(rep.get('rd_intensity_bps_delta'))})"
            )

        # Detect peak YoY step inflection from annual_series
        annual_steps = [
            s for s in (rep.get("annual_series") or []) if s.get("revenue_yoy_pct") is not None
        ]
        if annual_steps:
            peak_step = max(annual_steps, key=lambda s: abs(float(s["revenue_yoy_pct"])))
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
                research_and_development_pct_delta=rd_pct,
                operating_margin_bps_delta=op_bps,
                net_margin_bps_delta=ni_bps,
                material_findings=findings,
                divergences=divergences,
                targeted_10k_queries=targeted_queries,
            )
        )

    return evaluations


@traced("react.query_cached_10k")
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
            sec_id = sec_norm if sec_norm in ("item_1a", "item_7") else str(cit["section_id"])
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


_RETRIEVAL_TOOL_NAMES = frozenset(
    {"search_10k_chunks_rag", "query_cached_10k_context", "compare_disclosures_across_years"}
)
_UNEXECUTED_ACTION_MARKERS = ("/*ACTION*/", "default_api.", "/*PLANNING*/")


def _output_schema_error(exc: BaseException) -> str | None:
    """Short message when `exc` (or its cause chain) is an output-schema validation failure."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ValidationError):
            first = (current.errors() or [{}])[0]
            return f"{first.get('type', 'validation_error')}: {first.get('msg', str(current))}"[
                :200
            ]
        current = current.__cause__ or current.__context__
    return None


def _withheld_answer_markdown(reason: str) -> str:
    """Narrative shown when the live answer fails verification (no template answer is substituted)."""
    return (
        "### Answer withheld: the live analyst response failed verification\n\n"
        "The live FinancialAnalystAgent answer could not be verified against the retrieved 10-K "
        f"passages and deterministic figures ({reason}). No narrative is shown. The variance "
        "tables, material variance flags, disclosure deltas, and retrieved sources for this turn "
        "are unaffected; rephrase or narrow the question and try again.\n"
    )


def _incomplete_agent_output(text: str) -> str | None:
    """Return a reason when the live answer is planning text or a tool call written as prose."""
    if not text.strip():
        return "Empty live agent output"
    if "/*FINAL_ANSWER*/" in text:
        return None
    for marker in _UNEXECUTED_ACTION_MARKERS:
        if marker in text:
            return (
                f"Live agent output was incomplete (contains '{marker}': planning text or an "
                "unexecuted tool call instead of a final answer)"
            )
    return None


_CITED_CHUNK_ID_RE = re.compile(r"\[[^\]|]+\|[^\]|]+\|\s*([^\]|]+?)\s*\]\(")


def _merge_agent_retrieved_citations(
    merged_citations: list[dict[str, Any]],
    session_citations: Any,
) -> list[dict[str, Any]]:
    """Append citations the agent retrieved via tools (in place) and return only the new ones."""
    existing = {str(c.get("chunk_id")) for c in merged_citations}
    added: list[dict[str, Any]] = []
    for cit in session_citations or []:
        if not isinstance(cit, dict):
            continue
        cid = str(cit.get("chunk_id") or "")
        if cid and cid not in existing:
            existing.add(cid)
            merged_citations.append(cit)
            added.append(cit)
    return added


def _summarize_agent_retrieval(
    live_stream_result: AdkStreamResult | None,
    agent_retrieved_citations: list[dict[str, Any]],
    merged_citations: list[dict[str, Any]],
    answer_markdown: str,
) -> dict[str, Any] | None:
    """Report which 10-K evidence the agent pulled itself and whether its citations resolve."""
    if live_stream_result is None:
        return None
    retrieval_calls = [
        {"name": fc.get("name"), "args": fc.get("args")}
        for fc in live_stream_result.function_calls
        if fc.get("name") in _RETRIEVAL_TOOL_NAMES
    ]
    known_ids = {str(c.get("chunk_id")) for c in merged_citations}
    cited_ids = list(dict.fromkeys(m.strip() for m in _CITED_CHUNK_ID_RE.findall(answer_markdown)))
    return {
        "retrieval_tool_calls": retrieval_calls,
        "retrieval_tool_calls_count": len(retrieval_calls),
        "agent_retrieved_chunk_ids": [str(c.get("chunk_id")) for c in agent_retrieved_citations],
        "cited_chunk_ids": cited_ids,
        "unresolved_cited_chunk_ids": [cid for cid in cited_ids if cid not in known_ids],
    }


class StructuredReActExecutor:
    """Executes the structured ReAct loop (`evaluate_variances -> query_cached_10k_context -> live synthesis`).

    Every answer is written by the live `FinancialAnalystAgent`. When the live answer cannot be
    verified against the retrieved passages and deterministic figures, the narrative is withheld
    (`answer_verified=False`) instead of being replaced by template text.
    """

    def __init__(
        self,
        *,
        max_iterations: int = 5,
        reasoning_llm: BaseLlm | None = None,
        judge_llm: BaseLlm | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.reasoning_llm = reasoning_llm
        self.judge_llm = judge_llm

    @traced("react.execute", record_args=("intent", "session_id"))
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
        memory_context: dict[str, Any] | None = None,
        session_id: str | None = None,
        user_id: str = "analyst",
        on_cache_s2s_call: Callable[[str, int, str], None] | None = None,
        material_variance_flags: list[dict[str, Any]] | None = None,
        disclosure_deltas: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the variance pre-pass, cached 10-K querying, live synthesis, and Flash evaluation."""
        material_variance_flags = list(material_variance_flags or [])
        reasoning_model = (
            self.reasoning_llm.model
            if self.reasoning_llm is not None
            else resolve_model_for_task("financial_reasoning")
        )
        tracker = get_context_latency_tracker()
        runtime_res = resolve_live_adk_runtime(custom_llm=self.reasoning_llm)

        react_steps: list[ReActStep] = []
        turn_latency_records: list[ModelContextLatencyRecord] = []

        # ------------------------------------------------------------------
        # ReAct Iteration 1: Evaluate Calculated Variances (deterministic pre-pass)
        # ------------------------------------------------------------------
        t_eval = time.perf_counter()
        variance_evaluations = evaluate_calculated_variances(
            variance_reports,
            query=query,
            section_filter=scope.get("section_id", "ALL"),
        )
        eval_ms = (time.perf_counter() - t_eval) * 1000.0
        var_payload_tokens = estimate_token_count(json.dumps(variance_reports, default=str))

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
                model_used=None,
                context_tokens=var_payload_tokens + memory_tokens,
                cached_context_tokens=0,
                context_latency_ms=round(eval_ms + memory_prep_latency_ms, 3),
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

        merged_citations = list(citations)
        existing_cids = {c["chunk_id"] for c in merged_citations}
        for ec in extra_citations:
            if ec["chunk_id"] not in existing_cids:
                merged_citations.append(ec)
                existing_cids.add(ec["chunk_id"])

        cached_hits = sum(1 for e in cached_explanations if e.cache_hit)
        cached_toks_hit = sum(e.token_estimate for e in cached_explanations if e.cache_hit)
        uncached_toks_miss = sum(e.token_estimate for e in cached_explanations if not e.cache_hit)

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
            ev.ticker for ev in variance_evaluations if ev.ticker not in covered_tickers
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
        # ReAct Iteration 3: Live synthesis by FinancialAnalystAgent (ADK Runner, SSE)
        # ------------------------------------------------------------------
        t_synth = time.perf_counter()
        calc_map = {f"{r['ticker']}_{r['start_year']}_{r['end_year']}": r for r in variance_reports}
        schema_failure_reason: str | None = None
        try:
            live_stream_result = run_live_adk_financial_reasoning(
                query=query,
                intent=intent,
                scope=scope,
                variance_reports=variance_reports,
                variance_evaluations=[v.model_dump(mode="json") for v in variance_evaluations],
                cached_explanations=[c.model_dump(mode="json") for c in cached_explanations],
                citations=merged_citations,
                sector_benchmarks=sector_benchmarks,
                chunk_health=chunk_health,
                memory_context=memory_context,
                session_id=session_id,
                user_id=user_id,
                resolution=runtime_res,
                custom_llm=self.reasoning_llm,
                summarizer_llm=self.judge_llm,
                material_variance_flags=material_variance_flags,
                disclosure_deltas=disclosure_deltas,
            )
        except Exception as exc:
            schema_error = _output_schema_error(exc)
            if schema_error is None:
                raise
            # The agent's final text wasn't a valid AnalystReport (e.g. a plan or tool call
            # written as prose, or guardrail text replacing the JSON). Withhold the answer
            # instead of failing the turn.
            schema_failure_reason = (
                f"Live agent output was not a valid AnalystReport ({schema_error})"
            )
            live_stream_result = AdkStreamResult(
                agent_name="FinancialAnalystAgent",
                app_name="edgar_chat_app",
                session_id=session_id or "",
                backend=runtime_res.backend,
                final_text="",
            )
        candidate_md = live_stream_result.final_text
        # Passages the agent retrieved itself with search_10k_chunks_rag /
        # query_cached_10k_context become part of the turn's citation set (source
        # viewer, fact-check, and evaluator).
        agent_retrieved_citations = _merge_agent_retrieved_citations(
            merged_citations,
            live_stream_result.session_state.get("retrieved_citations"),
        )

        answer_markdown = candidate_md
        withheld_reason: str | None = None
        adk_response_blocked = False
        structured_report_dict: dict[str, Any] | None = None
        raw_structured_report_text = ""
        report_verification: ReportVerification | None = None
        adk_response_decision: dict[str, Any] | None = None
        adk_response_block_message = ""

        if schema_failure_reason is not None:
            withheld_reason = schema_failure_reason
        elif live_stream_result.session_state.get("model_armor_response_blocked") is True:
            adk_response_blocked = True
            raw_dec = live_stream_result.session_state.get("model_armor_last_response_decision")
            if isinstance(raw_dec, dict):
                adk_response_decision = raw_dec
            adk_response_block_message = str(
                live_stream_result.session_state.get("model_armor_response_block_message")
                or candidate_md
            )
        elif (
            structured := parse_analyst_report(
                live_stream_result.session_state.get("financial_analyst_report")
            )
        ) is not None:
            # Structured answer: verify each claim, then render compact Markdown.
            verified = verify_analyst_report(
                structured,
                variance_reports=variance_reports,
                citations=merged_citations,
                material_flags=material_variance_flags,
            )
            report_verification = verified.verification
            structured_report_dict = verified.report.model_dump(mode="json")
            # Unverified original, kept so the response guardrail also sees claims
            # that verification removed (e.g. a price target in the notes).
            raw_structured_report_text = structured.model_dump_json()
            if not report_verification.usable:
                withheld_reason = report_verification.unusable_reason
            else:
                answer_markdown = render_analyst_report_markdown(
                    verified,
                    variance_reports=variance_reports,
                    citations=merged_citations,
                    scope=scope,
                )
                check = verify_narrative_against_calculations(
                    response_text=answer_markdown,
                    calculated_variances=calc_map,
                    retrieved_citations=merged_citations,
                )
                if not check.passed:
                    withheld_reason = "; ".join(check.discrepancies)
        elif (incomplete_reason := _incomplete_agent_output(candidate_md)) is not None:
            withheld_reason = incomplete_reason
        else:
            check = verify_narrative_against_calculations(
                response_text=candidate_md,
                calculated_variances=calc_map,
                retrieved_citations=merged_citations,
            )
            if not check.passed:
                withheld_reason = "; ".join(check.discrepancies) or "Ungrounded live LLM output"

        answer_verified = withheld_reason is None and not adk_response_blocked
        if withheld_reason is not None:
            answer_markdown = _withheld_answer_markdown(withheld_reason)

        synth_ms = (time.perf_counter() - t_synth) * 1000.0
        turn_latency_records.extend(live_stream_result.latency_records)
        rec_synth = (
            live_stream_result.latency_records[-1] if live_stream_result.latency_records else None
        )
        ttft_note = (
            f", observed_ttft={live_stream_result.observed_ttft_ms:.2f}ms"
            if live_stream_result.observed_ttft_ms is not None
            else ""
        )
        react_steps.append(
            ReActStep(
                iteration=len(react_steps) + 1,
                phase="format_grounded_output",
                thought=(
                    live_stream_result.thoughts[-1]
                    if live_stream_result.thoughts
                    else (
                        f"Executed live ADK Runner ({reasoning_model}, SSE streaming) combining deterministic "
                        f"variances with {len(cached_explanations)} cached 10-K explanation(s) and "
                        f"{len(merged_citations)} grounded citation(s)."
                    )
                ),
                action="adk_runner.FinancialAnalystAgent(SSE)",
                action_input={
                    "model": reasoning_model,
                    "cached_explanations_count": len(cached_explanations),
                    "citation_count": len(merged_citations),
                },
                observation=(
                    f"Streamed grounded report via ADK Runner ({live_stream_result.partial_chunks_count} SSE chunk(s)"
                    f"{ttft_note}; answer_verified={answer_verified})."
                ),
                agent="FinancialAnalystAgent",
                model_used=reasoning_model,
                context_tokens=rec_synth.context_breakdown.total_context_tokens if rec_synth else 0,
                cached_context_tokens=(
                    rec_synth.context_breakdown.cached_context_tokens if rec_synth else 0
                ),
                context_latency_ms=rec_synth.context_latency_ms if rec_synth else 0.0,
                duration_ms=round(synth_ms, 3),
                status="completed" if answer_verified else "warning",
            )
        )

        # ------------------------------------------------------------------
        # ReAct Verification & Gemini 3.5 Flash Evaluation Step
        # ------------------------------------------------------------------
        if answer_verified:
            eval_report: NarrativeEvaluationReport = evaluate_variance_explanation(
                query=query,
                response_markdown=answer_markdown,
                variance_reports=variance_reports,
                citations=merged_citations,
                cached_explanations_count=len(cached_explanations),
                llm=self.judge_llm,
            )
            latest_flash_recs = tracker.recent_records(
                limit=1, model_name=eval_report.evaluator_model
            )
            if latest_flash_recs:
                turn_latency_records.append(latest_flash_recs[-1])
        else:
            # Nothing to judge: the narrative was withheld (or blocked). Skip the Flash call and
            # record a failing grade so telemetry and evals count the turn as unanswered.
            eval_report = NarrativeEvaluationReport(
                evaluator_model=resolve_model_for_task("evaluation"),
                grade="fail",
                overall_score=0.0,
                numeric_fidelity_score=0.0,
                citation_grounding_score=0.0,
                explanation_quality_score=0.0,
                thematic_consistency_score=0.0,
                rationale=(
                    "Not evaluated: the live answer was withheld"
                    + (
                        f" ({withheld_reason})."
                        if withheld_reason
                        else " by the response guardrail."
                    )
                ),
            )

        react_steps.append(
            ReActStep(
                iteration=len(react_steps) + 1,
                phase="fact_check_and_eval",
                thought=(
                    f"Run post-generation numeric fact-check and {eval_report.evaluator_model} quality evaluation "
                    f"to verify 100% calculation fidelity and citation grounding."
                ),
                action="evaluate_variance_explanation",
                action_input={"evaluator_model": eval_report.evaluator_model},
                observation=(
                    f"Grade={eval_report.grade} (overall_score={eval_report.overall_score:.2f}, "
                    f"numeric_fidelity={eval_report.numeric_fidelity_score:.2f}, "
                    f"citation_grounding={eval_report.citation_grounding_score:.2f})."
                ),
                agent="NarrativeEvaluatorAgent",
                model_used=eval_report.evaluator_model,
                context_tokens=eval_report.context_tokens,
                cached_context_tokens=0,
                context_latency_ms=eval_report.context_latency_ms,
                duration_ms=eval_report.total_latency_ms,
                status="completed" if eval_report.grade == "pass" else "warning",
            )
        )

        turn_latency_summary = tracker.summarize(turn_latency_records)

        adk_runtime_meta: dict[str, Any] = {
            **runtime_res.to_public_dict(),
            "executed_via_adk_runner": True,
            "observed_ttft_ms": live_stream_result.observed_ttft_ms,
            "streamed_chunks_count": live_stream_result.partial_chunks_count,
            "adk_events_count": live_stream_result.total_events_count,
            "adk_tool_calls": live_stream_result.function_calls,
            "answer_verified": answer_verified,
            "answer_withheld_reason": withheld_reason,
            "structured_report": structured_report_dict,
            "report_verification": (
                report_verification.model_dump(mode="json") if report_verification else None
            ),
            "agent_retrieval": _summarize_agent_retrieval(
                live_stream_result,
                agent_retrieved_citations,
                merged_citations,
                answer_markdown,
            ),
        }

        return {
            "answer_markdown": answer_markdown,
            "answer_verified": answer_verified,
            "citations": merged_citations,
            "variance_evaluations": variance_evaluations,
            "cached_explanations": cached_explanations,
            "react_steps": react_steps,
            "evaluation_report": eval_report,
            "turn_latency_records": turn_latency_records,
            "turn_latency_summary": turn_latency_summary,
            "reasoning_model": reasoning_model,
            "evaluation_model": eval_report.evaluator_model,
            "execution_mode": "live_adk",
            "adk_runtime": adk_runtime_meta,
            "raw_structured_report_text": raw_structured_report_text,
            "model_armor_response_blocked": adk_response_blocked,
            "model_armor_response_decision": adk_response_decision,
            "model_armor_response_block_message": adk_response_block_message,
        }
