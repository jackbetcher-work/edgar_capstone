"""Structured ReAct Execution Loop for SEC 10-K Financial Variance Reasoning.

Implements the structured ReAct loop (`Thought -> Action -> Observation -> ... -> Final Output`):
1. Evaluate Calculated Variances: Analyzes deterministic Revenue, Operating Income, Net Income,
   and margin bps variances (`CompanyVarianceReport`), isolating primary movers, YoY inflections,
   and operating-leverage divergences to formulate variance-targeted 10-K queries.
2. Query Cached 10-K Context for Explanation: Queries resident/warmed 10-K sections (`item_7` MD&A
   and `item_1a` Risk Factors) via `query_cached_10k_context` + metadata-filtered Hybrid RAG,
   adapting across iterations if peer or boundary-year coverage is incomplete.
3. Live Synthesis & Evaluate: the live `FinancialAnalystAgent` (`EDGAR_ANALYST_MODEL`) decides
   which deterministic engines the question needs (variance calculations, anomaly flags with MD&A
   explanations, disclosure deltas, sector map-reduce) and invokes them as ADK tools; nothing is
   pre-computed. Their outputs are hydrated from ADK `session_state` BEFORE the structured report
   is verified claim by claim (numeric and citation fact-checks), rendered, and fact-checked;
   answers that fail verification are withheld, never replaced by template text. Output quality is scored
   by the live `gemini-3.8-flash` (low-thinking) judge (synchronously in evals or asynchronously with real
   telemetry callback via `record_async_evaluation_result` on interactive turns — never emitting
   fabricated `1.0` placeholder scores).
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from google.adk.models.base_llm import BaseLlm
from pydantic import ValidationError

from edgar_capstone.agents.core.citations import merge_citations
from edgar_capstone.agents.core.schemas import (
    AnalystReport,
    Cached10KExplanation,
    FactCheckVerification,
    ModelContextLatencyRecord,
    NarrativeEvaluationReport,
    ReActStep,
    VarianceEvaluationItem,
)
from edgar_capstone.agents.evaluation.judge import (
    evaluate_variance_explanation,
    evaluate_variance_explanation_with_latency,
)
from edgar_capstone.agents.evaluation.structured_report import (
    ReportVerification,
    VerifiedAnalystReport,
    parse_analyst_report,
    render_analyst_report_markdown,
    verify_analyst_report,
)
from edgar_capstone.agents.orchestration.intent import QUERY_INTENT_VALUES
from edgar_capstone.agents.orchestration.scope_validation import extend_coverage_notices
from edgar_capstone.agents.runtime.callbacks import verify_narrative_against_calculations
from edgar_capstone.agents.runtime.latency_tracker import get_context_latency_tracker
from edgar_capstone.agents.runtime.live_runtime import (
    AdkStreamResult,
    LiveRuntimeResolution,
    resolve_live_adk_runtime,
    run_live_adk_financial_reasoning,
)
from edgar_capstone.agents.runtime.model_config import resolve_model_for_task
from edgar_capstone.agents.tools.runtime import (
    DETERMINISTIC_TOOL_NAMES,
    ToolOutputs,
    hydrate_tool_outputs,
)
from edgar_capstone.agents.tools.sec_tools import calculate_period_variances
from edgar_capstone.common.domain import (
    RETRIEVAL_TOOL_NAMES,
    MetricName,
    normalize_edgar_ticker,
)
from edgar_capstone.common.formatting import fmt_bps, fmt_pct, fmt_usd
from edgar_capstone.observability.tracing import current_trace_correlation, traced
from edgar_capstone.telemetry import get_telemetry_sink
from edgar_capstone.vector_db.chunker import estimate_token_count

logger = logging.getLogger(__name__)

_SHUTDOWN_WAIT_BUDGET_SECONDS = 12.0
ASYNC_JUDGE_SAMPLE_RATE_ENV = "EDGAR_ASYNC_JUDGE_SAMPLE_RATE"
DEFAULT_ASYNC_JUDGE_SAMPLE_RATE = 1.0


def resolve_async_judge_sample_rate() -> float:
    """Resolve the async LLM-as-a-Judge sampling probability in ``[0.0, 1.0]``.

    Controlled by ``EDGAR_ASYNC_JUDGE_SAMPLE_RATE`` (default ``1.0`` = evaluate every verified
    interactive turn; set ``0.0`` to disable background judging; set e.g. ``0.1``–``0.25`` at
    higher traffic to bound evaluation token spend). Invalid or non-finite values fall back to
    ``DEFAULT_ASYNC_JUDGE_SAMPLE_RATE`` and out-of-range finite values are clamped to ``[0.0, 1.0]``.
    """
    raw = os.environ.get(ASYNC_JUDGE_SAMPLE_RATE_ENV, "").strip()
    if not raw:
        return DEFAULT_ASYNC_JUDGE_SAMPLE_RATE
    try:
        parsed = float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to %.2f",
            ASYNC_JUDGE_SAMPLE_RATE_ENV,
            raw,
            DEFAULT_ASYNC_JUDGE_SAMPLE_RATE,
        )
        return DEFAULT_ASYNC_JUDGE_SAMPLE_RATE
    if not math.isfinite(parsed):
        logger.warning(
            "Non-finite %s=%r; falling back to %.2f",
            ASYNC_JUDGE_SAMPLE_RATE_ENV,
            raw,
            DEFAULT_ASYNC_JUDGE_SAMPLE_RATE,
        )
        return DEFAULT_ASYNC_JUDGE_SAMPLE_RATE
    return max(0.0, min(1.0, parsed))


def _should_sample_async_judge(sample_rate: float) -> bool:
    """Return True when the current turn is selected for background LLM-as-a-Judge scoring."""
    if sample_rate <= 0.0:
        return False
    if sample_rate >= 1.0:
        return True
    return random.random() < sample_rate


class _BackgroundJudgeManager:
    """Lazy-initialized thread pool for off-critical-path LLM-as-a-Judge evaluations."""

    def __init__(self, *, max_workers: int = 4) -> None:
        self._max_workers = max_workers
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self._pending: list[Future[Any]] = []
        self._shutting_down = threading.Event()

    @property
    def is_pool_initialized(self) -> bool:
        with self._lock:
            return self._pool is not None

    def is_shutting_down(self) -> bool:
        return self._shutting_down.is_set() or sys.is_finalizing()

    def submit(self, fn: Callable[[], Any]) -> Future[Any] | None:
        if self.is_shutting_down():
            return None
        ctx = contextvars.copy_context()
        with self._lock:
            if self._shutting_down.is_set():
                return None
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix="edgar-bg-judge",
                )
            try:
                fut = self._pool.submit(ctx.run, fn)
            except RuntimeError:
                return None
            self._pending[:] = [f for f in self._pending if not f.done()]
            self._pending.append(fut)
            return fut

    def flush(self, timeout: float = 30.0) -> int:
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        deadline = time.monotonic() + max(0.0, float(timeout))
        completed = 0
        for fut in pending:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                fut.result(timeout=remaining)
                completed += 1
            except Exception:
                logger.debug(
                    "Background judge future raised or timed out during flush", exc_info=True
                )
        return completed

    def shutdown(self, timeout: float = _SHUTDOWN_WAIT_BUDGET_SECONDS) -> None:
        self._shutting_down.set()
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
            pool = self._pool
            self._pool = None
        for fut in pending:
            fut.cancel()
        deadline = time.monotonic() + max(0.0, float(timeout))
        for fut in pending:
            if not fut.cancelled() and not fut.done():
                remaining = max(0.0, deadline - time.monotonic())
                with contextlib.suppress(Exception):
                    fut.result(timeout=remaining)
        if pool is not None:
            with contextlib.suppress(Exception):
                pool.shutdown(wait=False, cancel_futures=True)


_JUDGE_MANAGER = _BackgroundJudgeManager(max_workers=4)


def _on_interpreter_exit() -> None:
    _JUDGE_MANAGER.shutdown(timeout=_SHUTDOWN_WAIT_BUDGET_SECONDS)


atexit.register(_on_interpreter_exit)


def flush_pending_evaluations(timeout: float = 30.0) -> int:
    """Wait for any pending background LLM-as-a-Judge tasks using a shared monotonic deadline."""
    return _JUDGE_MANAGER.flush(timeout=timeout)


def _schedule_background_judge(
    *,
    query: str,
    response_markdown: str,
    variance_reports: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    cached_explanations_count: int,
    session_id: str | None = None,
    user_email: str = "analyst",
    intent: str = "longitudinal_single_company",
    tickers_analyzed: list[str] | None = None,
    turn_id: str | None = None,
    trace_id: str | None = None,
    fact_check: FactCheckVerification | None = None,
    llm: BaseLlm | None = None,
) -> Future[Any] | None:
    """Run live Flash LLM-as-a-Judge asynchronously and stream its real grade/score to telemetry."""
    if _JUDGE_MANAGER.is_shutting_down():
        return None

    corr = current_trace_correlation()
    eff_trace_id = trace_id or (corr.trace_id if corr is not None else None)
    eff_tickers = list(tickers_analyzed or [])

    def _run() -> NarrativeEvaluationReport | None:
        if _JUDGE_MANAGER.is_shutting_down():
            return None
        try:
            report = evaluate_variance_explanation(
                query=query,
                response_markdown=response_markdown,
                variance_reports=variance_reports,
                citations=citations,
                cached_explanations_count=cached_explanations_count,
                llm=llm,
                fact_check=fact_check,
            )
            if report.overall_score is not None:
                get_telemetry_sink().record_async_evaluation_result(
                    session_id=session_id or "",
                    user_email=user_email,
                    intent=intent,
                    evaluation_grade=report.grade,
                    evaluation_score=report.overall_score,
                    evaluator_model=report.evaluator_model,
                    query_text=query,
                    tickers_analyzed=eff_tickers,
                    rationale=report.rationale,
                    turn_id=turn_id,
                    trace_id=eff_trace_id,
                )
            return report
        except RuntimeError as exc:
            msg_low = str(exc).lower()
            if (
                "interpreter shutdown" in msg_low
                or "after shutdown" in msg_low
                or _JUDGE_MANAGER.is_shutting_down()
            ):
                return None
            logger.warning("Background async LLM-as-a-Judge evaluation failed: %s", exc)
            return None
        except Exception:
            if not _JUDGE_MANAGER.is_shutting_down():
                logger.warning("Background async LLM-as-a-Judge evaluation failed", exc_info=True)
            return None

    return _JUDGE_MANAGER.submit(_run)


def _fmt_score(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "pending"


@traced("react.evaluate_variances")
def evaluate_calculated_variances(
    variance_reports: list[dict[str, Any]],
    *,
    query: str = "",
    section_filter: str = "ALL",
    include_targeted_queries: bool = True,
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

        candidates: list[tuple[float, MetricName]] = [
            (abs(float(rev_pct)) if rev_pct is not None else -1.0, "revenue"),
            (abs(float(op_pct)) if op_pct is not None else -1.0, "operating_income"),
            (abs(float(ni_pct)) if ni_pct is not None else -1.0, "net_income"),
        ]
        candidates.sort(key=lambda x: x[0], reverse=True)
        primary_mover: MetricName = candidates[0][1] if candidates[0][0] >= 0 else "revenue"

        findings: list[str] = []
        if rev_pct is not None:
            cagr_txt = (
                f", {fmt_pct(rev.get('cagr_pct'))} CAGR"
                if rev.get("cagr_pct") is not None and fy_e > fy_s
                else ""
            )
            findings.append(
                f"Revenue {rev.get('direction', 'moved')} {fmt_pct(rev_pct)} "
                f"({fmt_usd(rev.get('start_value'))} -> {fmt_usd(rev.get('end_value'))}, "
                f"Δ {fmt_usd(rev.get('absolute_delta'))}{cagr_txt})"
            )
        if op_pct is not None:
            findings.append(
                f"Operating Income {op.get('direction', 'moved')} {fmt_pct(op_pct)} "
                f"to {fmt_usd(op.get('end_value'))} (Operating Margin {fmt_bps(op_bps)})"
            )
        if ni_pct is not None:
            findings.append(
                f"Net Income {ni.get('direction', 'moved')} {fmt_pct(ni_pct)} "
                f"to {fmt_usd(ni.get('end_value'))} (Net Margin {fmt_bps(ni_bps)})"
            )
        if rd_pct is not None:
            findings.append(
                f"R&D expense {rd.get('direction', 'moved')} {fmt_pct(rd_pct)} "
                f"({fmt_usd(rd.get('start_value'))} -> {fmt_usd(rd.get('end_value'))}; "
                f"R&D intensity {fmt_bps(rep.get('rd_intensity_bps_delta'))})"
            )

        # Detect peak YoY step inflection from annual_series
        annual_steps = [
            s for s in (rep.get("annual_series") or []) if s.get("revenue_yoy_pct") is not None
        ]
        if annual_steps:
            peak_step = max(annual_steps, key=lambda s: abs(float(s["revenue_yoy_pct"])))
            findings.append(
                f"Peak YoY revenue inflection in FY{peak_step['fiscal_year']} "
                f"({fmt_pct(peak_step['revenue_yoy_pct'])} vs FY{peak_step.get('prev_fiscal_year')})"
            )

        # Detect operating leverage or margin divergences requiring 10-K explanation
        divergences: list[str] = []
        if rev_pct is not None and op_pct is not None:
            spread = float(op_pct) - float(rev_pct)
            if spread >= 5.0:
                divergences.append(
                    f"Positive operating leverage: Operating Income growth ({fmt_pct(op_pct)}) "
                    f"exceeded Revenue growth ({fmt_pct(rev_pct)}) by +{spread:.2f} pts"
                )
            elif spread <= -5.0:
                divergences.append(
                    f"Operating margin pressure: Operating Income growth ({fmt_pct(op_pct)}) "
                    f"trailed Revenue growth ({fmt_pct(rev_pct)}) by {spread:.2f} pts"
                )
        if op_pct is not None and ni_pct is not None:
            ni_spread = float(ni_pct) - float(op_pct)
            if abs(ni_spread) >= 8.0:
                divergences.append(
                    f"Below-the-line variance: Net Income ({fmt_pct(ni_pct)}) diverged from "
                    f"Operating Income ({fmt_pct(op_pct)}) by {fmt_pct(ni_spread)}"
                )

        # Build targeted cached 10-K section queries driven by the variance evaluation
        targeted_queries: list[dict[str, Any]] = []
        if include_targeted_queries:
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
                primary_metric_mover=primary_mover,
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


_CITED_CHUNK_ID_RE = re.compile(r"\[[^\]|]+\|[^\]|]+\|\s*([^\]|]+?)\s*\]\(")


def _merge_agent_retrieved_citations(
    merged_citations: list[dict[str, Any]],
    session_citations: Any,
) -> list[dict[str, Any]]:
    """Append citations the agent retrieved via tools (in place) and return only the new ones."""
    valid_extras = [c for c in (session_citations or []) if isinstance(c, dict)]
    if not valid_extras:
        return []
    before_len = len(merged_citations)
    merge_citations(merged_citations, valid_extras)
    return list(merged_citations[before_len:])


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
        if fc.get("name") in RETRIEVAL_TOOL_NAMES
    ]
    known_ids = {str(c.get("chunk_id")) for c in merged_citations}
    raw_tags = list(dict.fromkeys(m.strip() for m in _CITED_CHUNK_ID_RE.findall(answer_markdown)))
    cited_xbrl_accessions = [t[4:] for t in raw_tags if t.startswith("acc:")]
    cited_ids = [t for t in raw_tags if not t.startswith("acc:")]
    return {
        "retrieval_tool_calls": retrieval_calls,
        "retrieval_tool_calls_count": len(retrieval_calls),
        "agent_retrieved_chunk_ids": [str(c.get("chunk_id")) for c in agent_retrieved_citations],
        "cited_chunk_ids": cited_ids,
        "cited_xbrl_accessions": cited_xbrl_accessions,
        "unresolved_cited_chunk_ids": [cid for cid in cited_ids if cid not in known_ids],
    }


@dataclass
class _ReasoningInput:
    """Bundled inputs for synthesis and verification-guided regeneration."""

    query: str
    intent: str
    scope: dict[str, Any]
    initial_outputs: ToolOutputs
    variance_evaluations: list[VarianceEvaluationItem]
    cached_explanations: list[Cached10KExplanation]
    merged_citations: list[dict[str, Any]]
    memory_context: dict[str, Any] | None
    session_id: str | None
    user_id: str
    runtime_res: LiveRuntimeResolution
    reasoning_model: str
    coverage_notices: list[dict[str, Any]] | None
    on_progress: Callable[[str, str, dict[str, Any] | None], None] | None


@dataclass
class _SynthesisOutcome:
    live_stream_result: AdkStreamResult
    agent_retrieved_citations: list[dict[str, Any]]
    answer_markdown: str
    answer_verified: bool
    withheld_reason: str | None
    structured_report_dict: dict[str, Any] | None
    report_verification: ReportVerification | None
    fact_check_verification: FactCheckVerification | None
    latency_records: list[ModelContextLatencyRecord]
    synth_ms: float
    tool_outputs: ToolOutputs = field(default_factory=ToolOutputs)
    coverage_notices: list[dict[str, Any]] = field(default_factory=list)
    agent_intent: str | None = None
    key_metric_backfill_tickers: list[str] = field(default_factory=list)


def _backfill_key_metric_variances(
    report: AnalystReport,
    tool_outputs: ToolOutputs,
    scope: dict[str, Any],
) -> tuple[ToolOutputs, list[str]]:
    """Deterministically compute variances for `key_metrics` the agent selected without them.

    `key_metrics` render inline XBRL provenance chips from `variance_reports`; when the agent
    selects a metric for an in-scope company but did not call `calculate_period_variances`,
    the deterministic engine fills the gap so the chips and the XBRL Fact Inspector stay backed
    by the same `variance_reports` returned with the turn. Out-of-scope tickers are left to
    `verify_analyst_report`, which drops them.
    """
    have = {str(r.get("ticker", "")).upper() for r in tool_outputs.variance_reports}
    allowed = {str(t).upper() for t in scope.get("tickers") or []} | {
        f.ticker.upper() for f in report.findings
    }
    need = sorted({ref.ticker.upper() for ref in report.key_metrics} & allowed - have)
    if not need or scope.get("start_year") is None or scope.get("end_year") is None:
        return tool_outputs, []
    res = calculate_period_variances(
        tickers=",".join(need),
        start_year=int(scope["start_year"]),
        end_year=int(scope["end_year"]),
    )
    if res.get("status") != "success" or not res.get("variance_reports"):
        return tool_outputs, []
    backfilled = ToolOutputs(
        variance_reports=list(res["variance_reports"]),
        missing_tickers=list(res.get("missing_tickers") or []),
    )
    return tool_outputs.merge(backfilled), need


def _missing_scope_coverage(
    verified: VerifiedAnalystReport,
    *,
    scope: dict[str, Any],
    citations: list[dict[str, Any]],
) -> tuple[list[str], list[int]]:
    """Return `(missing_tickers, missing_boundary_years)` absent from verified findings.

    Only reports a ticker or boundary fiscal year as missing when retrieved `citations`
    actually contain at least one passage for it, so unavailable filings never trigger a
    pointless repair pass.
    """
    by_id = {str(c.get("chunk_id")): c for c in citations if c.get("chunk_id")}
    cited_tickers: set[str] = set()
    cited_years: set[int] = set()
    for f in verified.report.findings:
        if f.ticker:
            cited_tickers.add(normalize_edgar_ticker(f.ticker))
        if f.fiscal_year:
            cited_years.add(int(f.fiscal_year))
        for cid in f.chunk_ids:
            c = by_id.get(cid)
            if c is None:
                continue
            if c.get("ticker"):
                cited_tickers.add(normalize_edgar_ticker(str(c["ticker"])))
            if c.get("fiscal_year") is not None:
                cited_years.add(int(c["fiscal_year"]))

    available_tickers = {
        normalize_edgar_ticker(str(c["ticker"])) for c in citations if c.get("ticker")
    }
    available_years = {int(c["fiscal_year"]) for c in citations if c.get("fiscal_year") is not None}

    scope_tickers = [
        normalize_edgar_ticker(str(t)) for t in (scope.get("tickers") or []) if str(t).strip()
    ]
    missing_tickers: list[str] = []
    if 2 <= len(scope_tickers) <= 6:
        missing_tickers = [
            tk for tk in scope_tickers if tk in available_tickers and tk not in cited_tickers
        ]

    missing_years: list[int] = []
    if 1 <= len(scope_tickers) <= 6:
        start_y = scope.get("start_year")
        end_y = scope.get("end_year")
        if start_y is not None and end_y is not None and int(start_y) != int(end_y):
            for by in (int(start_y), int(end_y)):
                if by in available_years and by not in cited_years:
                    missing_years.append(by)

    return missing_tickers, missing_years


def _is_retry_improvement(
    original: VerifiedAnalystReport,
    candidate: VerifiedAnalystReport,
    *,
    scope: dict[str, Any],
    citations: list[dict[str, Any]],
) -> bool:
    """Ensure a partial-rejection repair pass never regresses a usable draft."""
    if not original.verification.usable:
        return True
    if not candidate.verification.usable:
        return False
    orig_tk, orig_yr = _missing_scope_coverage(original, scope=scope, citations=citations)
    cand_tk, cand_yr = _missing_scope_coverage(candidate, scope=scope, citations=citations)
    orig_gaps = len(orig_tk) + len(orig_yr)
    cand_gaps = len(cand_tk) + len(cand_yr)
    if cand_gaps < orig_gaps:
        return True
    return (
        cand_gaps == orig_gaps
        and candidate.verification.findings_verified > original.verification.findings_verified
    )


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

    def _run_variance_eval_pass(
        self,
        *,
        query: str,
        intent: str,
        scope: dict[str, Any],
        variance_reports: list[dict[str, Any]],
        citations: list[dict[str, Any]],
        sector_benchmarks: list[dict[str, Any]],
        memory_tokens: int,
        memory_prep_latency_ms: float,
    ) -> tuple[list[VarianceEvaluationItem], ReActStep]:
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
                "I must evaluate primary metric movers, CAGR, and operating-margin divergences before querying cached 10-K sections."
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

        step = ReActStep(
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
        return variance_evaluations, step

    def _invoke_financial_reasoning(
        self,
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
        memory_context: dict[str, Any] | None,
        session_id: str | None,
        user_id: str,
        runtime_res: LiveRuntimeResolution,
        material_variance_flags: list[dict[str, Any]],
        disclosure_deltas: dict[str, Any] | None,
        coverage_notices: list[dict[str, Any]] | None,
        sector_map_summaries: dict[str, Any] | None,
        custom_llm: BaseLlm | None,
        verification_feedback: str | None = None,
        ephemeral: bool = False,
        enable_tools: bool = True,
        low_thinking: bool = False,
    ) -> AdkStreamResult:
        """Single unified invocation helper for initial synthesis and regeneration retry."""
        return run_live_adk_financial_reasoning(
            query=query,
            intent=intent,
            scope=scope,
            variance_reports=variance_reports,
            variance_evaluations=[v.model_dump(mode="json") for v in variance_evaluations],
            cached_explanations=[c.model_dump(mode="json") for c in cached_explanations],
            citations=citations,
            sector_benchmarks=sector_benchmarks,
            chunk_health=chunk_health,
            memory_context=memory_context,
            session_id=session_id,
            user_id=user_id,
            resolution=runtime_res,
            custom_llm=custom_llm,
            summarizer_llm=self.judge_llm,
            material_variance_flags=material_variance_flags,
            disclosure_deltas=disclosure_deltas,
            coverage_notices=coverage_notices,
            sector_map_summaries=sector_map_summaries,
            verification_feedback=verification_feedback,
            ephemeral=ephemeral,
            enable_tools=enable_tools,
            low_thinking=low_thinking,
        )

    def _maybe_regenerate_structured_report(
        self,
        req: _ReasoningInput,
        *,
        structured: AnalystReport,
        verified: VerifiedAnalystReport,
        live_stream_result: AdkStreamResult,
        tool_outputs: ToolOutputs,
        latency_records: list[ModelContextLatencyRecord],
        agent_retrieved_citations: list[dict[str, Any]],
    ) -> tuple[VerifiedAnalystReport, AdkStreamResult, ToolOutputs, str | None]:
        """One fast, tool-less, low-thinking verification repair pass.

        Triggers when (a) all findings failed verification (`usable=False`), or (b) partial
        rejections dropped an in-scope ticker or boundary fiscal year that is present in the
        turn's retrieved citations. When the original draft was already usable, the repaired
        report is adopted only if it improves coverage or verified finding count.
        """
        if self.reasoning_llm is not None:
            return verified, live_stream_result, tool_outputs, None
        missing_tickers, missing_years = _missing_scope_coverage(
            verified, scope=req.scope, citations=req.merged_citations
        )
        partial_coverage_gap = verified.verification.findings_rejected > 0 and (
            bool(missing_tickers) or bool(missing_years)
        )
        if verified.verification.usable and not partial_coverage_gap:
            return verified, live_stream_result, tool_outputs, None

        draft_findings = structured.findings
        feedback_lines: list[str] = []
        for fv in verified.verification.findings:
            if not fv.verified:
                if 0 <= fv.index < len(draft_findings):
                    f = draft_findings[fv.index]
                    cids = ", ".join(f.chunk_ids)
                    feedback_lines.append(
                        f"- Finding {fv.index} ({f.ticker} FY{f.fiscal_year}, chunks=[{cids}]): "
                        f"{'; '.join(fv.reasons)}"
                    )
                else:
                    feedback_lines.append(f"- Finding {fv.index}: {'; '.join(fv.reasons)}")
        if partial_coverage_gap:
            gap_parts: list[str] = []
            if missing_tickers:
                gap_parts.append(f"in-scope ticker(s) {', '.join(missing_tickers)}")
            if missing_years:
                gap_parts.append(
                    f"boundary fiscal year(s) {', '.join(f'FY{y}' for y in missing_years)}"
                )
            kept_cids = sorted(
                {cid for f in verified.report.findings for cid in f.chunk_ids if cid}
            )
            feedback_lines.append(
                "- Coverage gap after dropping rejected findings: missing "
                + " and ".join(gap_parts)
                + ". Keep your valid findings"
                + (f" (chunks: {', '.join(kept_cids)})" if kept_cids else "")
                + " and add/fix verified findings for the missing coverage using exact verbatim "
                "quotes copied from the retrieved passages above."
            )

        logger.info(
            "Verification repair triggered (usable=%s, rejected=%d/%d, missing_tickers=%s, "
            "missing_years=%s); running one tool-less low-thinking regeneration pass.",
            verified.verification.usable,
            verified.verification.findings_rejected,
            verified.verification.findings_total,
            missing_tickers,
            missing_years,
        )
        try:
            # The retry is seeded with every deterministic output and citation already retrieved
            # this turn, so it runs tool-less with low thinking as a single JSON repair call.
            retry_res = self._invoke_financial_reasoning(
                query=req.query,
                intent=req.intent,
                scope=req.scope,
                variance_reports=tool_outputs.variance_reports,
                variance_evaluations=req.variance_evaluations,
                cached_explanations=req.cached_explanations,
                citations=req.merged_citations,
                sector_benchmarks=tool_outputs.sector_benchmarks,
                chunk_health=tool_outputs.chunk_health,
                memory_context=req.memory_context,
                session_id=f"{req.session_id or 'sess'}-retry",
                user_id=req.user_id,
                runtime_res=req.runtime_res,
                material_variance_flags=tool_outputs.material_variance_flags,
                disclosure_deltas=tool_outputs.disclosure_deltas,
                coverage_notices=req.coverage_notices,
                sector_map_summaries=tool_outputs.sector_map_summaries,
                custom_llm=None,
                verification_feedback="\n".join(feedback_lines),
                ephemeral=True,
                enable_tools=False,
                low_thinking=True,
            )
            retry_tool_outputs = tool_outputs.merge(hydrate_tool_outputs(retry_res.session_state))
            agent_retrieved_citations.extend(
                _merge_agent_retrieved_citations(req.merged_citations, retry_tool_outputs.citations)
            )
            retry_structured = parse_analyst_report(
                retry_res.session_state.get("financial_analyst_report")
            )
            retry_intent: str | None = None
            if retry_structured is not None:
                retry_intent = retry_structured.intent
                retry_tool_outputs, _ = _backfill_key_metric_variances(
                    retry_structured, retry_tool_outputs, req.scope
                )
                retry_verified = verify_analyst_report(
                    retry_structured,
                    variance_reports=retry_tool_outputs.variance_reports,
                    citations=req.merged_citations,
                    material_flags=retry_tool_outputs.material_variance_flags,
                )
                if _is_retry_improvement(
                    verified,
                    retry_verified,
                    scope=req.scope,
                    citations=req.merged_citations,
                ):
                    latency_records.extend(live_stream_result.latency_records)
                    return retry_verified, retry_res, retry_tool_outputs, retry_intent
            elif not verified.verification.usable:
                latency_records.extend(live_stream_result.latency_records)
                return verified, retry_res, retry_tool_outputs, None

            # Partial repair did not improve on the already-usable draft; preserve original draft
            # while still recording the retry call's latency records for telemetry.
            latency_records.extend(retry_res.latency_records)
            return verified, live_stream_result, tool_outputs, None
        except Exception:
            logger.warning(
                "Verification-guided regeneration pass failed; keeping original verification state.",
                exc_info=True,
            )
            return verified, live_stream_result, tool_outputs, None

    def _synthesize_and_verify_with_retry(self, req: _ReasoningInput) -> _SynthesisOutcome:
        """Run the live agent, hydrate its tool outputs, then verify, render, and fact-check.

        Deterministic tool outputs (variance reports, material flags + MD&A explanations,
        disclosure deltas, sector map-reduce payloads, citations) are hydrated from the ADK
        `session_state` BEFORE `verify_analyst_report`, `render_analyst_report_markdown`, and
        `verify_narrative_against_calculations` run, so every claim is checked against exactly
        the deterministic outputs the agent chose to invoke.
        """
        t_synth = time.perf_counter()
        latency_records: list[ModelContextLatencyRecord] = []
        schema_failure_reason: str | None = None

        if req.on_progress is not None:
            req.on_progress(
                "synthesizing",
                f"Synthesizing grounded report via {req.reasoning_model}...",
                {"citation_count": len(req.merged_citations)},
            )

        init_out = req.initial_outputs
        try:
            live_stream_result = self._invoke_financial_reasoning(
                query=req.query,
                intent=req.intent,
                scope=req.scope,
                variance_reports=init_out.variance_reports,
                variance_evaluations=req.variance_evaluations,
                cached_explanations=req.cached_explanations,
                citations=req.merged_citations,
                sector_benchmarks=init_out.sector_benchmarks,
                chunk_health=init_out.chunk_health,
                memory_context=req.memory_context,
                session_id=req.session_id,
                user_id=req.user_id,
                runtime_res=req.runtime_res,
                material_variance_flags=init_out.material_variance_flags,
                disclosure_deltas=init_out.disclosure_deltas,
                coverage_notices=req.coverage_notices,
                sector_map_summaries=init_out.sector_map_summaries,
                custom_llm=self.reasoning_llm,
            )
        except Exception as exc:
            schema_error = _output_schema_error(exc)
            if schema_error is None:
                raise
            schema_failure_reason = (
                f"Live agent output was not a valid AnalystReport ({schema_error})"
            )
            live_stream_result = AdkStreamResult(
                agent_name="FinancialAnalystAgent",
                app_name="edgar_chat_app",
                session_id=req.session_id or "",
                backend=req.runtime_res.backend,
                final_text="",
            )

        # Hydrate every deterministic output the agent invoked this turn (pre-supplied outputs
        # were seeded into the same keys, so they round-trip unchanged).
        tool_outputs = init_out.merge(hydrate_tool_outputs(live_stream_result.session_state))
        candidate_md = live_stream_result.final_text
        agent_retrieved_citations = _merge_agent_retrieved_citations(
            req.merged_citations, tool_outputs.citations
        )

        answer_markdown = candidate_md
        withheld_reason: str | None = None
        structured_report_dict: dict[str, Any] | None = None
        report_verification: ReportVerification | None = None
        fact_check_verification: FactCheckVerification | None = None
        agent_intent: str | None = None
        backfill_tickers: list[str] = []

        def _final_notices() -> list[dict[str, Any]]:
            return extend_coverage_notices(
                list(req.coverage_notices or []),
                query=req.query,
                variance_reports=tool_outputs.variance_reports,
                sector_coverage=tool_outputs.sector_coverage,
            )

        if schema_failure_reason is not None:
            withheld_reason = schema_failure_reason
        elif (
            structured := parse_analyst_report(
                live_stream_result.session_state.get("financial_analyst_report")
            )
        ) is not None:
            agent_intent = structured.intent
            tool_outputs, backfill_tickers = _backfill_key_metric_variances(
                structured, tool_outputs, req.scope
            )
            verified = verify_analyst_report(
                structured,
                variance_reports=tool_outputs.variance_reports,
                citations=req.merged_citations,
                material_flags=tool_outputs.material_variance_flags,
            )
            verified, live_stream_result, tool_outputs, retry_intent = (
                self._maybe_regenerate_structured_report(
                    req,
                    structured=structured,
                    verified=verified,
                    live_stream_result=live_stream_result,
                    tool_outputs=tool_outputs,
                    latency_records=latency_records,
                    agent_retrieved_citations=agent_retrieved_citations,
                )
            )
            agent_intent = retry_intent or agent_intent
            report_verification = verified.verification
            structured_report_dict = verified.report.model_dump(mode="json")
            if not report_verification.usable:
                withheld_reason = report_verification.unusable_reason
            else:
                answer_markdown = render_analyst_report_markdown(
                    verified,
                    variance_reports=tool_outputs.variance_reports,
                    citations=req.merged_citations,
                    scope=req.scope,
                    coverage_notices=_final_notices(),
                )
                cited_ids = {cid for f in verified.report.findings for cid in f.chunk_ids}
                fact_check_verification = FactCheckVerification(
                    passed=True,
                    checked_metrics_count=len(verified.report.key_metrics),
                    matched_citations_count=len(cited_ids),
                    discrepancies=[],
                )
        elif not candidate_md.strip():
            withheld_reason = "Empty live agent output"
        else:
            check = verify_narrative_against_calculations(
                response_text=candidate_md,
                calculated_variances=tool_outputs.calculated_variances_map,
                retrieved_citations=req.merged_citations,
            )
            if check.passed:
                fact_check_verification = check
            else:
                withheld_reason = "; ".join(check.discrepancies) or "Ungrounded live LLM output"

        answer_verified = withheld_reason is None
        if withheld_reason is not None:
            answer_markdown = _withheld_answer_markdown(withheld_reason)
            fact_check_verification = None

        latency_records.extend(live_stream_result.latency_records)
        synth_ms = (time.perf_counter() - t_synth) * 1000.0
        return _SynthesisOutcome(
            live_stream_result=live_stream_result,
            agent_retrieved_citations=agent_retrieved_citations,
            answer_markdown=answer_markdown,
            answer_verified=answer_verified,
            withheld_reason=withheld_reason,
            structured_report_dict=structured_report_dict,
            report_verification=report_verification,
            fact_check_verification=fact_check_verification,
            latency_records=latency_records,
            synth_ms=synth_ms,
            tool_outputs=tool_outputs,
            coverage_notices=_final_notices(),
            agent_intent=agent_intent,
            key_metric_backfill_tickers=backfill_tickers,
        )

    def _run_post_synthesis_evaluation(
        self,
        *,
        query: str,
        intent: str,
        scope: dict[str, Any],
        variance_reports: list[dict[str, Any]],
        merged_citations: list[dict[str, Any]],
        cached_explanations_count: int,
        outcome: _SynthesisOutcome,
        session_id: str | None,
        user_id: str,
        turn_id: str | None,
        run_judge_sync: bool,
        skip_background_judge: bool,
    ) -> tuple[NarrativeEvaluationReport, str, list[ModelContextLatencyRecord]]:
        eval_latency_records: list[ModelContextLatencyRecord] = []
        if not outcome.answer_verified:
            report = NarrativeEvaluationReport(
                evaluator_model=resolve_model_for_task("evaluation"),
                grade="fail",
                overall_score=0.0,
                numeric_fidelity_score=0.0,
                citation_grounding_score=0.0,
                citation_precision_score=0.0,
                explanation_quality_score=0.0,
                thematic_consistency_score=0.0,
                unsupported_claims_count=(
                    outcome.report_verification.findings_rejected
                    if outcome.report_verification
                    else 1
                ),
                rationale=(
                    "Not evaluated: the live answer was withheld"
                    + (
                        f" ({outcome.withheld_reason})."
                        if outcome.withheld_reason
                        else " by the response guardrail."
                    )
                ),
            )
            return report, "skipped", eval_latency_records

        if run_judge_sync or self.judge_llm is not None:
            report, eval_latency_records = evaluate_variance_explanation_with_latency(
                query=query,
                response_markdown=outcome.answer_markdown,
                variance_reports=variance_reports,
                citations=merged_citations,
                cached_explanations_count=cached_explanations_count,
                llm=self.judge_llm,
                fact_check=outcome.fact_check_verification,
            )
            return report, "sync", eval_latency_records

        if skip_background_judge:
            report = NarrativeEvaluationReport(
                evaluator_model=resolve_model_for_task("evaluation"),
                grade="pending",
                overall_score=None,
                numeric_fidelity_score=None,
                citation_grounding_score=None,
                citation_precision_score=None,
                explanation_quality_score=None,
                thematic_consistency_score=None,
                unsupported_claims_count=0,
                rationale=(
                    "Deterministic claim verification passed; LLM-as-a-Judge deferred to "
                    "caller evaluation harness (scores pending)."
                ),
            )
            return report, "deferred_to_caller", eval_latency_records

        sample_rate = resolve_async_judge_sample_rate()
        if sample_rate <= 0.0:
            report = NarrativeEvaluationReport(
                evaluator_model=resolve_model_for_task("evaluation"),
                grade="pending",
                overall_score=None,
                numeric_fidelity_score=None,
                citation_grounding_score=None,
                citation_precision_score=None,
                explanation_quality_score=None,
                thematic_consistency_score=None,
                unsupported_claims_count=0,
                rationale=(
                    "Deterministic claim verification passed; live async LLM-as-a-Judge is "
                    f"disabled ({ASYNC_JUDGE_SAMPLE_RATE_ENV}=0)."
                ),
            )
            return report, "offline_only", eval_latency_records

        if not _should_sample_async_judge(sample_rate):
            report = NarrativeEvaluationReport(
                evaluator_model=resolve_model_for_task("evaluation"),
                grade="pending",
                overall_score=None,
                numeric_fidelity_score=None,
                citation_grounding_score=None,
                citation_precision_score=None,
                explanation_quality_score=None,
                thematic_consistency_score=None,
                unsupported_claims_count=0,
                rationale=(
                    "Deterministic claim verification passed; turn not selected for async "
                    f"LLM-as-a-Judge sampling ({ASYNC_JUDGE_SAMPLE_RATE_ENV}={sample_rate:.2f})."
                ),
            )
            return report, "sampled_out", eval_latency_records

        tickers_analyzed = sorted(
            {
                *(scope.get("tickers") or []),
                *(str(r.get("ticker")) for r in variance_reports if r.get("ticker")),
                *(str(c.get("ticker")) for c in merged_citations if c.get("ticker")),
            }
        )
        _schedule_background_judge(
            query=query,
            response_markdown=outcome.answer_markdown,
            variance_reports=variance_reports,
            citations=merged_citations,
            cached_explanations_count=cached_explanations_count,
            session_id=session_id,
            user_email=user_id,
            intent=intent,
            tickers_analyzed=tickers_analyzed,
            turn_id=turn_id,
            fact_check=outcome.fact_check_verification,
            llm=None,
        )
        report = NarrativeEvaluationReport(
            evaluator_model=resolve_model_for_task("evaluation"),
            grade="pending",
            overall_score=None,
            numeric_fidelity_score=None,
            citation_grounding_score=None,
            citation_precision_score=None,
            explanation_quality_score=None,
            thematic_consistency_score=None,
            unsupported_claims_count=0,
            rationale=(
                "Deterministic claim verification and numeric fact-check passed; "
                "live Flash LLM-as-a-Judge scheduled asynchronously off the request path "
                "(real grade and score will be streamed via record_async_evaluation_result)."
            ),
        )
        return report, "async_background", eval_latency_records

    @traced("react.execute", record_args=("intent", "session_id"))
    def execute_post_retrieval_reasoning(
        self,
        *,
        query: str,
        intent: str,
        scope: dict[str, Any],
        citations: list[dict[str, Any]],
        variance_reports: list[dict[str, Any]] | None = None,
        sector_benchmarks: list[dict[str, Any]] | None = None,
        chunk_health: dict[str, Any] | None = None,
        memory_tokens: int = 0,
        memory_prep_latency_ms: float = 0.0,
        memory_context: dict[str, Any] | None = None,
        session_id: str | None = None,
        telemetry_session_id: str | None = None,
        turn_id: str | None = None,
        user_id: str = "analyst",
        material_variance_flags: list[dict[str, Any]] | None = None,
        disclosure_deltas: dict[str, Any] | None = None,
        coverage_notices: list[dict[str, Any]] | None = None,
        sector_map_summaries: dict[str, Any] | None = None,
        run_judge_sync: bool = False,
        skip_background_judge: bool = False,
        on_progress: Callable[[str, str, dict[str, Any] | None], None] | None = None,
    ) -> dict[str, Any]:
        """Run the variance pre-pass, live synthesis, verification, and Flash evaluation."""
        initial_outputs = ToolOutputs(
            variance_reports=list(variance_reports or []),
            material_variance_flags=list(material_variance_flags or []),
            disclosure_deltas=disclosure_deltas,
            sector_coverage=(sector_map_summaries or {}).get("sector_coverage"),
            sector_map_summaries=sector_map_summaries,
            sector_benchmarks=list(sector_benchmarks or []),
            chunk_health=chunk_health,
        )
        reasoning_model = (
            self.reasoning_llm.model
            if self.reasoning_llm is not None
            else resolve_model_for_task("financial_reasoning")
        )
        tracker = get_context_latency_tracker()
        runtime_res = resolve_live_adk_runtime(custom_llm=self.reasoning_llm)

        # Iteration 1: Evaluate Calculated Variances
        variance_evaluations, step_1 = self._run_variance_eval_pass(
            query=query,
            intent=intent,
            scope=scope,
            variance_reports=initial_outputs.variance_reports,
            citations=citations,
            sector_benchmarks=initial_outputs.sector_benchmarks,
            memory_tokens=memory_tokens,
            memory_prep_latency_ms=memory_prep_latency_ms,
        )
        react_steps: list[ReActStep] = [step_1]

        # Pre-pass citations are already coverage-balanced by hybrid RAG; cached 10-K sections
        # are queried on demand by the agent via QueryCached10KTool instead of a redundant pre-pass.
        cached_explanations: list[Cached10KExplanation] = []
        merged_citations = list(citations)

        # Iteration 2: Live synthesis & claim verification (with optional tool-less repair pass)
        outcome = self._synthesize_and_verify_with_retry(
            _ReasoningInput(
                query=query,
                intent=intent,
                scope=scope,
                initial_outputs=initial_outputs,
                variance_evaluations=variance_evaluations,
                cached_explanations=cached_explanations,
                merged_citations=merged_citations,
                memory_context=memory_context,
                session_id=session_id,
                user_id=user_id,
                runtime_res=runtime_res,
                reasoning_model=reasoning_model,
                coverage_notices=coverage_notices,
                on_progress=on_progress,
            )
        )
        live_stream_result = outcome.live_stream_result
        tool_outputs = outcome.tool_outputs
        hydrated_reports = tool_outputs.variance_reports
        if hydrated_reports and not variance_evaluations:
            # Variance movers for the reports the agent chose to compute (memory highlights,
            # react_execution payload); deterministic and cheap.
            variance_evaluations = evaluate_calculated_variances(
                hydrated_reports,
                query=query,
                section_filter=scope.get("section_id", "ALL"),
                include_targeted_queries=False,
            )
        turn_latency_records = list(outcome.latency_records)
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
                    f"{ttft_note}; answer_verified={outcome.answer_verified})."
                ),
                agent="FinancialAnalystAgent",
                model_used=reasoning_model,
                context_tokens=rec_synth.context_breakdown.total_context_tokens if rec_synth else 0,
                cached_context_tokens=(
                    rec_synth.context_breakdown.cached_context_tokens if rec_synth else 0
                ),
                context_latency_ms=rec_synth.context_latency_ms if rec_synth else 0.0,
                duration_ms=round(outcome.synth_ms, 3),
                status="completed" if outcome.answer_verified else "warning",
            )
        )

        # Iteration 4: Post-synthesis Flash LLM-as-a-Judge evaluation (sync or sampled async)
        effective_eval_intent = (
            outcome.agent_intent
            if isinstance(outcome.agent_intent, str) and outcome.agent_intent in QUERY_INTENT_VALUES
            else intent
        )
        eval_report, judge_mode, eval_latency_records = self._run_post_synthesis_evaluation(
            query=query,
            intent=effective_eval_intent,
            scope=scope,
            variance_reports=hydrated_reports,
            merged_citations=merged_citations,
            cached_explanations_count=len(cached_explanations),
            outcome=outcome,
            session_id=telemetry_session_id or session_id,
            user_id=user_id,
            turn_id=turn_id,
            run_judge_sync=run_judge_sync,
            skip_background_judge=skip_background_judge,
        )
        turn_latency_records.extend(eval_latency_records)

        react_steps.append(
            ReActStep(
                iteration=len(react_steps) + 1,
                phase="fact_check_and_eval",
                thought=(
                    f"Run post-generation numeric fact-check and {eval_report.evaluator_model} quality evaluation "
                    f"({judge_mode}) to verify 100% calculation fidelity and citation grounding."
                ),
                action="evaluate_variance_explanation",
                action_input={
                    "evaluator_model": eval_report.evaluator_model,
                    "judge_mode": judge_mode,
                },
                observation=(
                    f"Grade={eval_report.grade} (mode={judge_mode}, "
                    f"overall_score={_fmt_score(eval_report.overall_score)}, "
                    f"numeric_fidelity={_fmt_score(eval_report.numeric_fidelity_score)}, "
                    f"citation_grounding={_fmt_score(eval_report.citation_grounding_score)})."
                ),
                agent="NarrativeEvaluatorAgent",
                model_used=eval_report.evaluator_model,
                context_tokens=eval_report.context_tokens,
                cached_context_tokens=0,
                context_latency_ms=eval_report.context_latency_ms,
                duration_ms=eval_report.total_latency_ms,
                status="completed" if eval_report.grade in ("pass", "pending") else "warning",
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
            "agent_invoked_tools": sorted(
                {
                    str(fc.get("name"))
                    for fc in live_stream_result.function_calls
                    if fc.get("name") in DETERMINISTIC_TOOL_NAMES
                }
            ),
            "agent_intent": outcome.agent_intent,
            "key_metric_backfill_tickers": outcome.key_metric_backfill_tickers,
            "answer_verified": outcome.answer_verified,
            "answer_withheld_reason": outcome.withheld_reason,
            "judge_mode": judge_mode,
            "structured_report": outcome.structured_report_dict,
            "report_verification": (
                outcome.report_verification.model_dump(mode="json")
                if outcome.report_verification
                else None
            ),
            "agent_retrieval": _summarize_agent_retrieval(
                live_stream_result,
                outcome.agent_retrieved_citations,
                merged_citations,
                outcome.answer_markdown,
            ),
        }

        raw_llm_text_for_guardrail = (
            live_stream_result.final_text
            if (
                outcome.report_verification is not None
                and (
                    not outcome.report_verification.headline_verified
                    or outcome.report_verification.findings_rejected > 0
                )
            )
            else ""
        )

        return {
            "answer_markdown": outcome.answer_markdown,
            "raw_llm_text_for_guardrail": raw_llm_text_for_guardrail,
            "answer_verified": outcome.answer_verified,
            "citations": merged_citations,
            "tool_outputs": tool_outputs,
            "coverage_notices": outcome.coverage_notices,
            "agent_intent": outcome.agent_intent,
            "agent_tool_calls": list(live_stream_result.function_calls),
            "variance_evaluations": variance_evaluations,
            "cached_explanations": cached_explanations,
            "react_steps": react_steps,
            "evaluation_report": eval_report,
            "judge_mode": judge_mode,
            "fact_check_verification": outcome.fact_check_verification,
            "turn_latency_records": turn_latency_records,
            "turn_latency_summary": turn_latency_summary,
            "reasoning_model": reasoning_model,
            "evaluation_model": eval_report.evaluator_model,
            "execution_mode": "live_adk",
            "adk_runtime": adk_runtime_meta,
        }
