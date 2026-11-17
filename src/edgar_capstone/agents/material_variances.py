"""Deterministic anomaly flagging over XBRL financials (`flag_material_variances`).

Separates *detecting* material variances from *explaining* them. The rules below run on the
same structured profiles the variance engine uses, so every flagged figure is reproducible:

- ``yoy_change``: |YoY %| of Revenue, Operating Income, Net Income, or R&D at or above
  ``yoy_pct`` (same rounding as ``annual_series``, so the figure matches the variance payload).
- ``sign_reversal``: Operating Income or Net Income crossed zero year over year.
- ``margin_shift``: YoY change in operating margin, net margin, or R&D intensity (bps).
- ``period_change``: start-to-end % change across a multi-year window.
- ``sector_divergence``: YoY revenue growth versus the GICS sector median (percentage points).

Operating Income / Net Income percentage flags also require the absolute change to be at
least ``min_delta_revenue_pct`` of revenue, so tiny bases (e.g. $2M -> $9M) don't dominate.

Flags are ranked (high severity first, then magnitude relative to threshold). The top flags
can be paired with an MD&A passage via ``attach_flag_explanations`` so the analyst receives a
grounded explanation of each anomaly.
"""

from __future__ import annotations

import logging
import math
import os
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Literal

from google.adk.tools import FunctionTool, ToolContext

from edgar_capstone.agents.citations import clean_excerpt
from edgar_capstone.agents.resilience import with_tool_resilience
from edgar_capstone.agents.schemas import (
    MaterialFlagExplanation,
    MaterialFlagSource,
    MaterialVarianceFlag,
)
from edgar_capstone.agents.tools import (
    _gcp_adapters,
    _parse_ticker_list,
    _record_gcp_backend_invocation,
    query_cached_10k_context,
)
from edgar_capstone.common.concurrency import run_parallel_with_context
from edgar_capstone.common.domain import MetricName, normalize_edgar_ticker
from edgar_capstone.common.formatting import (
    fmt_pct,
    fmt_usd,
    margin_bps_delta,
    pct_change,
    ratio_pct,
)
from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.observability.tracing import traced

logger = logging.getLogger(__name__)

DEFAULT_MAX_FLAGS = 10
MIN_SECTOR_PEERS = 5

RatioFlagMetric = Literal["operating_margin", "net_margin", "rd_intensity"]
FlagSeverity = Literal["high", "medium"]

_METRICS: tuple[tuple[MetricName, str], ...] = (
    ("revenue", "Revenue"),
    ("operating_income", "Operating income"),
    ("net_income", "Net income"),
    ("research_and_development", "R&D expense"),
)
_RATIOS: tuple[tuple[RatioFlagMetric, MetricName, str], ...] = (
    ("operating_margin", "operating_income", "Operating margin"),
    ("net_margin", "net_income", "Net margin"),
    ("rd_intensity", "research_and_development", "R&D intensity"),
)
_PROFIT_METRICS: frozenset[MetricName] = frozenset({"operating_income", "net_income"})


@dataclass(frozen=True)
class MaterialityThresholds:
    """Configurable materiality thresholds (env: ``EDGAR_MATERIALITY_<FIELD>``)."""

    yoy_pct: float = 25.0
    period_pct: float = 50.0
    margin_bps: float = 500.0
    rd_intensity_bps: float = 200.0
    sector_growth_gap_pp: float = 20.0
    min_delta_revenue_pct: float = 1.0

    @classmethod
    def from_env(cls) -> MaterialityThresholds:
        values: dict[str, float] = {}
        for name, default in asdict(cls()).items():
            raw = os.environ.get(f"EDGAR_MATERIALITY_{name.upper()}", "").strip()
            try:
                values[name] = float(raw) if raw else float(default)
            except ValueError:
                logger.warning("Ignoring invalid EDGAR_MATERIALITY_%s=%r", name.upper(), raw)
                values[name] = float(default)
        return cls(**values)


def _source(
    curr: FinancialDataProfile,
    prev: FinancialDataProfile | None,
    metric: str | None,
    *,
    denominator_metric: str | None = None,
) -> MaterialFlagSource:
    concept = getattr(curr.metric_provenance, metric, None) if metric else None
    prior_concept = (
        getattr(prev.metric_provenance, metric, None) if (metric and prev is not None) else None
    )
    denom_concept = (
        getattr(curr.metric_provenance, denominator_metric, None) if denominator_metric else None
    )
    prior_denom_concept = (
        getattr(prev.metric_provenance, denominator_metric, None)
        if (denominator_metric and prev is not None)
        else None
    )
    return MaterialFlagSource(
        accession_number=curr.accession_number,
        sec_filing_url=curr.sec_filing_url,
        period_of_report=curr.period_of_report,
        xbrl_concept=str(concept) if concept else None,
        prior_accession_number=prev.accession_number if prev else None,
        prior_sec_filing_url=prev.sec_filing_url if prev else None,
        prior_period_of_report=prev.period_of_report if prev else None,
        prior_xbrl_concept=str(prior_concept) if prior_concept else None,
        denominator_xbrl_concept=str(denom_concept) if denom_concept else None,
        prior_denominator_xbrl_concept=str(prior_denom_concept) if prior_denom_concept else None,
    )


def _severity(magnitude: float, threshold: float) -> FlagSeverity:
    return "high" if threshold > 0 and abs(magnitude) >= 2.0 * threshold else "medium"


_EXPLANATION_KEYWORDS = {
    "revenue": "revenue increase decrease primarily due to demand growth segment",
    "operating income": "operating income gross margin operating expenses primarily due to",
    "net income": "net income income tax provision other income expense primarily due to",
    "r&d expense": "research and development expenses compensation headcount primarily due to",
}


def _explanation_query(ticker: str, fy: int, label: str, increased: bool) -> str:
    direction = "increased" if increased else "decreased"
    focus = _EXPLANATION_KEYWORDS.get(label.lower(), label.lower())
    return f"{ticker} fiscal year {fy} {label.lower()} {direction} {focus} compared to prior year"


# ---------------------------------------------------------------------------
# Pure Rule Evaluators (unit-testable without any data backend)
# ---------------------------------------------------------------------------


def _sector_revenue_growth(
    sector_profiles: list[FinancialDataProfile],
) -> dict[tuple[str, int], dict[str, float]]:
    """Map (sector_lower, fiscal_year) -> {ticker: revenue YoY %} for peers with both years."""
    by_ticker: dict[str, dict[int, FinancialDataProfile]] = defaultdict(dict)
    for p in sector_profiles:
        by_ticker[normalize_edgar_ticker(p.ticker)][p.fiscal_year] = p
    growth: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for ticker, fy_map in by_ticker.items():
        for fy, curr in fy_map.items():
            prev = fy_map.get(fy - 1)
            if prev is None or curr.revenue is None or prev.revenue is None or prev.revenue <= 0:
                continue
            pct = pct_change(prev.revenue, curr.revenue)
            if pct is not None:
                growth[(str(curr.gics_sector or "").lower(), fy)][ticker] = pct
    return growth


def _flag_sign_reversal(
    ticker: str,
    curr: FinancialDataProfile,
    prev: FinancialDataProfile,
    metric: MetricName,
    label: str,
    pv: float,
    cv: float,
) -> MaterialVarianceFlag | None:
    if metric not in _PROFIT_METRICS or pv * cv >= 0:
        return None
    swing = "a loss to a profit" if cv > 0 else "a profit to a loss"
    return MaterialVarianceFlag(
        flag_id=f"{ticker}_{curr.fiscal_year}_{metric}_sign_reversal",
        ticker=ticker,
        company_name=curr.company_name,
        gics_sector=curr.gics_sector,
        fiscal_year=curr.fiscal_year,
        prior_fiscal_year=prev.fiscal_year,
        metric=metric,
        rule="sign_reversal",
        severity="high",
        value=round(cv - pv, 2),
        unit="usd",
        threshold=0.0,
        prior_value=pv,
        current_value=cv,
        description=(
            f"{label} swung from {swing} in FY{curr.fiscal_year}: "
            f"{fmt_usd(pv)} in FY{prev.fiscal_year} to {fmt_usd(cv)}."
        ),
        explanation_query=_explanation_query(ticker, curr.fiscal_year, label, cv > pv),
        source=_source(curr, prev, metric),
    )


def _flag_yoy(
    ticker: str,
    curr: FinancialDataProfile,
    prev: FinancialDataProfile,
    metric: MetricName,
    label: str,
    pv: float,
    cv: float,
    *,
    materiality_base: float,
    thr: MaterialityThresholds,
) -> MaterialVarianceFlag | None:
    pct = pct_change(pv, cv)
    if pct is None or abs(pct) < thr.yoy_pct:
        return None
    if (
        metric in _PROFIT_METRICS
        and materiality_base > 0
        and abs(cv - pv) < materiality_base * thr.min_delta_revenue_pct / 100.0
    ):
        return None
    verb = "rose" if pct > 0 else "fell"
    return MaterialVarianceFlag(
        flag_id=f"{ticker}_{curr.fiscal_year}_{metric}_yoy_change",
        ticker=ticker,
        company_name=curr.company_name,
        gics_sector=curr.gics_sector,
        fiscal_year=curr.fiscal_year,
        prior_fiscal_year=prev.fiscal_year,
        metric=metric,
        rule="yoy_change",
        severity=_severity(pct, thr.yoy_pct),
        value=pct,
        unit="pct",
        threshold=thr.yoy_pct,
        prior_value=pv,
        current_value=cv,
        description=(
            f"{label} {verb} {fmt_pct(pct)} YoY in FY{curr.fiscal_year} "
            f"({fmt_usd(pv)} → {fmt_usd(cv)}), beyond the "
            f"±{thr.yoy_pct:g}% materiality threshold."
        ),
        explanation_query=_explanation_query(ticker, curr.fiscal_year, label, pct > 0),
        source=_source(curr, prev, metric),
    )


def _flag_margin_shift(
    ticker: str,
    curr: FinancialDataProfile,
    prev: FinancialDataProfile,
    flag_metric: RatioFlagMetric,
    numerator: MetricName,
    label: str,
    thr: MaterialityThresholds,
) -> MaterialVarianceFlag | None:
    p_ratio = ratio_pct(getattr(prev, numerator), abs(prev.revenue)) if prev.revenue else None
    c_ratio = ratio_pct(getattr(curr, numerator), abs(curr.revenue)) if curr.revenue else None
    bps = margin_bps_delta(p_ratio, c_ratio)
    if p_ratio is None or c_ratio is None or bps is None:
        return None
    limit = thr.rd_intensity_bps if flag_metric == "rd_intensity" else thr.margin_bps
    if abs(bps) < limit:
        return None
    verb = "expanded" if bps > 0 else "contracted"
    return MaterialVarianceFlag(
        flag_id=f"{ticker}_{curr.fiscal_year}_{flag_metric}_margin_shift",
        ticker=ticker,
        company_name=curr.company_name,
        gics_sector=curr.gics_sector,
        fiscal_year=curr.fiscal_year,
        prior_fiscal_year=prev.fiscal_year,
        metric=flag_metric,
        rule="margin_shift",
        severity=_severity(bps, limit),
        value=bps,
        unit="bps",
        threshold=limit,
        prior_value=p_ratio,
        current_value=c_ratio,
        description=(
            f"{label} {verb} {bps:+,.1f} bps YoY in FY{curr.fiscal_year} "
            f"({p_ratio:.2f}% → {c_ratio:.2f}%)."
        ),
        explanation_query=(
            f"{ticker} FY{curr.fiscal_year} {label.lower()} gross margin cost of revenue "
            "operating expenses drivers"
        ),
        source=_source(curr, prev, numerator, denominator_metric="revenue"),
    )


def _flag_sector_divergence(
    ticker: str,
    curr: FinancialDataProfile,
    prev: FinancialDataProfile,
    sector_growth: dict[tuple[str, int], dict[str, float]],
    thr: MaterialityThresholds,
) -> MaterialVarianceFlag | None:
    if not sector_growth or curr.revenue is None or not prev.revenue or prev.revenue <= 0:
        return None
    own = pct_change(prev.revenue, curr.revenue)
    norm_tk = normalize_edgar_ticker(ticker)
    peers = {
        t: g
        for t, g in sector_growth.get(
            (str(curr.gics_sector or "").lower(), curr.fiscal_year), {}
        ).items()
        if t != norm_tk
    }
    if own is None or len(peers) < MIN_SECTOR_PEERS:
        return None
    median = round(statistics.median(peers.values()), 2)
    gap = round(own - median, 2)
    if abs(gap) < thr.sector_growth_gap_pp:
        return None
    side = "above" if gap > 0 else "below"
    return MaterialVarianceFlag(
        flag_id=f"{ticker}_{curr.fiscal_year}_revenue_sector_divergence",
        ticker=ticker,
        company_name=curr.company_name,
        gics_sector=curr.gics_sector,
        fiscal_year=curr.fiscal_year,
        prior_fiscal_year=prev.fiscal_year,
        metric="revenue",
        rule="sector_divergence",
        severity=_severity(gap, thr.sector_growth_gap_pp),
        value=gap,
        unit="pp",
        threshold=thr.sector_growth_gap_pp,
        prior_value=prev.revenue,
        current_value=curr.revenue,
        benchmark_value=median,
        peer_count=len(peers),
        description=(
            f"FY{curr.fiscal_year} revenue growth of {fmt_pct(own)} was "
            f"{abs(gap):,.2f} pp {side} the {curr.gics_sector} median "
            f"({fmt_pct(median)} across {len(peers)} peers)."
        ),
        explanation_query=_explanation_query(ticker, curr.fiscal_year, "Revenue", own > median),
        source=_source(curr, prev, "revenue"),
    )


def _flag_period_change(
    ticker: str,
    s_prof: FinancialDataProfile,
    e_prof: FinancialDataProfile,
    metric: MetricName,
    label: str,
    *,
    base: float,
    thr: MaterialityThresholds,
) -> MaterialVarianceFlag | None:
    sv, ev = getattr(s_prof, metric), getattr(e_prof, metric)
    if sv is None or ev is None or sv * ev < 0:
        return None
    pct = pct_change(sv, ev)
    if pct is None or abs(pct) < thr.period_pct:
        return None
    if (
        metric in _PROFIT_METRICS
        and base > 0
        and abs(ev - sv) < base * thr.min_delta_revenue_pct / 100.0
    ):
        return None
    verb = "rose" if pct > 0 else "fell"
    return MaterialVarianceFlag(
        flag_id=f"{ticker}_{s_prof.fiscal_year}_{e_prof.fiscal_year}_{metric}_period_change",
        ticker=ticker,
        company_name=e_prof.company_name,
        gics_sector=e_prof.gics_sector,
        fiscal_year=e_prof.fiscal_year,
        prior_fiscal_year=s_prof.fiscal_year,
        metric=metric,
        rule="period_change",
        severity=_severity(pct, thr.period_pct),
        value=pct,
        unit="pct",
        threshold=thr.period_pct,
        prior_value=sv,
        current_value=ev,
        description=(
            f"{label} {verb} {fmt_pct(pct)} from FY{s_prof.fiscal_year} to FY{e_prof.fiscal_year} "
            f"({fmt_usd(sv)} → {fmt_usd(ev)})."
        ),
        explanation_query=_explanation_query(ticker, e_prof.fiscal_year, label, pct > 0),
        source=_source(e_prof, s_prof, metric),
    )


def compute_material_variance_flags(
    profiles: list[FinancialDataProfile],
    *,
    tickers: set[str],
    start_year: int,
    end_year: int,
    thresholds: MaterialityThresholds | None = None,
    sector_profiles: list[FinancialDataProfile] | None = None,
    max_flags: int | None = DEFAULT_MAX_FLAGS,
) -> dict[str, Any]:
    """Detect material variances for ``tickers`` across ``start_year``..``end_year``.

    YoY steps are evaluated for every fiscal year in the window whose prior year is present
    (mirroring ``annual_series``), so a FY2024-FY2026 window also tests FY2024 vs FY2023.
    """
    thr = thresholds or MaterialityThresholds()
    norm_filter = {normalize_edgar_ticker(t) for t in tickers}
    by_ticker: dict[str, dict[int, FinancialDataProfile]] = defaultdict(dict)
    for p in profiles:
        if normalize_edgar_ticker(p.ticker) in norm_filter:
            by_ticker[p.ticker][p.fiscal_year] = p

    sector_growth = _sector_revenue_growth(sector_profiles) if sector_profiles else {}
    flags: list[MaterialVarianceFlag] = []

    for ticker, fy_map in sorted(by_ticker.items()):
        window = sorted(fy for fy in fy_map if start_year <= fy <= end_year)
        if not window:
            continue

        for fy in window:
            curr = fy_map[fy]
            prev = fy_map.get(fy - 1)
            if prev is None:
                continue
            materiality_base = abs(curr.revenue or prev.revenue or 0.0)

            for metric, label in _METRICS:
                pv, cv = getattr(prev, metric), getattr(curr, metric)
                if pv is None or cv is None:
                    continue
                reversal = _flag_sign_reversal(ticker, curr, prev, metric, label, pv, cv)
                if reversal is not None:
                    flags.append(reversal)
                    continue
                yoy = _flag_yoy(
                    ticker,
                    curr,
                    prev,
                    metric,
                    label,
                    pv,
                    cv,
                    materiality_base=materiality_base,
                    thr=thr,
                )
                if yoy is not None:
                    flags.append(yoy)

            for flag_metric, numerator, label in _RATIOS:
                shift = _flag_margin_shift(ticker, curr, prev, flag_metric, numerator, label, thr)
                if shift is not None:
                    flags.append(shift)

            div = _flag_sector_divergence(ticker, curr, prev, sector_growth, thr)
            if div is not None:
                flags.append(div)

        eff_start, eff_end = window[0], window[-1]
        if eff_end - eff_start >= 2:
            s_prof, e_prof = fy_map[eff_start], fy_map[eff_end]
            base = abs(e_prof.revenue or s_prof.revenue or 0.0)
            for metric, label in _METRICS:
                period = _flag_period_change(
                    ticker, s_prof, e_prof, metric, label, base=base, thr=thr
                )
                if period is not None:
                    flags.append(period)

    def _rank(flag: MaterialVarianceFlag) -> tuple[int, float, str]:
        ratio = math.inf if flag.threshold <= 0 else abs(flag.value) / flag.threshold
        return (0 if flag.severity == "high" else 1, -ratio, flag.flag_id)

    flags.sort(key=_rank)
    total = len(flags)
    kept = flags if max_flags is None else flags[: max(0, int(max_flags))]
    return {
        "status": "success",
        "total_flags": total,
        "returned_flags": len(kept),
        "high_severity_count": sum(1 for f in flags if f.severity == "high"),
        "thresholds": asdict(thr),
        "flags": [f.model_dump(mode="json") for f in kept],
    }


# ---------------------------------------------------------------------------
# Data access + ADK tool
# ---------------------------------------------------------------------------


def _validate_flag_request(
    tickers: str, start_year: int, end_year: int
) -> tuple[set[str], int, int] | dict[str, Any]:
    ticker_filter = _parse_ticker_list(tickers)
    if not ticker_filter:
        return {
            "status": "error",
            "message": "Please specify at least one ticker symbol (e.g., 'NVDA' or 'NVDA,AAPL').",
        }
    lo, hi = sorted((int(start_year), int(end_year)))
    return ticker_filter, lo, hi


def _compute_and_annotate_missing(
    company_profiles: list[FinancialDataProfile],
    sector_profiles: list[FinancialDataProfile],
    *,
    ticker_filter: set[str],
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    result = compute_material_variance_flags(
        company_profiles,
        tickers=ticker_filter,
        start_year=start_year,
        end_year=end_year,
        thresholds=MaterialityThresholds.from_env(),
        sector_profiles=sector_profiles,
    )
    found = {normalize_edgar_ticker(p.ticker) for p in company_profiles}
    result["missing_tickers"] = sorted(ticker_filter - found)
    return result


def _load_profiles(
    ticker_filter: set[str],
    start_year: int,
    end_year: int,
    tool_context: ToolContext | None,
) -> tuple[list[FinancialDataProfile], list[FinancialDataProfile]]:
    """Return (company profiles, sector peer profiles) from BigQuery (1 batched peer query)."""
    adapters = _gcp_adapters()
    profiles = adapters.bigquery.fetch_profiles(
        tickers=ticker_filter, start_year=start_year - 1, end_year=end_year
    )
    sectors = sorted({p.gics_sector for p in profiles if p.gics_sector})
    sector_profiles: list[FinancialDataProfile] = []
    if len(sectors) == 1:
        sector_profiles = adapters.bigquery.fetch_profiles(
            tickers=set(),
            start_year=start_year - 1,
            end_year=end_year,
            gics_sector=sectors[0],
        )
    elif len(sectors) > 1:
        sector_profiles = adapters.bigquery.fetch_profiles(
            tickers=set(),
            start_year=start_year - 1,
            end_year=end_year,
            gics_sectors=set(sectors),
        )
    _record_gcp_backend_invocation(
        tool_context,
        tool_name="flag_material_variances",
        data_source="gcp",
        services_hit=["bigquery"],
    )
    return profiles, sector_profiles


def flag_material_variances_from_cached_profiles(
    all_profiles: list[FinancialDataProfile],
    *,
    tickers: str,
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    """Compute material variance flags from an already-fetched profile catalog (0 BigQuery RPCs)."""
    validated = _validate_flag_request(tickers, start_year, end_year)
    if isinstance(validated, dict):
        return validated
    ticker_filter, eff_start, eff_end = validated
    lo, hi = eff_start - 1, eff_end
    in_window = [p for p in all_profiles if lo <= p.fiscal_year <= hi]
    company_profiles = [p for p in in_window if normalize_edgar_ticker(p.ticker) in ticker_filter]
    sectors = {str(p.gics_sector or "").lower() for p in company_profiles if p.gics_sector}
    sector_profiles = [p for p in in_window if str(p.gics_sector or "").lower() in sectors]
    return _compute_and_annotate_missing(
        company_profiles,
        sector_profiles,
        ticker_filter=ticker_filter,
        start_year=eff_start,
        end_year=eff_end,
    )


@with_tool_resilience(tool_name="flag_material_variances")
def flag_material_variances(
    tickers: str,
    start_year: int,
    end_year: int,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Deterministically flags material financial variances (anomalies) that need a 10-K explanation.

    Rules: large YoY changes in Revenue, Operating Income, Net Income, or R&D; operating or net
    income sign reversals; large operating-margin, net-margin, or R&D-intensity shifts (bps);
    large multi-year changes; and revenue growth far from the GICS sector median. Thresholds are
    configured server-side. Each flag carries its filing accession number, XBRL concept, and a
    suggested MD&A query for explaining it. Never recompute these figures.

    Args:
        tickers: Comma-separated ticker symbols (e.g., 'NVDA' or 'NVDA,AAPL').
        start_year: First fiscal year of the window (e.g., 2024).
        end_year: Last fiscal year of the window (e.g., 2026).

    Returns:
        Dictionary with 'status', 'total_flags', 'thresholds', and ranked 'flags'.
    """
    validated = _validate_flag_request(tickers, start_year, end_year)
    if isinstance(validated, dict):
        return validated
    ticker_filter, eff_start, eff_end = validated

    profiles, sector_profiles = _load_profiles(ticker_filter, eff_start, eff_end, tool_context)
    result = _compute_and_annotate_missing(
        profiles,
        sector_profiles,
        ticker_filter=ticker_filter,
        start_year=eff_start,
        end_year=eff_end,
    )
    if tool_context is not None and hasattr(tool_context, "state"):
        tool_context.state["material_variance_flags"] = result["flags"]
    return result


@traced("anomaly.attach_explanations")
def attach_flag_explanations(
    flags: list[dict[str, Any]],
    *,
    max_flags: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retrieve an MD&A passage explaining each of the top flags (one per company-year) in parallel.

    Only passages located in a specific filing chunk are attached. Returns
    ``(flags, citations)``; the flags list is updated in place with an ``explanation`` entry.
    """
    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()
    for flag in flags:
        if len(selected) >= max_flags:
            break
        key = (str(flag["ticker"]), int(flag["fiscal_year"]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(flag)

    if not selected:
        return flags, []

    def _fetch_one(flag: dict[str, Any]) -> dict[str, Any] | None:
        res = query_cached_10k_context(
            ticker=str(flag["ticker"]),
            fiscal_year=int(flag["fiscal_year"]),
            section_id="item_7",
            variance_query=str(flag.get("explanation_query") or flag["description"]),
            max_passages=2,
        )
        if res.get("status") != "success":
            return None
        by_id = {str(c["chunk_id"]): c for c in res.get("matched_citations") or []}
        passage = next(
            (
                p
                for p in res.get("passages") or []
                if p.get("chunk_match") == "exact" and str(p.get("chunk_id")) in by_id
            ),
            None,
        )
        if passage is None:
            return None
        cit = by_id[str(passage["chunk_id"])]
        flag["explanation"] = MaterialFlagExplanation(
            chunk_id=str(cit["chunk_id"]),
            section_id=cit.get("section_id") or "item_7",
            section_title=str(cit.get("section_title") or ""),
            excerpt=clean_excerpt(str(passage.get("text") or ""), 420),
            sec_filing_url=str(cit.get("sec_filing_url") or ""),
        ).model_dump(mode="json")
        return cit

    fetched = run_parallel_with_context(selected, _fetch_one, max_workers=min(4, len(selected)))
    citations = [cit for cit in fetched if cit is not None]
    return flags, citations


def material_flag_payload_figures(flags: list[dict[str, Any]] | None) -> list[tuple[str, float]]:
    """(kind, value) figures a narrative may quote from flags: pct, bps, pp-as-pct, and USD."""
    out: list[tuple[str, float]] = []
    for flag in flags or []:
        unit, value = flag.get("unit"), flag.get("value")
        if value is None:
            continue
        if unit in ("pct", "pp"):
            out.append(("pct", float(value)))
        elif unit == "bps":
            out.append(("bps", float(value)))
        elif unit == "usd":
            out.append(("amount", abs(float(value))))
        if flag.get("benchmark_value") is not None:
            out.append(("pct", float(flag["benchmark_value"])))
        if flag.get("metric") in ("operating_margin", "net_margin", "rd_intensity"):
            for key in ("prior_value", "current_value"):
                if flag.get(key) is not None:
                    out.append(("pct", float(flag[key])))
        else:
            for key in ("prior_value", "current_value"):
                if flag.get(key) is not None:
                    out.append(("amount", abs(float(flag[key]))))
    return out


AnomalyFlagTool = FunctionTool(flag_material_variances)
