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

import contextvars
import logging
import math
import os
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

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
from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.observability.tracing import traced

logger = logging.getLogger(__name__)

DEFAULT_MAX_FLAGS = 10
MIN_SECTOR_PEERS = 5

_METRICS: tuple[tuple[str, str], ...] = (
    ("revenue", "Revenue"),
    ("operating_income", "Operating income"),
    ("net_income", "Net income"),
    ("research_and_development", "R&D expense"),
)
_RATIOS: tuple[tuple[str, str, str], ...] = (
    # (flag metric, numerator field, label)
    ("operating_margin", "operating_income", "Operating margin"),
    ("net_margin", "net_income", "Net margin"),
    ("rd_intensity", "research_and_development", "R&D intensity"),
)
_PROFIT_METRICS = frozenset({"operating_income", "net_income"})


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


# ---------------------------------------------------------------------------
# Formatting (matches the variance engine's rounding and the UI's notation)
# ---------------------------------------------------------------------------


def _fmt_usd(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e12:
        return f"{sign}${v / 1e12:,.2f}T"
    if v >= 1e9:
        return f"{sign}${v / 1e9:,.2f}B"
    if v >= 1e6:
        return f"{sign}${v / 1e6:,.2f}M"
    return f"{sign}${v:,.0f}"


def _fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def _pct_change(prev: float, curr: float) -> float | None:
    if prev == 0:
        return None
    return round(((curr - prev) / abs(prev)) * 100.0, 2)


def _ratio_pct(numerator: float | None, revenue: float | None) -> float | None:
    if numerator is None or not revenue:
        return None
    return round((numerator / abs(revenue)) * 100.0, 2)


def _norm_ticker(ticker: str) -> str:
    return ticker.upper().replace(".", "-")


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


def _severity(magnitude: float, threshold: float) -> str:
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
# Core rules (pure; unit-testable without any data backend)
# ---------------------------------------------------------------------------


def _sector_revenue_growth(
    sector_profiles: list[FinancialDataProfile],
) -> dict[tuple[str, int], dict[str, float]]:
    """Map (sector_lower, fiscal_year) -> {ticker: revenue YoY %} for peers with both years."""
    by_ticker: dict[str, dict[int, FinancialDataProfile]] = defaultdict(dict)
    for p in sector_profiles:
        by_ticker[_norm_ticker(p.ticker)][p.fiscal_year] = p
    growth: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for ticker, fy_map in by_ticker.items():
        for fy, curr in fy_map.items():
            prev = fy_map.get(fy - 1)
            if prev is None or curr.revenue is None or prev.revenue is None or prev.revenue <= 0:
                continue
            pct = _pct_change(prev.revenue, curr.revenue)
            if pct is not None:
                growth[(str(curr.gics_sector or "").lower(), fy)][ticker] = pct
    return growth


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
    norm_filter = {_norm_ticker(t) for t in tickers}
    by_ticker: dict[str, dict[int, FinancialDataProfile]] = defaultdict(dict)
    for p in profiles:
        if _norm_ticker(p.ticker) in norm_filter:
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

            # 1-2. Metric YoY changes and profit sign reversals
            for metric, label in _METRICS:
                pv, cv = getattr(prev, metric), getattr(curr, metric)
                if pv is None or cv is None:
                    continue
                if metric in _PROFIT_METRICS and pv * cv < 0:
                    swing = "a loss to a profit" if cv > 0 else "a profit to a loss"
                    flags.append(
                        MaterialVarianceFlag(
                            flag_id=f"{ticker}_{fy}_{metric}_sign_reversal",
                            ticker=ticker,
                            company_name=curr.company_name,
                            gics_sector=curr.gics_sector,
                            fiscal_year=fy,
                            prior_fiscal_year=prev.fiscal_year,
                            metric=metric,  # type: ignore[arg-type]
                            rule="sign_reversal",
                            severity="high",
                            value=round(cv - pv, 2),
                            unit="usd",
                            threshold=0.0,
                            prior_value=pv,
                            current_value=cv,
                            description=(
                                f"{label} swung from {swing} in FY{fy}: "
                                f"{_fmt_usd(pv)} in FY{prev.fiscal_year} to {_fmt_usd(cv)}."
                            ),
                            explanation_query=_explanation_query(ticker, fy, label, cv > pv),
                            source=_source(curr, prev, metric),
                        )
                    )
                    continue
                pct = _pct_change(pv, cv)
                if pct is None or abs(pct) < thr.yoy_pct:
                    continue
                if (
                    metric in _PROFIT_METRICS
                    and materiality_base > 0
                    and abs(cv - pv) < materiality_base * thr.min_delta_revenue_pct / 100.0
                ):
                    continue
                verb = "rose" if pct > 0 else "fell"
                flags.append(
                    MaterialVarianceFlag(
                        flag_id=f"{ticker}_{fy}_{metric}_yoy_change",
                        ticker=ticker,
                        company_name=curr.company_name,
                        gics_sector=curr.gics_sector,
                        fiscal_year=fy,
                        prior_fiscal_year=prev.fiscal_year,
                        metric=metric,  # type: ignore[arg-type]
                        rule="yoy_change",
                        severity=_severity(pct, thr.yoy_pct),  # type: ignore[arg-type]
                        value=pct,
                        unit="pct",
                        threshold=thr.yoy_pct,
                        prior_value=pv,
                        current_value=cv,
                        description=(
                            f"{label} {verb} {_fmt_pct(pct)} YoY in FY{fy} "
                            f"({_fmt_usd(pv)} → {_fmt_usd(cv)}), beyond the "
                            f"±{thr.yoy_pct:g}% materiality threshold."
                        ),
                        explanation_query=_explanation_query(ticker, fy, label, pct > 0),
                        source=_source(curr, prev, metric),
                    )
                )

            # 3. Margin / intensity shifts (bps)
            for flag_metric, numerator, label in _RATIOS:
                p_ratio = _ratio_pct(getattr(prev, numerator), prev.revenue)
                c_ratio = _ratio_pct(getattr(curr, numerator), curr.revenue)
                if p_ratio is None or c_ratio is None:
                    continue
                bps = round((c_ratio - p_ratio) * 100.0, 1)
                limit = thr.rd_intensity_bps if flag_metric == "rd_intensity" else thr.margin_bps
                if abs(bps) < limit:
                    continue
                verb = "expanded" if bps > 0 else "contracted"
                flags.append(
                    MaterialVarianceFlag(
                        flag_id=f"{ticker}_{fy}_{flag_metric}_margin_shift",
                        ticker=ticker,
                        company_name=curr.company_name,
                        gics_sector=curr.gics_sector,
                        fiscal_year=fy,
                        prior_fiscal_year=prev.fiscal_year,
                        metric=flag_metric,  # type: ignore[arg-type]
                        rule="margin_shift",
                        severity=_severity(bps, limit),  # type: ignore[arg-type]
                        value=bps,
                        unit="bps",
                        threshold=limit,
                        prior_value=p_ratio,
                        current_value=c_ratio,
                        description=(
                            f"{label} {verb} {bps:+,.1f} bps YoY in FY{fy} "
                            f"({p_ratio:.2f}% → {c_ratio:.2f}%)."
                        ),
                        explanation_query=(
                            f"{ticker} FY{fy} {label.lower()} gross margin cost of revenue "
                            "operating expenses drivers"
                        ),
                        source=_source(curr, prev, numerator, denominator_metric="revenue"),
                    )
                )

            # 5. Revenue growth versus the GICS sector median
            if sector_growth and curr.revenue is not None and prev.revenue and prev.revenue > 0:
                own = _pct_change(prev.revenue, curr.revenue)
                peers = {
                    t: g
                    for t, g in sector_growth.get(
                        (str(curr.gics_sector or "").lower(), fy), {}
                    ).items()
                    if t != _norm_ticker(ticker)
                }
                if own is not None and len(peers) >= MIN_SECTOR_PEERS:
                    median = round(statistics.median(peers.values()), 2)
                    gap = round(own - median, 2)
                    if abs(gap) >= thr.sector_growth_gap_pp:
                        side = "above" if gap > 0 else "below"
                        flags.append(
                            MaterialVarianceFlag(
                                flag_id=f"{ticker}_{fy}_revenue_sector_divergence",
                                ticker=ticker,
                                company_name=curr.company_name,
                                gics_sector=curr.gics_sector,
                                fiscal_year=fy,
                                prior_fiscal_year=prev.fiscal_year,
                                metric="revenue",
                                rule="sector_divergence",
                                severity=_severity(gap, thr.sector_growth_gap_pp),  # type: ignore[arg-type]
                                value=gap,
                                unit="pp",
                                threshold=thr.sector_growth_gap_pp,
                                prior_value=prev.revenue,
                                current_value=curr.revenue,
                                benchmark_value=median,
                                peer_count=len(peers),
                                description=(
                                    f"FY{fy} revenue growth of {_fmt_pct(own)} was "
                                    f"{abs(gap):,.2f} pp {side} the {curr.gics_sector} median "
                                    f"({_fmt_pct(median)} across {len(peers)} peers)."
                                ),
                                explanation_query=_explanation_query(
                                    ticker, fy, "Revenue", own > median
                                ),
                                source=_source(curr, prev, "revenue"),
                            )
                        )

        # 4. Multi-year window change (only when it spans more than one YoY step)
        eff_start, eff_end = window[0], window[-1]
        if eff_end - eff_start >= 2:
            s_prof, e_prof = fy_map[eff_start], fy_map[eff_end]
            base = abs(e_prof.revenue or s_prof.revenue or 0.0)
            for metric, label in _METRICS:
                sv, ev = getattr(s_prof, metric), getattr(e_prof, metric)
                if sv is None or ev is None or sv * ev < 0:
                    continue
                pct = _pct_change(sv, ev)
                if pct is None or abs(pct) < thr.period_pct:
                    continue
                if (
                    metric in _PROFIT_METRICS
                    and base > 0
                    and abs(ev - sv) < base * thr.min_delta_revenue_pct / 100.0
                ):
                    continue
                verb = "rose" if pct > 0 else "fell"
                flags.append(
                    MaterialVarianceFlag(
                        flag_id=f"{ticker}_{eff_start}_{eff_end}_{metric}_period_change",
                        ticker=ticker,
                        company_name=e_prof.company_name,
                        gics_sector=e_prof.gics_sector,
                        fiscal_year=eff_end,
                        prior_fiscal_year=eff_start,
                        metric=metric,  # type: ignore[arg-type]
                        rule="period_change",
                        severity=_severity(pct, thr.period_pct),  # type: ignore[arg-type]
                        value=pct,
                        unit="pct",
                        threshold=thr.period_pct,
                        prior_value=sv,
                        current_value=ev,
                        description=(
                            f"{label} {verb} {_fmt_pct(pct)} from FY{eff_start} to FY{eff_end} "
                            f"({_fmt_usd(sv)} → {_fmt_usd(ev)})."
                        ),
                        explanation_query=_explanation_query(ticker, eff_end, label, pct > 0),
                        source=_source(e_prof, s_prof, metric),
                    )
                )

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


def _load_profiles(
    ticker_filter: set[str],
    start_year: int,
    end_year: int,
    tool_context: ToolContext | None,
) -> tuple[list[FinancialDataProfile], list[FinancialDataProfile]]:
    """Return (company profiles, sector peer profiles) for the window from BigQuery."""
    adapters = _gcp_adapters()
    profiles = adapters.bigquery.fetch_profiles(
        tickers=ticker_filter, start_year=start_year - 1, end_year=end_year
    )
    sectors = sorted({p.gics_sector for p in profiles if p.gics_sector})
    sector_profiles: list[FinancialDataProfile] = []
    if len(sectors) == 1:
        sector_profiles.extend(
            adapters.bigquery.fetch_profiles(
                tickers=set(),
                start_year=start_year - 1,
                end_year=end_year,
                gics_sector=sectors[0],
            )
        )
    elif len(sectors) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(sectors))) as pool:
            futs = [
                pool.submit(
                    contextvars.copy_context().run,
                    adapters.bigquery.fetch_profiles,
                    tickers=set(),
                    start_year=start_year - 1,
                    end_year=end_year,
                    gics_sector=sector,
                )
                for sector in sectors
            ]
            for fut in futs:
                sector_profiles.extend(fut.result())
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
    ticker_filter = _parse_ticker_list(tickers)
    if not ticker_filter:
        return {
            "status": "error",
            "message": "Please specify at least one ticker symbol (e.g., 'NVDA' or 'NVDA,AAPL').",
        }
    if start_year > end_year:
        start_year, end_year = end_year, start_year
    lo, hi = int(start_year) - 1, int(end_year)
    in_window = [p for p in all_profiles if lo <= p.fiscal_year <= hi]
    company_profiles = [p for p in in_window if _norm_ticker(p.ticker) in ticker_filter]
    sectors = {str(p.gics_sector or "").lower() for p in company_profiles if p.gics_sector}
    sector_profiles = [p for p in in_window if str(p.gics_sector or "").lower() in sectors]
    result = compute_material_variance_flags(
        company_profiles,
        tickers=ticker_filter,
        start_year=int(start_year),
        end_year=int(end_year),
        thresholds=MaterialityThresholds.from_env(),
        sector_profiles=sector_profiles,
    )
    found = {_norm_ticker(p.ticker) for p in company_profiles}
    result["missing_tickers"] = sorted(ticker_filter - found)
    return result


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
    ticker_filter = _parse_ticker_list(tickers)
    if not ticker_filter:
        return {
            "status": "error",
            "message": "Please specify at least one ticker symbol (e.g., 'NVDA' or 'NVDA,AAPL').",
        }
    if start_year > end_year:
        start_year, end_year = end_year, start_year

    profiles, sector_profiles = _load_profiles(
        ticker_filter, int(start_year), int(end_year), tool_context
    )
    result = compute_material_variance_flags(
        profiles,
        tickers=ticker_filter,
        start_year=int(start_year),
        end_year=int(end_year),
        thresholds=MaterialityThresholds.from_env(),
        sector_profiles=sector_profiles,
    )
    found = {_norm_ticker(p.ticker) for p in profiles}
    result["missing_tickers"] = sorted(ticker_filter - found)

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

    def _fetch_one(flag: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        res = query_cached_10k_context(
            ticker=str(flag["ticker"]),
            fiscal_year=int(flag["fiscal_year"]),
            section_id="item_7",
            variance_query=str(flag.get("explanation_query") or flag["description"]),
            max_passages=2,
        )
        if res.get("status") != "success":
            return flag, None
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
            return flag, None
        cit = by_id[str(passage["chunk_id"])]
        flag["explanation"] = MaterialFlagExplanation(
            chunk_id=str(cit["chunk_id"]),
            section_id=cit.get("section_id") or "item_7",
            section_title=str(cit.get("section_title") or ""),
            excerpt=clean_excerpt(str(passage.get("text") or ""), 420),
            sec_filing_url=str(cit.get("sec_filing_url") or ""),
        ).model_dump(mode="json")
        return flag, cit

    citations: list[dict[str, Any]] = []
    if len(selected) == 1:
        _, cit = _fetch_one(selected[0])
        if cit is not None:
            citations.append(cit)
    else:
        with ThreadPoolExecutor(max_workers=min(4, len(selected))) as pool:
            futs = [
                pool.submit(contextvars.copy_context().run, _fetch_one, flag) for flag in selected
            ]
            for fut in futs:
                _, cit = fut.result()
                if cit is not None:
                    citations.append(cit)
    return flags, citations


def material_flag_payload_figures(flags: list[dict[str, Any]] | None) -> list[tuple[str, float]]:
    """(kind, value) figures a narrative may quote from flags: pct, bps, pp-as-pct, and USD."""
    out: list[tuple[str, float]] = []
    for flag in flags or []:
        unit, value = flag.get("unit"), flag.get("value")
        if value is None:
            continue
        if unit == "pct" or unit == "pp":
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
