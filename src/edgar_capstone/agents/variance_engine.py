"""Deterministic financial metric conversion and period-over-period variance engine."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from edgar_capstone.agents.schemas import (
    AnnualStepVariance,
    CompanyVarianceReport,
    MetricRecord,
    MetricVariance,
    NumericCitation,
)
from edgar_capstone.common.domain import (
    CORE_METRICS,
    RATIO_METRICS,
    MetricName,
    normalize_edgar_ticker,
)
from edgar_capstone.common.formatting import margin_bps_delta, pct_change, ratio_pct
from edgar_capstone.dataset.models import FinancialDataProfile


def _profile_to_metric_record(p: FinancialDataProfile) -> MetricRecord:
    """Convert a `FinancialDataProfile` into a `MetricRecord` with computed margins."""
    abs_rev = abs(p.revenue) if p.revenue else None
    return MetricRecord(
        profile_id=p.profile_id,
        ticker=p.ticker,
        cik=p.cik,
        company_name=p.company_name,
        gics_sector=p.gics_sector,
        gics_sub_industry=p.gics_sub_industry,
        fiscal_year=p.fiscal_year,
        period_of_report=p.period_of_report,
        filing_date=p.filing_date,
        accession_number=p.accession_number,
        currency=p.currency,
        revenue=p.revenue,
        operating_income=p.operating_income,
        net_income=p.net_income,
        research_and_development=p.research_and_development,
        operating_margin_pct=ratio_pct(p.operating_income, abs_rev),
        net_margin_pct=ratio_pct(p.net_income, abs_rev),
        rd_intensity_pct=ratio_pct(p.research_and_development, abs_rev),
        sec_filing_url=p.sec_filing_url,
        unstructured_gcs_uri=p.unstructured_gcs_uri,
        structured_gcs_uri=p.structured_gcs_uri,
        provenance=p.metric_provenance.model_dump(mode="json"),
    )


def _xbrl_concept_for_metric(profile: FinancialDataProfile, metric_name: str) -> str | None:
    """Return the XBRL concept name recorded in `metric_provenance` for a metric."""
    val = getattr(profile.metric_provenance, metric_name, None)
    return str(val).strip() if val else None


def _numeric_citation_for_profile(
    profile: FinancialDataProfile, metric_name: str, value: float | None
) -> NumericCitation:
    """Build a traceable `NumericCitation` from a company-year `FinancialDataProfile`."""
    return NumericCitation(
        ticker=profile.ticker,
        fiscal_year=profile.fiscal_year,
        period_of_report=profile.period_of_report or None,
        filing_date=profile.filing_date or None,
        accession_number=profile.accession_number or None,
        sec_filing_url=profile.sec_filing_url or None,
        metric=metric_name,
        xbrl_concept=_xbrl_concept_for_metric(profile, metric_name),
        value=value,
    )


def _compute_single_metric_variance(
    metric_name: MetricName,
    start_year: int,
    end_year: int,
    start_val: float | None,
    end_val: float | None,
    *,
    start_profile: FinancialDataProfile | None = None,
    end_profile: FinancialDataProfile | None = None,
) -> MetricVariance:
    """Deterministically compute USD delta, percentage delta, CAGR, and XBRL provenance."""
    start_period = (start_profile.period_of_report or None) if start_profile else None
    end_period = (end_profile.period_of_report or None) if end_profile else None
    start_acc = (start_profile.accession_number or None) if start_profile else None
    end_acc = (end_profile.accession_number or None) if end_profile else None
    start_url = (start_profile.sec_filing_url or None) if start_profile else None
    end_url = (end_profile.sec_filing_url or None) if end_profile else None
    start_concept = _xbrl_concept_for_metric(start_profile, metric_name) if start_profile else None
    end_concept = _xbrl_concept_for_metric(end_profile, metric_name) if end_profile else None

    derived_from: list[NumericCitation] = []
    if start_profile is not None and start_val is not None:
        derived_from.append(_numeric_citation_for_profile(start_profile, metric_name, start_val))
    if end_profile is not None and end_val is not None and end_profile is not start_profile:
        derived_from.append(_numeric_citation_for_profile(end_profile, metric_name, end_val))

    provenance_kwargs = {
        "start_period_of_report": start_period,
        "end_period_of_report": end_period,
        "start_accession_number": start_acc,
        "end_accession_number": end_acc,
        "start_sec_filing_url": start_url,
        "end_sec_filing_url": end_url,
        "start_xbrl_concept": start_concept,
        "end_xbrl_concept": end_concept,
        "derived_from": derived_from,
    }

    if start_val is None or end_val is None:
        return MetricVariance(
            metric_name=metric_name,
            start_year=start_year,
            end_year=end_year,
            start_value=start_val,
            end_value=end_val,
            direction="unavailable",
            **provenance_kwargs,
        )

    abs_delta = round(end_val - start_val, 2)
    pct_delta = pct_change(start_val, end_val)
    years_span = max(0, end_year - start_year)
    cagr_pct: float | None = None
    if years_span >= 1 and start_val > 0 and end_val > 0:
        cagr_pct = round((((end_val / start_val) ** (1.0 / years_span)) - 1.0) * 100.0, 2)

    direction: Literal["increase", "decrease", "flat"]
    if abs_delta > 0:
        direction = "increase"
    elif abs_delta < 0:
        direction = "decrease"
    else:
        direction = "flat"

    return MetricVariance(
        metric_name=metric_name,
        start_year=start_year,
        end_year=end_year,
        start_value=start_val,
        end_value=end_val,
        absolute_delta=abs_delta,
        percentage_delta=pct_delta,
        cagr_pct=cagr_pct,
        direction=direction,
        **provenance_kwargs,
    )


def _build_annual_step(
    curr: FinancialDataProfile,
    prev: FinancialDataProfile | None,
    prov_dict: dict[str, str | None],
) -> AnnualStepVariance:
    """Build a single year-over-year `AnnualStepVariance` row across all core metrics."""
    metric_fields: dict[str, float | None] = {}
    for metric in CORE_METRICS:
        curr_val = getattr(curr, metric)
        metric_fields[metric] = curr_val
        yoy_delta: float | None = None
        yoy_pct: float | None = None
        if prev is not None:
            prev_val = getattr(prev, metric)
            if curr_val is not None and prev_val is not None:
                yoy_delta = round(curr_val - prev_val, 2)
                yoy_pct = pct_change(prev_val, curr_val)
        metric_fields[f"{metric}_yoy_delta"] = yoy_delta
        metric_fields[f"{metric}_yoy_pct"] = yoy_pct

    return AnnualStepVariance(
        fiscal_year=curr.fiscal_year,
        prev_fiscal_year=prev.fiscal_year if prev else None,
        period_of_report=curr.period_of_report or None,
        filing_date=curr.filing_date or None,
        accession_number=curr.accession_number or None,
        prior_period_of_report=(prev.period_of_report or None) if prev else None,
        prior_accession_number=(prev.accession_number or None) if prev else None,
        prior_sec_filing_url=(prev.sec_filing_url or None) if prev else None,
        metric_provenance=prov_dict,
        sec_filing_url=curr.sec_filing_url,
        **metric_fields,
    )


def _build_variance_reports_from_profiles(
    profiles: list[FinancialDataProfile],
    *,
    ticker_filter: set[str],
    start_year: int,
    end_year: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compute deterministic `CompanyVarianceReport` dicts with full XBRL & fiscal-period provenance."""
    norm_filter = {normalize_edgar_ticker(t) for t in ticker_filter}
    by_ticker: dict[str, dict[int, FinancialDataProfile]] = defaultdict(dict)
    for p in profiles:
        if normalize_edgar_ticker(p.ticker) in norm_filter:
            by_ticker[p.ticker][p.fiscal_year] = p

    reports: list[dict[str, Any]] = []
    found_norm_tickers: set[str] = set()
    requested_years = list(range(int(start_year), int(end_year) + 1))

    for ticker, fy_map in sorted(by_ticker.items()):
        avail_years = sorted(fy for fy in fy_map if start_year <= fy <= end_year)
        if not avail_years:
            continue
        found_norm_tickers.add(normalize_edgar_ticker(ticker))
        missing_years = [fy for fy in requested_years if fy not in avail_years]

        eff_start, eff_end = avail_years[0], avail_years[-1]
        start_prof, end_prof = fy_map[eff_start], fy_map[eff_end]

        variances: dict[MetricName, MetricVariance] = {
            metric: _compute_single_metric_variance(
                metric,
                eff_start,
                eff_end,
                getattr(start_prof, metric),
                getattr(end_prof, metric),
                start_profile=start_prof,
                end_profile=end_prof,
            )
            for metric in CORE_METRICS
        }

        start_rec = _profile_to_metric_record(start_prof)
        end_rec = _profile_to_metric_record(end_prof)

        ratio_bps: dict[str, float | None] = {}
        derived_margin_inputs: dict[str, list[NumericCitation]] = {}
        for ratio_field, num_metric, bps_field in RATIO_METRICS:
            bps = margin_bps_delta(getattr(start_rec, ratio_field), getattr(end_rec, ratio_field))
            ratio_bps[bps_field] = bps
            if bps is not None:
                derived_margin_inputs[bps_field] = [
                    _numeric_citation_for_profile(
                        start_prof, num_metric, getattr(start_prof, num_metric)
                    ),
                    _numeric_citation_for_profile(start_prof, "revenue", start_prof.revenue),
                    _numeric_citation_for_profile(
                        end_prof, num_metric, getattr(end_prof, num_metric)
                    ),
                    _numeric_citation_for_profile(end_prof, "revenue", end_prof.revenue),
                ]

        annual_series: list[AnnualStepVariance] = []
        source_urls: dict[int, str] = {}
        period_end_dates: dict[int, str] = {}
        filing_dates: dict[int, str] = {}
        accession_numbers: dict[int, str] = {}
        metric_provenance_by_year: dict[int, dict[str, str | None]] = {}

        for fy in avail_years:
            curr = fy_map[fy]
            source_urls[fy] = curr.sec_filing_url
            if curr.period_of_report:
                period_end_dates[fy] = curr.period_of_report
            if curr.filing_date:
                filing_dates[fy] = curr.filing_date
            if curr.accession_number:
                accession_numbers[fy] = curr.accession_number
            prov_dict = {metric: _xbrl_concept_for_metric(curr, metric) for metric in CORE_METRICS}
            metric_provenance_by_year[fy] = prov_dict
            annual_series.append(_build_annual_step(curr, fy_map.get(fy - 1), prov_dict))

        report = CompanyVarianceReport(
            ticker=ticker,
            company_name=end_prof.company_name,
            cik=end_prof.cik,
            gics_sector=end_prof.gics_sector,
            gics_sub_industry=end_prof.gics_sub_industry,
            start_year=eff_start,
            end_year=eff_end,
            requested_start_year=int(start_year),
            requested_end_year=int(end_year),
            available_fiscal_years=avail_years,
            missing_fiscal_years=missing_years,
            currency=end_prof.currency,
            revenue_variance=variances["revenue"],
            operating_income_variance=variances["operating_income"],
            net_income_variance=variances["net_income"],
            research_and_development_variance=variances["research_and_development"],
            operating_margin_bps_delta=ratio_bps["operating_margin_bps_delta"],
            net_margin_bps_delta=ratio_bps["net_margin_bps_delta"],
            rd_intensity_bps_delta=ratio_bps["rd_intensity_bps_delta"],
            annual_series=annual_series,
            source_urls=source_urls,
            period_end_dates=period_end_dates,
            filing_dates=filing_dates,
            accession_numbers=accession_numbers,
            metric_provenance=metric_provenance_by_year,
            derived_margin_inputs=derived_margin_inputs,
        )
        reports.append(report.model_dump(mode="json"))

    missing_tickers = sorted(t for t in norm_filter if t not in found_norm_tickers)
    return reports, missing_tickers
