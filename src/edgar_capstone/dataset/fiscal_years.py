"""Fiscal-year resolution and multi-year DEI offset reconciliation for SEC 10-K filings."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


def resolve_fiscal_year(
    *,
    dei_fiscal_year: Any,
    report_date_str: str,
) -> int:
    """Determine the canonical fiscal year for a 10-K filing.

    Validates XBRL `dei:DocumentFiscalYearFocus` (`entity_info['fiscal_year']`) against
    `report_date_str` to guard against filer DEI copy-paste typos (e.g., a 2022-12-31
    10-K tagged with DEI fiscal_year=2021):
      - Period end in June-December (`month >= 6`): fiscal year must equal `report_date.year`.
      - Period end in January-May (`month <= 5`): fiscal year must be in
        `{report_date.year - 1, report_date.year}` (accommodating both NVDA Jan 2026 -> FY2026
        and HD Feb 2026 -> FY2025 conventions).
    """
    parsed_date = date.fromisoformat(str(report_date_str).strip()[:10])
    # 52/53-week calendar-year companies whose fiscal year spills a few days into early Jan
    is_jan_spillover = parsed_date.month == 1 and parsed_date.day <= 10
    date_fallback_fy = parsed_date.year - 1 if is_jan_spillover else parsed_date.year

    if is_jan_spillover:
        # A fiscal year ending Jan 1-10 is always the prior calendar year's 52/53-week year
        # (e.g., DPZ 2022-01-02 = FY2021). Some filers tag DEI with the calendar year of the
        # period end, which previously shifted labels forward one year and caused FY
        # collisions that silently dropped a real 10-K (DPZ FY2022 lost, FY2021 gap).
        if dei_fiscal_year is not None and str(dei_fiscal_year).strip() != str(date_fallback_fy):
            logger.warning(
                "Ignoring DEI fiscal_year=%s for Jan 1-10 52/53-week period_of_report=%s; using %d",
                dei_fiscal_year,
                report_date_str,
                date_fallback_fy,
            )
        return date_fallback_fy

    if dei_fiscal_year is not None:
        try:
            fy = int(str(dei_fiscal_year).strip())
            if 1990 <= fy <= 2100:
                if parsed_date.month >= 6:
                    if fy == parsed_date.year:
                        return fy
                    logger.warning(
                        "Ignoring implausible DEI fiscal_year=%d for period_of_report=%s; using %d",
                        fy,
                        report_date_str,
                        parsed_date.year,
                    )
                    return parsed_date.year
                if fy in (parsed_date.year - 1, parsed_date.year):
                    return fy
                logger.warning(
                    "Ignoring out-of-range DEI fiscal_year=%d for period_of_report=%s; using %d",
                    fy,
                    report_date_str,
                    date_fallback_fy,
                )
        except ValueError:
            pass

    return date_fallback_fy


def reconcile_company_fiscal_years(
    report_dates_and_fys: list[tuple[str, int]],
) -> list[int]:
    """Normalize Jan-May fiscal year offsets across a company's multi-year 10-K series.

    Some Jan-May filers (e.g., CrowdStrike Jan 31) have DEI `DocumentFiscalYearFocus`
    tagged as `report_year` on most filings (2026-01-31 -> FY2026) but `report_year - 1`
    on a subset (2025-01-31 -> DEI 2024), creating artificial collisions and year gaps.
    When all filings share a Jan-May fiscal year-end (`month <= 5`, `day > 10`) and
    exhibit mixed `{0, -1}` offsets, apply the majority offset (breaking ties with the
    most recent filing).
    """
    if len(report_dates_and_fys) < 2:
        return [fy for _, fy in report_dates_and_fys]

    parsed = [(date.fromisoformat(rd.strip()[:10]), fy) for rd, fy in report_dates_and_fys]
    # Only normalize when the entire series consists of non-spillover Jan-May year-ends
    if not all(d.month <= 5 and not (d.month == 1 and d.day <= 10) for d, _ in parsed):
        return [fy for _, fy in report_dates_and_fys]

    months = {d.month for d, _ in parsed}
    if max(months) - min(months) > 1:
        # Fiscal year-end month changed materially over time; keep individual resolutions
        return [fy for _, fy in report_dates_and_fys]

    offsets = [fy - d.year for d, fy in parsed]
    distinct_offsets = set(offsets)
    if len(distinct_offsets) <= 1 or not distinct_offsets.issubset({-1, 0}):
        return [fy for _, fy in report_dates_and_fys]

    counts = Counter(offsets)
    # Newest filing is first in SEC EDGAR iteration order; sort by date desc to be explicit
    newest_offset = max(zip(parsed, offsets, strict=False), key=lambda item: item[0][0])[1]
    canonical_offset = (
        0 if (counts[0] > counts[-1] or (counts[0] == counts[-1] and newest_offset == 0)) else -1
    )
    return [d.year + canonical_offset for d, _ in parsed]


__all__ = [
    "reconcile_company_fiscal_years",
    "resolve_fiscal_year",
]
