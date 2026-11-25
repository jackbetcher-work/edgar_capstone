"""SEC EDGAR 10-K unstructured text (Item 1A & Item 7) and XBRL financials extractor.

Delegates fiscal-year resolution, Markdown section parsing, and XBRL financial
extraction to cohesive sibling modules (`fiscal_years`, `section_parser`,
`financial_extractor`) while re-exporting all public symbols for backward compatibility.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import date
from typing import Any, TypeVar

import edgar
import httpx
from edgar import CompanyNotFoundError

from edgar_capstone.common.domain import format_cik
from edgar_capstone.dataset.financial_extractor import (
    RESEARCH_AND_DEVELOPMENT_CONCEPTS,
    IncomeStatementView,
    _filter_non_dimensional_income_df,
    _load_income_statement_df,
    extract_financial_metrics,
    extract_research_and_development_expense,
)
from edgar_capstone.dataset.fiscal_years import (
    reconcile_company_fiscal_years,
    resolve_fiscal_year,
)
from edgar_capstone.dataset.models import (
    CompanyConstituent,
    ExtractedFilingBundle,
    FinancialDataProfile,
)
from edgar_capstone.dataset.section_parser import (
    MIN_SUBSTANTIVE_SECTION_CHARS,
    _extract_single_section,
    clean_markdown_section,
    format_combined_10k_markdown,
    is_substantive_section_markdown,
    rewrite_markdown_fiscal_year,
    truncate_at_section_boundary,
)
from edgar_capstone.gcp.defaults import DEFAULT_SEC_USER_AGENT

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Marker embedded in per-company warnings when a 10-K could not be downloaded/parsed, so the
# pipeline can avoid pruning previously staged files for that company on transient failures.
PARSE_FAILURE_MARKER = "Failed to parse 10-K"

# Only retry genuine transient HTTP / OS / socket / timeout failures; let programming errors
# (ValueError, TypeError, KeyError, AttributeError) propagate immediately.
_TRANSIENT_SEC_ERRORS: tuple[type[BaseException], ...] = (httpx.HTTPError, OSError)


def _retry_sec_call(
    fn: Callable[[], _T],
    *,
    description: str,
    max_attempts: int = 3,
    base_delay: float = 1.5,
) -> _T:
    """Retry transient SEC EDGAR HTTP / socket operations with exponential backoff."""
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except CompanyNotFoundError:
            raise
        except _TRANSIENT_SEC_ERRORS as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Transient error during %s (attempt %d/%d: %s); retrying in %.1fs...",
                description,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def configure_sec_identity(user_agent: str = DEFAULT_SEC_USER_AGENT) -> None:
    """Configure the mandatory SEC EDGAR User-Agent header."""
    cleaned = user_agent.strip()
    if "@" not in cleaned or " " not in cleaned:
        raise ValueError(
            "SEC EDGAR requires a User-Agent in the format 'Organization Name email@domain.com'. "
            f"Got: {user_agent!r}"
        )
    edgar.set_identity(cleaned)


def _resolve_edgar_company(constituent: CompanyConstituent) -> Any:
    """Resolve an `edgar.Company` by ticker first, falling back to CIK if needed."""
    try:
        company = _retry_sec_call(
            lambda: edgar.Company(constituent.edgar_ticker),
            description=f"resolve ticker {constituent.edgar_ticker}",
        )
        if getattr(company, "not_found", False):
            raise CompanyNotFoundError(f"Ticker {constituent.edgar_ticker} not found")
        return company
    except Exception:
        if constituent.cik:
            return _retry_sec_call(
                lambda: edgar.Company(int(constituent.cik)),
                description=f"resolve CIK {constituent.cik} ({constituent.ticker})",
            )
        raise


def _stage_company_filings(
    constituent: CompanyConstituent,
    filings: Any,
    *,
    start_year: int,
    end_year: int,
    existing_accessions: dict[str, ExtractedFilingBundle] | None,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Collect candidate 10-K filings (reusing valid cached bundles or parsing `.obj()`)."""
    staged_entries: list[dict[str, Any]] = []

    for filing in filings:
        # Skip transition reports like 10-KT
        if str(getattr(filing, "form", "")).strip().upper() != "10-K":
            continue

        report_date_raw = str(getattr(filing, "report_date", "") or "").strip()
        filing_date_raw = str(getattr(filing, "filing_date", "") or "").strip()
        accession_no = str(getattr(filing, "accession_no", "") or "").strip()

        if not report_date_raw:
            continue

        # Fast pre-filter before downloading full HTML/XBRL:
        # Any report_date prior to Jan 11 of `start_year` belongs to `< start_year`
        # (e.g., Dec 31, 2019 10-Ks filed in Feb 2020).
        if report_date_raw < f"{start_year}-01-11" or report_date_raw > f"{end_year + 1}-06-30":
            continue

        if existing_accessions and accession_no in existing_accessions:
            cached_bundle = existing_accessions[accession_no]
            raw_fy = resolve_fiscal_year(
                dei_fiscal_year=cached_bundle.profile.fiscal_year,
                report_date_str=report_date_raw,
            )
            staged_entries.append(
                {
                    "kind": "cached",
                    "report_date": report_date_raw,
                    "filing_date": filing_date_raw,
                    "accession_no": accession_no,
                    "raw_fy": raw_fy,
                    "bundle": cached_bundle,
                }
            )
            continue

        try:
            tenk = _retry_sec_call(
                filing.obj,
                description=f"parse 10-K {constituent.ticker} {accession_no}",
            )
        except Exception as exc:
            msg = f"[{constituent.ticker}] {PARSE_FAILURE_MARKER} {accession_no}: {exc}"
            logger.warning(msg)
            warnings.append(msg)
            continue

        fin = getattr(tenk, "financials", None)
        xb = getattr(fin, "xb", None) if fin is not None else None
        entity_info: dict[str, Any] = (
            getattr(xb, "entity_info", None) or {} if xb is not None else {}
        )

        raw_fy = resolve_fiscal_year(
            dei_fiscal_year=entity_info.get("fiscal_year"),
            report_date_str=report_date_raw,
        )
        staged_entries.append(
            {
                "kind": "parsed",
                "filing": filing,
                "tenk": tenk,
                "report_date": report_date_raw,
                "filing_date": filing_date_raw,
                "accession_no": accession_no,
                "raw_fy": raw_fy,
            }
        )

    return staged_entries


def _reconcile_cached_bundle(
    cached_bundle: ExtractedFilingBundle,
    *,
    ticker: str,
    bucket_name: str,
    fiscal_year: int,
) -> ExtractedFilingBundle:
    """Update cached bundle metadata and Markdown headings when FY reconciliation shifted the year."""
    if cached_bundle.profile.fiscal_year == fiscal_year:
        return cached_bundle

    prof = cached_bundle.profile.model_copy(
        update={
            "profile_id": f"{ticker}_{fiscal_year}",
            "fiscal_year": fiscal_year,
            "unstructured_blob_path": f"unstructured/{ticker}/{fiscal_year}_10k.md",
            "structured_blob_path": f"structured/{ticker}/{fiscal_year}_financials.json",
            "unstructured_gcs_uri": f"gs://{bucket_name}/unstructured/{ticker}/{fiscal_year}_10k.md",
            "structured_gcs_uri": f"gs://{bucket_name}/structured/{ticker}/{fiscal_year}_financials.json",
        }
    )
    updated_md = rewrite_markdown_fiscal_year(
        cached_bundle.markdown_content, fiscal_year=fiscal_year
    )
    return ExtractedFilingBundle(profile=prof, markdown_content=updated_md)


def _build_parsed_bundle(
    entry: dict[str, Any],
    *,
    constituent: CompanyConstituent,
    official_name: str,
    cik_10: str,
    bucket_name: str,
    fiscal_year: int,
    warnings: list[str],
) -> ExtractedFilingBundle:
    """Extract sections and XBRL metrics for one parsed 10-K entry."""
    accession_no = entry["accession_no"]
    report_date_raw = entry["report_date"]
    filing_date_raw = entry["filing_date"]
    filing = entry["filing"]
    tenk = entry["tenk"]
    sec_url = (
        getattr(filing, "filing_url", None)
        or getattr(filing, "homepage_url", None)
        or f"https://www.sec.gov/Archives/edgar/data/{int(cik_10)}/{accession_no.replace('-', '')}/{accession_no}-index.htm"
    )

    md_cache: dict[str, Any] = {}
    item_1a_md = _extract_single_section(
        tenk,
        section_keys=("part_i_item_1a", "item_1a"),
        fallback_attr="risk_factors",
        item_key="Item 1A",
        item_id="item_1a",
        filing=filing,
        markdown_cache=md_cache,
    )
    item_7_md = _extract_single_section(
        tenk,
        section_keys=("part_ii_item_7", "item_7"),
        fallback_attr="management_discussion",
        item_key="Item 7",
        item_id="item_7",
        filing=filing,
        markdown_cache=md_cache,
    )

    extracted_sections: list[str] = []
    if item_1a_md:
        extracted_sections.append("item_1a")
    else:
        w = f"[{constituent.ticker} FY{fiscal_year}] Item 1A (Risk Factors) missing or empty."
        logger.warning(w)
        warnings.append(w)

    if item_7_md:
        extracted_sections.append("item_7")
    else:
        w = f"[{constituent.ticker} FY{fiscal_year}] Item 7 (MD&A) missing or empty."
        logger.warning(w)
        warnings.append(w)

    # Load and pre-filter the income statement DataFrame ONCE per filing.
    inc_df = _load_income_statement_df(tenk)
    inc_view = IncomeStatementView.from_dataframe(inc_df, report_date_str=report_date_raw)

    revenue, operating_income, net_income, provenance = extract_financial_metrics(
        tenk,
        report_date_str=report_date_raw,
        inc_view=inc_view,
    )
    research_and_development, rd_tag = extract_research_and_development_expense(
        tenk,
        report_date_str=report_date_raw,
        inc_view=inc_view,
    )
    provenance.research_and_development = rd_tag

    missing_metrics = [
        name
        for name, val in (
            ("revenue", revenue),
            ("operating_income", operating_income),
            ("net_income", net_income),
        )
        if val is None
    ]
    if missing_metrics:
        w = (
            f"[{constituent.ticker} FY{fiscal_year}] Missing structured financial "
            f"metrics: {', '.join(missing_metrics)}"
        )
        logger.warning(w)
        warnings.append(w)

    if revenue is not None and revenue > 0:
        for name, val in (("operating_income", operating_income), ("net_income", net_income)):
            if val is not None and val > 1.5 * revenue:
                w = (
                    f"[{constituent.ticker} FY{fiscal_year}] Implausible {name}={val:,.0f} "
                    f"exceeds 1.5x revenue={revenue:,.0f} ({provenance.revenue}); "
                    "revenue concept may be a mis-tagged subtotal."
                )
                logger.warning(w)
                warnings.append(w)

    unstructured_blob = f"unstructured/{constituent.ticker}/{fiscal_year}_10k.md"
    structured_blob = f"structured/{constituent.ticker}/{fiscal_year}_financials.json"

    markdown_doc = format_combined_10k_markdown(
        ticker=constituent.ticker,
        company_name=official_name,
        cik=cik_10,
        gics_sector=constituent.gics_sector,
        gics_sub_industry=constituent.gics_sub_industry,
        fiscal_year=fiscal_year,
        period_of_report=report_date_raw,
        filing_date=filing_date_raw,
        accession_number=accession_no,
        sec_filing_url=sec_url,
        item_1a_markdown=item_1a_md,
        item_7_markdown=item_7_md,
    )

    profile = FinancialDataProfile(
        profile_id=f"{constituent.ticker}_{fiscal_year}",
        ticker=constituent.ticker,
        cik=cik_10,
        company_name=official_name,
        gics_sector=constituent.gics_sector,
        gics_sub_industry=constituent.gics_sub_industry,
        form_type="10-K",
        fiscal_year=fiscal_year,
        fiscal_period="FY",
        period_of_report=report_date_raw,
        filing_date=filing_date_raw,
        accession_number=accession_no,
        currency="USD",
        revenue=revenue,
        operating_income=operating_income,
        net_income=net_income,
        research_and_development=research_and_development,
        metric_provenance=provenance,
        extracted_sections=extracted_sections,
        unstructured_blob_path=unstructured_blob,
        structured_blob_path=structured_blob,
        unstructured_gcs_uri=f"gs://{bucket_name}/{unstructured_blob}",
        structured_gcs_uri=f"gs://{bucket_name}/{structured_blob}",
        sec_filing_url=sec_url,
    )

    logger.info(
        "[%s] Extracted FY%d 10-K (Rev=%s, OpInc=%s, NetInc=%s, R&D=%s, MD=%d chars)",
        constituent.ticker,
        fiscal_year,
        f"${revenue:,.0f}" if revenue is not None else "None",
        f"${operating_income:,.0f}" if operating_income is not None else "None",
        f"${net_income:,.0f}" if net_income is not None else "None",
        f"${research_and_development:,.0f}" if research_and_development is not None else "None",
        len(markdown_doc),
    )
    return ExtractedFilingBundle(profile=profile, markdown_content=markdown_doc)


def extract_company_10k_bundles(
    constituent: CompanyConstituent,
    *,
    bucket_name: str,
    start_year: int = 2020,
    end_year: int | None = None,
    existing_accessions: dict[str, ExtractedFilingBundle] | None = None,
) -> tuple[list[ExtractedFilingBundle], list[str]]:
    """Fetch and extract all 10-K Data Profiles for a single company across [start_year, end_year]."""
    if end_year is None:
        end_year = date.today().year

    warnings: list[str] = []
    company = _resolve_edgar_company(constituent)
    cik_10 = format_cik(getattr(company, "cik", None) or constituent.cik) or "0000000000"
    official_name = getattr(company, "name", None) or constituent.company_name or constituent.ticker

    filings = _retry_sec_call(
        lambda: company.get_filings(form="10-K", amendments=False).filter(
            filing_date=f"{start_year}-01-01:"
        ),
        description=f"fetch 10-K filings index for {constituent.ticker}",
    )

    staged_entries = _stage_company_filings(
        constituent,
        filings,
        start_year=start_year,
        end_year=end_year,
        existing_accessions=existing_accessions,
        warnings=warnings,
    )
    reconciled_fys = reconcile_company_fiscal_years(
        [(entry["report_date"], entry["raw_fy"]) for entry in staged_entries]
    )

    bundles_by_fy: dict[int, ExtractedFilingBundle] = {}
    for entry, fiscal_year in zip(staged_entries, reconciled_fys, strict=False):
        if fiscal_year < start_year or fiscal_year > end_year:
            continue
        # Filings are iterated newest-first; keep the most recent 10-K per fiscal year
        if fiscal_year in bundles_by_fy:
            kept = bundles_by_fy[fiscal_year].profile
            w = (
                f"[{constituent.ticker} FY{fiscal_year}] Fiscal-year collision: dropping 10-K "
                f"{entry['accession_no']} (period {entry['report_date']}) in favor of "
                f"{kept.accession_number} (period {kept.period_of_report})."
            )
            logger.warning(w)
            warnings.append(w)
            continue

        if entry["kind"] == "cached":
            cached_bundle = _reconcile_cached_bundle(
                entry["bundle"],
                ticker=constituent.ticker,
                bucket_name=bucket_name,
                fiscal_year=fiscal_year,
            )
            logger.info(
                "[%s] Reusing locally staged FY%d 10-K (%s)",
                constituent.ticker,
                fiscal_year,
                entry["accession_no"],
            )
            bundles_by_fy[fiscal_year] = cached_bundle
            continue

        bundles_by_fy[fiscal_year] = _build_parsed_bundle(
            entry,
            constituent=constituent,
            official_name=official_name,
            cik_10=cik_10,
            bucket_name=bucket_name,
            fiscal_year=fiscal_year,
            warnings=warnings,
        )

    ordered_bundles = [bundles_by_fy[fy] for fy in sorted(bundles_by_fy.keys())]
    return ordered_bundles, warnings


__all__ = [
    "DEFAULT_SEC_USER_AGENT",
    "MIN_SUBSTANTIVE_SECTION_CHARS",
    "PARSE_FAILURE_MARKER",
    "RESEARCH_AND_DEVELOPMENT_CONCEPTS",
    "IncomeStatementView",
    "_extract_single_section",
    "_filter_non_dimensional_income_df",
    "clean_markdown_section",
    "configure_sec_identity",
    "extract_company_10k_bundles",
    "extract_financial_metrics",
    "extract_research_and_development_expense",
    "format_combined_10k_markdown",
    "is_substantive_section_markdown",
    "reconcile_company_fiscal_years",
    "resolve_fiscal_year",
    "truncate_at_section_boundary",
]
