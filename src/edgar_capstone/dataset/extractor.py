"""SEC EDGAR 10-K unstructured text (Item 1A & Item 7) and XBRL financials extractor."""

from __future__ import annotations

from datetime import date
import logging
import math
import re
from typing import Any

import edgar
from edgar import CompanyNotFoundError

from edgar_capstone.dataset.constituents import format_cik
from edgar_capstone.dataset.models import (
    CompanyConstituent,
    ExtractedFilingBundle,
    FinancialDataProfile,
    MetricProvenance,
)

logger = logging.getLogger(__name__)

DEFAULT_SEC_USER_AGENT = (
    "FDE SEC Edgar Capstone sec-analyst@fde-sec-edgar-sandbox-dev.iam.gserviceaccount.com"
)

# Ordered US-GAAP fallback concepts when high-level helpers return None
# (e.g., banking, insurance, energy, or utility single-step income statements).
REVENUE_FALLBACK_CONCEPTS: tuple[str, ...] = (
    "us-gaap_Revenues",
    "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax",
    "us-gaap_SalesRevenueNet",
    "us-gaap_RevenuesNetOfInterestExpense",
    "us-gaap_TotalRevenuesAndOtherIncome",
    "us-gaap_RegulatedAndUnregulatedOperatingRevenue",
    "us-gaap_HealthCareOrganizationRevenue",
)

OPERATING_INCOME_FALLBACK_CONCEPTS: tuple[str, ...] = (
    "us-gaap_OperatingIncomeLoss",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
)

NET_INCOME_FALLBACK_CONCEPTS: tuple[str, ...] = (
    "us-gaap_NetIncomeLoss",
    "us-gaap_ProfitLoss",
    "us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic",
)

_TOC_TAG_RE = re.compile(
    r"(?im)^\s*(?:<u>\s*Table of Contents\s*</u>|\[Table of Contents\]\([^)]*\)|Table of Contents)\s*$"
)
_LEADING_ITEM_HEADER_RE = re.compile(
    r"^\s*(?:#{1,4}\s*|\*\*)?Item\s+(?:1A|7)\b[^\n]*?(?:\*\*)?\s*\n+",
    re.IGNORECASE,
)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def configure_sec_identity(user_agent: str = DEFAULT_SEC_USER_AGENT) -> None:
    """Configure the mandatory SEC EDGAR User-Agent header."""
    cleaned = user_agent.strip()
    if "@" not in cleaned or " " not in cleaned:
        raise ValueError(
            "SEC EDGAR requires a User-Agent in the format 'Organization Name email@domain.com'. "
            f"Got: {user_agent!r}"
        )
    edgar.set_identity(cleaned)


def _coerce_finite_float(value: Any) -> float | None:
    """Convert numeric values to finite Python float, filtering NaN/Inf/None."""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def resolve_fiscal_year(
    *,
    dei_fiscal_year: Any,
    report_date_str: str,
) -> int:
    """Determine the canonical fiscal year for a 10-K filing.

    Prefers the authoritative XBRL `dei:DocumentFiscalYearFocus` (`entity_info['fiscal_year']`),
    which handles January/February retail & tech fiscal year conventions accurately
    (e.g., NVDA Jan 25, 2026 -> FY2026; HD Feb 1, 2026 -> FY2025).
    Falls back to `report_date` if DEI metadata is unavailable.
    """
    if dei_fiscal_year is not None:
        try:
            fy = int(str(dei_fiscal_year).strip())
            if 1990 <= fy <= 2100:
                return fy
        except ValueError:
            pass

    parsed_date = date.fromisoformat(str(report_date_str).strip()[:10])
    # 52/53-week calendar-year companies whose fiscal year spills a few days into Jan
    if parsed_date.month == 1 and parsed_date.day <= 10:
        return parsed_date.year - 1
    return parsed_date.year


def clean_markdown_section(raw_text: str) -> str:
    """Normalize extracted 10-K section Markdown for RAG chunking and source viewing."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TOC_TAG_RE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = text.strip()
    # Strip redundant leading "Item 1A. Risk Factors" / "Item 7. MD&A" heading if present
    # since we render standardized H2 section headers in the combined document.
    text = _LEADING_ITEM_HEADER_RE.sub("", text, count=1).strip()
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text


def _extract_single_section(
    tenk: Any,
    *,
    section_keys: tuple[str, ...],
    fallback_attr: str,
    item_key: str,
) -> str | None:
    """Extract a 10-K section as Markdown (preserving tables), with plain-text fallbacks."""
    sections = getattr(tenk, "sections", None)
    if sections:
        for key in section_keys:
            sec_obj = sections.get(key)
            if sec_obj is None:
                continue
            md_attr = getattr(sec_obj, "markdown", None)
            raw = md_attr() if callable(md_attr) else md_attr
            if isinstance(raw, str):
                cleaned = clean_markdown_section(raw)
                if len(cleaned) >= 150:
                    return cleaned

    # Fallback 1: direct TenK property (e.g., tenk.risk_factors / tenk.management_discussion)
    prop_val = getattr(tenk, fallback_attr, None)
    if isinstance(prop_val, str):
        cleaned = clean_markdown_section(prop_val)
        if len(cleaned) >= 100:
            return cleaned

    # Fallback 2: item indexing (e.g., tenk["Item 1A"])
    try:
        indexed_val = tenk[item_key]
        if isinstance(indexed_val, str):
            cleaned = clean_markdown_section(indexed_val)
            if cleaned:
                return cleaned
    except Exception:
        pass

    return None


def format_combined_10k_markdown(
    *,
    ticker: str,
    company_name: str,
    cik: str,
    gics_sector: str,
    gics_sub_industry: str,
    fiscal_year: int,
    period_of_report: str,
    filing_date: str,
    accession_number: str,
    sec_filing_url: str,
    item_1a_markdown: str | None,
    item_7_markdown: str | None,
) -> str:
    """Compose a unified, citation-friendly Markdown document for Item 1A and Item 7."""
    safe_company_yaml = company_name.replace('"', '\\"')
    safe_sector_yaml = gics_sector.replace('"', '\\"')
    safe_sub_industry_yaml = gics_sub_industry.replace('"', '\\"')

    parts: list[str] = [
        "---",
        f'ticker: "{ticker}"',
        f'company_name: "{safe_company_yaml}"',
        f'cik: "{cik}"',
        f'gics_sector: "{safe_sector_yaml}"',
        f'gics_sub_industry: "{safe_sub_industry_yaml}"',
        'form_type: "10-K"',
        f"fiscal_year: {fiscal_year}",
        f'period_of_report: "{period_of_report}"',
        f'filing_date: "{filing_date}"',
        f'accession_number: "{accession_number}"',
        f'sec_filing_url: "{sec_filing_url}"',
        "---",
        "",
        f"# {company_name} ({ticker}) — FY{fiscal_year} Form 10-K",
        "",
        f"- **CIK:** `{cik}`",
        f"- **GICS Sector:** {gics_sector} ({gics_sub_industry})",
        f"- **Fiscal Year:** {fiscal_year} (Period Ended {period_of_report})",
        f"- **Filing Date:** {filing_date}",
        f"- **Accession Number:** `{accession_number}`",
        f"- **Source Filing:** {sec_filing_url}",
        "",
        "---",
        "",
        "## Item 1A. Risk Factors",
        "",
        (
            item_1a_markdown
            if item_1a_markdown
            else "*Item 1A (Risk Factors) could not be extracted or was omitted in this filing.*"
        ),
        "",
        "---",
        "",
        "## Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations",
        "",
        (
            item_7_markdown
            if item_7_markdown
            else "*Item 7 (MD&A) could not be extracted or was incorporated by reference in this filing.*"
        ),
        "",
    ]
    return "\n".join(parts)


def _lookup_concept_from_income_df(
    inc_df: Any,
    concepts: tuple[str, ...],
) -> tuple[float | None, str | None]:
    """Search non-dimensional income statement DataFrame rows for the first matching concept."""
    if inc_df is None or getattr(inc_df, "empty", True):
        return None, None

    df = inc_df
    if "dimension" in df.columns:
        df = df[df["dimension"] == False]  # noqa: E712
    if "is_breakdown" in df.columns:
        df = df[df["is_breakdown"] == False]  # noqa: E712
    if "abstract" in df.columns:
        df = df[df["abstract"] == False]  # noqa: E712

    if df.empty or "concept" not in df.columns:
        return None, None

    # Identify primary annual value column (e.g., "2025-12-31 (FY)" or first YYYY-MM-DD column)
    fy_cols = [c for c in df.columns if "(FY)" in str(c)]
    if not fy_cols:
        fy_cols = [c for c in df.columns if re.match(r"^\d{4}-\d{2}-\d{2}", str(c))]
    if not fy_cols:
        return None, None
    primary_col = fy_cols[0]

    for target_concept in concepts:
        matched = df[df["concept"] == target_concept]
        if matched.empty:
            continue
        for raw_val in matched[primary_col].tolist():
            val = _coerce_finite_float(raw_val)
            if val is not None:
                canonical_tag = target_concept.replace("_", ":", 1)
                return val, canonical_tag

    return None, None


def extract_financial_metrics(
    tenk: Any,
) -> tuple[float | None, float | None, float | None, MetricProvenance]:
    """Extract Revenue, Operating Income, and Net Income (in USD) with XBRL provenance."""
    provenance = MetricProvenance()
    fin = getattr(tenk, "financials", None)
    if fin is None:
        provenance.notes.append("No XBRL financials object available on filing.")
        return None, None, None, provenance

    revenue = _coerce_finite_float(fin.get_revenue())
    if revenue is not None:
        provenance.revenue = "edgartools:Financials.get_revenue"

    operating_income = _coerce_finite_float(fin.get_operating_income())
    if operating_income is not None:
        provenance.operating_income = (
            "edgartools:Financials.get_operating_income (us-gaap:OperatingIncomeLoss)"
        )

    net_income = _coerce_finite_float(fin.get_net_income())
    if net_income is not None:
        provenance.net_income = (
            "edgartools:Financials.get_net_income (us-gaap:NetIncomeLoss)"
        )

    # Fallback inspection of the primary Income Statement DataFrame if any metric is None
    if revenue is None or operating_income is None or net_income is None:
        inc_df = None
        try:
            inc_stmt = fin.income_statement()
            if inc_stmt is not None:
                inc_df = inc_stmt.to_dataframe()
        except Exception as exc:
            logger.debug("Income statement DataFrame fallback unavailable: %s", exc)

        if inc_df is not None:
            if revenue is None:
                rev_val, rev_tag = _lookup_concept_from_income_df(
                    inc_df, REVENUE_FALLBACK_CONCEPTS
                )
                if rev_val is not None:
                    revenue = rev_val
                    provenance.revenue = rev_tag

            if operating_income is None:
                op_val, op_tag = _lookup_concept_from_income_df(
                    inc_df, OPERATING_INCOME_FALLBACK_CONCEPTS
                )
                if op_val is not None:
                    operating_income = op_val
                    provenance.operating_income = op_tag
                    if op_tag and "OperatingIncomeLoss" not in op_tag:
                        provenance.notes.append(
                            f"Filer does not report standard GAAP OperatingIncomeLoss; "
                            f"populated operating_income from {op_tag} (pre-tax continuing operations income)."
                        )

            if net_income is None:
                ni_val, ni_tag = _lookup_concept_from_income_df(
                    inc_df, NET_INCOME_FALLBACK_CONCEPTS
                )
                if ni_val is not None:
                    net_income = ni_val
                    provenance.net_income = ni_tag

    return revenue, operating_income, net_income, provenance


def _resolve_edgar_company(constituent: CompanyConstituent) -> Any:
    """Resolve an `edgar.Company` by ticker first, falling back to CIK if needed."""
    try:
        company = edgar.Company(constituent.edgar_ticker)
        if getattr(company, "not_found", False):
            raise CompanyNotFoundError(f"Ticker {constituent.edgar_ticker} not found")
        return company
    except Exception:
        if constituent.cik:
            return edgar.Company(int(constituent.cik))
        raise


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
    official_name = (
        getattr(company, "name", None) or constituent.company_name or constituent.ticker
    )

    filings = company.get_filings(form="10-K", amendments=False).filter(
        filing_date=f"{start_year}-01-01:"
    )

    bundles_by_fy: dict[int, ExtractedFilingBundle] = {}

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
        if report_date_raw < f"{start_year}-01-11":
            continue
        if report_date_raw > f"{end_year + 1}-06-30":
            continue

        # Fast resume: reuse locally staged bundle if accession matches
        if existing_accessions and accession_no in existing_accessions:
            cached_bundle = existing_accessions[accession_no]
            fy_cached = cached_bundle.profile.fiscal_year
            if start_year <= fy_cached <= end_year and fy_cached not in bundles_by_fy:
                logger.info(
                    "[%s] Reusing locally staged FY%d 10-K (%s)",
                    constituent.ticker,
                    fy_cached,
                    accession_no,
                )
                bundles_by_fy[fy_cached] = cached_bundle
            continue

        try:
            tenk = filing.obj()
        except Exception as exc:
            msg = f"[{constituent.ticker}] Failed to parse 10-K {accession_no}: {exc}"
            logger.warning(msg)
            warnings.append(msg)
            continue

        fin = getattr(tenk, "financials", None)
        xb = getattr(fin, "xb", None) if fin is not None else None
        entity_info: dict[str, Any] = (
            getattr(xb, "entity_info", None) or {} if xb is not None else {}
        )

        fiscal_year = resolve_fiscal_year(
            dei_fiscal_year=entity_info.get("fiscal_year"),
            report_date_str=report_date_raw,
        )
        if fiscal_year < start_year or fiscal_year > end_year:
            continue

        # Filings are iterated newest-first; keep the most recent 10-K per fiscal year
        if fiscal_year in bundles_by_fy:
            continue

        sec_url = (
            getattr(filing, "filing_url", None)
            or getattr(filing, "homepage_url", None)
            or f"https://www.sec.gov/Archives/edgar/data/{int(cik_10)}/{accession_no.replace('-', '')}/{accession_no}-index.htm"
        )

        item_1a_md = _extract_single_section(
            tenk,
            section_keys=("part_i_item_1a", "item_1a"),
            fallback_attr="risk_factors",
            item_key="Item 1A",
        )
        item_7_md = _extract_single_section(
            tenk,
            section_keys=("part_ii_item_7", "item_7"),
            fallback_attr="management_discussion",
            item_key="Item 7",
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

        revenue, operating_income, net_income, provenance = extract_financial_metrics(tenk)
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

        unstructured_blob = f"unstructured/{constituent.ticker}/{fiscal_year}_10k.md"
        structured_blob = (
            f"structured/{constituent.ticker}/{fiscal_year}_financials.json"
        )

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
            metric_provenance=provenance,
            extracted_sections=extracted_sections,
            unstructured_blob_path=unstructured_blob,
            structured_blob_path=structured_blob,
            unstructured_gcs_uri=f"gs://{bucket_name}/{unstructured_blob}",
            structured_gcs_uri=f"gs://{bucket_name}/{structured_blob}",
            sec_filing_url=sec_url,
        )

        bundles_by_fy[fiscal_year] = ExtractedFilingBundle(
            profile=profile,
            markdown_content=markdown_doc,
        )
        logger.info(
            "[%s] Extracted FY%d 10-K (Rev=%s, OpInc=%s, NetInc=%s, MD=%d chars)",
            constituent.ticker,
            fiscal_year,
            f"${revenue:,.0f}" if revenue is not None else "None",
            f"${operating_income:,.0f}" if operating_income is not None else "None",
            f"${net_income:,.0f}" if net_income is not None else "None",
            len(markdown_doc),
        )

    ordered_bundles = [bundles_by_fy[fy] for fy in sorted(bundles_by_fy.keys())]
    return ordered_bundles, warnings
