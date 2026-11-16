"""SEC EDGAR 10-K unstructured text (Item 1A & Item 7) and XBRL financials extractor."""

from __future__ import annotations

import logging
import math
import re
import time
import warnings as py_warnings
from collections import Counter
from collections.abc import Callable
from datetime import date
from typing import Any, Literal, TypeVar

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

_T = TypeVar("_T")

DEFAULT_SEC_USER_AGENT = (
    "FDE SEC Edgar Capstone sec-analyst-sa@fde-sec-edgar-sandbox-dev.iam.gserviceaccount.com"
)

# Consolidated top-line revenue concepts checked first on non-dimensional income statements.
CONSOLIDATED_TOTAL_REVENUE_CONCEPTS: tuple[str, ...] = (
    "us-gaap_Revenues",
    "us-gaap_SalesRevenueNet",
    "us-gaap_RevenuesNetOfInterestExpense",
    "us-gaap_TotalRevenuesAndOtherIncome",
    "us-gaap_RealEstateRevenueNet",
    "us-gaap_RegulatedAndUnregulatedOperatingRevenue",
)

# Component / standard-specific revenue concepts (ASC 606, ASC 842 REIT leases, healthcare, insurance).
# When no explicit consolidated total-revenue tag exists (e.g., Equity REITs reporting separate
# OperatingLeaseLeaseIncome and minor ASC 606 fee lines), we select the largest positive line.
COMPONENT_REVENUE_CONCEPTS: tuple[str, ...] = (
    "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax",
    "us-gaap_OperatingLeaseLeaseIncome",
    "us-gaap_LeaseIncome",
    "us-gaap_HealthCareOrganizationRevenue",
    "us-gaap_PremiumsEarnedNet",
    "us-gaap_InterestAndDividendIncomeOperating",
)

# Combined tuple preserved for backwards compatibility
REVENUE_FALLBACK_CONCEPTS: tuple[str, ...] = (
    *CONSOLIDATED_TOTAL_REVENUE_CONCEPTS,
    *COMPONENT_REVENUE_CONCEPTS,
)

OPERATING_INCOME_CONCEPTS: tuple[str, ...] = ("us-gaap_OperatingIncomeLoss",)

OPERATING_INCOME_FALLBACK_CONCEPTS: tuple[str, ...] = (
    "us-gaap_OperatingIncomeLoss",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
)

NET_INCOME_FALLBACK_CONCEPTS: tuple[str, ...] = (
    "us-gaap_NetIncomeLoss",
    "us-gaap_ProfitLoss",
    "us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic",
)

# Research & Development expense concepts, in priority order. Issuers that do not report an
# R&D line (banks, REITs, utilities, most retailers) get `None`, never 0.
RESEARCH_AND_DEVELOPMENT_CONCEPTS: tuple[str, ...] = (
    "us-gaap_ResearchAndDevelopmentExpense",
    "us-gaap_ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    "us-gaap_ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
)

_RESEARCH_AND_DEVELOPMENT_LABEL_RE = re.compile(
    r"^\s*(?:total\s+)?research(?:,)?\s+(?:and|&)\s+development(?:\s+expenses?)?\s*:?\s*$",
    re.IGNORECASE,
)

_TOTAL_REVENUE_LABEL_RE = re.compile(
    r"^\s*total\s+(?:revenues?|net\s+revenues?|operating\s+revenues?|net\s+sales|sales\s+and\s+revenues?)\s*:?\s*$",
    re.IGNORECASE,
)
# Statement top line for filers that present "Total revenues and other" (e.g., APA, EXE). Used to
# sanity-check the chosen revenue concept: APA tags `us-gaap:Revenues` on a $1.1B sub-line in FY2021
# and reports it at 2x the true total in FY2023-FY2025, while `apa_RevenuesAndOther` is reliable.
_TOTAL_REVENUES_AND_OTHER_LABEL_RE = re.compile(
    r"^\s*total\s+revenues?,?\s+and\s+other(?:\s+income)?\s*:?\s*$",
    re.IGNORECASE,
)
_REVENUE_TOP_LINE_TOLERANCE = (0.5, 1.5)
_OPERATING_INCOME_LABEL_RE = re.compile(
    r"^\s*(?:operating\s+(?:income|profit|loss|income\s*\(loss\)|loss\s*\(income\))|(?:income|loss)\s+from\s+operations)\s*:?\s*$",
    re.IGNORECASE,
)
_PRETAX_INCOME_LABEL_RE = re.compile(
    r"^\s*(?:net\s+)?(?:income|earnings|loss|\(loss\)\s*income|income\s*\(loss\)|loss\s*\(income\))\s+"
    r"(?:from\s+continuing\s+operations\s+)?before\s+"
    r"(?:provision\s+for\s+|federal\s+and\s+state\s+)?income\s+taxes"
    r"(?:\s+and\s+[^\n]{1,60})?\s*:?\s*$",
    re.IGNORECASE,
)

_TOC_TAG_RE = re.compile(
    r"(?im)^\s*(?:<u>\s*Table\s+of\s+Contents\s*</u>|\*\*Table\s+of\s+Contents\*\*|\[Table\s+of\s+Contents\]\([^)]*\)|Table\s+of\s+Contents)\s*$"
)
_RUNNING_PAGE_HEADER_RE = re.compile(
    r"(?im)^(?:"
    r"#{1,4}\s*MANAGEMENT['\u2019]?S\s+DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+FINANCIAL\s+CONDITION\s+AND\s+RESULTS\s+OF\s+OPERATIONS\s*|"
    r"\|\s*(?:Results\s+of\s+Operations|Risk\s+Factors|Forward-looking\s+Statements|MD&A)\s*\(continued\)\s*\|\s*"
    r")$"
)
_LEADING_ITEM_HEADER_RE = re.compile(
    r"^\s*(?:\|\s*---[^\n]*\n+\s*)?(?:\|\s*)?(?:#{1,4}\s*|\*\*)?(?:PART\s+I{1,2}[\s.\\—–|-]*)?Item\s+(?:1\\?\.?A|7)\\?\.?\b[^\n]*?(?:\*\*)?(?:\s*\|)?\s*(?:\n\s*\|\s*---[^\n]*)?\n+",
    re.IGNORECASE,
)
_LEADING_TABLE_RULE_RE = re.compile(r"^\s*(?:\|\s*---[^\n]*\|?\s*\n+)+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_DATE_COLUMN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")

# Headings of the NEXT 10-K item, used to cut sections that bleed past their own boundary
# (e.g., ETR/CCL "Item 7" = cross-ref stub followed by Items 7A-14; AEP/CVX MD&A running into
# audited financial statements; CSX/ED Item 1A running into Item 1B/1C/2). A boundary line must
# be a short title-style line: the item label followed by its canonical title keyword, or the
# bare label alone (e.g., "| ITEM 1B. |"). Prose cross-references such as
# "Item 8 of this report." or "Item 8. Financial Statements...—Note 4" do not qualify.
_ITEM_LABEL_PREFIX = (
    r"(?im)^[ \t]*(?:\|\s*)?(?:#{1,6}\s*|\*\*\s*)?(?:PART\s+I{1,3}[\s.,:|\\—–-]*)?ITEM\s+"
)
_ITEM_TITLE_GAP = r"[\s*|\\.:—–-]*(?:<u>\s*)?(?:\*\*\s*)?"
_SECTION_BOUNDARY_RES: dict[str, re.Pattern[str]] = {
    "item_1a": re.compile(
        _ITEM_LABEL_PREFIX
        + r"(?:"
        + r"1\\?\.?\s*B\\?\.?"
        + _ITEM_TITLE_GAP
        + r"Unresolve"
        + r"|1\\?\.?\s*C\\?\.?"
        + _ITEM_TITLE_GAP
        + r"Cybersecurity"
        + r"|2\\?\.?"
        + _ITEM_TITLE_GAP
        + r"Propert"
        + r"|3\\?\.?"
        + _ITEM_TITLE_GAP
        + r"Legal\s+Proceeding"
        + r"|(?:1\\?\.?\s*[BC]|2|3)\\?\.?[\s*|\\.:]*$"
        + r")[^\n]*$"
    ),
    "item_7": re.compile(
        _ITEM_LABEL_PREFIX
        + r"(?:"
        + r"7\\?\.?\s*A\\?\.?"
        + _ITEM_TITLE_GAP
        + r"Quantitative"
        + r"|8\\?\.?"
        + _ITEM_TITLE_GAP
        + r"Financial\s+Statements"
        + r"|9\\?\.?"
        + _ITEM_TITLE_GAP
        + r"Changes\s+in"
        + r"|9\\?\.?\s*A\\?\.?"
        + _ITEM_TITLE_GAP
        + r"Controls"
        + r"|(?:7\\?\.?\s*A|8|9A?)\\?\.?[\s*|\\.:]*$"
        + r")[^\n]*$"
    ),
}
_AUDITOR_REPORT_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:\|\s*|\*\*\s*|#{1,6}\s*)Report\s+of\s+Independent\s+Registered\s+Public\s+Accounting\s+Firm\b[^\n]*$"
)
_MAX_BOUNDARY_LINE_CHARS = 120

# Some filers place a next-item heading (or a stub for it) BEFORE the real section body, e.g.
# RF "Item 7A" heading leading a combined Item 7/7A MD&A, or SO an Item 7 page index followed by a
# one-line Item 7A stub and then the full "COMBINED MANAGEMENT'S DISCUSSION AND ANALYSIS". Cutting at
# such a heading discards the entire section, so a boundary is skipped when the text before it is
# only a fragment and the text after it (up to the next boundary) is clearly the section's content.
_SECTION_CONTENT_MARKER_RES: dict[str, tuple[re.Pattern[str], ...]] = {
    "item_7": (
        re.compile(r"(?i)\bresults\s+of\s+operations\b"),
        re.compile(r"(?i)\bliquidity\b"),
        re.compile(r"(?i)\bcritical\s+accounting\s+(?:policies|estimates)\b"),
        re.compile(r"(?i)\boverview\b"),
        re.compile(r"(?i)\bmanagement\\?['’]?s\s+discussion\s+and\s+analysis\b"),
    ),
}
_MIN_SECTION_CONTENT_MARKERS = 3

# Minimum lengths for validating extracted 10-K sections
MIN_ABSOLUTE_SECTION_CHARS = 350
MIN_SUBSTANTIVE_SECTION_CHARS = 2000

# Marker embedded in per-company warnings when a 10-K could not be downloaded/parsed, so the
# pipeline can avoid pruning previously staged files for that company on transient failures.
PARSE_FAILURE_MARKER = "Failed to parse 10-K"

_CROSS_REF_STUB_RE = re.compile(
    r"(?is)(?:"
    r"incorporated\s+(?:herein\s+)?by\s+reference|"
    r"reference\s+is\s+made\s+to\s+(?:the\s+section|page|item|exhibit)|"
    r"information\s+(?:required|called\s+for)\s+by\s+(?:this\s+)?item(?:\s+(?:1\\?\.?a|7))?\s+(?:is\s+(?:set\s+forth|shown|presented)|appears)|"
    r"(?:this\s+)?information\s+appears\s+under\s+[“\"']?(?:management|risk\s+factors)|"
    r"index\s+to\s+management['\u2019]?s\s+discussion\s+and\s+analysis|"
    r"is\s+shown\s+in\s+.*?exhibit\s+(?:13|99)|"
    r"\A[\W_]*(?:#{1,6}\s*)?refer\s+to\b|"
    r"\bon\s+pages?\s+\d+\s*(?:through|to|and|-|–|—)\s*\d+"
    r")"
)
_WRONG_ITEM_1A_RE = re.compile(
    r"^\s*(?:#{1,6}\s*|\*\*)?Item\s+1(?:\\?\.)?\s+Business\b",
    re.IGNORECASE,
)
_WRONG_ITEM_7_FOOTNOTE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*|\*\*)?(?:Note\s+)?7\\?\.[\s\xa0]+(?!Management)",
    re.IGNORECASE,
)

# Full-document Markdown fallback regexes (handling `# ITEM 1.A.`, `| ITEM 7. | ... |`, etc.)
_TOC_PAGE_NUMBER_TAIL_RE = re.compile(r"(?:\|\s*\d{1,3}\s*\|\s*$|\s+\d{1,3}\s*\|\s*$)")

_MD_ITEM_1A_START_RE = re.compile(
    r"(?im)^(?:\|\s*)?(?:#{1,4}\s*|\*\*)?(?:PART\s+I[\s.|—–-]*)?ITEM\s+1\.?A\.?\b[^\n]{0,120}?risk\s+factors[^\n]*$"
)
_MD_ITEM_1A_END_RE = re.compile(
    r"(?im)^(?:\|\s*)?(?:#{1,4}\s*|\*\*)?(?:PART\s+I{1,2}[\s.|—–-]*)?ITEM\s+(?:1\.?[BC]|2|3)\b[^\n]*$"
)
_MD_ITEM_7_START_RE = re.compile(
    r"(?im)^(?:\|\s*)?(?:#{1,4}\s*|\*\*)?(?:PART\s+II[\s.|—–-]*)?ITEM\s+7(?!\s*\.?A)\b[^\n]{0,140}?management['\u2019]?s\s+discussion[^\n]*$"
)
_MD_ITEM_7_END_RE = re.compile(
    r"(?im)^(?:\|\s*)?(?:#{1,4}\s*|\*\*)?(?:PART\s+(?:II|III)[\s.|—–-]*)?ITEM\s+(?:7\.?A|8|9)\b[^\n]*$"
)

# Annual Report / Financial Section standalone headings (e.g. XOM, CVX, BK EX-13, CLX EX-99.1)
_AR_MDA_START_RE = re.compile(
    r"(?im)^(?:\|\s*|\*\*\s*|#{1,4}\s*)?Management['\u2019]?s\s+Discussion\s+and\s+Analysis"
    r"(?:\s+of\s+Financial\s+Condition(?:\s+and(?:\s+Results\s+of\s+Operations)?)?)?[\s*:|]*$"
)
_AR_MDA_END_RE = re.compile(
    r"(?im)^(?:\|\s*|\*\*\s*|#{1,4}\s*)(?:"
    r"Report\s+of\s+Independent\s+Registered\s+Public\s+Accounting\s+Firm|"
    r"Management['\u2019]?s\s+Report\s+on\s+Internal\s+Control|"
    r"Report\s+of\s+Management\b|"
    r"Consolidated\s+Statements?\s+of\s+(?:Income|Operations|Earnings)\b|"
    r"Statements?\s+of\s+Consolidated\s+(?:Income|Operations|Earnings)\b|"
    r"Risk\s+Factors\s*(?:\|\s*)?$|"
    r"Forward-Looking\s+Statements\s*(?:\|\s*)?$"
    r")[^\n]*$"
)
_AR_RISK_START_RE = re.compile(
    r"(?im)^(?:\|\s*|\*\*\s*|#{1,4}\s*)(?:MD&A\s*[–—-]\s*)?Risk\s+Factors(?:\s*\*\*|\s*\|)?\s*$"
)
_AR_RISK_END_RE = re.compile(
    r"(?im)^(?:\|\s*|\*\*\s*|#{1,4}\s*)(?:"
    r"Forward-Looking\s+Statements\b|"
    r"Glossary\b|"
    r"Report\s+of\s+Independent\s+Registered\s+Public\s+Accounting\s+Firm|"
    r"Consolidated\s+Statements?\s+of\s+(?:Income|Operations|Earnings)\b|"
    r"Quantitative\s+and\s+Qualitative\s+Disclosures\s+About\s+Market\s+Risk"
    r")[^\n]*$"
)


def _retry_sec_call(
    fn: Callable[[], _T],
    *,
    description: str,
    max_attempts: int = 3,
    base_delay: float = 1.5,
) -> _T:
    """Retry transient SEC EDGAR HTTP / socket operations with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except CompanyNotFoundError:
            raise
        except Exception as exc:
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


def clean_markdown_section(raw_text: str) -> str:
    """Normalize extracted 10-K section Markdown for RAG chunking and source viewing."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TOC_TAG_RE.sub("", text)
    text = _RUNNING_PAGE_HEADER_RE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = text.strip()
    # Strip redundant leading "Item 1A. Risk Factors" / "Item 7. MD&A" heading if present
    # since we render standardized H2 section headers in the combined document.
    text = _LEADING_ITEM_HEADER_RE.sub("", text, count=1).strip()
    text = _LEADING_TABLE_RULE_RE.sub("", text).strip()
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text


def truncate_at_section_boundary(
    text: str,
    *,
    item_id: Literal["item_1a", "item_7"],
) -> str:
    """Cut `text` at the first heading of a subsequent 10-K item (or the auditor's report for MD&A).

    A boundary heading is skipped (not cut) when the text before it is only a fragment and the
    text after it, up to the next boundary, is substantive section content (see
    `_SECTION_CONTENT_MARKER_RES`). This preserves sections whose real body follows a misplaced or
    stub next-item heading (RF combined Item 7/7A, SO Item 7 index + Item 7A stub).
    """
    if not text:
        return text
    patterns = [_SECTION_BOUNDARY_RES[item_id]]
    if item_id == "item_7":
        patterns.append(_AUDITOR_REPORT_HEADING_RE)

    candidates: set[int] = set()
    for pattern in patterns:
        for m in pattern.finditer(text):
            line = m.group(0)
            if len(line.strip()) > _MAX_BOUNDARY_LINE_CHARS:
                continue
            if _TOC_PAGE_NUMBER_TAIL_RE.search(line) or re.search(r"(?i)\bnotes?\b", line):
                continue
            candidates.add(m.start())

    ordered = sorted(candidates)
    for idx, pos in enumerate(ordered):
        next_pos = ordered[idx + 1] if idx + 1 < len(ordered) else len(text)
        if _is_misplaced_leading_boundary(text[:pos], text[pos:next_pos], item_id=item_id):
            continue
        return _MULTI_NEWLINE_RE.sub("\n\n", text[:pos]).strip()
    return text


def _is_misplaced_leading_boundary(
    prefix: str,
    segment: str,
    *,
    item_id: Literal["item_1a", "item_7"],
) -> bool:
    """Return True if a boundary heading precedes the real section body and must not be cut at."""
    markers = _SECTION_CONTENT_MARKER_RES.get(item_id)
    if not markers:
        return False
    prefix = prefix.strip()
    if len(prefix) >= MIN_SUBSTANTIVE_SECTION_CHARS and is_substantive_section_markdown(
        prefix, item_id=item_id
    ):
        # Normal bleed: a full section precedes the next-item heading -> cut.
        return False
    if len(segment) < MIN_SUBSTANTIVE_SECTION_CHARS:
        return False
    found = sum(1 for marker in markers if marker.search(segment))
    return found >= _MIN_SECTION_CONTENT_MARKERS


def _clean_and_bound_section(raw_text: str, *, item_id: Literal["item_1a", "item_7"]) -> str:
    """Normalize section Markdown and trim any bleed into the following 10-K item."""
    return truncate_at_section_boundary(clean_markdown_section(raw_text), item_id=item_id)


def is_substantive_section_markdown(
    cleaned_text: str | None,
    *,
    item_id: Literal["item_1a", "item_7"],
    min_chars: int = MIN_ABSOLUTE_SECTION_CHARS,
) -> bool:
    """Return True if `cleaned_text` is genuine Item 1A / Item 7 prose rather than a stub or wrong anchor."""
    if not cleaned_text or len(cleaned_text) < min_chars:
        return False

    head = cleaned_text[:300]
    if item_id == "item_1a" and _WRONG_ITEM_1A_RE.match(head):
        return False
    if item_id == "item_7" and _WRONG_ITEM_7_FOOTNOTE_RE.match(head):
        return False

    if len(cleaned_text) < 2500 and _CROSS_REF_STUB_RE.search(cleaned_text):
        return False

    return True


def _slice_markdown_between(
    full_md: str,
    *,
    start_re: re.Pattern[str],
    end_re: re.Pattern[str],
    item_id: Literal["item_1a", "item_7"],
    min_slice_chars: int = 1500,
    max_slice_chars: int = 450_000,
) -> str | None:
    """Find the best substantive section slice between `start_re` and `end_re` in full Markdown."""
    best_candidate: str | None = None
    for m_start in start_re.finditer(full_md):
        header_line = m_start.group(0)
        if _TOC_PAGE_NUMBER_TAIL_RE.search(header_line):
            continue

        body_start = m_start.end()
        end_pos = min(len(full_md), body_start + max_slice_chars)
        for m_end in end_re.finditer(full_md, body_start + 400):
            if _TOC_PAGE_NUMBER_TAIL_RE.search(m_end.group(0)):
                continue
            end_pos = m_end.start()
            break

        if end_pos - body_start < min_slice_chars:
            continue

        cleaned = _clean_and_bound_section(full_md[body_start:end_pos], item_id=item_id)
        if is_substantive_section_markdown(cleaned, item_id=item_id, min_chars=min_slice_chars):
            if best_candidate is None or len(cleaned) > len(best_candidate):
                best_candidate = cleaned

    return best_candidate


def _extract_section_from_full_markdown(
    full_md: str,
    *,
    item_id: Literal["item_1a", "item_7"],
    include_annual_report_headings: bool = True,
) -> str | None:
    """Fallback parser extracting Item 1A or Item 7 directly from full 10-K or Exhibit Markdown."""
    if not full_md:
        return None

    if item_id == "item_1a":
        candidate = _slice_markdown_between(
            full_md,
            start_re=_MD_ITEM_1A_START_RE,
            end_re=_MD_ITEM_1A_END_RE,
            item_id="item_1a",
        )
        if candidate:
            return candidate
        if include_annual_report_headings:
            return _slice_markdown_between(
                full_md,
                start_re=_AR_RISK_START_RE,
                end_re=_AR_RISK_END_RE,
                item_id="item_1a",
            )
        return None

    candidate = _slice_markdown_between(
        full_md,
        start_re=_MD_ITEM_7_START_RE,
        end_re=_MD_ITEM_7_END_RE,
        item_id="item_7",
    )
    if candidate:
        return candidate
    if include_annual_report_headings:
        return _slice_markdown_between(
            full_md,
            start_re=_AR_MDA_START_RE,
            end_re=_AR_MDA_END_RE,
            item_id="item_7",
            min_slice_chars=2500,
        )
    return None


def _load_annual_report_exhibit_markdowns(filing: Any) -> list[str]:
    """Load Markdown from `EX-13*` or `EX-99.1` Annual Report attachments when referenced."""
    markdowns: list[str] = []
    attachments = getattr(filing, "attachments", None)
    if not attachments:
        return markdowns

    try:
        att_iter = list(attachments)
    except Exception:
        return markdowns

    for att in att_iter:
        doc_type = (
            str(
                getattr(att, "document_type", None)
                or getattr(att, "form", None)
                or getattr(att, "type", None)
                or ""
            )
            .strip()
            .upper()
        )
        if not (doc_type.startswith("EX-13") or doc_type in ("EX-99.1", "EX-99")):
            continue
        doc_name = str(getattr(att, "document", "") or "").lower()
        if doc_name and not doc_name.endswith((".htm", ".html", ".txt")):
            continue
        md_fn = getattr(att, "markdown", None)
        if not callable(md_fn):
            continue
        try:
            att_md = md_fn()
            if isinstance(att_md, str) and len(att_md) >= 5000:
                markdowns.append(att_md)
        except Exception as exc:
            logger.debug("Unable to render exhibit %s markdown: %s", doc_type, exc)

    return markdowns


def _extract_single_section(
    tenk: Any,
    *,
    section_keys: tuple[str, ...],
    fallback_attr: str,
    item_key: str,
    item_id: Literal["item_1a", "item_7"] | None = None,
    filing: Any | None = None,
    markdown_cache: dict[str, Any] | None = None,
) -> str | None:
    """Extract a 10-K section as Markdown (preserving tables), with full-filing & exhibit fallbacks."""
    resolved_item_id: Literal["item_1a", "item_7"] = item_id or (
        "item_1a" if "1a" in item_key.lower() else "item_7"
    )

    candidates: list[str] = []

    sections = getattr(tenk, "sections", None)
    if sections:
        for key in section_keys:
            sec_obj = sections.get(key)
            if sec_obj is None:
                continue
            md_attr = getattr(sec_obj, "markdown", None)
            raw = md_attr() if callable(md_attr) else md_attr
            if isinstance(raw, str):
                cleaned = _clean_and_bound_section(raw, item_id=resolved_item_id)
                if is_substantive_section_markdown(cleaned, item_id=resolved_item_id):
                    if len(cleaned) >= MIN_SUBSTANTIVE_SECTION_CHARS:
                        return cleaned
                    candidates.append(cleaned)

    # Fallback 1: direct TenK property (e.g., tenk.risk_factors / tenk.management_discussion)
    prop_val = getattr(tenk, fallback_attr, None)
    if isinstance(prop_val, str):
        cleaned = _clean_and_bound_section(prop_val, item_id=resolved_item_id)
        if is_substantive_section_markdown(cleaned, item_id=resolved_item_id):
            if len(cleaned) >= MIN_SUBSTANTIVE_SECTION_CHARS:
                return cleaned
            candidates.append(cleaned)

    # Fallback 2: item indexing (e.g., tenk["Item 1A"])
    try:
        with py_warnings.catch_warnings():
            py_warnings.simplefilter("ignore", FutureWarning)
            indexed_val = tenk[item_key]
        if isinstance(indexed_val, str):
            cleaned = _clean_and_bound_section(indexed_val, item_id=resolved_item_id)
            if is_substantive_section_markdown(cleaned, item_id=resolved_item_id):
                if len(cleaned) >= MIN_SUBSTANTIVE_SECTION_CHARS:
                    return cleaned
                candidates.append(cleaned)
    except Exception:
        pass

    # Fallback 3: Full 10-K Markdown scan (recovers GLW, ATO, CHD, CTSH, CLX Item 1A, XOM, CVX)
    if filing is not None:
        cache = markdown_cache if markdown_cache is not None else {}
        if "filing_md" not in cache:
            md_fn = getattr(filing, "markdown", None)
            try:
                raw_f_md = md_fn() if callable(md_fn) else None
                cache["filing_md"] = raw_f_md if isinstance(raw_f_md, str) else None
            except Exception as exc:
                logger.debug("Unable to render full filing markdown: %s", exc)
                cache["filing_md"] = None

        filing_md: str | None = cache.get("filing_md")
        if filing_md and len(filing_md) >= 5000:
            from_full_md = _extract_section_from_full_markdown(
                filing_md,
                item_id=resolved_item_id,
                include_annual_report_headings=True,
            )
            if from_full_md and len(from_full_md) >= MIN_SUBSTANTIVE_SECTION_CHARS:
                return from_full_md
            if from_full_md:
                candidates.append(from_full_md)

        # Fallback 4: EX-13 / EX-99.1 Annual Report exhibit scan (recovers BK/BNY, CLX, CCL, AVY)
        if "exhibit_mds" not in cache:
            cache["exhibit_mds"] = _load_annual_report_exhibit_markdowns(filing)
        for ex_md in cache["exhibit_mds"]:
            from_ex_md = _extract_section_from_full_markdown(
                ex_md,
                item_id=resolved_item_id,
                include_annual_report_headings=True,
            )
            if from_ex_md and len(from_ex_md) >= MIN_SUBSTANTIVE_SECTION_CHARS:
                return from_ex_md
            if from_ex_md:
                candidates.append(from_ex_md)

    if candidates:
        return max(candidates, key=len)

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


def _filter_non_dimensional_income_df(
    inc_df: Any,
    *,
    report_date_str: str | None = None,
) -> tuple[Any | None, str | None]:
    """Filter income statement DataFrame to non-dimensional, non-breakdown rows and pick primary FY column."""
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

    fy_cols = [c for c in df.columns if "(FY)" in str(c)]
    date_cols = [c for c in df.columns if _DATE_COLUMN_RE.match(str(c))]
    # Full-year "(FY)" columns first, then any other dated period columns (e.g., a fresh-start
    # successor period labelled "2021-12-31" without the "(FY)" suffix).
    candidate_cols = fy_cols + [c for c in date_cols if c not in fy_cols]
    if not candidate_cols:
        return None, None

    if not report_date_str:
        return df, candidate_cols[0]

    report_str = str(report_date_str).strip()
    try:
        target_date: date | None = date.fromisoformat(report_str[:10])
    except ValueError:
        target_date = None

    if target_date is not None:
        for col in candidate_cols:
            m = _DATE_COLUMN_RE.match(str(col))
            if not m:
                continue
            try:
                col_date = date.fromisoformat(m.group(0))
            except ValueError:
                continue
            if abs((col_date - target_date).days) <= 7:
                return df, col

    report_prefix = report_str[:7]  # YYYY-MM
    for col in candidate_cols:
        if str(col).startswith(report_prefix):
            return df, col

    # No column covers the reporting period. Previously this fell back to the first "(FY)"
    # column, which silently attributed PRIOR-YEAR values to the current fiscal year
    # (e.g., EXE FY2021 profile carried FY2020 operating/net income). Refuse instead.
    logger.debug(
        "No income statement column matches period_of_report=%s (columns=%s)",
        report_str,
        candidate_cols,
    )
    return None, None


def _canonical_xbrl_tag(raw_concept: str) -> str:
    """Format `us-gaap_Revenues` -> `us-gaap:Revenues`."""
    return str(raw_concept).replace("_", ":", 1)


def _lookup_concept_from_income_df(
    inc_df: Any,
    concepts: tuple[str, ...],
    *,
    report_date_str: str | None = None,
    pick_largest_positive: bool = False,
) -> tuple[float | None, str | None]:
    """Search non-dimensional income statement rows for matching concepts."""
    df, primary_col = _filter_non_dimensional_income_df(inc_df, report_date_str=report_date_str)
    if df is None or primary_col is None:
        return None, None

    candidates: list[tuple[float, str]] = []
    for target_concept in concepts:
        matched = df[df["concept"] == target_concept]
        if matched.empty:
            continue
        for raw_val in matched[primary_col].tolist():
            val = _coerce_finite_float(raw_val)
            if val is not None:
                tag = _canonical_xbrl_tag(target_concept)
                if not pick_largest_positive:
                    return val, tag
                candidates.append((val, tag))

    if candidates:
        # Prefer largest positive component when disambiguating REIT lease vs fee lines
        pos_candidates = [c for c in candidates if c[0] > 0]
        if pos_candidates:
            return max(pos_candidates, key=lambda item: item[0])
        return candidates[0]

    return None, None


def _lookup_by_label_or_suffix(
    inc_df: Any,
    *,
    concept_suffixes: tuple[str, ...] = (),
    label_re: re.Pattern[str] | None = None,
    report_date_str: str | None = None,
) -> tuple[float | None, str | None]:
    """Match company-extension XBRL concepts (e.g., `bk_...`) or standardized line-item labels."""
    df, primary_col = _filter_non_dimensional_income_df(inc_df, report_date_str=report_date_str)
    if df is None or primary_col is None:
        return None, None

    if concept_suffixes:
        for _, row in df.iterrows():
            concept_str = str(row.get("concept", "") or "")
            if any(concept_str.endswith(sfx) or sfx in concept_str for sfx in concept_suffixes):
                val = _coerce_finite_float(row.get(primary_col))
                if val is not None:
                    return val, _canonical_xbrl_tag(concept_str)

    if label_re is not None and "label" in df.columns:
        for _, row in df.iterrows():
            label_str = str(row.get("label", "") or "").strip()
            concept_str = str(row.get("concept", "") or "")
            if "PerShare" in concept_str or "Shares" in concept_str:
                continue
            if label_re.match(label_str):
                val = _coerce_finite_float(row.get(primary_col))
                if val is not None:
                    return val, _canonical_xbrl_tag(concept_str)

    return None, None


def _is_exclusively_dimensional_in_df(
    inc_df: Any,
    target_val: float,
    *,
    report_date_str: str | None = None,
) -> bool:
    """Return True if `target_val` appears only on dimensional/breakdown rows of `inc_df`."""
    if inc_df is None or getattr(inc_df, "empty", True):
        return False

    non_dim_df, primary_col = _filter_non_dimensional_income_df(
        inc_df, report_date_str=report_date_str
    )
    if primary_col is None or primary_col not in inc_df.columns:
        return False

    if non_dim_df is not None:
        for raw_val in non_dim_df[primary_col].tolist():
            val = _coerce_finite_float(raw_val)
            if val is not None and math.isclose(val, target_val, rel_tol=1e-6, abs_tol=1.0):
                return False

    has_dim_col = "dimension" in inc_df.columns
    has_brk_col = "is_breakdown" in inc_df.columns
    if not has_dim_col and not has_brk_col:
        return False

    dim_mask = False
    if has_dim_col:
        dim_mask = dim_mask | (inc_df["dimension"] == True)  # noqa: E712
    if has_brk_col:
        dim_mask = dim_mask | (inc_df["is_breakdown"] == True)  # noqa: E712

    dim_rows = inc_df[dim_mask]
    for raw_val in dim_rows[primary_col].tolist():
        val = _coerce_finite_float(raw_val)
        if val is not None and math.isclose(val, target_val, rel_tol=1e-6, abs_tol=1.0):
            return True

    return False


def extract_financial_metrics(
    tenk: Any,
    *,
    report_date_str: str | None = None,
) -> tuple[float | None, float | None, float | None, MetricProvenance]:
    """Extract Revenue, Operating Income, and Net Income (in USD) with XBRL provenance.

    Prioritizes non-dimensional (`dimension=False, is_breakdown=False`) Income Statement
    DataFrame rows whenever available so dimensional segment rows (e.g., APA's Kinetik
    breakdown) and minor ASC 606 ancillary lines on Equity REITs (e.g., CPT fee income vs
    ASC 842 `OperatingLeaseLeaseIncome`) never override consolidated totals.
    """
    provenance = MetricProvenance()
    fin = getattr(tenk, "financials", None)
    if fin is None:
        provenance.notes.append("No XBRL financials object available on filing.")
        return None, None, None, provenance

    inc_df = None
    try:
        inc_stmt = fin.income_statement()
        if inc_stmt is not None:
            inc_df = inc_stmt.to_dataframe()
    except Exception as exc:
        logger.debug("Income statement DataFrame unavailable: %s", exc)

    # --- 1. Revenue ---
    revenue: float | None = None
    if inc_df is not None:
        # Step 1a: Consolidated total revenue tags (us-gaap_Revenues, SalesRevenueNet, RealEstateRevenueNet, etc.)
        rev_val, rev_tag = _lookup_concept_from_income_df(
            inc_df,
            CONSOLIDATED_TOTAL_REVENUE_CONCEPTS,
            report_date_str=report_date_str,
        )
        # Step 1b: Explicit company-extension "Total revenue(s)" line (e.g., BK's bk_TotalRevenues...)
        if rev_val is None:
            rev_val, rev_tag = _lookup_by_label_or_suffix(
                inc_df,
                concept_suffixes=("_TotalRevenues",),
                label_re=_TOTAL_REVENUE_LABEL_RE,
                report_date_str=report_date_str,
            )
        # Step 1c: Component revenue tags — pick largest positive (resolves REIT OperatingLeaseLeaseIncome vs minor fees)
        if rev_val is None:
            rev_val, rev_tag = _lookup_concept_from_income_df(
                inc_df,
                COMPONENT_REVENUE_CONCEPTS,
                report_date_str=report_date_str,
                pick_largest_positive=True,
            )
        if rev_val is not None:
            revenue = rev_val
            provenance.revenue = rev_tag

    if revenue is None:
        helper_rev = _coerce_finite_float(fin.get_revenue())
        if helper_rev is not None and not _is_exclusively_dimensional_in_df(
            inc_df, helper_rev, report_date_str=report_date_str
        ):
            revenue = helper_rev
            provenance.revenue = "edgartools:Financials.get_revenue"

    # Step 1d: Reconcile against the statement's "Total revenues and other" top line when present.
    if revenue is not None and inc_df is not None:
        top_line, top_tag = _lookup_by_label_or_suffix(
            inc_df,
            concept_suffixes=("_RevenuesAndOther",),
            label_re=_TOTAL_REVENUES_AND_OTHER_LABEL_RE,
            report_date_str=report_date_str,
        )
        low, high = _REVENUE_TOP_LINE_TOLERANCE
        if (
            top_line is not None
            and top_line > 0
            and not (low * top_line <= revenue <= high * top_line)
        ):
            provenance.notes.append(
                f"Revenue concept {provenance.revenue}={revenue:,.0f} is outside {low}x-{high}x of the "
                f"statement top line {top_tag} ('Total revenues and other')={top_line:,.0f}; "
                "using the top line (mis-tagged or double-counted revenue concept)."
            )
            revenue = top_line
            provenance.revenue = top_tag

    # --- 2. Operating Income (with Pre-Tax Continuing Operations fallback) ---
    operating_income: float | None = None
    if inc_df is not None:
        op_val, op_tag = _lookup_concept_from_income_df(
            inc_df,
            OPERATING_INCOME_CONCEPTS,
            report_date_str=report_date_str,
        )
        if op_val is None:
            op_val, op_tag = _lookup_by_label_or_suffix(
                inc_df,
                concept_suffixes=("_OperatingIncomeLoss",),
                label_re=_OPERATING_INCOME_LABEL_RE,
                report_date_str=report_date_str,
            )
        if op_val is not None:
            operating_income = op_val
            provenance.operating_income = op_tag

    if operating_income is None:
        helper_op = _coerce_finite_float(fin.get_operating_income())
        if helper_op is not None and not _is_exclusively_dimensional_in_df(
            inc_df, helper_op, report_date_str=report_date_str
        ):
            operating_income = helper_op
            provenance.operating_income = (
                "edgartools:Financials.get_operating_income (us-gaap:OperatingIncomeLoss)"
            )

    if operating_income is None and inc_df is not None:
        pretax_val, pretax_tag = _lookup_concept_from_income_df(
            inc_df,
            OPERATING_INCOME_FALLBACK_CONCEPTS[1:],
            report_date_str=report_date_str,
        )
        if pretax_val is None:
            pretax_val, pretax_tag = _lookup_by_label_or_suffix(
                inc_df,
                concept_suffixes=(
                    "_IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
                    "BeforeEarningsFromAffiliatesIncomeTaxes",
                ),
                label_re=_PRETAX_INCOME_LABEL_RE,
                report_date_str=report_date_str,
            )
        # Tax-exempt single-step Equity REITs (e.g., BXP) report ProfitLoss with no tax line
        if pretax_val is None:
            tax_val, _ = _lookup_concept_from_income_df(
                inc_df,
                ("us-gaap_IncomeTaxExpenseBenefit", "us-gaap_CurrentIncomeTaxExpenseBenefit"),
                report_date_str=report_date_str,
            )
            if tax_val is None:
                reit_pl, reit_tag = _lookup_concept_from_income_df(
                    inc_df,
                    ("us-gaap_ProfitLoss",),
                    report_date_str=report_date_str,
                )
                if reit_pl is not None:
                    pretax_val, pretax_tag = reit_pl, reit_tag

        if pretax_val is not None:
            operating_income = pretax_val
            provenance.operating_income = pretax_tag
            if pretax_tag and "OperatingIncomeLoss" not in pretax_tag:
                provenance.notes.append(
                    f"Filer does not report standard GAAP OperatingIncomeLoss; "
                    f"populated operating_income from {pretax_tag} (pre-tax continuing operations income)."
                )

    # --- 3. Net Income ---
    net_income: float | None = None
    if inc_df is not None:
        ni_val, ni_tag = _lookup_concept_from_income_df(
            inc_df,
            NET_INCOME_FALLBACK_CONCEPTS,
            report_date_str=report_date_str,
        )
        if ni_val is not None:
            net_income = ni_val
            provenance.net_income = ni_tag

    if net_income is None:
        helper_ni = _coerce_finite_float(fin.get_net_income())
        if helper_ni is not None and not _is_exclusively_dimensional_in_df(
            inc_df, helper_ni, report_date_str=report_date_str
        ):
            net_income = helper_ni
            provenance.net_income = "edgartools:Financials.get_net_income (us-gaap:NetIncomeLoss)"

    return revenue, operating_income, net_income, provenance


def extract_research_and_development_expense(
    tenk: Any,
    *,
    report_date_str: str | None = None,
) -> tuple[float | None, str | None]:
    """Extract annual Research & Development expense (USD) and its XBRL concept.

    Reads the non-dimensional income statement rows for the reporting-period column, trying
    `RESEARCH_AND_DEVELOPMENT_CONCEPTS` first and then a company-extension concept / line label
    ("Research and development"). Returns `(None, None)` when the filer reports no R&D line.
    """
    fin = getattr(tenk, "financials", None)
    if fin is None:
        return None, None
    try:
        inc_stmt = fin.income_statement()
        inc_df = inc_stmt.to_dataframe() if inc_stmt is not None else None
    except Exception as exc:
        logger.debug("Income statement DataFrame unavailable for R&D: %s", exc)
        return None, None
    if inc_df is None:
        return None, None

    rd_val, rd_tag = _lookup_concept_from_income_df(
        inc_df,
        RESEARCH_AND_DEVELOPMENT_CONCEPTS,
        report_date_str=report_date_str,
    )
    if rd_val is None:
        rd_val, rd_tag = _lookup_by_label_or_suffix(
            inc_df,
            concept_suffixes=("_ResearchAndDevelopmentExpense",),
            label_re=_RESEARCH_AND_DEVELOPMENT_LABEL_RE,
            report_date_str=report_date_str,
        )
    if rd_val is None:
        return None, None
    return rd_val, rd_tag


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

    # First pass: collect candidate filings (reusing valid cached bundles or parsing `.obj()`)
    # so we can reconcile Jan-May multi-year fiscal_year offsets before deduplicating by FY.
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
        if report_date_raw < f"{start_year}-01-11":
            continue
        if report_date_raw > f"{end_year + 1}-06-30":
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

        accession_no = entry["accession_no"]
        report_date_raw = entry["report_date"]
        filing_date_raw = entry["filing_date"]

        if entry["kind"] == "cached":
            cached_bundle: ExtractedFilingBundle = entry["bundle"]
            if cached_bundle.profile.fiscal_year != fiscal_year:
                # Update fiscal year metadata if series reconciliation adjusted it
                prof = cached_bundle.profile.model_copy(
                    update={
                        "profile_id": f"{constituent.ticker}_{fiscal_year}",
                        "fiscal_year": fiscal_year,
                        "unstructured_blob_path": f"unstructured/{constituent.ticker}/{fiscal_year}_10k.md",
                        "structured_blob_path": f"structured/{constituent.ticker}/{fiscal_year}_financials.json",
                        "unstructured_gcs_uri": f"gs://{bucket_name}/unstructured/{constituent.ticker}/{fiscal_year}_10k.md",
                        "structured_gcs_uri": f"gs://{bucket_name}/structured/{constituent.ticker}/{fiscal_year}_financials.json",
                    }
                )
                updated_md = re.sub(
                    r"(?m)^fiscal_year:\s*\d+$",
                    f"fiscal_year: {fiscal_year}",
                    cached_bundle.markdown_content,
                    count=1,
                )
                cached_bundle = ExtractedFilingBundle(profile=prof, markdown_content=updated_md)
            logger.info(
                "[%s] Reusing locally staged FY%d 10-K (%s)",
                constituent.ticker,
                fiscal_year,
                accession_no,
            )
            bundles_by_fy[fiscal_year] = cached_bundle
            continue

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

        revenue, operating_income, net_income, provenance = extract_financial_metrics(
            tenk,
            report_date_str=report_date_raw,
        )
        research_and_development, rd_tag = extract_research_and_development_expense(
            tenk,
            report_date_str=report_date_raw,
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

        bundles_by_fy[fiscal_year] = ExtractedFilingBundle(
            profile=profile,
            markdown_content=markdown_doc,
        )
        logger.info(
            "[%s] Extracted FY%d 10-K (Rev=%s, OpInc=%s, NetInc=%s, R&D=%s, MD=%d chars)",
            constituent.ticker,
            fiscal_year,
            f"${revenue:,.0f}" if revenue is not None else "None",
            f"${operating_income:,.0f}" if operating_income is not None else "None",
            f"${net_income:,.0f}" if net_income is not None else "None",
            (
                f"${research_and_development:,.0f}"
                if research_and_development is not None
                else "None"
            ),
            len(markdown_doc),
        )

    ordered_bundles = [bundles_by_fy[fy] for fy in sorted(bundles_by_fy.keys())]
    return ordered_bundles, warnings
