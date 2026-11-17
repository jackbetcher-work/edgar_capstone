"""SEC 10-K section Markdown cleaning, boundary truncation, fallback parsing, and formatting."""

from __future__ import annotations

import logging
import re
import warnings as py_warnings
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Minimum lengths for validating extracted 10-K sections
MIN_ABSOLUTE_SECTION_CHARS = 350
MIN_SUBSTANTIVE_SECTION_CHARS = 2000

_TOC_TAG_RE = re.compile(
    r"(?im)^\s*(?:<u>\s*Table\s+of\s+Contents\s*</u>|\*\*Table\s+of\s+Contents\*\*|"
    r"\[Table\s+of\s+Contents\]\([^)]*\)|Table\s+of\s+Contents)\s*$"
)
_RUNNING_PAGE_HEADER_RE = re.compile(
    r"(?im)^(?:"
    r"#{1,4}\s*MANAGEMENT['\u2019]?S\s+DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+FINANCIAL\s+CONDITION\s+AND\s+RESULTS\s+OF\s+OPERATIONS\s*|"
    r"\|\s*(?:Results\s+of\s+Operations|Risk\s+Factors|Forward-looking\s+Statements|MD&A)\s*\(continued\)\s*\|\s*"
    r")$"
)
_LEADING_ITEM_HEADER_RE = re.compile(
    r"^\s*(?:\|\s*---[^\n]*\n+\s*)?(?:\|\s*)?(?:#{1,4}\s*|\*\*)?(?:PART\s+I{1,2}[\s.\\\u2014\u2013|-]*)?"
    r"Item\s+(?:1\\?\.?A|7)\\?\.?\b[^\n]*?(?:\*\*)?(?:\s*\|)?\s*(?:\n\s*\|\s*---[^\n]*)?\n+",
    re.IGNORECASE,
)
_LEADING_TABLE_RULE_RE = re.compile(r"^\s*(?:\|\s*---[^\n]*\|?\s*\n+)+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Headings of the NEXT 10-K item, used to cut sections that bleed past their own boundary
# (e.g., ETR/CCL "Item 7" = cross-ref stub followed by Items 7A-14; AEP/CVX MD&A running into
# audited financial statements; CSX/ED Item 1A running into Item 1B/1C/2). A boundary line must
# be a short title-style line: the item label followed by its canonical title keyword, or the
# bare label alone (e.g., "| ITEM 1B. |"). Prose cross-references such as
# "Item 8 of this report." or "Item 8. Financial Statements...—Note 4" do not qualify.
_ITEM_LABEL_PREFIX = (
    r"(?im)^[ \t]*(?:\|\s*)?(?:#{1,6}\s*|\*\*\s*)?(?:PART\s+I{1,3}[\s.,:|\\\u2014\u2013-]*)?ITEM\s+"
)
_ITEM_TITLE_GAP = r"[\s*|\\.:\u2014\u2013-]*(?:<u>\s*)?(?:\*\*\s*)?"
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
        re.compile(r"(?i)\bmanagement\\?['\u2019]?s\s+discussion\s+and\s+analysis\b"),
    ),
}
_MIN_SECTION_CONTENT_MARKERS = 3

_CROSS_REF_STUB_RE = re.compile(
    r"(?is)(?:"
    r"incorporated\s+(?:herein\s+)?by\s+reference|"
    r"reference\s+is\s+made\s+to\s+(?:the\s+section|page|item|exhibit)|"
    r"information\s+(?:required|called\s+for)\s+by\s+(?:this\s+)?item(?:\s+(?:1\\?\.?a|7))?\s+"
    r"(?:is\s+(?:set\s+forth|shown|presented)|appears)|"
    r"(?:this\s+)?information\s+appears\s+under\s+[\u201c\"']?(?:management|risk\s+factors)|"
    r"index\s+to\s+management['\u2019]?s\s+discussion\s+and\s+analysis|"
    r"is\s+shown\s+in\s+.*?exhibit\s+(?:13|99)|"
    r"\A[\W_]*(?:#{1,6}\s*)?refer\s+to\b|"
    r"\bon\s+pages?\s+\d+\s*(?:through|to|and|-|\u2013|\u2014)\s*\d+"
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
    r"(?im)^(?:\|\s*)?(?:#{1,4}\s*|\*\*)?(?:PART\s+I[\s.|\u2014\u2013-]*)?"
    r"ITEM\s+1\.?A\.?\b[^\n]{0,120}?risk\s+factors[^\n]*$"
)
_MD_ITEM_1A_END_RE = re.compile(
    r"(?im)^(?:\|\s*)?(?:#{1,4}\s*|\*\*)?(?:PART\s+I{1,2}[\s.|\u2014\u2013-]*)?"
    r"ITEM\s+(?:1\.?[BC]|2|3)\b[^\n]*$"
)
_MD_ITEM_7_START_RE = re.compile(
    r"(?im)^(?:\|\s*)?(?:#{1,4}\s*|\*\*)?(?:PART\s+II[\s.|\u2014\u2013-]*)?"
    r"ITEM\s+7(?!\s*\.?A)\b[^\n]{0,140}?management['\u2019]?s\s+discussion[^\n]*$"
)
_MD_ITEM_7_END_RE = re.compile(
    r"(?im)^(?:\|\s*)?(?:#{1,4}\s*|\*\*)?(?:PART\s+(?:II|III)[\s.|\u2014\u2013-]*)?"
    r"ITEM\s+(?:7\.?A|8|9)\b[^\n]*$"
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
    r"(?im)^(?:\|\s*|\*\*\s*|#{1,4}\s*)(?:MD&A\s*[\u2013\u2014-]\s*)?Risk\s+Factors(?:\s*\*\*|\s*\|)?\s*$"
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

    return not (len(cleaned_text) < 2500 and _CROSS_REF_STUB_RE.search(cleaned_text))


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
        if is_substantive_section_markdown(
            cleaned, item_id=item_id, min_chars=min_slice_chars
        ) and (best_candidate is None or len(cleaned) > len(best_candidate)):
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


def _accept_candidate(
    raw: Any,
    *,
    item_id: Literal["item_1a", "item_7"],
    candidates: list[str],
    already_cleaned: bool = False,
) -> str | None:
    """Clean, validate, and either immediately accept a substantive section or record as candidate."""
    if not isinstance(raw, str):
        return None
    cleaned = raw if already_cleaned else _clean_and_bound_section(raw, item_id=item_id)
    if not already_cleaned and not is_substantive_section_markdown(cleaned, item_id=item_id):
        return None
    if len(cleaned) >= MIN_SUBSTANTIVE_SECTION_CHARS:
        return cleaned
    if cleaned:
        candidates.append(cleaned)
    return None


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
            accepted = _accept_candidate(raw, item_id=resolved_item_id, candidates=candidates)
            if accepted is not None:
                return accepted

    # Fallback 1: direct TenK property (e.g., tenk.risk_factors / tenk.management_discussion)
    accepted = _accept_candidate(
        getattr(tenk, fallback_attr, None), item_id=resolved_item_id, candidates=candidates
    )
    if accepted is not None:
        return accepted

    # Fallback 2: item indexing (e.g., tenk["Item 1A"])
    try:
        with py_warnings.catch_warnings():
            py_warnings.simplefilter("ignore", FutureWarning)
            indexed_val = tenk[item_key]
        accepted = _accept_candidate(indexed_val, item_id=resolved_item_id, candidates=candidates)
        if accepted is not None:
            return accepted
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
            accepted = _accept_candidate(
                from_full_md,
                item_id=resolved_item_id,
                candidates=candidates,
                already_cleaned=True,
            )
            if accepted is not None:
                return accepted

        # Fallback 4: EX-13 / EX-99.1 Annual Report exhibit scan (recovers BK/BNY, CLX, CCL, AVY)
        if "exhibit_mds" not in cache:
            cache["exhibit_mds"] = _load_annual_report_exhibit_markdowns(filing)
        for ex_md in cache["exhibit_mds"]:
            from_ex_md = _extract_section_from_full_markdown(
                ex_md,
                item_id=resolved_item_id,
                include_annual_report_headings=True,
            )
            accepted = _accept_candidate(
                from_ex_md,
                item_id=resolved_item_id,
                candidates=candidates,
                already_cleaned=True,
            )
            if accepted is not None:
                return accepted

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


def rewrite_markdown_fiscal_year(markdown_content: str, *, fiscal_year: int) -> str:
    """Update YAML frontmatter, H1 heading, and Fiscal Year header bullet after FY reconciliation."""
    updated = re.sub(
        r"(?m)^fiscal_year:\s*\d+$",
        f"fiscal_year: {fiscal_year}",
        markdown_content,
        count=1,
    )
    updated = re.sub(
        r"(?m)^(# [^\n]+ \u2014 FY)\d+( Form 10-K)$",
        rf"\g<1>{fiscal_year}\g<2>",
        updated,
        count=1,
    )
    updated = re.sub(
        r"(?m)^(- \*\*Fiscal Year:\*\* )\d+",
        rf"\g<1>{fiscal_year}",
        updated,
        count=1,
    )
    return updated


__all__ = [
    "MIN_ABSOLUTE_SECTION_CHARS",
    "MIN_SUBSTANTIVE_SECTION_CHARS",
    "clean_markdown_section",
    "format_combined_10k_markdown",
    "is_substantive_section_markdown",
    "rewrite_markdown_fiscal_year",
    "truncate_at_section_boundary",
]
