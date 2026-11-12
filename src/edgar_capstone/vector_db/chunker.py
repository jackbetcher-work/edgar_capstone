"""Section-aware, heading-preserving Markdown chunker for SEC 10-K filings."""

from __future__ import annotations

import math
import re
from typing import Literal

from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.vector_db.models import FilingChunk

_YAML_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n+", re.DOTALL)
_ITEM_1A_HEADER_RE = re.compile(
    r"^##\s+Item\s+1A\.?\s+Risk\s+Factors[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_ITEM_7_HEADER_RE = re.compile(
    r"^##\s+Item\s+7\.?\s+Management['’]?s\s+Discussion[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_HORIZONTAL_RULE_RE = re.compile(r"^\s*---\s*$", re.MULTILINE)
_SUBHEADING_MD_RE = re.compile(r"^(#{3,6})\s+(.+?)\s*$")
_SUBHEADING_BOLD_RE = re.compile(r"^\*\*([^*\n]{4,140})\*\*\s*$")

_MISSING_SECTION_MARKERS: tuple[str, ...] = (
    "*Item 1A (Risk Factors) could not be extracted",
    "*Item 7 (MD&A) could not be extracted",
)

SECTION_BASE_TITLES: dict[Literal["item_1a", "item_7"], str] = {
    "item_1a": "Item 1A. Risk Factors",
    "item_7": "Item 7. MD&A",
}


def estimate_token_count(text: str) -> int:
    """Estimate token count for English financial & Markdown text (~4 chars/token)."""
    cleaned = text.strip()
    if not cleaned:
        return 0
    word_est = int(math.ceil(len(cleaned.split()) * 1.33))
    char_est = int(math.ceil(len(cleaned) / 4.0))
    return max(1, int(round((word_est + char_est) / 2.0)))


def extract_sections_from_combined_markdown(
    markdown_content: str,
) -> dict[Literal["item_1a", "item_7"], str]:
    """Extract raw Item 1A and Item 7 Markdown bodies from a curated 10-K `.md` file."""
    body = _YAML_FRONTMATTER_RE.sub("", markdown_content)

    m_1a = _ITEM_1A_HEADER_RE.search(body)
    m_7 = _ITEM_7_HEADER_RE.search(body)

    sections: dict[Literal["item_1a", "item_7"], str] = {}

    if m_1a:
        start_1a = m_1a.end()
        end_1a = m_7.start() if (m_7 and m_7.start() > start_1a) else len(body)
        raw_1a = body[start_1a:end_1a].strip()
        raw_1a = _HORIZONTAL_RULE_RE.sub("", raw_1a).strip()
        if raw_1a and not any(marker in raw_1a for marker in _MISSING_SECTION_MARKERS):
            sections["item_1a"] = raw_1a

    if m_7:
        start_7 = m_7.end()
        raw_7 = body[start_7:].strip()
        raw_7 = _HORIZONTAL_RULE_RE.sub("", raw_7).strip()
        if raw_7 and not any(marker in raw_7 for marker in _MISSING_SECTION_MARKERS):
            sections["item_7"] = raw_7

    return sections


def _detect_subheading(block: str) -> str | None:
    """Detect if a single-line block is a Markdown or bold subsection heading."""
    stripped = block.strip()
    if "\n" in stripped:
        return None
    md_match = _SUBHEADING_MD_RE.match(stripped)
    if md_match:
        heading = md_match.group(2).strip().rstrip(".")
        return heading.replace("\\.", ".").replace("\\-", "-")

    bold_match = _SUBHEADING_BOLD_RE.match(stripped)
    if bold_match:
        heading = bold_match.group(1).strip()
        # Treat concise standalone bold lines as subsection headers
        if len(heading) <= 130 and not heading.endswith(":"):
            return heading.replace("\\.", ".").replace("\\-", "-")

    return None


def _split_oversized_block(block: str, max_chars: int) -> list[str]:
    """Split a single huge paragraph or table into sub-blocks <= max_chars."""
    if len(block) <= max_chars:
        return [block]

    lines = block.split("\n")
    # If multi-line (e.g., a large Markdown table), group by lines preserving table headers
    if len(lines) > 2:
        table_header: list[str] = []
        if lines[0].lstrip().startswith("|") and len(lines) >= 2 and "---" in lines[1]:
            table_header = [lines[0], lines[1]]
            content_lines = lines[2:]
        else:
            content_lines = lines

        pieces: list[str] = []
        current_lines: list[str] = list(table_header)
        current_len = sum(len(x) + 1 for x in current_lines)

        for line in content_lines:
            if current_lines and current_len + len(line) + 1 > max_chars and len(current_lines) > len(table_header):
                pieces.append("\n".join(current_lines).strip())
                current_lines = list(table_header)
                current_len = sum(len(x) + 1 for x in current_lines)
            current_lines.append(line)
            current_len += len(line) + 1

        if len(current_lines) > len(table_header):
            pieces.append("\n".join(current_lines).strip())
        if pieces:
            return pieces

    # Sentence-level fallback for single giant paragraphs
    sentences = re.split(r"(?<=[.!?])\s+", block)
    pieces = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        if current and current_len + len(sent) + 1 > max_chars:
            pieces.append(" ".join(current).strip())
            current = []
            current_len = 0
        if len(sent) > max_chars:
            for start in range(0, len(sent), max_chars):
                pieces.append(sent[start : start + max_chars].strip())
        else:
            current.append(sent)
            current_len += len(sent) + 1
    if current:
        pieces.append(" ".join(current).strip())
    return [p for p in pieces if p]


def split_section_into_passages(
    section_text: str,
    *,
    section_id: Literal["item_1a", "item_7"],
    target_chars: int = 2200,
    max_chars: int = 3600,
    min_chars: int = 150,
    overlap_chars: int = 200,
) -> list[tuple[str, str]]:
    """Split a 10-K section into `(section_title, passage_text)` tuples.

    Preserves subsection headings and Markdown tables while maintaining semantic
    continuity across chunk boundaries.
    """
    base_title = SECTION_BASE_TITLES[section_id]
    raw_blocks = [b.strip() for b in re.split(r"\n{2,}", section_text) if b.strip()]
    if not raw_blocks:
        return []

    current_subheading: str | None = None
    tagged_blocks: list[tuple[str, str, bool]] = []

    for block in raw_blocks:
        detected = _detect_subheading(block)
        if detected is not None:
            current_subheading = detected
            full_title = f"{base_title} — {current_subheading}"
            tagged_blocks.append((full_title, block, True))
        else:
            full_title = (
                f"{base_title} — {current_subheading}"
                if current_subheading
                else base_title
            )
            for sub_block in _split_oversized_block(block, max_chars=max_chars):
                tagged_blocks.append((full_title, sub_block, False))

    chunks: list[tuple[str, str]] = []
    buf_blocks: list[str] = []
    buf_title: str = tagged_blocks[0][0]
    buf_len = 0

    for title, text, is_heading in tagged_blocks:
        block_len = len(text)
        # Flush buffer if adding this block exceeds target_chars and we already have substantive text
        if (
            buf_blocks
            and (buf_len + block_len + 2 > target_chars or (is_heading and buf_len >= min_chars * 3))
            and buf_len >= min_chars
        ):
            chunk_body = "\n\n".join(buf_blocks).strip()
            chunks.append((buf_title, chunk_body))

            # Carry over trailing non-table paragraph for semantic overlap if within same subheading
            overlap_seed: list[str] = []
            if overlap_chars > 0 and not is_heading and buf_blocks:
                last_block = buf_blocks[-1]
                if len(last_block) <= overlap_chars * 2 and not last_block.lstrip().startswith("|"):
                    overlap_seed = [last_block]

            buf_blocks = overlap_seed
            buf_len = sum(len(x) for x in buf_blocks)
            buf_title = title

        if not buf_blocks:
            buf_title = title
        buf_blocks.append(text)
        buf_len += block_len + 2

    if buf_blocks:
        final_body = "\n\n".join(buf_blocks).strip()
        if len(final_body) < min_chars and chunks:
            prev_title, prev_body = chunks[-1]
            if len(prev_body) + len(final_body) + 2 <= max_chars:
                chunks[-1] = (prev_title, f"{prev_body}\n\n{final_body}")
            else:
                chunks.append((buf_title, final_body))
        else:
            chunks.append((buf_title, final_body))

    return chunks


def chunk_filing_document(
    profile: FinancialDataProfile,
    markdown_content: str,
    *,
    embedding_model: str = "text-embedding-005",
    target_chars: int = 2200,
    max_chars: int = 3600,
    min_chars: int = 150,
    overlap_chars: int = 200,
) -> list[FilingChunk]:
    """Chunk a single company-year 10-K Markdown document into metadata-tagged `FilingChunk`s."""
    sections = extract_sections_from_combined_markdown(markdown_content)
    all_chunks: list[FilingChunk] = []

    for section_id in ("item_1a", "item_7"):
        sec_text = sections.get(section_id)
        if not sec_text:
            continue

        passages = split_section_into_passages(
            sec_text,
            section_id=section_id,
            target_chars=target_chars,
            max_chars=max_chars,
            min_chars=min_chars,
            overlap_chars=overlap_chars,
        )

        for idx, (sec_title, passage) in enumerate(passages):
            context_header = (
                f"[{profile.ticker} ({profile.company_name}) | "
                f"FY{profile.fiscal_year} Form 10-K | {sec_title}]"
            )
            enriched_text = f"{context_header}\n\n{passage}"
            char_count = len(enriched_text)
            token_est = estimate_token_count(enriched_text)
            safe_profile = profile.profile_id.replace(".", "_")
            chunk_id = f"{safe_profile}_{section_id}_{idx:04d}"

            all_chunks.append(
                FilingChunk(
                    chunk_id=chunk_id,
                    profile_id=profile.profile_id,
                    ticker=profile.ticker,
                    cik=profile.cik,
                    company_name=profile.company_name,
                    gics_sector=profile.gics_sector,
                    gics_sub_industry=profile.gics_sub_industry,
                    form_type=profile.form_type,
                    fiscal_year=profile.fiscal_year,
                    period_of_report=profile.period_of_report,
                    filing_date=profile.filing_date,
                    accession_number=profile.accession_number,
                    section_id=section_id,
                    section_title=sec_title,
                    chunk_index=idx,
                    chunk_text=enriched_text,
                    char_count=char_count,
                    token_estimate=token_est,
                    embedding=[],
                    embedding_model=embedding_model,
                    unstructured_gcs_uri=profile.unstructured_gcs_uri,
                    structured_gcs_uri=profile.structured_gcs_uri,
                    sec_filing_url=profile.sec_filing_url,
                )
            )

    return all_chunks
