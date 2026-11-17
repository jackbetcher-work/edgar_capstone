"""Shared helpers for canonical inline 10-K citation tags and excerpt cleanup."""

from __future__ import annotations

import re
from typing import Any

from edgar_capstone.common.domain import SECTION_TITLES

_MD_ESCAPE_RE = re.compile(r"\\([.\-(),&$%])")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_WHITESPACE_RE = re.compile(r"\s+")


def format_inline_citation(
    cit: dict[str, Any],
    *,
    fallback_ticker: str = "",
    fallback_fy: int | str = "",
    default_url: str = "",
) -> str:
    """Format `[TICKER FYxxxx | Section | chunk_id](sec_filing_url)`, the tag the UI parses."""
    ticker = cit.get("ticker") or fallback_ticker
    fy = cit.get("fiscal_year") or fallback_fy
    section = cit.get("section_title") or SECTION_TITLES.get(
        str(cit.get("section_id") or ""), "10-K"
    )
    url = cit.get("sec_filing_url") or default_url
    return f"[{ticker} FY{fy} | {section} | {cit.get('chunk_id')}]({url})"


def clean_excerpt(text: str | None, max_len: int = 320) -> str:
    """Strip Markdown escapes/emphasis and collapse whitespace, clipping at a word boundary."""
    cleaned = _MD_ESCAPE_RE.sub(r"\1", text or "")
    cleaned = _BOLD_RE.sub(r"\1", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if len(cleaned) <= max_len:
        return cleaned
    clipped = cleaned[:max_len].rsplit(" ", 1)[0]
    return f"{clipped}..."


def merge_citations(
    existing: list[dict[str, Any]], extra: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Append citations whose `chunk_id` is not already present (in place); return `existing`."""
    seen = {str(c.get("chunk_id")) for c in existing}
    for cit in extra or []:
        cid = str(cit.get("chunk_id") or "")
        if cid and cid not in seen:
            seen.add(cid)
            existing.append(cit)
    return existing
