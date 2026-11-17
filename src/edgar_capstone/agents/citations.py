"""Shared helpers for canonical inline 10-K citation tags and excerpt cleanup."""

from __future__ import annotations

import re
from typing import Any

_SECTION_LABELS = {"item_1a": "Item 1A Risk Factors", "item_7": "Item 7 MD&A"}


def format_inline_citation(cit: dict[str, Any]) -> str:
    """Format `[TICKER FYxxxx | Section | chunk_id](sec_filing_url)`, the tag the UI parses."""
    sec_label = _SECTION_LABELS.get(str(cit.get("section_id") or ""), "10-K")
    url = cit.get("sec_filing_url") or "#source-viewer"
    return f"[{cit['ticker']} FY{cit['fiscal_year']} | {sec_label} | {cit['chunk_id']}]({url})"


def clean_excerpt(text: str | None, max_len: int = 320) -> str:
    """Strip Markdown escapes/emphasis and collapse whitespace, clipping at a word boundary."""
    cleaned = re.sub(r"\\([.\-(),&$%])", r"\1", text or "")
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
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
