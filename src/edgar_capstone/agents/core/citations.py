"""Shared helpers for canonical inline 10-K citation tags and excerpt cleanup."""

from __future__ import annotations

import re
from typing import Any

from edgar_capstone.common.domain import SECTION_TITLES

_MD_ESCAPE_RE = re.compile(r"\\([.\-(),&$%])")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_TOKEN_RE = re.compile(r"[a-z0-9]+")

NEAR_DUPLICATE_THRESHOLD = 0.90
MIN_CONTAINED_WORDS = 15


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


def _passage_fingerprint(excerpt: str) -> tuple[str, set[str], int]:
    tokens = _WORD_TOKEN_RE.findall((excerpt or "").lower())
    if len(tokens) < 3:
        shingles = {" ".join(tokens)} if tokens else set()
    else:
        shingles = {f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}" for i in range(len(tokens) - 2)}
    return " ".join(tokens), shingles, len(tokens)


def _is_duplicate_passage(
    fp_a: tuple[str, set[str], int],
    fp_b: tuple[str, set[str], int],
    *,
    same_section: bool,
) -> bool:
    joined_a, shingles_a, count_a = fp_a
    joined_b, shingles_b, count_b = fp_b
    if not joined_a or not joined_b:
        return False
    if joined_a == joined_b:
        return True
    shorter_j, shorter_n, longer_j = (
        (joined_a, count_a, joined_b) if count_a <= count_b else (joined_b, count_b, joined_a)
    )
    if shorter_n >= MIN_CONTAINED_WORDS and shorter_j in longer_j:
        return True
    if not same_section or not shingles_a or not shingles_b:
        return False
    inter = len(shingles_a & shingles_b)
    union = len(shingles_a) + len(shingles_b) - inter
    return union > 0 and (inter / union) >= NEAR_DUPLICATE_THRESHOLD


def deduplicate_citations_by_text(
    citations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Collapse exact and near-duplicate 10-K citation excerpts once on the backend.

    Returns `(deduped_citations, alias_of)` where `alias_of` maps every original `chunk_id`
    to its group's primary `chunk_id`. Duplicate members are attached to the primary under
    `duplicate_sources` so the UI can show multi-year / share-class appearances without
    re-running 3-gram deduplication on every render.
    """
    groups: list[dict[str, Any]] = []
    alias_of: dict[str, str] = {}

    for cit in citations:
        cid = str(cit.get("chunk_id") or "")
        if not cid or cid in alias_of:
            continue
        fp = _passage_fingerprint(str(cit.get("excerpt") or ""))
        sec = str(cit.get("section_id") or "")
        matched = next(
            (
                g
                for g in groups
                if _is_duplicate_passage(g["fp"], fp, same_section=(g["sec"] == sec))
            ),
            None,
        )
        if matched is None:
            groups.append({"primary": dict(cit), "duplicates": [], "fp": fp, "sec": sec})
            alias_of[cid] = cid
            continue

        if fp[2] > matched["fp"][2]:
            prev_primary = matched["primary"]
            prev_cid = str(prev_primary.get("chunk_id") or "")
            matched["duplicates"].append(prev_primary)
            matched["primary"] = dict(cit)
            matched["fp"] = fp
            for k, v in list(alias_of.items()):
                if v == prev_cid:
                    alias_of[k] = cid
            alias_of[cid] = cid
        else:
            matched["duplicates"].append(dict(cit))
            alias_of[cid] = str(matched["primary"].get("chunk_id") or "")

    out: list[dict[str, Any]] = []
    for g in groups:
        primary = g["primary"]
        if g["duplicates"]:
            primary["duplicate_sources"] = [
                {
                    "chunk_id": d.get("chunk_id"),
                    "ticker": d.get("ticker"),
                    "fiscal_year": d.get("fiscal_year"),
                    "section_id": d.get("section_id"),
                    "section_title": d.get("section_title"),
                    "sec_filing_url": d.get("sec_filing_url"),
                }
                for d in g["duplicates"]
            ]
        out.append(primary)
    return out, alias_of
