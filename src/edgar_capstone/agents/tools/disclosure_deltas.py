"""Disclosure Delta pipeline (`compare_disclosures_across_years`).

Turns "how has the wording on <theme> changed?" into deterministic, citable grounding facts:

1. **Theme retrieval:** for each company and fiscal year, hybrid search returns the passages
   most relevant to the theme (same `search_10k_chunks_rag` used everywhere else).
2. **Alignment:** each theme passage is compared with the *other* year's full section using
   word-trigram coverage, which is insensitive to chunk boundaries moving between filings.
   The best-overlapping chunk in the other year becomes the counterpart citation.
3. **Classification:** coverage >= ``UNCHANGED_MIN`` -> *unchanged*; a counterpart chunk that
   shares >= ``MODIFIED_MIN`` of the passage's trigrams -> *modified* (with word-level added /
   removed phrases); otherwise *added* (current-year passage) or *removed* (prior-year passage).
4. **Theme mentions:** occurrences of the theme's key terms and domain synonyms (supplied by the
   calling LLM via ``theme_terms`` in the same tool call, with a lightweight token fallback when
   omitted) per fiscal year across the compared sections.

Every delta cites the prior-year and/or current-year chunk it was computed from, so Gemini
summarizes changes it did not have to detect itself, and the analyst can open both passages.
"""

from __future__ import annotations

import difflib
import functools
import itertools
import re
import threading
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from google.adk.tools import FunctionTool, ToolContext

from edgar_capstone.agents.core.citations import clean_excerpt, merge_citations
from edgar_capstone.agents.core.resilience import with_tool_resilience
from edgar_capstone.agents.core.schemas import (
    CompanyDisclosureEvolution,
    DisclosureDelta,
    DisclosureYearPair,
    GroundedCitation,
)
from edgar_capstone.agents.prompts.builders import compact_disclosure_deltas_for_tool
from edgar_capstone.agents.tools.runtime import (
    STATE_DISCLOSURE_DELTA_SUMMARY,
    STATE_DISCLOSURE_DELTAS,
    STATE_RETRIEVED_CITATIONS,
    cached_turn_profiles,
    merge_disclosure_deltas,
)
from edgar_capstone.agents.tools.sec_tools import (
    _gcp_adapters,
    _parse_ticker_list,
    get_full_10k_section,
    search_10k_chunks_rag,
)
from edgar_capstone.common.concurrency import run_parallel_with_context
from edgar_capstone.common.domain import (
    SECTION_TITLES,
    SectionId,
    normalize_edgar_ticker,
)
from edgar_capstone.common.env import env_positive_int

UNCHANGED_MIN = 0.90
MODIFIED_MIN = 0.40
SHINGLE_SIZE = 3
DEFAULT_MAX_COMPANIES = 5
DEFAULT_PASSAGES_PER_YEAR = 6
DEFAULT_MAX_DELTAS_PER_PAIR = 8
_MAX_TERM_SUPPLEMENT_CHUNKS = 12
_MAX_PHRASES = 5
_MAX_PHRASE_WORDS = 18
_EXCERPT_CHARS = 420

ChangeType = Literal["added", "removed", "modified", "unchanged"]

_SECTIONS: tuple[SectionId, ...] = ("item_1a", "item_7")
_CHANGE_ORDER: dict[ChangeType, int] = {"modified": 0, "added": 1, "removed": 2, "unchanged": 3}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_MD_ESCAPE_RE = re.compile(r"\\([^\w\s])")
_NON_WORD_RE = re.compile(r"[^\w]")
_TERM_SPLIT_RE = re.compile(r"[,;\n]+")
_MAX_THEME_TERMS = 20
_MAX_TERM_CHARS = 80
_STATE_MERGE_LOCK = threading.Lock()

# Minimal stopword filter used only as a fallback when the caller omits explicit `theme_terms`.
_FALLBACK_STOPWORDS: frozenset[str] = frozenset(
    {
        "10k",
        "a",
        "across",
        "all",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "between",
        "by",
        "change",
        "changed",
        "changes",
        "companies",
        "company",
        "disclosure",
        "disclosures",
        "filing",
        "filings",
        "fiscal",
        "for",
        "from",
        "fy",
        "has",
        "have",
        "how",
        "in",
        "is",
        "item",
        "of",
        "on",
        "or",
        "risk",
        "risks",
        "the",
        "to",
        "was",
        "were",
        "with",
        "wording",
        "year",
        "years",
    }
)


# ---------------------------------------------------------------------------
# Text similarity primitives
# ---------------------------------------------------------------------------


def _clean_text(text: str | None) -> str:
    return (
        _MD_ESCAPE_RE.sub(r"\1", text or "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("*", " ")
    )


_CHUNK_HEADER_RE = re.compile(r"^\s*\[[^\n]*\bForm 10-K\b[^\n]*\]\s*")


def strip_chunk_header(text: str) -> str:
    """Remove the chunker's context header line from a chunk's text."""
    return _CHUNK_HEADER_RE.sub("", text or "", count=1)


def word_tokens(text: str | None) -> list[str]:
    """Lowercase word tokens with Markdown escapes and emphasis removed."""
    return _TOKEN_RE.findall(_clean_text(text).lower())


def shingles(tokens: list[str], size: int = SHINGLE_SIZE) -> set[tuple[str, ...]]:
    """Word n-gram set; texts shorter than ``size`` collapse to a single n-gram."""
    if not tokens:
        return set()
    if len(tokens) < size:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def coverage(passage: set[tuple[str, ...]], other: set[tuple[str, ...]]) -> float:
    """Share of ``passage`` n-grams present in ``other`` (0.0 when ``passage`` is empty)."""
    if not passage:
        return 0.0
    return len(passage & other) / len(passage)


def wording_changes(prior_text: str, current_text: str) -> tuple[list[str], list[str]]:
    """Word-level phrases added in ``current_text`` and removed from ``prior_text``.

    Leading/trailing unmatched runs are ignored: they come from chunk boundaries that moved
    between filings, not from edits to the passage itself.
    """
    prior_words = _clean_text(prior_text).split()
    curr_words = _clean_text(current_text).split()

    def _key(word: str) -> str:
        return _NON_WORD_RE.sub("", word.lower())

    matcher = difflib.SequenceMatcher(
        None, [_key(w) for w in prior_words], [_key(w) for w in curr_words], autojunk=False
    )
    opcodes = matcher.get_opcodes()
    while opcodes and opcodes[0][0] != "equal":
        opcodes = opcodes[1:]
    while opcodes and opcodes[-1][0] != "equal":
        opcodes = opcodes[:-1]

    added: list[str] = []
    removed: list[str] = []

    def _phrase(words: list[str]) -> str | None:
        kept_words = [w for w in words if _key(w)]
        if not kept_words:
            return None
        text = " ".join(kept_words[:_MAX_PHRASE_WORDS])
        return text + (" …" if len(kept_words) > _MAX_PHRASE_WORDS else "")

    for tag, i1, i2, j1, j2 in opcodes:
        if tag in ("insert", "replace") and len(added) < _MAX_PHRASES:
            phrase = _phrase(curr_words[j1:j2])
            if phrase:
                added.append(phrase)
        if tag in ("delete", "replace") and len(removed) < _MAX_PHRASES:
            phrase = _phrase(prior_words[i1:i2])
            if phrase:
                removed.append(phrase)
    return added, removed


def parse_theme_terms(raw_terms: str | Sequence[str] | None) -> list[str]:
    """Normalize LLM-supplied theme phrases/synonyms (comma-separated string or sequence)."""
    if not raw_terms:
        return []
    items = _TERM_SPLIT_RE.split(raw_terms) if isinstance(raw_terms, str) else list(raw_terms)
    terms: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = " ".join(_clean_text(str(item)).strip(" \"'").split()).lower()
        if len(cleaned) < 2 or len(cleaned) > _MAX_TERM_CHARS or cleaned.isdigit():
            continue
        if not any(ch.isalnum() for ch in cleaned):
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            terms.append(cleaned)
            if len(terms) >= _MAX_THEME_TERMS:
                break
    return terms


def extract_theme_terms(theme: str) -> list[str]:
    """Lightweight fallback tokenization when the caller omits explicit ``theme_terms``."""
    terms: list[str] = []
    seen: set[str] = set()
    for tok in word_tokens(theme):
        if tok in _FALLBACK_STOPWORDS or tok.isdigit() or len(tok) < 2:
            continue
        if tok not in seen:
            seen.add(tok)
            terms.append(tok)
            if len(terms) >= _MAX_THEME_TERMS:
                break
    return terms


def resolve_theme_terms(theme: str, theme_terms: str | Sequence[str] | None = None) -> list[str]:
    """Prefer LLM-supplied ``theme_terms`` (phrases + synonyms); fall back to ``theme`` tokens."""
    parsed = parse_theme_terms(theme_terms)
    return parsed if parsed else extract_theme_terms(theme)


@functools.lru_cache(maxsize=256)
def _terms_regex_cached(terms: tuple[str, ...]) -> re.Pattern[str] | None:
    if not terms:
        return None
    alternation = "|".join(
        re.escape(t).replace(r"\ ", r"\s+") for t in sorted(terms, key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternation})(?:s|es)?(?!\w)", re.IGNORECASE)


def _terms_regex(terms: Sequence[str]) -> re.Pattern[str] | None:
    return _terms_regex_cached(tuple(terms))


def count_theme_mentions(text: str | None, terms: Sequence[str]) -> int:
    """Count non-overlapping theme-term occurrences (longest phrase wins, plurals included)."""
    pattern = _terms_regex(terms)
    if pattern is None or not text:
        return 0
    return len(pattern.findall(_clean_text(text)))


def _coerce_section_id(raw: str) -> SectionId:
    return "item_7" if str(raw).strip().lower() == "item_7" else "item_1a"


# ---------------------------------------------------------------------------
# Pure alignment over prepared section material
# ---------------------------------------------------------------------------


@dataclass
class SectionMaterial:
    """One company-year section: full text (for coverage) and canonical chunks (for citations)."""

    ticker: str
    company_name: str
    fiscal_year: int
    section_id: str
    full_text: str = ""
    chunks: list[dict[str, Any]] = field(default_factory=list)

    def effective_text(self) -> str:
        """Return full section text, falling back to concatenated chunk excerpts when empty."""
        if self.full_text:
            return self.full_text
        parts = [_passage_text(c) for c in self.chunks]
        return " ".join(p for p in parts if p)

    @functools.cached_property
    def all_shingles(self) -> set[tuple[str, ...]]:
        text = self.full_text or "\n".join(_passage_text(c) for c in self.chunks)
        return shingles(word_tokens(text))

    @functools.cached_property
    def chunk_shingles(self) -> list[tuple[dict[str, Any], set[tuple[str, ...]]]]:
        return [(c, shingles(word_tokens(_passage_text(c)))) for c in self.chunks]

    def best_counterpart(
        self, passage: set[tuple[str, ...]]
    ) -> tuple[dict[str, Any] | None, float]:
        best: dict[str, Any] | None = None
        best_cov = 0.0
        for chunk, chunk_sh in self.chunk_shingles:
            cov = coverage(passage, chunk_sh)
            if cov > best_cov:
                best, best_cov = chunk, cov
        return best, best_cov


def chunk_row_to_citation(row: dict[str, Any], score: float = 0.0) -> dict[str, Any]:
    """Normalize a BigQuery `filing_chunks` row into a `GroundedCitation` dict."""
    return GroundedCitation.from_row(row, score=score).model_dump(mode="json")


def _passage_text(cit: dict[str, Any]) -> str:
    """Chunk/passage body without the "[TICKER (Name) | FYxxxx Form 10-K | Section]" header.

    The header embeds the fiscal year, which would otherwise surface as a wording change.
    """
    return strip_chunk_header(str(cit.get("excerpt") or cit.get("chunk_text") or ""))


def _classify_current_passages(
    *,
    ticker: str,
    company_name: str,
    section_id: SectionId,
    section_title: str,
    py: int,
    cy: int,
    prior: SectionMaterial,
    current_passages: list[dict[str, Any]],
    cited: dict[str, dict[str, Any]],
    pair_keys: set[tuple[str | None, str | None]],
    deltas: list[DisclosureDelta],
) -> None:
    """Classify current-year theme passages as unchanged, modified, or added against `prior`."""
    for cit in current_passages:
        text = _passage_text(cit)
        sh = shingles(word_tokens(text))
        if not sh:
            continue
        sec_cov = coverage(sh, prior.all_shingles)
        counterpart, best_cov = prior.best_counterpart(sh)
        cp_cit = chunk_row_to_citation(counterpart) if counterpart is not None else None
        relevance = float(cit.get("score") or 0.0)
        kind: ChangeType
        if sec_cov >= UNCHANGED_MIN:
            kind = "unchanged"
            added_p: list[str] = []
            removed_p: list[str] = []
        elif counterpart is not None and best_cov >= MODIFIED_MIN:
            kind = "modified"
            added_p, removed_p = wording_changes(_passage_text(counterpart), text)
        else:
            kind, cp_cit, added_p, removed_p = "added", None, [], []
        if kind == "unchanged" and best_cov < MODIFIED_MIN:
            cp_cit = None

        prior_cid = str(cp_cit["chunk_id"]) if cp_cit else None
        curr_cid = str(cit["chunk_id"])
        cited.setdefault(curr_cid, cit)
        if cp_cit is not None and prior_cid is not None:
            cited.setdefault(prior_cid, cp_cit)
        key = (prior_cid, curr_cid)
        if key in pair_keys:
            continue
        pair_keys.add(key)
        deltas.append(
            DisclosureDelta(
                delta_id=f"{ticker}_{section_id}_{py}_{cy}_{kind}_{curr_cid}",
                ticker=ticker,
                company_name=company_name,
                section_id=section_id,
                section_title=section_title,
                change_type=kind,
                prior_fiscal_year=py,
                current_fiscal_year=cy,
                prior_chunk_id=prior_cid,
                current_chunk_id=curr_cid,
                prior_excerpt=clean_excerpt(_passage_text(cp_cit), _EXCERPT_CHARS)
                if cp_cit
                else None,
                current_excerpt=clean_excerpt(text, _EXCERPT_CHARS),
                similarity=round(sec_cov, 3),
                added_phrases=added_p,
                removed_phrases=removed_p,
                theme_relevance=round(relevance, 4),
            )
        )


def _classify_prior_passages(
    *,
    ticker: str,
    company_name: str,
    section_id: SectionId,
    section_title: str,
    py: int,
    cy: int,
    current: SectionMaterial,
    prior_passages: list[dict[str, Any]],
    cited: dict[str, dict[str, Any]],
    pair_keys: set[tuple[str | None, str | None]],
    deltas: list[DisclosureDelta],
) -> None:
    """Classify unmatched prior-year theme passages as removed or modified against `current`."""
    used_current = {d.current_chunk_id for d in deltas}
    used_prior = {d.prior_chunk_id for d in deltas if d.prior_chunk_id}
    for cit in prior_passages:
        prior_cid = str(cit["chunk_id"])
        if prior_cid in used_prior:
            continue
        text = _passage_text(cit)
        sh = shingles(word_tokens(text))
        if not sh:
            continue
        sec_cov = coverage(sh, current.all_shingles)
        if sec_cov >= UNCHANGED_MIN:
            continue  # still present; nothing new to report from the prior side
        counterpart, best_cov = current.best_counterpart(sh)
        relevance = float(cit.get("score") or 0.0)
        if counterpart is not None and best_cov >= MODIFIED_MIN:
            curr_cid = str(counterpart["chunk_id"])
            if curr_cid in used_current:
                continue
            cp_cit = chunk_row_to_citation(counterpart)
            added_p, removed_p = wording_changes(text, _passage_text(counterpart))
            cited.setdefault(prior_cid, cit)
            cited.setdefault(curr_cid, cp_cit)
            key = (prior_cid, curr_cid)
            if key in pair_keys:
                continue
            pair_keys.add(key)
            deltas.append(
                DisclosureDelta(
                    delta_id=f"{ticker}_{section_id}_{py}_{cy}_modified_{curr_cid}",
                    ticker=ticker,
                    company_name=company_name,
                    section_id=section_id,
                    section_title=section_title,
                    change_type="modified",
                    prior_fiscal_year=py,
                    current_fiscal_year=cy,
                    prior_chunk_id=prior_cid,
                    current_chunk_id=curr_cid,
                    prior_excerpt=clean_excerpt(text, _EXCERPT_CHARS),
                    current_excerpt=clean_excerpt(_passage_text(cp_cit), _EXCERPT_CHARS),
                    similarity=round(sec_cov, 3),
                    added_phrases=added_p,
                    removed_phrases=removed_p,
                    theme_relevance=round(relevance, 4),
                )
            )
            continue

        cited.setdefault(prior_cid, cit)
        key = (prior_cid, None)
        if key in pair_keys:
            continue
        pair_keys.add(key)
        deltas.append(
            DisclosureDelta(
                delta_id=f"{ticker}_{section_id}_{py}_{cy}_removed_{prior_cid}",
                ticker=ticker,
                company_name=company_name,
                section_id=section_id,
                section_title=section_title,
                change_type="removed",
                prior_fiscal_year=py,
                current_fiscal_year=cy,
                prior_chunk_id=prior_cid,
                current_chunk_id=None,
                prior_excerpt=clean_excerpt(text, _EXCERPT_CHARS),
                current_excerpt=None,
                similarity=round(sec_cov, 3),
                theme_relevance=round(relevance, 4),
            )
        )


def _select_top_deltas(
    deltas: list[DisclosureDelta],
    max_deltas: int,
) -> list[DisclosureDelta]:
    """Select up to `max_deltas` preserving substantive change-type diversity and relevance.

    Guarantees representation for each present substantive change type (`modified`, `added`,
    `removed`) so several modified paragraphs cannot crowd out a newly added or removed risk
    factor, then fills remaining slots by `theme_relevance` (substantive changes ahead of
    `unchanged`), and returns the selected items in canonical display order.
    """
    limit = max(0, int(max_deltas))
    if limit == 0 or not deltas:
        return []

    substantive_kinds: tuple[ChangeType, ...] = ("modified", "added", "removed")
    by_kind: dict[ChangeType, list[DisclosureDelta]] = {k: [] for k in _CHANGE_ORDER}
    for d in deltas:
        by_kind[d.change_type].append(d)
    for items in by_kind.values():
        items.sort(key=lambda d: (-d.theme_relevance, d.delta_id))

    selected: list[DisclosureDelta] = []
    seen_ids: set[str] = set()

    def _add(item: DisclosureDelta) -> None:
        if item.delta_id not in seen_ids and len(selected) < limit:
            seen_ids.add(item.delta_id)
            selected.append(item)

    # Pass 1: reserve up to 2 highest-relevance deltas from each substantive kind (round-robin)
    for slot in range(2):
        for kind in substantive_kinds:
            bucket = by_kind[kind]
            if slot < len(bucket):
                _add(bucket[slot])

    # Pass 2: fill remaining slots with highest-relevance substantive changes across all kinds
    remaining_substantive = [
        d for kind in substantive_kinds for d in by_kind[kind] if d.delta_id not in seen_ids
    ]
    remaining_substantive.sort(
        key=lambda d: (-d.theme_relevance, _CHANGE_ORDER[d.change_type], d.delta_id)
    )
    for d in remaining_substantive:
        if len(selected) >= limit:
            break
        _add(d)

    # Pass 3: fill any leftover slots with unchanged passages
    for d in by_kind["unchanged"]:
        if len(selected) >= limit:
            break
        _add(d)

    selected.sort(key=lambda d: (_CHANGE_ORDER[d.change_type], -d.theme_relevance, d.delta_id))
    return selected


def align_year_pair(
    *,
    ticker: str,
    company_name: str,
    section_id: str,
    prior: SectionMaterial,
    current: SectionMaterial,
    prior_passages: list[dict[str, Any]],
    current_passages: list[dict[str, Any]],
    max_deltas: int = DEFAULT_MAX_DELTAS_PER_PAIR,
) -> tuple[DisclosureYearPair, list[dict[str, Any]]]:
    """Classify theme passages of two consecutive filings; return the pair and cited chunks."""
    sec_id = _coerce_section_id(section_id)
    section_title = SECTION_TITLES[sec_id]
    py, cy = prior.fiscal_year, current.fiscal_year
    deltas: list[DisclosureDelta] = []
    cited: dict[str, dict[str, Any]] = {}
    pair_keys: set[tuple[str | None, str | None]] = set()

    _classify_current_passages(
        ticker=ticker,
        company_name=company_name,
        section_id=sec_id,
        section_title=section_title,
        py=py,
        cy=cy,
        prior=prior,
        current_passages=current_passages,
        cited=cited,
        pair_keys=pair_keys,
        deltas=deltas,
    )
    _classify_prior_passages(
        ticker=ticker,
        company_name=company_name,
        section_id=sec_id,
        section_title=section_title,
        py=py,
        cy=cy,
        current=current,
        prior_passages=prior_passages,
        cited=cited,
        pair_keys=pair_keys,
        deltas=deltas,
    )

    counts: dict[str, int] = {kind: 0 for kind in _CHANGE_ORDER}
    for d in deltas:
        counts[d.change_type] += 1
    kept = _select_top_deltas(deltas, max_deltas)
    kept_ids = {cid for d in kept for cid in (d.prior_chunk_id, d.current_chunk_id) if cid}
    pair = DisclosureYearPair(
        prior_fiscal_year=py, current_fiscal_year=cy, counts=counts, deltas=kept
    )
    return pair, [c for cid, c in cited.items() if cid in kept_ids]


def _supplement_theme_passages(
    passages: list[dict[str, Any]],
    material: SectionMaterial,
    theme_terms: Sequence[str],
    *,
    max_extra: int = _MAX_TERM_SUPPLEMENT_CHUNKS,
) -> list[dict[str, Any]]:
    """Augment vector-retrieved passages with unseen section chunks matching `theme_terms`.

    Multi-topic or multi-chunk risk factors (e.g. PFAS + earplug litigation spanning 8-10 chunks)
    can overflow `per_year_k` vector hits with unchanged paragraphs, hiding modified chunks.
    `SectionMaterial.chunks` is already loaded in memory from `fetch_sections_chunks_batch`, so
    scanning it for `theme_terms` adds zero extra BigQuery RPCs.
    """
    if not theme_terms or not material.chunks or max_extra <= 0:
        return passages
    seen_ids = {str(c.get("chunk_id") or "") for c in passages}
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for idx, chunk in enumerate(material.chunks):
        cid = str(chunk.get("chunk_id") or "")
        if not cid or cid in seen_ids:
            continue
        hits = count_theme_mentions(_passage_text(chunk), theme_terms)
        if hits > 0:
            chunk_idx = int(
                chunk.get("chunk_index") if chunk.get("chunk_index") is not None else idx
            )
            candidates.append((hits, chunk_idx, cid, chunk))
    if not candidates:
        return passages
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    extras = [
        chunk_row_to_citation(chunk, score=round(min(0.45, 0.15 + 0.05 * hits), 4))
        for hits, _, _, chunk in candidates[:max_extra]
    ]
    return [*passages, *extras]


def build_company_evolution(
    *,
    ticker: str,
    company_name: str,
    gics_sector: str,
    window_years: list[int],
    available_years: list[int],
    theme_terms: list[str],
    sections: dict[tuple[int, str], SectionMaterial],
    theme_passages: dict[int, list[dict[str, Any]]],
    max_deltas_per_pair: int = DEFAULT_MAX_DELTAS_PER_PAIR,
) -> tuple[CompanyDisclosureEvolution, list[dict[str, Any]]]:
    """Align every consecutive pair of available years for one company."""
    years = sorted(y for y in available_years if y in window_years)
    citations: dict[str, dict[str, Any]] = {}
    year_pairs: list[DisclosureYearPair] = []

    mentions: dict[str, int] = {}
    for fy in years:
        texts = [
            text for (y, _sec), m in sections.items() if y == fy and (text := m.effective_text())
        ]
        if texts:
            mentions[str(fy)] = sum(count_theme_mentions(t, theme_terms) for t in texts)

    for py, cy in itertools.pairwise(years):
        merged_pair: DisclosureYearPair | None = None
        for section_id in _SECTIONS:
            prior = sections.get((py, section_id))
            current = sections.get((cy, section_id))
            prior_p = [c for c in theme_passages.get(py, []) if c.get("section_id") == section_id]
            curr_p = [c for c in theme_passages.get(cy, []) if c.get("section_id") == section_id]
            if prior is None or current is None or not (prior_p or curr_p):
                continue
            prior_p = _supplement_theme_passages(prior_p, prior, theme_terms)
            curr_p = _supplement_theme_passages(curr_p, current, theme_terms)
            pair, cited = align_year_pair(
                ticker=ticker,
                company_name=company_name,
                section_id=section_id,
                prior=prior,
                current=current,
                prior_passages=prior_p,
                current_passages=curr_p,
                max_deltas=max_deltas_per_pair,
            )
            for cit in cited:
                citations.setdefault(str(cit["chunk_id"]), cit)
            if merged_pair is None:
                merged_pair = pair
            else:
                for kind, n in pair.counts.items():
                    merged_pair.counts[kind] = merged_pair.counts.get(kind, 0) + n
                merged_pair.deltas.extend(pair.deltas)
        if merged_pair is not None:
            merged_pair.deltas = _select_top_deltas(merged_pair.deltas, max_deltas_per_pair)
            year_pairs.append(merged_pair)

    kept_ids = {
        cid
        for p in year_pairs
        for d in p.deltas
        for cid in (d.prior_chunk_id, d.current_chunk_id)
        if cid
    }
    evolution = CompanyDisclosureEvolution(
        ticker=ticker,
        company_name=company_name,
        gics_sector=gics_sector,
        fiscal_years_available=years,
        missing_fiscal_years=[y for y in window_years if y not in years],
        theme_mentions=mentions,
        year_pairs=year_pairs,
    )
    return evolution, [c for cid, c in citations.items() if cid in kept_ids]


# ---------------------------------------------------------------------------
# Data access (live GCP) + ADK tool
# ---------------------------------------------------------------------------


def _load_section_material(
    ticker: str, company_name: str, fiscal_year: int, section_id: str
) -> SectionMaterial | None:
    """Load full 10-K section markdown from GCS when BigQuery had zero chunks for the slice."""
    norm_ticker = normalize_edgar_ticker(ticker)
    entry = get_full_10k_section(f"{norm_ticker}_{fiscal_year}_{section_id}")
    if entry is None:
        return None
    return SectionMaterial(
        ticker=norm_ticker,
        company_name=str(entry.get("company_name") or company_name),
        fiscal_year=int(fiscal_year),
        section_id=section_id,
        full_text=str(entry.get("full_text") or ""),
        chunks=[],
    )


def _company_years(
    tickers: list[str], start_year: int, end_year: int
) -> dict[str, tuple[str, str, list[int]]]:
    """ticker -> (company_name, gics_sector, fiscal years with a filing in the window)."""
    wanted = {normalize_edgar_ticker(t) for t in tickers}
    cached = cached_turn_profiles()
    if cached:
        profiles = [
            p
            for p in cached
            if normalize_edgar_ticker(p.ticker) in wanted
            and start_year <= int(p.fiscal_year) <= end_year
        ]
    else:
        profiles = _gcp_adapters().bigquery.fetch_profiles(
            tickers=wanted, start_year=start_year, end_year=end_year
        )
    out: dict[str, tuple[str, str, list[int]]] = {}
    for p in profiles:
        norm = normalize_edgar_ticker(p.ticker)
        if norm not in wanted:
            continue
        name, sector, years = out.get(norm, (p.company_name, p.gics_sector, []))
        years.append(p.fiscal_year)
        out[norm] = (p.company_name or name, p.gics_sector or sector, sorted(set(years)))
    return out


def _resolve_target_tickers(
    *,
    theme: str,
    tickers: str,
    gics_sector: str,
    sec_filter: str,
    start_year: int,
    end_year: int,
    max_companies: int,
) -> tuple[list[str], list[str], str]:
    explicit = sorted(_parse_ticker_list(tickers))
    if explicit:
        return explicit, explicit[:max_companies], "explicit"
    broad = search_10k_chunks_rag(
        query=theme,
        tickers="ALL",
        gics_sector=gics_sector or "ALL",
        section_id=sec_filter,
        start_year=start_year,
        end_year=end_year,
        top_k=25,
    )
    selected: list[str] = []
    for cit in broad.get("citations") or []:
        sym = normalize_edgar_ticker(str(cit["ticker"]))
        if sym not in selected:
            selected.append(sym)
        if len(selected) >= max_companies:
            break
    return explicit, selected, "theme_relevance"


def _retrieve_passages_and_materials(
    *,
    theme: str,
    selected: list[str],
    company_meta: dict[str, tuple[str, str, list[int]]],
    sec_filter: str,
    per_year_k: int,
) -> tuple[
    dict[tuple[str, int], list[dict[str, Any]]],
    dict[tuple[str, int, str], SectionMaterial | None],
]:
    """Retrieve theme passages and section chunks in parallel with a single batched BQ section query.

    Replaces the prior per-section `get_full_10k_section` (BQ + GCS) + per-section BQ chunk fan-out
    with one batched `fetch_sections_chunks_batch` RPC running concurrently alongside passage search.
    Falls back to `get_full_10k_section` only for a slice with zero BigQuery chunks.
    """
    retrieval_keys = [(t, fy) for t in selected for fy in company_meta.get(t, ("", "", []))[2]]
    all_years = [fy for _, fy in retrieval_keys]
    batch_tickers = {normalize_edgar_ticker(t) for t in selected}
    batch_section = sec_filter if sec_filter in _SECTIONS else None

    tasks: list[tuple[str, Any]] = [("search", key) for key in retrieval_keys]
    if batch_tickers and all_years:
        tasks.append(("batch_sections", (batch_tickers, min(all_years), max(all_years))))

    def _run_task(task: tuple[str, Any]) -> tuple[str, Any, Any]:
        kind, payload = task
        if kind == "search":
            tk, fy = payload
            res = search_10k_chunks_rag(
                query=theme,
                tickers=tk,
                gics_sector="ALL",
                section_id=sec_filter,
                start_year=fy,
                end_year=fy,
                top_k=per_year_k,
            )
            return kind, payload, list(res.get("citations") or [])
        tks, min_y, max_y = payload
        rows = _gcp_adapters().bigquery.fetch_sections_chunks_batch(
            tickers=tks,
            start_year=min_y,
            end_year=max_y,
            section_id=batch_section,
            include_embedding=False,
        )
        return kind, payload, rows

    completed = run_parallel_with_context(tasks, _run_task, max_workers=8)
    passages: dict[tuple[str, int], list[dict[str, Any]]] = {}
    chunks_by_slice: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for kind, payload, value in completed:
        if kind == "search":
            passages[payload] = value
        else:
            for row in value:
                row_tk = normalize_edgar_ticker(str(row.get("ticker") or ""))
                row_fy = int(row["fiscal_year"])
                row_sec = str(row.get("section_id") or "").strip().lower()
                if row_tk and row_sec in _SECTIONS:
                    chunks_by_slice[(row_tk, row_fy, row_sec)].append(row)

    section_keys = sorted(
        {
            (t, fy, str(c.get("section_id")))
            for (t, fy), cits in passages.items()
            for c in cits
            if c.get("section_id") in _SECTIONS
        }
    )
    expanded: set[tuple[str, int, str]] = set(section_keys)
    for t, _fy, sec in section_keys:
        for y in company_meta.get(t, ("", "", []))[2]:
            expanded.add((t, y, sec))
    material_keys = sorted(expanded)

    materials: dict[tuple[str, int, str], SectionMaterial | None] = {}
    missing_keys: list[tuple[str, int, str]] = []
    for tk, fy, sec in material_keys:
        norm_tk = normalize_edgar_ticker(tk)
        sec_chunks = chunks_by_slice.get((norm_tk, int(fy), sec.lower()), [])
        if sec_chunks:
            company_name = str(
                sec_chunks[0].get("company_name") or company_meta.get(tk, (tk, "", []))[0]
            )
            materials[(tk, fy, sec)] = SectionMaterial(
                ticker=norm_tk,
                company_name=company_name,
                fiscal_year=int(fy),
                section_id=sec,
                full_text="",
                chunks=sec_chunks,
            )
        else:
            missing_keys.append((tk, fy, sec))

    if missing_keys:

        def _load_fallback(item: tuple[str, int, str]) -> SectionMaterial | None:
            tk, fy, sec = item
            return _load_section_material(tk, company_meta.get(tk, (tk, "", []))[0], fy, sec)

        fallback_list = run_parallel_with_context(missing_keys, _load_fallback, max_workers=8)
        for key, mat in zip(missing_keys, fallback_list, strict=True):
            materials[key] = mat

    return passages, materials


@with_tool_resilience(tool_name="compare_disclosures_across_years")
def compare_disclosures_across_years(
    theme: str,
    tickers: str,
    gics_sector: str,
    section_id: str,
    start_year: int,
    end_year: int,
    theme_terms: str = "",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Compares how companies' 10-K wording on a theme changed between consecutive fiscal years.

    For each company and year, retrieves the passages most relevant to the theme, aligns them with
    the adjacent year's filing, and classifies each as 'added', 'removed', 'modified' (with the
    exact added/removed phrases), or 'unchanged'. Also counts theme-term mentions per year. Each
    delta carries the prior-year and/or current-year chunk_id to cite; cite both for 'modified'.

    Args:
        theme: Theme to track (e.g., 'AI risk', 'export controls China', 'cybersecurity').
        tickers: Comma-separated tickers (e.g., 'NVDA,AMD'), or 'ALL' to pick the companies most
            relevant to the theme within `gics_sector`.
        gics_sector: GICS Sector filter used when tickers is 'ALL' (e.g., 'Information Technology'), or 'ALL'.
        section_id: 'item_1a' (Risk Factors), 'item_7' (MD&A), or 'ALL'.
        start_year: First fiscal year to compare (e.g., 2024).
        end_year: Last fiscal year to compare (e.g., 2026).
        theme_terms: Optional comma-separated key phrases and domain synonyms to count across the
            compared filings (e.g., 'AI, artificial intelligence, generative AI, GenAI' or
            'export controls, BIS, Entity List'). Omit to fall back to tokens from `theme`.

    Returns:
        Dictionary with 'status', 'companies' (year pairs with deltas and counts, theme mentions,
        missing years), 'summary' counts, and 'citations' for every cited chunk.
    """
    if not (theme or "").strip():
        return {"status": "error", "message": "Please provide a theme to track."}
    start_year, end_year = sorted((int(start_year), int(end_year)))
    if start_year == end_year:
        return {
            "status": "error",
            "message": "Disclosure comparison needs at least two fiscal years (start_year < end_year).",
        }
    sec_norm = (section_id or "ALL").strip().lower()
    sec_filter = sec_norm if sec_norm in _SECTIONS else "ALL"
    max_companies = env_positive_int("EDGAR_DISCLOSURE_DELTA_MAX_COMPANIES", DEFAULT_MAX_COMPANIES)
    per_year_k = env_positive_int(
        "EDGAR_DISCLOSURE_DELTA_PASSAGES_PER_YEAR", DEFAULT_PASSAGES_PER_YEAR
    )

    explicit, selected, selected_by = _resolve_target_tickers(
        theme=theme,
        tickers=tickers,
        gics_sector=gics_sector,
        sec_filter=sec_filter,
        start_year=start_year,
        end_year=end_year,
        max_companies=max_companies,
    )

    window = list(range(start_year, end_year + 1))
    company_meta = _company_years(selected, start_year, end_year)
    resolved_terms = resolve_theme_terms(theme, theme_terms)
    passages, materials = _retrieve_passages_and_materials(
        theme=theme,
        selected=selected,
        company_meta=company_meta,
        sec_filter=sec_filter,
        per_year_k=per_year_k,
    )

    companies: list[dict[str, Any]] = []
    all_citations: dict[str, dict[str, Any]] = {}
    summary = {"added": 0, "removed": 0, "modified": 0, "unchanged": 0, "year_pairs": 0}
    for t in selected:
        name, sector, years = company_meta.get(t, (t, "Unknown", []))
        sections = {
            (fy, sec): m for (tk, fy, sec), m in materials.items() if tk == t and m is not None
        }
        evolution, cited = build_company_evolution(
            ticker=t,
            company_name=name,
            gics_sector=sector,
            window_years=window,
            available_years=years,
            theme_terms=resolved_terms,
            sections=sections,
            theme_passages={fy: passages.get((t, fy), []) for fy in years},
        )
        for cit in cited:
            all_citations.setdefault(str(cit["chunk_id"]), cit)
        for pair in evolution.year_pairs:
            summary["year_pairs"] += 1
            for kind in ("added", "removed", "modified", "unchanged"):
                summary[kind] += pair.counts.get(kind, 0)
        companies.append(evolution.model_dump(mode="json"))

    summary["companies_analyzed"] = len(companies)
    citations = list(all_citations.values())
    result: dict[str, Any] = {
        "status": "success",
        "theme": theme.strip(),
        "theme_terms": resolved_terms,
        "section_id": sec_filter,
        "start_year": start_year,
        "end_year": end_year,
        "companies_selected_by": selected_by,
        "missing_tickers": [t for t in explicit if t not in company_meta],
        "summary": summary,
        "companies": companies,
        "citations": citations,
    }
    if tool_context is not None and hasattr(tool_context, "state"):
        with _STATE_MERGE_LOCK:
            existing = list(tool_context.state.get(STATE_RETRIEVED_CITATIONS) or [])
            tool_context.state[STATE_RETRIEVED_CITATIONS] = merge_citations(existing, citations)
            # Full payload (not just the summary) so the turn's DisclosureDeltaCard is hydrated
            # from session_state; repeated calls in one turn merge per ticker and year pair.
            merged = merge_disclosure_deltas(
                tool_context.state.get(STATE_DISCLOSURE_DELTAS), result
            )
            tool_context.state[STATE_DISCLOSURE_DELTAS] = merged
            tool_context.state[STATE_DISCLOSURE_DELTA_SUMMARY] = (merged or result)["summary"]
    if tool_context is not None and hasattr(tool_context, "state"):
        return compact_disclosure_deltas_for_tool(result)
    return result


DisclosureDeltaTool = FunctionTool(compare_disclosures_across_years)
