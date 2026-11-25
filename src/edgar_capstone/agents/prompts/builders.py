"""Prompt formatting, payload compaction, and judge citation selection helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from edgar_capstone.common.domain import CORE_METRICS

# Evidence lead budget: generous enough that the analyst can extract verbatim quotes directly
# without burning extra ReAct tool round-trips for passages already retrieved (or truncating
# mid-chunk subsections / tables), while keeping prompt prefill lean for sub-3s TTFT.
_EVIDENCE_LEAD_EXCERPT_CHARS = 1200
_MAX_EVIDENCE_LEADS = 12
_REPAIR_EVIDENCE_LEAD_EXCERPT_CHARS = 1600
_REPAIR_MAX_EVIDENCE_LEADS = 18
_LEAD_HEADER_RE = re.compile(r"^\[[^\]]{1,180}\]\s*")
_LEAD_WORD_RE = re.compile(r"[a-z0-9]+&[a-z0-9]+|[a-z0-9]{2,}")
_LEAD_YEAR_RE = re.compile(r"^(?:19\d{2}|20\d{2})$")
_LEAD_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "between",
        "by",
        "compare",
        "compared",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "over",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "through",
        "to",
        "vs",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)
_LEAD_ACRONYM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "r&d": ("r&d", "research", "development"),
    "sg&a": ("sg&a", "selling", "administrative"),
    "md&a": ("md&a",),
}
_FEEDBACK_CHUNK_ID_RE = re.compile(
    r"\b[A-Z][A-Z0-9.\-]{0,6}_\d{4}_(?:item_1a|item_7)_(?:vais_)?\d+\b"
)
_MAX_PROMPT_FLAGS = 8
_MAX_PROMPT_DELTAS_PER_PAIR = 4

_JUDGE_MIN_CITATIONS = 16
_JUDGE_MAX_CITATIONS = 30
_JUDGE_EXCERPT_CHARS = 4000

_DROP_MV_KEYS = frozenset({"start_citation", "end_citation", "derived_from", "currency"})
_DROP_REPORT_KEYS = frozenset(
    {
        "derived_margin_inputs",
        "metric_provenance",
        "source_urls",
        "filing_dates",
        "cik",
        "currency",
    }
)
_DROP_ANNUAL_STEP_KEYS = frozenset({"citations", "sec_filing_url", "filing_date"})


def _compact_json(obj: Any) -> str:
    """Serialize JSON compactly (dropping indentation whitespace to cut prefill latency)."""
    return json.dumps(obj, separators=(", ", ": "), default=str)


def compact_variance_reports_for_prompt(
    variance_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strip repetitive nested XBRL citation metadata from the LLM prompt copy.

    Full `NumericCitation` structures (`start_citation`, `end_citation`, `derived_from`,
    `derived_margin_inputs`, `metric_provenance`, `source_urls`) are ~18 KB/company and are
    rendered deterministically by `render_analyst_report_markdown()`; keeping only compact
    numbers, dates, and margins cuts prompt prefill tokens by >10x without losing any fact.
    """
    compact_list: list[dict[str, Any]] = []
    for rep in variance_reports:
        if not isinstance(rep, dict):
            continue
        slim: dict[str, Any] = {k: v for k, v in rep.items() if k not in _DROP_REPORT_KEYS}
        for metric in CORE_METRICS:
            mv_key = f"{metric}_variance"
            mv = slim.get(mv_key)
            if isinstance(mv, dict):
                slim[mv_key] = {
                    mk: mv_val
                    for mk, mv_val in mv.items()
                    if mk not in _DROP_MV_KEYS and mv_val is not None
                }
        series = slim.get("annual_series")
        if isinstance(series, list):
            slim["annual_series"] = [
                {
                    sk: sv
                    for sk, sv in step.items()
                    if sk not in _DROP_ANNUAL_STEP_KEYS and sv is not None
                }
                for step in series
                if isinstance(step, dict)
            ]
        compact_list.append(slim)
    return compact_list


def _deduplicate_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen_all: set[str] = set()
    for c in citations:
        cid = str(c.get("chunk_id") or "").strip()
        if not cid or cid in seen_all:
            continue
        seen_all.add(cid)
        unique.append(c)
    return unique


def _parse_scope_years(scope: dict[str, Any] | None) -> list[int]:
    if not scope:
        return []
    years: list[int] = []
    for key in ("start_year", "end_year"):
        raw = scope.get(key)
        if raw is None:
            continue
        try:
            years.append(int(raw))
        except (TypeError, ValueError):
            return []
    return list(dict.fromkeys(years))


def _group_citations_by_ticker_year(
    citations: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], list[str]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    ticker_order: list[str] = []
    for c in citations:
        tk = str(c.get("ticker") or "").upper()
        try:
            fy = int(c.get("fiscal_year") or 0)
        except (TypeError, ValueError):
            fy = 0
        if tk and tk not in ticker_order:
            ticker_order.append(tk)
        grouped.setdefault((tk, fy), []).append(c)
    return grouped, ticker_order


def _extract_lead_query_terms(query: str) -> set[str]:
    """Extract domain-relevant query terms for centering evidence lead excerpts."""
    raw = _LEAD_WORD_RE.findall((query or "").lower())
    terms: set[str] = set()
    for tok in raw:
        if tok in _LEAD_ACRONYM_EXPANSIONS:
            terms.update(_LEAD_ACRONYM_EXPANSIONS[tok])
            continue
        if tok in _LEAD_STOPWORDS or _LEAD_YEAR_RE.match(tok):
            continue
        terms.add(tok)
    return terms


def _extract_query_centered_excerpt(text: str, query: str, max_chars: int) -> str:
    """Return a whitespace-collapsed excerpt up to `max_chars`, centered on query term matches.

    When a chunk fits inside `max_chars`, it is returned untouched. Otherwise, any leading
    bracketed chunk header (`[TICKER ...]`) is preserved and the body window is centered on
    the highest query-term-density region so mid-chunk subsections (e.g. R&D after SG&A) are
    not hidden behind an unrelated prefix. Every returned span remains verbatim in the chunk.
    """
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= max_chars:
        return collapsed
    terms = _extract_lead_query_terms(query)
    if not terms:
        return collapsed[:max_chars]

    header_match = _LEAD_HEADER_RE.match(collapsed)
    header = header_match.group(0) if header_match else ""
    body = collapsed[len(header) :]
    body_budget = max_chars - len(header)
    if body_budget < 120:
        return collapsed[:max_chars]

    body_lower = body.lower()
    hit_positions: list[int] = []
    for term in terms:
        pos = body_lower.find(term)
        while pos != -1:
            hit_positions.append(pos)
            pos = body_lower.find(term, pos + len(term))
    if not hit_positions:
        return collapsed[:max_chars]

    candidate_starts = [0]
    for pos in sorted(set(hit_positions)):
        start = max(0, pos - 120)
        if start > 0:
            ws = body.find(" ", start)
            if ws != -1 and ws < pos:
                start = ws + 1
        if start not in candidate_starts:
            candidate_starts.append(start)

    best_start = 0
    best_score: tuple[int, int, int] = (-1, -1, 0)
    for start in candidate_starts:
        window = body_lower[start : start + body_budget]
        distinct = sum(1 for term in terms if term in window)
        total = sum(window.count(term) for term in terms)
        score = (distinct, total, -start)
        if score > best_score:
            best_score = score
            best_start = start

    if best_start == 0:
        return f"{header}{body[:body_budget]}"
    prefix = "... "
    snippet = body[best_start : best_start + max(0, body_budget - len(prefix))]
    return f"{header}{prefix}{snippet}"


def compact_evidence_leads(
    citations: list[dict[str, Any]],
    *,
    scope: dict[str, Any] | None = None,
    query: str = "",
    priority_chunk_ids: set[str] | None = None,
    max_leads: int = _MAX_EVIDENCE_LEADS,
    excerpt_chars: int = _EVIDENCE_LEAD_EXCERPT_CHARS,
) -> list[dict[str, Any]]:
    """Citable passages already retrieved during the deterministic pre-pass.

    Uses boundary-first round-robin selection so every `(ticker, start_year)` and
    `(ticker, end_year)` present in `citations` is guaranteed representation before filling
    remaining lead slots in retrieval score order. When `priority_chunk_ids` is given (e.g.
    during a tool-less verification repair pass), those chunks are included first.
    """
    unique = _deduplicate_citations(citations)
    boundary_years = _parse_scope_years(scope)
    grouped, ticker_order = _group_citations_by_ticker_year(unique)

    ordered: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()

    def _pick(c: dict[str, Any]) -> None:
        cid = str(c.get("chunk_id") or "").strip()
        if cid and cid not in chosen_ids and len(ordered) < max_leads:
            chosen_ids.add(cid)
            ordered.append(c)

    # Pass 0: Guarantee chunks explicitly referenced by a verification-repair pass
    if priority_chunk_ids:
        for c in unique:
            if len(ordered) >= max_leads:
                break
            if str(c.get("chunk_id") or "").strip() in priority_chunk_ids:
                _pick(c)

    # Pass 1: Guarantee at least 1 boundary-year passage per (ticker, start_yr) & (ticker, end_yr)
    for tk in ticker_order:
        for by in boundary_years:
            bucket = grouped.get((tk, by)) or []
            if bucket:
                sec_pref = next((b for b in bucket if b.get("section_id") == "item_7"), bucket[0])
                _pick(sec_pref)

    # Pass 2: One passage from any remaining (ticker, fiscal_year) bucket (round-robin)
    for bucket in grouped.values():
        if len(ordered) >= max_leads:
            break
        first_unpicked = next(
            (c for c in bucket if str(c.get("chunk_id") or "").strip() not in chosen_ids),
            None,
        )
        if first_unpicked is not None:
            _pick(first_unpicked)

    # Pass 3: Fill remaining slots up to max_leads in original retrieval order
    for c in unique:
        if len(ordered) >= max_leads:
            break
        _pick(c)

    leads: list[dict[str, Any]] = [
        {
            "chunk_id": str(c.get("chunk_id") or "").strip(),
            "ticker": c.get("ticker"),
            "fiscal_year": c.get("fiscal_year"),
            "period_of_report": c.get("period_of_report"),
            "section_id": c.get("section_id"),
            "section_title": c.get("section_title"),
            "excerpt": _extract_query_centered_excerpt(
                str(c.get("excerpt") or ""), query, excerpt_chars
            ),
        }
        for c in ordered
    ]
    return leads


def _scope_structure_label(scope: dict[str, Any]) -> str:
    tickers = [t for t in str(scope.get("tickers_csv") or "ALL").split(",") if t and t != "ALL"]
    sector = str(scope.get("gics_sector") or "ALL")
    if not tickers and sector not in ("ALL", "", "*"):
        return f"sector-wide (no companies named; GICS sector `{sector}`)"
    if len(tickers) > 1:
        return f"multi-company ({len(tickers)} companies)"
    if len(tickers) == 1:
        return "single company"
    return "corpus-wide (no company or sector filter)"


def build_tool_selection_policy(scope: dict[str, Any], *, has_prior_outputs: bool = False) -> str:
    """Turn-specific scope parameters for the tool-selection policy defined in the system prompt."""
    tickers_csv = scope.get("tickers_csv", "ALL")
    years = f"start_year={scope.get('start_year')}, end_year={scope.get('end_year')}"
    status_line = (
        "Deterministic outputs already produced this turn are included above; do not re-invoke those tools."
        if has_prior_outputs
        else "No deterministic outputs were pre-computed; invoke only the engines this question requires."
    )
    return "\n".join(
        [
            "## Turn Scope & Tool Parameters",
            status_line,
            f"- Scope structure: {_scope_structure_label(scope)}.",
            f"- Default tool scope: `tickers={tickers_csv}`, `gics_sector={scope.get('gics_sector', 'ALL')}`, "
            f"`section_id={scope.get('section_id', 'ALL')}`, `{years}`.",
        ]
    )


def _citation_coverage_line(scope: dict[str, Any]) -> str | None:
    """Scope-specific citation-coverage requirement (no extra model call or re-prompt)."""
    tickers = [str(t) for t in (scope.get("tickers") or []) if t]
    years = _parse_scope_years(scope)
    multi_company = 2 <= len(tickers) <= 6
    multi_year = len(years) == 2 and years[0] != years[1]
    if not (multi_company or multi_year):
        return None
    parts: list[str] = []
    if multi_company:
        parts.append(f"at least one cited finding for each of {', '.join(tickers)}")
    if multi_year:
        yr_clause = f"findings citing both the FY{years[0]} and the FY{years[1]} filings"
        if multi_company:
            yr_clause += f" (including each company's FY{years[1]} filing)"
        if abs(years[1] - years[0]) >= 2:
            yr_clause += (
                ", plus any intermediate fiscal year where `annual_series` shows a major "
                "spike, drop, or reversal in the requested metric"
            )
        parts.append(yr_clause)
    return (
        "- Citation coverage for this question: "
        + " and ".join(parts)
        + " (use the matching Evidence Leads; skip a company-year only if nothing covers it)."
    )


def build_retrieval_instructions(
    scope: dict[str, Any],
    variance_evaluations: list[dict[str, Any]],
) -> str:
    """Concise turn-specific retrieval & citation-coverage guidance (rules live in system prompt)."""
    suggested: list[dict[str, Any]] = [
        {
            "ticker": tq.get("ticker"),
            "fiscal_year": tq.get("fiscal_year"),
            "section_id": tq.get("section_id"),
            "driver_focus": tq.get("driver_focus"),
        }
        for ev in variance_evaluations
        for tq in (ev.get("targeted_10k_queries") or [])
    ]
    lines = [
        "## Evidence Coverage",
        "Synthesize directly from the Evidence Leads and tool outputs above if they cover the question; "
        "call retrieval tools only for missing company-year or section gaps.",
    ]
    coverage_line = _citation_coverage_line(scope)
    if coverage_line:
        lines.append(coverage_line)
    if suggested:
        lines.append(
            "- Suggested retrieval targets:\n" f"```json\n{_compact_json(suggested[:8])}\n```"
        )
    return "\n".join(lines)


def _compact_flag_base(flag: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the shared lean field projection and raw explanation dict for one material flag."""
    src = flag.get("source") or {}
    exp = flag.get("explanation") or {}
    entry: dict[str, Any] = {
        "flag_id": flag.get("flag_id"),
        "ticker": flag.get("ticker"),
        "fiscal_year": flag.get("fiscal_year"),
        "prior_fiscal_year": flag.get("prior_fiscal_year"),
        "metric": flag.get("metric"),
        "rule": flag.get("rule"),
        "severity": flag.get("severity"),
        "value": flag.get("value"),
        "unit": flag.get("unit"),
        "description": flag.get("description"),
        "period_of_report": src.get("period_of_report"),
        "prior_period_of_report": src.get("prior_period_of_report"),
        "accession_number": src.get("accession_number"),
        "xbrl_concept": src.get("xbrl_concept"),
        "explanation_query": flag.get("explanation_query"),
    }
    return entry, exp


def _compact_material_flag_entries(flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact material variance flags to lean LLM-ready dicts (full payloads stay in session_state)."""
    compact: list[dict[str, Any]] = []
    for f in flags[:_MAX_PROMPT_FLAGS]:
        entry, exp = _compact_flag_base(f)
        entry["explanation_chunk_id"] = exp.get("chunk_id")
        entry["explanation_excerpt"] = (
            " ".join(str(exp.get("excerpt") or "").split())[:600] if exp.get("excerpt") else None
        )
        compact.append({k: v for k, v in entry.items() if v is not None})
    return compact


def compact_material_flags_for_tool(result: dict[str, Any]) -> dict[str, Any]:
    """Return a token-lean `flag_material_variances` tool digest for the LLM context.

    The full flags list, thresholds, and `explanation_citations` remain in ADK `session_state`
    for UI cards, claim verification, and citation rendering.
    """
    if result.get("status") != "success":
        return result
    compact_flags: list[dict[str, Any]] = []
    for f in list(result.get("flags") or [])[:_MAX_PROMPT_FLAGS]:
        entry, exp = _compact_flag_base(f)
        if exp:
            entry["explanation"] = {
                k: v
                for k, v in {
                    "chunk_id": exp.get("chunk_id"),
                    "section_id": exp.get("section_id"),
                    "excerpt": " ".join(str(exp.get("excerpt") or "").split())[:600] or None,
                }.items()
                if v is not None
            }
        compact_flags.append({k: v for k, v in entry.items() if v is not None})
    payload: dict[str, Any] = {
        "status": "success",
        "total_flags": result.get("total_flags", len(compact_flags)),
        "returned_flags": len(compact_flags),
        "high_severity_count": result.get("high_severity_count", 0),
        "missing_tickers": list(result.get("missing_tickers") or []),
        "flags": compact_flags,
    }
    if result.get("summary") is not None:
        payload["summary"] = {
            k: v for k, v in result["summary"].items() if k != "thresholds" and v is not None
        }
    return payload


def build_material_flags_prompt_section(flags: list[dict[str, Any]] | None) -> str:
    """Compact, authoritative anomaly flags for the analyst (including XBRL provenance & excerpt)."""
    if not flags:
        return ""
    compact = _compact_material_flag_entries(flags)
    return (
        "## Authoritative Material Variance Flags (deterministic anomaly detection)\n"
        "(These anomalies were detected by fixed rules; never recompute or re-threshold them. "
        "Explain the high-severity flags first with MD&A evidence: use `explanation_chunk_id` and "
        "`explanation_excerpt` when present, otherwise call `query_cached_10k_context` with the "
        "flag's `explanation_query`.)\n"
        f"```json\n{_compact_json(compact)}\n```"
    )


def _compact_disclosure_delta_companies(
    companies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compact_companies: list[dict[str, Any]] = []
    for comp in companies:
        pairs = []
        for pair in comp.get("year_pairs") or []:
            substantive = [
                d for d in (pair.get("deltas") or []) if d.get("change_type") != "unchanged"
            ]
            # Round-robin across substantive change types so added/removed risk factors are not
            # hidden behind several modified paragraphs in the LLM tool digest.
            by_kind: dict[str, list[dict[str, Any]]] = {"modified": [], "added": [], "removed": []}
            for d in substantive:
                kind = str(d.get("change_type") or "")
                by_kind.setdefault(kind, []).append(d)
            ordered: list[dict[str, Any]] = []
            max_bucket = max((len(v) for v in by_kind.values()), default=0)
            for slot in range(max_bucket):
                for kind in ("modified", "added", "removed"):
                    bucket = by_kind.get(kind) or []
                    if slot < len(bucket):
                        ordered.append(bucket[slot])
                        if len(ordered) >= _MAX_PROMPT_DELTAS_PER_PAIR:
                            break
                if len(ordered) >= _MAX_PROMPT_DELTAS_PER_PAIR:
                    break

            compact_deltas = []
            for d in ordered:
                delta_entry: dict[str, Any] = {
                    "change_type": d["change_type"],
                    "section_id": d.get("section_id"),
                    "prior_chunk_id": d.get("prior_chunk_id"),
                    "current_chunk_id": d.get("current_chunk_id"),
                    "added_phrases": d.get("added_phrases") or [],
                    "removed_phrases": d.get("removed_phrases") or [],
                    "excerpt": " ".join(
                        str(d.get("current_excerpt") or d.get("prior_excerpt") or "").split()
                    )[:_EVIDENCE_LEAD_EXCERPT_CHARS],
                }
                if d["change_type"] == "modified" and d.get("prior_excerpt"):
                    delta_entry["prior_excerpt"] = " ".join(str(d["prior_excerpt"]).split())[:350]
                compact_deltas.append(delta_entry)
            pairs.append(
                {
                    "years": f"FY{pair['prior_fiscal_year']}->FY{pair['current_fiscal_year']}",
                    "counts": pair.get("counts"),
                    "deltas": compact_deltas,
                }
            )
        compact_companies.append(
            {
                "ticker": comp["ticker"],
                "theme_mentions": comp.get("theme_mentions") or {},
                "missing_fiscal_years": comp.get("missing_fiscal_years") or [],
                "year_pairs": pairs,
            }
        )
    return compact_companies


def compact_disclosure_deltas_for_tool(result: dict[str, Any]) -> dict[str, Any]:
    """Return a token-lean `compare_disclosures_across_years` tool digest for the LLM context.

    The full disclosure delta payload and `citations` remain in ADK `session_state` for UI
    rendering, claim verification, and citation tables.
    """
    if result.get("status") != "success":
        return result
    return {
        "status": "success",
        "theme": result.get("theme"),
        "theme_terms": result.get("theme_terms") or [],
        "section_id": result.get("section_id"),
        "start_year": result.get("start_year"),
        "end_year": result.get("end_year"),
        "companies_selected_by": result.get("companies_selected_by"),
        "missing_tickers": list(result.get("missing_tickers") or []),
        "summary": result.get("summary") or {},
        "companies": _compact_disclosure_delta_companies(list(result.get("companies") or [])),
    }


def build_disclosure_deltas_prompt_section(result: dict[str, Any] | None) -> str:
    """Compact year-over-year disclosure deltas (theme wording changes) for the analyst."""
    if not result or result.get("status") != "success" or not result.get("companies"):
        return ""
    payload = {
        "theme": result.get("theme"),
        "summary": result.get("summary"),
        "companies": _compact_disclosure_delta_companies(list(result["companies"])),
    }
    return (
        "## Authoritative Year-over-Year Disclosure Deltas (deterministic wording comparison)\n"
        "(Each delta compares a theme passage with the adjacent year's filing. Summarize the trend "
        "from these; for a `modified` delta cite BOTH `prior_chunk_id` and `current_chunk_id` in the "
        "finding's chunk_ids and quote the current-year chunk; for `removed` quote the prior-year "
        "chunk. Report missing fiscal years as data gaps.)\n"
        f"```json\n{_compact_json(payload)}\n```"
    )


def _sector_map_prompt_section(
    sector_map_summaries: dict[str, Any],
    disclosure_deltas: dict[str, Any] | None,
) -> str:
    compact_batches = [
        {
            "tickers_covered": b.get("tickers_covered"),
            "dominant_themes": b.get("dominant_themes"),
            "wording_shifts": b.get("wording_shifts"),
            "evidence": [
                {
                    "ticker": ev.get("ticker"),
                    "fiscal_year": ev.get("fiscal_year"),
                    "chunk_id": ev.get("chunk_id"),
                    "verbatim_quote": ev.get("verbatim_quote"),
                    "takeaway": ev.get("takeaway"),
                }
                for ev in (b.get("evidence") or [])
            ],
        }
        for b in sector_map_summaries.get("batch_summaries") or []
    ]
    compact_sector_payload = {
        "sector_coverage": sector_map_summaries.get("sector_coverage"),
        "delta_summary": (disclosure_deltas or {}).get("summary"),
        "batch_summaries": compact_batches,
    }
    return (
        "## Sector-Wide Map-Reduce Batch Summaries (covering all matching issuers in the sector)\n"
        "(Synthesize sector-wide trends from these verified batch summaries. Every claim in "
        "`findings` MUST cite one of the real company `chunk_id`s provided below or in evidence leads "
        "with a verbatim `evidence_quote`.)\n"
        f"```json\n{_compact_json(compact_sector_payload)}\n```"
    )


def build_financial_reasoning_prompt(
    *,
    query: str,
    intent: str,
    scope: dict[str, Any],
    variance_reports: list[dict[str, Any]],
    variance_evaluations: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    sector_benchmarks: list[dict[str, Any]],
    chunk_health: dict[str, Any] | None,
    memory_summary_text: str = "",
    material_variance_flags: list[dict[str, Any]] | None = None,
    disclosure_deltas: dict[str, Any] | None = None,
    coverage_notices: list[dict[str, Any]] | None = None,
    sector_map_summaries: dict[str, Any] | None = None,
    verification_feedback: str | None = None,
) -> str:
    """Assemble the deterministic facts and retrieval brief for `FinancialAnalystAgent`."""
    sub_industry_line = (
        f"\n- GICS Sub-Industry: `{scope['gics_sub_industry']}`"
        if scope.get("gics_sub_industry")
        else ""
    )
    sections: list[str] = [
        f"## Analyst Query\n{query.strip()}",
        (
            f"## Resolved Scope\n"
            f"- Scope structure: {_scope_structure_label(scope)} (provisional routing "
            f"hint: `{intent}`; classify the final intent yourself in `AnalystReport.intent`)\n"
            f"- Tickers: `{scope.get('tickers_csv', 'ALL')}`\n"
            f"- GICS Sector: `{scope.get('gics_sector', 'ALL')}`"
            f"{sub_industry_line}\n"
            f"- Section Filter: `{scope.get('section_id', 'ALL')}`\n"
            f"- Fiscal Period: `FY{scope.get('start_year')}–FY{scope.get('end_year')}`"
        ),
    ]
    if coverage_notices:
        notice_lines = [
            f"- [{n.get('severity', 'info').upper()}] {n.get('message')}"
            for n in coverage_notices
            if n.get("message")
        ]
        if notice_lines:
            sections.append(
                "## System Coverage & Fiscal Calendar Notices (already rendered verbatim)\n"
                + "\n".join(notice_lines)
            )
    if memory_summary_text.strip():
        sections.append(f"## Recalled Conversational Session Memory\n{memory_summary_text.strip()}")
    if variance_reports:
        compact_vr = compact_variance_reports_for_prompt(variance_reports)
        sections.append(
            "## Authoritative Deterministic Period-over-Period Variance Calculations\n"
            "(Use these exact USD figures, percentage deltas, CAGRs, margin bps shifts, and "
            "`period_end_dates` — never alter or recompute them. When comparing peers whose fiscal "
            "years end in different months, acknowledge the fiscal period-end dates clearly.)\n"
            f"```json\n{_compact_json(compact_vr)}\n```"
        )
    if variance_evaluations:
        compact_evals = [
            {k: v for k, v in ev.items() if k != "targeted_10k_queries"}
            for ev in variance_evaluations
        ]
        sections.append(
            "## Pre-Evaluated ReAct Variance Movers & Operating Leverage Diagnostics\n"
            f"```json\n{_compact_json(compact_evals)}\n```"
        )
    if sector_benchmarks:
        sections.append(
            "## Authoritative Sector Quantitative Benchmarks\n"
            f"```json\n{_compact_json(sector_benchmarks[:5])}\n```"
        )
    if chunk_health:
        sections.append(
            "## Vector Corpus & Chunk Health Diagnostics\n"
            f"```json\n{_compact_json(chunk_health)}\n```"
        )
    flags_section = build_material_flags_prompt_section(material_variance_flags)
    if flags_section:
        sections.append(flags_section)

    has_sector_map = bool(sector_map_summaries and sector_map_summaries.get("batch_summaries"))
    if has_sector_map and sector_map_summaries is not None:
        sections.append(_sector_map_prompt_section(sector_map_summaries, disclosure_deltas))
    else:
        deltas_section = build_disclosure_deltas_prompt_section(disclosure_deltas)
        if deltas_section:
            sections.append(deltas_section)

    if verification_feedback:
        feedback_chunk_ids = set(_FEEDBACK_CHUNK_ID_RE.findall(verification_feedback))
        leads = compact_evidence_leads(
            citations,
            scope=scope,
            query=query,
            priority_chunk_ids=feedback_chunk_ids or None,
            max_leads=_REPAIR_MAX_EVIDENCE_LEADS,
            excerpt_chars=_REPAIR_EVIDENCE_LEAD_EXCERPT_CHARS,
        )
    else:
        leads = compact_evidence_leads(citations, scope=scope, query=query)
    if leads:
        sections.append(
            "## 10-K Evidence Leads (pre-retrieved passages — cite directly or extend with tools)\n"
            f"```json\n{_compact_json(leads)}\n```"
        )
    if verification_feedback:
        sections.append(
            "## Verification Correction Required (Prior Draft Rejected)\n"
            f"{verification_feedback.strip()}\n"
            "Tools are disabled for this repair pass — all deterministic outputs and 10-K evidence "
            "are already included above. Fix every finding so each `evidence_quote` is copied "
            "verbatim from its cited `chunk_id`, and every dollar/percentage figure in `statement` "
            "matches the exact figure in `evidence_quote` or the deterministic variance payload "
            "above (avoid scaling or rewriting table numbers if units are ambiguous)."
        )
    else:
        sections.append(
            build_tool_selection_policy(
                scope,
                has_prior_outputs=bool(
                    variance_reports
                    or material_variance_flags
                    or disclosure_deltas
                    or sector_map_summaries
                    or sector_benchmarks
                    or chunk_health
                ),
            )
        )
        sections.append(build_retrieval_instructions(scope, variance_evaluations))
    sections.append(
        "## Output Reminder\n"
        "Return a valid `AnalystReport` JSON (`intent`, `headline`, `key_metrics`, `findings`) "
        f"citing each company's `FY{scope.get('end_year')}` filing, `FY{scope.get('start_year')}`, "
        "and any major intermediate-year inflection when available in multi-year or peer queries, "
        "with verbatim `evidence_quote` snippets and only requested `key_metrics`."
    )
    return "\n\n".join(sections)


def select_citations_for_judge(
    response_markdown: str,
    citations: list[dict[str, Any]],
    *,
    min_citations: int = _JUDGE_MIN_CITATIONS,
    max_citations: int = _JUDGE_MAX_CITATIONS,
) -> list[dict[str, Any]]:
    """Give the judge every passage the report references (full excerpts), then other evidence.

    With agent-driven retrieval, the passages the analyst cites are usually ones it fetched
    itself, which sit at the end of the turn's citation list. Taking the first N citations, or
    truncating their excerpts, hid the supporting text from the judge, so valid claims and
    citations were scored as hallucinated. A chunk counts as referenced when its chunk_id
    appears anywhere in the report: inline tags or the closing citation table.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for c in citations:
        cid = str(c.get("chunk_id") or "")
        if cid:
            by_id.setdefault(cid, c)
    referenced = sorted(
        (cid for cid in by_id if cid in response_markdown),
        key=response_markdown.index,
    )
    limit = min(max_citations, max(min_citations, len(referenced)))
    ordered = [by_id[cid] for cid in referenced]
    referenced_set = set(referenced)
    ordered += [c for cid, c in by_id.items() if cid not in referenced_set]
    selected: list[dict[str, Any]] = []
    for c in ordered[:limit]:
        trimmed = dict(c)
        trimmed["excerpt"] = str(c.get("excerpt") or "")[:_JUDGE_EXCERPT_CHARS]
        selected.append(trimmed)
    return selected


def build_narrative_evaluation_prompt(
    *,
    query: str,
    response_markdown: str,
    variance_reports: list[dict[str, Any]],
    judge_citations: list[dict[str, Any]],
    reference_explanation: str | None = None,
) -> str:
    """Assemble the evaluation prompt for `NarrativeEvaluatorAgent` (Gemini 3.8 Flash, low thinking)."""
    if variance_reports:
        variance_block = (
            "### Deterministic Variance Calculations\n"
            f"```json\n{json.dumps(variance_reports, default=str)}\n```\n\n"
        )
    else:
        variance_block = (
            "### Deterministic Variance Calculations\n"
            "None for this turn: the analyst did not invoke the variance engine because the "
            "question is qualitative (risk factors, disclosure wording, themes). This is "
            "expected. Do NOT penalize the report for lacking variance tables, Key Metrics, or "
            "XBRL figures; score `numeric_fidelity_score` only on figures the report actually "
            "states (1.0 when every stated figure matches its cited passage, or when it states "
            "none), and score `explanation_quality_score` on how well the cited passages answer "
            "the question asked.\n\n"
        )
    eval_prompt = (
        "Evaluate the following synthesized SEC 10-K variance report against the deterministic "
        "calculation outputs and retrieved citations. Return JSON matching NarrativeEvaluationReport, "
        "including `citation_precision_score` (fraction of inline citations whose cited passage "
        "actually supports the exact claim alongside it, 0.0 to 1.0) and `unsupported_claims_count` "
        "(integer count of narrative claims not supported by the cited passage or deterministic payload).\n\n"
        f"### Analyst Query\n{query}\n\n"
        f"{variance_block}"
        "### Retrieved 10-K Citations\n"
        f"```json\n{json.dumps(judge_citations, default=str)}\n```\n\n"
        f"### Synthesized Narrative Report Under Evaluation\n{response_markdown}\n"
    )
    if reference_explanation and reference_explanation.strip():
        eval_prompt += (
            "\n### Golden Reference Explanation (analyst-authored ground truth)\n"
            f"{reference_explanation.strip()}\n\n"
            "Score `explanation_quality_score` and `thematic_consistency_score` by how well the "
            "report recovers the drivers and themes in this reference. Do not penalize the report "
            "for omitting reference claims that its retrieved citations do not support, and never "
            "reward claims that appear only in the reference but not in the citations.\n"
        )
    return eval_prompt


__all__ = [
    "build_disclosure_deltas_prompt_section",
    "build_financial_reasoning_prompt",
    "build_material_flags_prompt_section",
    "build_narrative_evaluation_prompt",
    "build_retrieval_instructions",
    "build_tool_selection_policy",
    "compact_disclosure_deltas_for_tool",
    "compact_evidence_leads",
    "compact_material_flags_for_tool",
    "compact_variance_reports_for_prompt",
    "select_citations_for_judge",
]
