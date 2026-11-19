"""Prompt formatting, payload compaction, and judge citation selection helpers."""

from __future__ import annotations

import json
from typing import Any

# Evidence lead budget: generous enough that the analyst can extract verbatim quotes directly
# without burning extra ReAct tool round-trips for passages already retrieved, while keeping
# prompt prefill lean for sub-3s TTFT.
_EVIDENCE_LEAD_EXCERPT_CHARS = 650
_MAX_EVIDENCE_LEADS = 12
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
        for mv_key in (
            "revenue_variance",
            "operating_income_variance",
            "net_income_variance",
            "research_and_development_variance",
        ):
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


def compact_evidence_leads(
    cached_explanations: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    *,
    scope: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Citable passages already retrieved during the deterministic pre-pass.

    Uses boundary-first round-robin selection so every `(ticker, start_year)` and
    `(ticker, end_year)` present in `citations` is guaranteed representation before filling
    remaining lead slots in retrieval score order.
    """
    unique = _deduplicate_citations(citations)
    boundary_years = _parse_scope_years(scope)
    grouped, ticker_order = _group_citations_by_ticker_year(unique)

    ordered: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()

    def _pick(c: dict[str, Any]) -> None:
        cid = str(c.get("chunk_id") or "").strip()
        if cid and cid not in chosen_ids and len(ordered) < _MAX_EVIDENCE_LEADS:
            chosen_ids.add(cid)
            ordered.append(c)

    # Pass 1: Guarantee at least 1 boundary-year passage per (ticker, start_yr) & (ticker, end_yr)
    for tk in ticker_order:
        for by in boundary_years:
            bucket = grouped.get((tk, by)) or []
            if bucket:
                sec_pref = next((b for b in bucket if b.get("section_id") == "item_7"), bucket[0])
                _pick(sec_pref)

    # Pass 2: One passage from any remaining (ticker, fiscal_year) bucket (round-robin)
    for bucket in grouped.values():
        if len(ordered) >= _MAX_EVIDENCE_LEADS:
            break
        first_unpicked = next(
            (c for c in bucket if str(c.get("chunk_id") or "").strip() not in chosen_ids),
            None,
        )
        if first_unpicked is not None:
            _pick(first_unpicked)

    # Pass 3: Fill remaining slots up to _MAX_EVIDENCE_LEADS in original retrieval order
    for c in unique:
        if len(ordered) >= _MAX_EVIDENCE_LEADS:
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
            "excerpt": " ".join(str(c.get("excerpt") or "").split())[:_EVIDENCE_LEAD_EXCERPT_CHARS],
        }
        for c in ordered
    ]
    sections_queried = sorted(
        {str(e.get("cache_key")) for e in cached_explanations if e.get("cache_key")}
    )
    if sections_queried:
        leads.append({"pre_queried_sections": sections_queried})
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
    """Agent-driven deterministic tool policy: invoke engines ONLY when the question needs them."""
    tickers_csv = scope.get("tickers_csv", "ALL")
    years = f"start_year={scope.get('start_year')}, end_year={scope.get('end_year')}"
    preamble = (
        "The deterministic outputs you already produced this turn are included above; do NOT "
        "re-invoke those tools. Invoke another engine only if the question needs it."
        if has_prior_outputs
        else "No variance tables, anomaly flags, or disclosure deltas were computed for this "
        "request. Invoke a deterministic engine ONLY when the question needs its output; "
        "whatever you invoke is shown to the user as a table or card next to your answer, "
        "so do not invoke engines the question does not ask about."
    )
    return "\n".join(
        [
            "## Deterministic Tool Selection (you decide; nothing is pre-computed)",
            preamble,
            "- `calculate_period_variances(tickers, start_year, end_year)`: call when the question "
            "asks about reported financial figures (revenue, operating income, net income, R&D "
            "spend, margins, growth, CAGR) or compares companies on them. Do NOT call it for "
            "purely qualitative questions (risk-factor wording, disclosure themes, strategy). "
            f"Scope: `tickers={tickers_csv}`, `{years}`.",
            "- `flag_material_variances(tickers, start_year, end_year, explain_top_flags)`: call "
            "ONLY when the question asks about anomalies: unusual swings, inflections, a profit "
            "turning into a loss (or back), margin shifts, or why a figure spiked or collapsed. "
            "Usually pair it with `calculate_period_variances`. Keep `explain_top_flags=true` so "
            "the top flags arrive with MD&A passages (`explanation.chunk_id`).",
            "- `compare_disclosures_across_years(theme, tickers, gics_sector, section_id, "
            "start_year, end_year, theme_terms)`: call when the question asks how specific "
            "companies' 10-K wording on a theme changed across fiscal years. Populate "
            "`theme_terms` with comma-separated key phrases and domain synonyms to count.",
            "- `sector_thematic_map_reduce(theme, gics_sector, section_id, start_year, end_year, "
            "theme_terms)`: call for sector-wide thematic questions (no companies named, a GICS "
            "sector in scope); it scans every issuer in the sector. Populate `theme_terms` with "
            "comma-separated key phrases and synonyms. Do not also call "
            "`compare_disclosures_across_years` for the same sector theme.",
            "- `query_looker_financial_explore(explore_name, ...)`: `sector_benchmarks` / "
            "`peer_comparison` for sector-level financial benchmarks, `chunk_health` for vector "
            "index / corpus health questions.",
            f"- Scope structure: {_scope_structure_label(scope)}.",
        ]
    )


def _citation_coverage_line(scope: dict[str, Any]) -> str | None:
    """Scope-specific citation-coverage requirement (no extra model call or re-prompt)."""
    tickers = [str(t) for t in (scope.get("tickers") or []) if t]
    years = _parse_scope_years(scope)
    multi_company = len(tickers) > 1
    multi_year = len(years) == 2 and years[0] != years[1]
    if not (multi_company or multi_year):
        return None
    parts: list[str] = []
    if multi_company:
        parts.append(f"at least one cited finding for each of {', '.join(tickers)}")
    if multi_year:
        parts.append(f"findings citing both the FY{years[0]} and the FY{years[1]} filings")
    return (
        "- Citation coverage for this question: "
        + " and ".join(parts)
        + " (use the matching Evidence Leads; skip a company-year only if nothing covers it)."
    )


def build_retrieval_instructions(
    scope: dict[str, Any],
    variance_evaluations: list[dict[str, Any]],
) -> str:
    """Build retrieval guidance advising the analyst to synthesize first when leads suffice."""
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
        "## 10-K Retrieval Guidance (Synthesize First When Leads Suffice)",
        "Full 10-K sections are NOT in this request. The **10-K Evidence Leads** above (and any "
        "tool output you request: flag explanations, disclosure deltas, sector map-reduce "
        "summaries) include verbatim filing passages with canonical `chunk_id`s. If they already "
        "cover every ticker and fiscal year needed to answer the user's question accurately, "
        "synthesize your `AnalystReport` directly WITHOUT calling extra retrieval tools.",
        "Call retrieval tools ONLY to fill a specific missing gap (e.g. a missing start/end fiscal "
        "year or an unretrieved ticker/section):",
        "- `query_cached_10k_context(ticker, fiscal_year, section_id, variance_query, max_passages)`: "
        "ranked passages (with `chunk_id`) from one company-year's `item_7` or `item_1a`, for "
        "explaining a specific variance driver.",
        "- `search_10k_chunks_rag(query, tickers, gics_sector, section_id, start_year, end_year, "
        "top_k)`: hybrid search across filings, for peer comparisons, thematic questions, and "
        "changes across fiscal years.",
        f"- Scope: tickers `{scope.get('tickers_csv', 'ALL')}`, sector "
        f"`{scope.get('gics_sector', 'ALL')}`, sections `{scope.get('section_id', 'ALL')}`, "
        f"fiscal years `FY{scope.get('start_year')}`–`FY{scope.get('end_year')}`.",
        "- If you do call tools, keep result sets small (`max_passages` <= 4, `top_k` <= 8), "
        "issue independent calls in parallel in one step, and never re-fetch evidence already "
        "present in the prompt (at most 6 retrieval calls).",
    ]
    coverage_line = _citation_coverage_line(scope)
    if coverage_line:
        lines.append(coverage_line)
    if suggested:
        lines.append(
            "- Suggested retrieval targets from the variance evaluation:\n"
            f"```json\n{_compact_json(suggested[:8])}\n```"
        )
    return "\n".join(lines)


def _compact_material_flag_entries(flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact material variance flags to lean LLM-ready dicts (full payloads stay in session_state)."""
    compact: list[dict[str, Any]] = []
    for f in flags[:_MAX_PROMPT_FLAGS]:
        src = f.get("source") or {}
        exp = f.get("explanation") or {}
        entry: dict[str, Any] = {
            "flag_id": f.get("flag_id"),
            "ticker": f.get("ticker"),
            "fiscal_year": f.get("fiscal_year"),
            "prior_fiscal_year": f.get("prior_fiscal_year"),
            "metric": f.get("metric"),
            "rule": f.get("rule"),
            "severity": f.get("severity"),
            "value": f.get("value"),
            "unit": f.get("unit"),
            "description": f.get("description"),
            "period_of_report": src.get("period_of_report"),
            "prior_period_of_report": src.get("prior_period_of_report"),
            "accession_number": src.get("accession_number"),
            "xbrl_concept": src.get("xbrl_concept"),
            "explanation_query": f.get("explanation_query"),
            "explanation_chunk_id": exp.get("chunk_id"),
            "explanation_excerpt": " ".join(str(exp.get("excerpt") or "").split())[:600]
            if exp.get("excerpt")
            else None,
        }
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
        src = f.get("source") or {}
        exp = f.get("explanation") or {}
        entry: dict[str, Any] = {
            "flag_id": f.get("flag_id"),
            "ticker": f.get("ticker"),
            "fiscal_year": f.get("fiscal_year"),
            "prior_fiscal_year": f.get("prior_fiscal_year"),
            "metric": f.get("metric"),
            "rule": f.get("rule"),
            "severity": f.get("severity"),
            "value": f.get("value"),
            "unit": f.get("unit"),
            "description": f.get("description"),
            "period_of_report": src.get("period_of_report"),
            "prior_period_of_report": src.get("prior_period_of_report"),
            "accession_number": src.get("accession_number"),
            "xbrl_concept": src.get("xbrl_concept"),
            "explanation_query": f.get("explanation_query"),
        }
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
            compact_deltas = []
            for d in pair.get("deltas") or []:
                if d.get("change_type") == "unchanged":
                    continue
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
                if len(compact_deltas) >= _MAX_PROMPT_DELTAS_PER_PAIR:
                    break
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
    cached_explanations: list[dict[str, Any]],
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
    sections: list[str] = [
        f"## Analyst Query\n{query.strip()}",
        (
            f"## Resolved Scope\n"
            f"- Scope structure: {_scope_structure_label(scope)} (provisional routing "
            f"hint: `{intent}`; classify the final intent yourself in `AnalystReport.intent`)\n"
            f"- Tickers: `{scope.get('tickers_csv', 'ALL')}`\n"
            f"- GICS Sector: `{scope.get('gics_sector', 'ALL')}`\n"
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

    leads = compact_evidence_leads(cached_explanations, citations, scope=scope)
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
        "## Answer Format\n"
        "Answer with a concise `AnalystReport` JSON (`intent`, `headline`, `key_metrics`, "
        "`findings`). `intent` is one of `thematic_trend_tracking` (wording / disclosure or "
        "theme change across years or a sector; takes precedence), "
        "`multi_company_peer_comparison`, `longitudinal_single_company`, or "
        "`chunk_health_monitoring`. "
        "There is NO `notes` field: every qualitative claim must be a verified finding in `findings` "
        "with `chunk_ids` from your tools or the evidence leads plus a verbatim `evidence_quote` "
        "(10-40 words) copied exactly from one of those chunks. For multi-year or peer queries, "
        "include at least one verified finding citing EACH in-scope company's start fiscal year "
        f"(`FY{scope.get('start_year')}`) AND end fiscal year (`FY{scope.get('end_year')}`) "
        "when passages for those years appear in the evidence leads. Figures in a finding "
        "must appear in its quote or in the deterministic payload / flags. Findings that fail "
        "verification are dropped. `key_metrics` are references only (`ticker` + `metric`); exact "
        "values, `period_of_report` dates, and XBRL provenance citations are rendered automatically "
        "from `calculate_period_variances`. List ONLY the metrics the question asks about (e.g. "
        "only `research_and_development` for an R&D question) and leave `key_metrics` EMPTY for "
        "qualitative, risk-factor, or wording-change questions."
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
    cached_explanations_count: int = 0,
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
        f"### Retrieved 10-K Citations (cached_explanations_count={cached_explanations_count})\n"
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


# Backward-compatible underscore aliases retained for internal callers & tests.
_compact_variance_reports_for_prompt = compact_variance_reports_for_prompt
_compact_evidence_leads = compact_evidence_leads
_retrieval_instructions = build_retrieval_instructions
_material_flags_prompt_section = build_material_flags_prompt_section
_disclosure_deltas_prompt_section = build_disclosure_deltas_prompt_section
_build_financial_reasoning_prompt = build_financial_reasoning_prompt

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
