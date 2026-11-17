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
        "Full 10-K sections are NOT in this request. The **10-K Evidence Leads**, "
        "**Material Variance Flag excerpts**, **Disclosure Deltas**, and **Sector Map-Reduce "
        "Summaries** above already include verbatim retrieved filing passages with canonical "
        "`chunk_id`s. If they already cover every ticker and fiscal year needed to answer the "
        "user's question accurately, synthesize your `AnalystReport` directly WITHOUT calling "
        "extra tools.",
        "Call retrieval tools ONLY to fill a specific missing gap (e.g. a missing start/end fiscal "
        "year or an unretrieved ticker/section):",
        "- `query_cached_10k_context(ticker, fiscal_year, section_id, variance_query, max_passages)`: "
        "ranked passages (with `chunk_id`) from one company-year's `item_7` or `item_1a`, for "
        "explaining a specific variance driver.",
        "- `search_10k_chunks_rag(query, tickers, gics_sector, section_id, start_year, end_year, "
        "top_k)`: hybrid search across filings, for peer comparisons, thematic questions, and "
        "changes across fiscal years.",
        "- `compare_disclosures_across_years(theme, tickers, gics_sector, section_id, start_year, "
        "end_year)`: deterministic year-over-year wording comparison (added / removed / modified "
        "passages with both-year chunk_ids). Use only if disclosure deltas are not already above.",
        "- `flag_material_variances(tickers, start_year, end_year)`: deterministic anomaly flags "
        "for companies or years the payload does not cover.",
        f"- Scope: tickers `{scope.get('tickers_csv', 'ALL')}`, sector "
        f"`{scope.get('gics_sector', 'ALL')}`, sections `{scope.get('section_id', 'ALL')}`, "
        f"fiscal years `FY{scope.get('start_year')}`–`FY{scope.get('end_year')}`.",
        "- If you do call tools, keep result sets small (`max_passages` <= 4, `top_k` <= 8), "
        "issue independent calls in parallel in one step, and never re-fetch evidence already "
        "present in the prompt (at most 6 retrieval calls).",
    ]
    if suggested:
        lines.append(
            "- Suggested retrieval targets from the variance evaluation:\n"
            f"```json\n{_compact_json(suggested[:8])}\n```"
        )
    return "\n".join(lines)


def build_material_flags_prompt_section(flags: list[dict[str, Any]] | None) -> str:
    """Compact, authoritative anomaly flags for the analyst (including XBRL provenance & excerpt)."""
    if not flags:
        return ""
    compact = []
    for f in flags[:_MAX_PROMPT_FLAGS]:
        src = f.get("source") or {}
        exp = f.get("explanation") or {}
        compact.append(
            {
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
        )
    return (
        "## Authoritative Material Variance Flags (deterministic anomaly detection)\n"
        "(These anomalies were detected by fixed rules; never recompute or re-threshold them. "
        "Explain the high-severity flags first with MD&A evidence: use `explanation_chunk_id` and "
        "`explanation_excerpt` when present, otherwise call `query_cached_10k_context` with the "
        "flag's `explanation_query`.)\n"
        f"```json\n{_compact_json(compact)}\n```"
    )


def build_disclosure_deltas_prompt_section(result: dict[str, Any] | None) -> str:
    """Compact year-over-year disclosure deltas (theme wording changes) for the analyst."""
    if not result or result.get("status") != "success" or not result.get("companies"):
        return ""
    compact_companies: list[dict[str, Any]] = []
    for comp in result["companies"]:
        pairs = []
        for pair in comp.get("year_pairs") or []:
            pairs.append(
                {
                    "years": f"FY{pair['prior_fiscal_year']}->FY{pair['current_fiscal_year']}",
                    "counts": pair.get("counts"),
                    "deltas": [
                        {
                            "change_type": d["change_type"],
                            "prior_chunk_id": d.get("prior_chunk_id"),
                            "current_chunk_id": d.get("current_chunk_id"),
                            "added_phrases": d.get("added_phrases") or [],
                            "removed_phrases": d.get("removed_phrases") or [],
                            "excerpt": " ".join(
                                str(
                                    d.get("current_excerpt") or d.get("prior_excerpt") or ""
                                ).split()
                            )[:_EVIDENCE_LEAD_EXCERPT_CHARS],
                        }
                        for d in pair.get("deltas") or []
                        if d["change_type"] != "unchanged"
                    ][:_MAX_PROMPT_DELTAS_PER_PAIR],
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
    payload = {
        "theme": result.get("theme"),
        "summary": result.get("summary"),
        "companies": compact_companies,
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
            f"## Resolved Scope & Intent\n"
            f"- Intent: `{intent}`\n"
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
    sections.append(build_retrieval_instructions(scope, variance_evaluations))
    if verification_feedback:
        sections.append(
            "## Verification Correction Required (Prior Draft Rejected)\n"
            f"{verification_feedback.strip()}\n"
            "Fix every finding so each `evidence_quote` is copied verbatim from its cited "
            "`chunk_id`, and every dollar/percentage figure in `statement` matches the exact "
            "figure in `evidence_quote` or the deterministic variance payload above (avoid scaling "
            "or rewriting table numbers if units are ambiguous)."
        )
    sections.append(
        "## Answer Format\n"
        "Answer with a concise `AnalystReport` JSON (`headline`, `key_metrics`, `findings`). "
        "There is NO `notes` field: every qualitative claim must be a verified finding in `findings` "
        "with `chunk_ids` from your tools or the evidence leads plus a verbatim `evidence_quote` "
        "(10-40 words) copied exactly from one of those chunks. For multi-year or peer queries, "
        "include at least one verified finding citing EACH in-scope company's start fiscal year "
        f"(`FY{scope.get('start_year')}`) AND end fiscal year (`FY{scope.get('end_year')}`) "
        "when passages for those years appear in the evidence leads. Figures in a finding "
        "must appear in its quote or in the deterministic payload / flags. Findings that fail "
        "verification are dropped. `key_metrics` are references only (`ticker` + `metric`); exact "
        "values, `period_of_report` dates, and XBRL provenance citations are rendered automatically "
        "from the engine."
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
    """Assemble the evaluation prompt for `NarrativeEvaluatorAgent` (Gemini 3.5 Flash)."""
    eval_prompt = (
        "Evaluate the following synthesized SEC 10-K variance report against the deterministic "
        "calculation outputs and retrieved citations. Return JSON matching NarrativeEvaluationReport, "
        "including `citation_precision_score` (fraction of inline citations whose cited passage "
        "actually supports the exact claim alongside it, 0.0 to 1.0) and `unsupported_claims_count` "
        "(integer count of narrative claims not supported by the cited passage or deterministic payload).\n\n"
        f"### Analyst Query\n{query}\n\n"
        "### Deterministic Variance Calculations\n"
        f"```json\n{json.dumps(variance_reports, default=str)}\n```\n\n"
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
    "compact_evidence_leads",
    "compact_variance_reports_for_prompt",
    "select_citations_for_judge",
]
