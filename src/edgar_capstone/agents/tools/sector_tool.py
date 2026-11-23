"""Agent-invocable full-sector thematic map-reduce tool (`sector_thematic_map_reduce`).

`FinancialAnalystAgent` calls this tool for sector-wide thematic questions instead of the chat
pipeline pre-running map-reduce for every sector-scoped turn. The tool streams the same SSE
`sector_retrieval` / `sector_map_start` / `sector_map_progress` events through the active
`ToolTurnContext`, writes the full payload (sector coverage, batch summaries, disclosure deltas,
citations) into ADK `session_state` for hydration, and returns a compact digest to the model.

`run_sector_thematic_map_reduce` is imported lazily: `sector_map_reduce` depends on
`live_runtime`, which builds the analyst agent from `factories`, which registers this tool.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import FunctionTool, ToolContext

from edgar_capstone.agents.core.citations import merge_citations
from edgar_capstone.agents.core.resilience import with_tool_resilience
from edgar_capstone.agents.tools.runtime import (
    STATE_DISCLOSURE_DELTA_SUMMARY,
    STATE_DISCLOSURE_DELTAS,
    STATE_RETRIEVED_CITATIONS,
    STATE_SECTOR_COVERAGE,
    STATE_SECTOR_MAP_SUMMARIES,
    cached_turn_profiles,
    current_tool_turn_context,
    merge_disclosure_deltas,
    turn_map_llm,
)
from edgar_capstone.agents.tools.sec_tools import _record_gcp_backend_invocation
from edgar_capstone.common.domain import canonicalize_gics_sector

_DIGEST_MAX_CITATIONS = 12
_DIGEST_EXCERPT_CHARS = 450


def _compact_sector_digest(result: dict[str, Any]) -> dict[str, Any]:
    """Compact map-reduce digest returned to the analyst (full payload stays in session_state)."""
    cov = result.get("sector_coverage") or {}
    deltas = result.get("disclosure_deltas") or {}
    batches = [
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
        for b in (result.get("sector_map_summaries") or {}).get("batch_summaries") or []
    ]
    citations = [
        {
            "chunk_id": c.get("chunk_id"),
            "ticker": c.get("ticker"),
            "fiscal_year": c.get("fiscal_year"),
            "section_id": c.get("section_id"),
            "excerpt": " ".join(str(c.get("excerpt") or "").split())[:_DIGEST_EXCERPT_CHARS],
        }
        for c in (result.get("citations") or [])[:_DIGEST_MAX_CITATIONS]
    ]
    return {
        "status": result.get("status", "success"),
        "sector_coverage": {
            k: cov.get(k)
            for k in (
                "gics_sector",
                "start_year",
                "end_year",
                "issuers_in_sector",
                "issuers_analyzed",
                "issuers_with_theme_matches",
                "filings_by_year",
                "missing_fy_counts",
                "theme_mentions_by_year",
                "total_chunks_evaluated",
                "batches_processed",
            )
        },
        "delta_summary": deltas.get("summary"),
        "batch_summaries": batches,
        "citations": citations,
        "instructions": (
            "Synthesize sector-wide trends from these verified batch summaries. Every finding must "
            "cite a chunk_id listed here (or in the evidence leads) with a verbatim evidence_quote."
        ),
    }


@with_tool_resilience(tool_name="sector_thematic_map_reduce")
def sector_thematic_map_reduce(
    theme: str,
    gics_sector: str,
    section_id: str,
    start_year: int,
    end_year: int,
    theme_terms: str = "",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Runs full-sector batched map-reduce over EVERY issuer in a GICS sector for one theme.

    Use this for sector-wide thematic questions (no specific companies named) such as how risk
    disclosures on a theme spread or shifted across a whole sector over several fiscal years.
    It ranks theme passages for every company-year in the sector (one BigQuery vector query),
    deterministically aligns year-over-year wording changes per company, summarizes the
    companies in parallel Flash batches with verified verbatim evidence, and streams progress.

    Args:
        theme: Theme to track across the sector (e.g., 'generative AI risk', 'wildfire liability').
        gics_sector: GICS Sector to scan (e.g., 'Information Technology', 'Utilities').
        section_id: 'item_1a' (Risk Factors), 'item_7' (MD&A), or 'ALL'.
        start_year: First fiscal year of the window (e.g., 2024).
        end_year: Last fiscal year of the window (e.g., 2026).
        theme_terms: Optional comma-separated key phrases and domain synonyms to count across the
            sector's filings (e.g., 'AI, artificial intelligence, generative AI, GenAI'). Omit to
            fall back to tokens from `theme`.

    Returns:
        Dictionary with 'status', 'sector_coverage', 'delta_summary', verified 'batch_summaries'
        (each evidence item carries a chunk_id and verbatim quote), and top 'citations'.
    """
    if not (theme or "").strip():
        return {"status": "error", "message": "Please provide a theme to track across the sector."}
    sector = canonicalize_gics_sector(gics_sector)
    if not sector:
        return {
            "status": "error",
            "message": (
                "sector_thematic_map_reduce needs a specific GICS sector (e.g. 'Utilities'); "
                "use compare_disclosures_across_years or search_10k_chunks_rag for named companies."
            ),
        }
    from edgar_capstone.agents.tools.sector_map_reduce import run_sector_thematic_map_reduce

    turn_ctx = current_tool_turn_context()
    result = run_sector_thematic_map_reduce(
        theme=theme,
        gics_sector=sector,
        section_id=section_id,
        start_year=start_year,
        end_year=end_year,
        theme_terms=theme_terms,
        sector_profiles=cached_turn_profiles() or None,
        map_llm=turn_map_llm(),
        on_progress=turn_ctx.on_progress if turn_ctx is not None else None,
    )
    if tool_context is not None and hasattr(tool_context, "state"):
        state = tool_context.state
        state[STATE_SECTOR_COVERAGE] = result.get("sector_coverage")
        state[STATE_SECTOR_MAP_SUMMARIES] = result.get("sector_map_summaries")
        deltas = result.get("disclosure_deltas")
        if isinstance(deltas, dict):
            state[STATE_DISCLOSURE_DELTAS] = merge_disclosure_deltas(
                state.get(STATE_DISCLOSURE_DELTAS), deltas
            )
            state[STATE_DISCLOSURE_DELTA_SUMMARY] = deltas.get("summary")
        existing = list(state.get(STATE_RETRIEVED_CITATIONS) or [])
        state[STATE_RETRIEVED_CITATIONS] = merge_citations(
            existing, list(result.get("citations") or [])
        )
    _record_gcp_backend_invocation(
        tool_context,
        tool_name="sector_thematic_map_reduce",
        data_source="gcp",
        services_hit=["bigquery", "vertex_ai"],
    )
    return _compact_sector_digest(result)


SectorThematicMapReduceTool = FunctionTool(sector_thematic_map_reduce)


__all__ = ["SectorThematicMapReduceTool", "sector_thematic_map_reduce"]
