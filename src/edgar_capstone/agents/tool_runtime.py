"""Turn-scoped runtime for agent-invoked deterministic tools.

`FinancialAnalystAgent` decides which deterministic engines a question needs
(`calculate_period_variances`, `flag_material_variances`, `compare_disclosures_across_years`,
`sector_thematic_map_reduce`, `query_looker_financial_explore`). Each tool writes its full,
deterministic output into ADK `session_state` under the keys below, and the chat pipeline
hydrates those outputs back (`hydrate_tool_outputs`) BEFORE claim verification, Markdown
rendering, and the numeric fact-check run. Nothing is pre-computed for the agent.

Two pieces of turn-scoped plumbing live here:

- `turn_scoped_state_seed()` resets every tool-output key at the start of an ADK run. Chat
  sessions are persistent, so without the reset a prior turn's variance tables or flags would
  leak into the current turn's response.
- `ToolTurnContext` (a `contextvars.ContextVar`) carries non-serializable per-turn handles into
  tools: the SSE progress callback (sector map-reduce progress + `agent_tool` events), an
  injected map-phase LLM, and the cached BigQuery profile catalog (0-RPC variance/flag math).
  ADK copies `contextvars` into its tool thread pool, so tools see the caller's context.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str, dict[str, Any] | None], None]

# ---------------------------------------------------------------------------
# ADK session_state keys written by agent-invoked deterministic tools
# ---------------------------------------------------------------------------

STATE_CALCULATED_VARIANCES = "calculated_variances"
STATE_VARIANCE_MISSING_TICKERS = "variance_missing_tickers"
STATE_MATERIAL_FLAGS = "material_variance_flags"
STATE_MATERIAL_FLAG_SUMMARY = "material_variance_summary"
STATE_DISCLOSURE_DELTAS = "disclosure_deltas"
STATE_DISCLOSURE_DELTA_SUMMARY = "disclosure_delta_summary"
STATE_SECTOR_COVERAGE = "sector_coverage"
STATE_SECTOR_MAP_SUMMARIES = "sector_map_summaries"
STATE_CHUNK_HEALTH = "chunk_health"
STATE_LOOKER_RESULTS = "looker_query_results"
STATE_RETRIEVED_CITATIONS = "retrieved_citations"

# Deterministic engines whose outputs surface as UI payloads (cards, badges, XBRL chips).
DETERMINISTIC_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "calculate_period_variances",
        "flag_material_variances",
        "compare_disclosures_across_years",
        "sector_thematic_map_reduce",
        "query_looker_financial_explore",
        "fetch_financial_metrics",
    }
)


# ---------------------------------------------------------------------------
# Turn context (contextvars) for non-serializable per-turn handles
# ---------------------------------------------------------------------------


@dataclass
class ToolTurnContext:
    """Per-turn handles visible to tools running on ADK's tool thread pool."""

    on_progress: ProgressCallback | None = None
    map_llm: Any | None = None
    cached_profiles: list[Any] = field(default_factory=list)
    # Wall-clock timing of every agent-invoked tool (for S2S-attested agent_trace steps).
    tool_started_at: dict[str, float] = field(default_factory=dict)
    tool_durations_ms: list[tuple[str, float]] = field(default_factory=list)

    def tool_duration_ms(self, tool_name: str) -> float:
        """Total wall-clock ms spent in `tool_name` this turn (0.0 when never invoked)."""
        return round(sum(ms for name, ms in self.tool_durations_ms if name == tool_name), 2)


_TURN_CONTEXT: contextvars.ContextVar[ToolTurnContext | None] = contextvars.ContextVar(
    "edgar_tool_turn_context", default=None
)


def current_tool_turn_context() -> ToolTurnContext | None:
    """Return the active turn context (None outside a chat turn, e.g. direct tool calls)."""
    return _TURN_CONTEXT.get()


@contextlib.contextmanager
def tool_turn_context(
    *,
    on_progress: ProgressCallback | None = None,
    map_llm: Any | None = None,
    cached_profiles: list[Any] | None = None,
) -> Iterator[ToolTurnContext]:
    """Bind a `ToolTurnContext` for the duration of one chat turn."""
    ctx = ToolTurnContext(
        on_progress=on_progress,
        map_llm=map_llm,
        cached_profiles=list(cached_profiles or []),
    )
    token = _TURN_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _TURN_CONTEXT.reset(token)


def emit_tool_progress(stage: str, message: str, meta: dict[str, Any] | None = None) -> None:
    """Forward a progress event to the active turn's SSE callback (no-op outside a turn)."""
    ctx = _TURN_CONTEXT.get()
    if ctx is None or ctx.on_progress is None:
        return
    try:
        ctx.on_progress(stage, message, meta)
    except Exception:  # progress streaming must never break a tool
        logger.debug("Tool progress callback raised", exc_info=True)


def cached_turn_profiles() -> list[Any]:
    """Cached BigQuery company-year profiles for this turn (empty outside a chat turn)."""
    ctx = _TURN_CONTEXT.get()
    return list(ctx.cached_profiles) if ctx is not None else []


def turn_map_llm() -> Any | None:
    """Injected map-phase LLM for sector map-reduce (None -> live Flash)."""
    ctx = _TURN_CONTEXT.get()
    return ctx.map_llm if ctx is not None else None


# ---------------------------------------------------------------------------
# Session-state seeding and hydration
# ---------------------------------------------------------------------------


@dataclass
class ToolOutputs:
    """Deterministic tool outputs hydrated from ADK `session_state` for one turn."""

    variance_reports: list[dict[str, Any]] = field(default_factory=list)
    missing_tickers: list[str] = field(default_factory=list)
    material_variance_flags: list[dict[str, Any]] = field(default_factory=list)
    material_variance_summary: dict[str, Any] | None = None
    disclosure_deltas: dict[str, Any] | None = None
    sector_coverage: dict[str, Any] | None = None
    sector_map_summaries: dict[str, Any] | None = None
    sector_benchmarks: list[dict[str, Any]] = field(default_factory=list)
    chunk_health: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def calculated_variances_map(self) -> dict[str, dict[str, Any]]:
        return variance_map(self.variance_reports)

    def merge(self, other: ToolOutputs) -> ToolOutputs:
        """Combine two hydrations (e.g. initial run + verification-guided retry)."""
        reports = {**variance_map(self.variance_reports), **variance_map(other.variance_reports)}
        flags: dict[str, dict[str, Any]] = {}
        for f in [*self.material_variance_flags, *other.material_variance_flags]:
            flags[_flag_key(f)] = f
        return ToolOutputs(
            variance_reports=list(reports.values()),
            missing_tickers=sorted({*self.missing_tickers, *other.missing_tickers}),
            material_variance_flags=list(flags.values()),
            material_variance_summary=(
                other.material_variance_summary or self.material_variance_summary
            ),
            disclosure_deltas=merge_disclosure_deltas(
                self.disclosure_deltas, other.disclosure_deltas
            ),
            sector_coverage=other.sector_coverage or self.sector_coverage,
            sector_map_summaries=other.sector_map_summaries or self.sector_map_summaries,
            sector_benchmarks=other.sector_benchmarks or self.sector_benchmarks,
            chunk_health=other.chunk_health or self.chunk_health,
            citations=_merge_citation_lists(self.citations, other.citations),
        )


def variance_map(variance_reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Key variance reports as `TICKER_START_END` (the fact-check `calculated_variances` key)."""
    return {
        f"{r.get('ticker', 'UNK')}_{r.get('start_year', 0)}_{r.get('end_year', 0)}": r
        for r in variance_reports
        if isinstance(r, dict)
    }


def _flag_key(flag: dict[str, Any]) -> str:
    return str(
        flag.get("flag_id")
        or f"{flag.get('ticker')}_{flag.get('fiscal_year')}_{flag.get('metric')}_{flag.get('rule')}"
    )


def _merge_citation_lists(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for c in [*first, *second]:
        if isinstance(c, dict) and c.get("chunk_id"):
            merged.setdefault(str(c["chunk_id"]), c)
    return list(merged.values())


def merge_disclosure_deltas(
    existing: dict[str, Any] | None, new: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Merge two `compare_disclosures_across_years` payloads (companies keyed by ticker)."""
    if not new or new.get("status") != "success":
        return existing if existing else new
    if not existing or existing.get("status") != "success":
        return new
    companies: dict[str, dict[str, Any]] = {
        str(c.get("ticker")): c for c in existing.get("companies") or []
    }
    for c in new.get("companies") or []:
        companies[str(c.get("ticker"))] = c
    summary = {
        k: int((existing.get("summary") or {}).get(k, 0) or 0)
        + int((new.get("summary") or {}).get(k, 0) or 0)
        for k in ("added", "removed", "modified", "unchanged", "year_pairs")
    }
    summary["companies_analyzed"] = len(companies)
    starts = [int(y) for y in (existing.get("start_year"), new.get("start_year")) if y]
    ends = [int(y) for y in (existing.get("end_year"), new.get("end_year")) if y]
    return {
        **new,
        "start_year": min(starts) if starts else None,
        "end_year": max(ends) if ends else None,
        "missing_tickers": sorted(
            {*(existing.get("missing_tickers") or []), *(new.get("missing_tickers") or [])}
        ),
        "summary": summary,
        "companies": list(companies.values()),
        "citations": _merge_citation_lists(
            list(existing.get("citations") or []), list(new.get("citations") or [])
        ),
    }


def turn_scoped_state_seed(
    *,
    citations: list[dict[str, Any]],
    prior: ToolOutputs | None = None,
) -> dict[str, Any]:
    """Reset every tool-output key for a new ADK run (optionally carrying `prior` outputs).

    `prior` is used by the verification-guided regeneration pass so the retry sees the
    deterministic outputs the agent already produced without re-invoking the tools.
    """
    p = prior or ToolOutputs()
    return {
        STATE_CALCULATED_VARIANCES: p.calculated_variances_map,
        STATE_VARIANCE_MISSING_TICKERS: list(p.missing_tickers),
        STATE_MATERIAL_FLAGS: list(p.material_variance_flags),
        STATE_MATERIAL_FLAG_SUMMARY: p.material_variance_summary,
        STATE_DISCLOSURE_DELTAS: p.disclosure_deltas,
        STATE_DISCLOSURE_DELTA_SUMMARY: (p.disclosure_deltas or {}).get("summary"),
        STATE_SECTOR_COVERAGE: p.sector_coverage,
        STATE_SECTOR_MAP_SUMMARIES: p.sector_map_summaries,
        STATE_CHUNK_HEALTH: p.chunk_health,
        STATE_LOOKER_RESULTS: (
            [
                {
                    "status": "success",
                    "explore_name": "sector_benchmarks",
                    "rows": p.sector_benchmarks,
                }
            ]
            if p.sector_benchmarks
            else []
        ),
        STATE_RETRIEVED_CITATIONS: _merge_citation_lists(list(citations), p.citations),
    }


def hydrate_tool_outputs(session_state: dict[str, Any] | None) -> ToolOutputs:
    """Read every deterministic tool output the agent produced this turn from `session_state`."""
    state = session_state or {}
    calc = state.get(STATE_CALCULATED_VARIANCES) or {}
    reports = [r for r in calc.values() if isinstance(r, dict)] if isinstance(calc, dict) else []
    reports.sort(key=lambda r: (str(r.get("ticker", "")), int(r.get("start_year") or 0)))

    flags = [f for f in (state.get(STATE_MATERIAL_FLAGS) or []) if isinstance(f, dict)]
    flag_summary = state.get(STATE_MATERIAL_FLAG_SUMMARY)
    deltas = state.get(STATE_DISCLOSURE_DELTAS)
    sector_cov = state.get(STATE_SECTOR_COVERAGE)
    sector_map = state.get(STATE_SECTOR_MAP_SUMMARIES)
    chunk_health = state.get(STATE_CHUNK_HEALTH)

    benchmarks: list[dict[str, Any]] = []
    for res in state.get(STATE_LOOKER_RESULTS) or []:
        if (
            isinstance(res, dict)
            and res.get("status") == "success"
            and res.get("explore_name") == "sector_benchmarks"
        ):
            benchmarks.extend(r for r in res.get("rows") or [] if isinstance(r, dict))

    return ToolOutputs(
        variance_reports=reports,
        missing_tickers=sorted(
            {str(t) for t in (state.get(STATE_VARIANCE_MISSING_TICKERS) or []) if t}
        ),
        material_variance_flags=flags,
        material_variance_summary=flag_summary if isinstance(flag_summary, dict) else None,
        disclosure_deltas=deltas if isinstance(deltas, dict) else None,
        sector_coverage=sector_cov if isinstance(sector_cov, dict) else None,
        sector_map_summaries=sector_map if isinstance(sector_map, dict) else None,
        sector_benchmarks=benchmarks,
        chunk_health=chunk_health if isinstance(chunk_health, dict) else None,
        citations=[c for c in (state.get(STATE_RETRIEVED_CITATIONS) or []) if isinstance(c, dict)],
    )


__all__ = [
    "DETERMINISTIC_TOOL_NAMES",
    "STATE_CALCULATED_VARIANCES",
    "STATE_CHUNK_HEALTH",
    "STATE_DISCLOSURE_DELTAS",
    "STATE_DISCLOSURE_DELTA_SUMMARY",
    "STATE_LOOKER_RESULTS",
    "STATE_MATERIAL_FLAGS",
    "STATE_MATERIAL_FLAG_SUMMARY",
    "STATE_RETRIEVED_CITATIONS",
    "STATE_SECTOR_COVERAGE",
    "STATE_SECTOR_MAP_SUMMARIES",
    "STATE_VARIANCE_MISSING_TICKERS",
    "ToolOutputs",
    "ToolTurnContext",
    "cached_turn_profiles",
    "current_tool_turn_context",
    "emit_tool_progress",
    "hydrate_tool_outputs",
    "merge_disclosure_deltas",
    "tool_turn_context",
    "turn_map_llm",
    "turn_scoped_state_seed",
    "variance_map",
]
