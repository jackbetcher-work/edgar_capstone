"""Conversational Hybrid RAG orchestration engine for the SEC EDGAR Natural Language Analyst UI."""

from __future__ import annotations

import calendar
import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from google.adk.models.base_llm import BaseLlm

from edgar_capstone.agents.orchestration.intent import (
    infer_intent_from_tool_calls,
    is_sector_wide_scope,
    resolve_turn_intent,
)
from edgar_capstone.agents.orchestration.react_loop import StructuredReActExecutor
from edgar_capstone.agents.orchestration.scope_validation import (
    MAX_CORPUS_YEAR,
    MIN_CORPUS_YEAR,
    SECTOR_ALIASES,
    EntityCatalog,
    build_coverage_notices,
    extract_removed_tickers,
    is_additive_followup,
    is_broad_corpus_query,
    validate_query_scope_and_coverage,
)
from edgar_capstone.agents.prompts.store import get_prompt_store_status_metadata
from edgar_capstone.agents.runtime.callbacks import verify_narrative_against_calculations
from edgar_capstone.agents.runtime.context_memory import get_conversation_memory_manager
from edgar_capstone.agents.runtime.latency_tracker import get_context_latency_tracker
from edgar_capstone.agents.runtime.live_runtime import get_live_runtime_status_metadata
from edgar_capstone.agents.runtime.model_config import (
    get_model_strategy_metadata,
    resolve_model_for_task,
)
from edgar_capstone.agents.tools.runtime import (
    DETERMINISTIC_TOOL_NAMES,
    ToolOutputs,
    ToolTurnContext,
    tool_turn_context,
)
from edgar_capstone.agents.tools.sec_tools import (
    _parse_ticker_list,
    _profile_to_metric_record,
    get_active_gcp_tool_adapters,
    get_context_cache_runtime_stats,
    search_10k_chunks_rag,
)
from edgar_capstone.common.concurrency import run_parallel_with_context
from edgar_capstone.observability.tracing import traced
from edgar_capstone.security.agent_identity import (
    VerifiedAgentS2SClaims,
    authorize_agent_s2s_hop,
    authorize_agent_tool_invocation,
    get_agent_s2s_broker,
)
from edgar_capstone.security.model_armor import (
    GuardrailDecision,
    get_model_armor_guardrail,
)
from edgar_capstone.security.rbac import CUSTOM_RUNTIME_SA_ACCOUNT_ID
from edgar_capstone.telemetry import get_telemetry_sink

logger = logging.getLogger(__name__)

# Active runtime agent roster exported to the API & server layer (Wave 4 contract).
PRIMARY_ORCHESTRATOR_NAME: str = "FinancialAnalystAgent"
ACTIVE_AGENT_NAMES: tuple[str, ...] = ("FinancialAnalystAgent", "NarrativeEvaluatorAgent")

# SPIFFE S2S identity key registered in `security.agent_identity.PER_AGENT_IDENTITIES` for
# the intent-classification hop & upstream pipeline attestation.
_S2S_ROUTER_IDENTITY_NAME: str = "RootOrchestrator"

_CORPUS_YEARS_GROUP = "|".join(str(y) for y in range(MIN_CORPUS_YEAR, MAX_CORPUS_YEAR + 1))
# Scope / metadata-filter resolution only (fiscal-year window and 10-K section filter). These are
# not intent classification: intent is emitted by FinancialAnalystAgent (`AnalystReport.intent`).
_YEAR_MENTION_RE = re.compile(rf"\b(?:FY\s*)?({_CORPUS_YEARS_GROUP})\b", re.IGNORECASE)
_SINCE_YEAR_RE = re.compile(rf"\bsince\s+(?:FY\s*)?(?:{_CORPUS_YEARS_GROUP})\b", re.IGNORECASE)
_ONLY_YEAR_RE = re.compile(
    rf"\b(?:just|only)\s+(?:FY\s*)?(?:{_CORPUS_YEARS_GROUP})\b", re.IGNORECASE
)
# Calendar-dated events ("the July 2024 outage", "after March 10, 2023"). A month pins a date
# that must be mapped onto each issuer's fiscal calendar; bare years are left as fiscal years.
_MONTH_NUMBERS: dict[str, int] = {
    name: idx
    for idx, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for name in names
}
_MONTH_GROUP = "|".join(sorted(_MONTH_NUMBERS, key=len, reverse=True))
_EVENT_DATE_RE = re.compile(
    rf"(?:\b(?P<after>after|following|since|post)[\s-]+(?:the\s+)?)?"
    rf"\b(?P<month>{_MONTH_GROUP})\.?\s+(?:(?P<day>\d{{1,2}})(?:st|nd|rd|th)?,?\s+)?"
    rf"(?P<year>{_CORPUS_YEARS_GROUP})\b",
    re.IGNORECASE,
)
_ITEM_1A_ONLY_RE = re.compile(r"\b(?:item\s*1a|risk\s+factors?)\b", re.IGNORECASE)
_ITEM_7_ONLY_RE = re.compile(r"\b(?:item\s*7|md&a|management'?s\s+discussion)\b", re.IGNORECASE)

_COMPANY_CATALOG_TTL_SECONDS = 600.0

ProgressCallback = Callable[[str, str, dict[str, Any] | None], None]


@dataclass
class _InFlightCatalogFetch:
    """Single-flight coordination state so concurrent cold misses share one BigQuery query."""

    done: threading.Event = field(default_factory=threading.Event)
    profiles: list[Any] = field(default_factory=list)
    catalog: EntityCatalog | None = None
    error: BaseException | None = None


class EntityCatalogCache:
    """Thread-safe single-flight TTL cache for the BigQuery S&P 500 profile catalog."""

    def __init__(self, ttl_seconds: float = _COMPANY_CATALOG_TTL_SECONDS) -> None:
        self._ttl_seconds = float(ttl_seconds)
        self._lock = threading.Lock()
        self._cached: tuple[float, list[Any], EntityCatalog] | None = None
        self._in_flight: _InFlightCatalogFetch | None = None

    def _get_fresh_locked(self, now: float) -> tuple[EntityCatalog, list[Any]] | None:
        if self._cached is not None and (now - self._cached[0]) < self._ttl_seconds:
            return self._cached[2], self._cached[1]
        return None

    def get_catalog_and_profiles(
        self,
        *,
        allow_fetch: bool = True,
        raise_on_error: bool = False,
    ) -> tuple[EntityCatalog, list[Any]]:
        """Return `(EntityCatalog, profiles)`, coalescing concurrent cold misses into one query."""
        now = time.monotonic()
        leader_flight: _InFlightCatalogFetch | None = None
        wait_flight: _InFlightCatalogFetch | None = None

        with self._lock:
            fresh = self._get_fresh_locked(now)
            if fresh is not None:
                return fresh
            if not allow_fetch:
                return EntityCatalog([]), []
            if self._in_flight is not None:
                wait_flight = self._in_flight
            else:
                leader_flight = _InFlightCatalogFetch()
                self._in_flight = leader_flight

        if wait_flight is not None:
            wait_flight.done.wait()
            if wait_flight.error is not None:
                if raise_on_error:
                    raise wait_flight.error
                return EntityCatalog([]), []
            return (wait_flight.catalog or EntityCatalog([])), wait_flight.profiles

        assert leader_flight is not None
        try:
            profiles = get_active_gcp_tool_adapters().bigquery.fetch_profiles(
                tickers=set(), start_year=1900, end_year=2100
            )
            catalog = EntityCatalog(profiles)
            leader_flight.profiles = profiles
            leader_flight.catalog = catalog
            with self._lock:
                self._cached = (time.monotonic(), profiles, catalog)
                self._in_flight = None
            leader_flight.done.set()
            return catalog, profiles
        except Exception as exc:
            leader_flight.error = exc
            with self._lock:
                self._in_flight = None
            leader_flight.done.set()
            if raise_on_error:
                raise
            logger.warning("Failed to load entity catalog from BigQuery: %s", exc)
            return EntityCatalog([]), []

    def load_profiles(self) -> list[Any]:
        """Return cached company-year profiles, raising if BigQuery fails."""
        _, profiles = self.get_catalog_and_profiles(allow_fetch=True, raise_on_error=True)
        return profiles

    def clear(self) -> None:
        """Clear cached profiles (for internal cache invalidation)."""
        with self._lock:
            self._cached = None


_ENTITY_CATALOG_CACHE = EntityCatalogCache()


def _load_all_profiles_from_bigquery() -> list[Any]:
    """All company-year profiles from BigQuery `financial_metrics` (cached for 10 minutes)."""
    return _ENTITY_CATALOG_CACHE.load_profiles()


def _get_entity_catalog(*, allow_fetch: bool = True) -> tuple[EntityCatalog, list[Any]]:
    """Return `(EntityCatalog, profiles)` from the single-flight 10-minute cache."""
    return _ENTITY_CATALOG_CACHE.get_catalog_and_profiles(
        allow_fetch=allow_fetch,
        raise_on_error=False,
    )


def list_curated_companies() -> dict[str, Any]:
    """Return all curated S&P 500 companies, sectors, and fiscal year bounds from BigQuery."""
    profiles = _load_all_profiles_from_bigquery()

    by_ticker: dict[str, list[Any]] = defaultdict(list)
    sectors: set[str] = set()
    all_years: list[int] = []

    for prof in profiles:
        by_ticker[prof.ticker].append(prof)
        if prof.gics_sector:
            sectors.add(prof.gics_sector)
        all_years.append(prof.fiscal_year)

    companies: list[dict[str, Any]] = []
    for ticker, t_profiles in sorted(by_ticker.items()):
        t_profiles.sort(key=lambda p: p.fiscal_year)
        latest = t_profiles[-1]
        latest_rec = _profile_to_metric_record(latest)
        companies.append(
            {
                "ticker": ticker,
                "company_name": latest.company_name,
                "cik": latest.cik,
                "gics_sector": latest.gics_sector,
                "gics_sub_industry": latest.gics_sub_industry,
                "fiscal_years": [p.fiscal_year for p in t_profiles],
                "latest_fiscal_year": latest.fiscal_year,
                "latest_period_of_report": latest.period_of_report or None,
                "latest_filing_date": latest.filing_date or None,
                "latest_accession_number": latest.accession_number or None,
                "latest_revenue": latest.revenue,
                "latest_operating_income": latest.operating_income,
                "latest_net_income": latest.net_income,
                "latest_research_and_development": latest.research_and_development,
                "latest_operating_margin_pct": latest_rec.operating_margin_pct,
                "latest_net_margin_pct": latest_rec.net_margin_pct,
                "latest_metric_provenance": latest.metric_provenance.model_dump(mode="json"),
                "sec_filing_url": latest.sec_filing_url,
            }
        )

    return {
        "status": "success",
        "company_count": len(companies),
        "total_profiles": len(profiles),
        "min_year": min(all_years) if all_years else MIN_CORPUS_YEAR,
        "max_year": max(all_years) if all_years else MAX_CORPUS_YEAR,
        "gics_sectors": sorted(sectors),
        "companies": companies,
    }


def _resolve_section_filter(query: str, section_id: str) -> str:
    if section_id and section_id.strip().upper() not in ("ALL", "*"):
        return section_id
    has_1a = bool(_ITEM_1A_ONLY_RE.search(query))
    has_7 = bool(_ITEM_7_ONLY_RE.search(query))
    if has_1a and not has_7:
        return "item_1a"
    if has_7 and not has_1a:
        return "item_7"
    return "ALL"


def _parse_event_date(query: str) -> tuple[date, bool] | None:
    """Return `(event_date, is_after)` for the first calendar-dated event in the query.

    `is_after` is True for "after / following / since / post-" phrasing, i.e. the user wants
    the change from before the event through the latest filings. Missing days default to the
    15th so the date falls inside the named month.
    """
    m = _EVENT_DATE_RE.search(query)
    if not m:
        return None
    year = int(m.group("year"))
    month = _MONTH_NUMBERS[m.group("month").lower()]
    day = int(m.group("day")) if m.group("day") else 15
    day = max(1, min(day, calendar.monthrange(year, month)[1]))
    return date(year, month, day), bool(m.group("after"))


def _event_fiscal_years(
    event_date: date,
    tickers: list[str],
    catalog: EntityCatalog | None,
) -> list[int]:
    """Fiscal year containing `event_date` for each ticker (calendar year as fallback)."""
    fys = [
        fy
        for tk in tickers
        if catalog is not None and (fy := catalog.fiscal_year_for_date(tk, event_date)) is not None
    ]
    return fys or [event_date.year]


def _resolve_year_window(
    query: str,
    start_year: int,
    end_year: int,
    *,
    memory_inherited: bool,
    memory_context: dict[str, Any] | None,
    tickers: list[str] | None = None,
    catalog: EntityCatalog | None = None,
) -> tuple[int, int]:
    mentioned_years = sorted({int(y) for y in _YEAR_MENTION_RE.findall(query)})
    eff_start = start_year
    eff_end = end_year
    event = _parse_event_date(query) if len(mentioned_years) == 1 else None
    is_default_window = start_year == 2024 and end_year == 2026
    if event is not None and (event[1] or is_default_window):
        # A dated event resolves on the issuer's fiscal calendar, always keeping the prior
        # fiscal year as the pre-event baseline. "After / since" extends through the latest
        # filings; a bare date ("the July 2024 outage") covers baseline + event year only.
        event_date, is_after = event
        event_fys = _event_fiscal_years(event_date, tickers or [], catalog)
        eff_start = max(MIN_CORPUS_YEAR, min(event_fys) - 1)
        eff_end = max(end_year, max(event_fys)) if is_after else max(event_fys)
    elif mentioned_years:
        if len(mentioned_years) >= 2:
            eff_start = min(mentioned_years)
            eff_end = max(mentioned_years)
        elif _SINCE_YEAR_RE.search(query):
            eff_start = mentioned_years[0]
            eff_end = max(end_year, mentioned_years[0])
        elif _ONLY_YEAR_RE.search(query):
            eff_start = mentioned_years[0]
            eff_end = mentioned_years[0]
        elif start_year == 2024 and end_year == 2026:
            yr = mentioned_years[0]
            eff_start = max(MIN_CORPUS_YEAR, yr - 1)
            eff_end = yr
    elif (
        memory_inherited
        and start_year == 2024
        and end_year == 2026
        and isinstance(memory_context, dict)
        and memory_context.get("last_start_year")
        and memory_context.get("last_end_year")
    ):
        eff_start = int(memory_context["last_start_year"])
        eff_end = int(memory_context["last_end_year"])

    if eff_start > eff_end:
        eff_start, eff_end = eff_end, eff_start
    return eff_start, eff_end


@traced("chat.resolve_scope", record_args=("tickers", "gics_sector", "section_id"))
def resolve_query_scope(
    query: str,
    tickers: str = "ALL",
    gics_sector: str = "ALL",
    section_id: str = "ALL",
    start_year: int = 2024,
    end_year: int = 2026,
    memory_context: dict[str, Any] | None = None,
    *,
    catalog: EntityCatalog | None = None,
) -> dict[str, Any]:
    """Combine explicit UI filter selections, full-catalog NL detection, and multi-turn memory."""
    if catalog is None:
        catalog, _ = _get_entity_catalog(allow_fetch=True)

    explicit_tickers = sorted(_parse_ticker_list(tickers))
    detected_tickers, unknown_entities = catalog.detect_tickers_in_query(query)

    prior_tickers: list[str] = []
    if isinstance(memory_context, dict) and memory_context.get("has_prior_memory"):
        raw_prior = memory_context.get("remembered_tickers") or []
        prior_tickers = [str(t) for t in raw_prior]
    has_prior_mem = bool(prior_tickers)

    memory_inherited = False
    removed_tickers = extract_removed_tickers(query, catalog) if has_prior_mem else []

    if explicit_tickers:
        effective_tickers = explicit_tickers
    elif removed_tickers and prior_tickers:
        effective_tickers = [t for t in prior_tickers if t not in removed_tickers]
        memory_inherited = True
    elif detected_tickers and prior_tickers and is_additive_followup(query):
        merged = list(prior_tickers)
        for tk in detected_tickers:
            if tk not in merged:
                merged.append(tk)
        effective_tickers = merged
        memory_inherited = True
    else:
        effective_tickers = detected_tickers

    effective_sector = gics_sector
    if not effective_sector or effective_sector.strip().upper() in ("ALL", "*"):
        effective_sector = "ALL"
        if not effective_tickers:
            for pattern, sec_name in SECTOR_ALIASES:
                if pattern.search(query):
                    effective_sector = sec_name
                    break

    broad_corpus_query = is_broad_corpus_query(query)
    if (
        not effective_tickers
        and effective_sector == "ALL"
        and not broad_corpus_query
        and not unknown_entities
        and has_prior_mem
    ):
        effective_tickers = list(prior_tickers)
        memory_inherited = True

    valid_tickers = (
        [t for t in effective_tickers if catalog.contains_ticker(t)]
        if catalog.tickers
        else list(effective_tickers)
    )
    effective_section = _resolve_section_filter(query, section_id)
    eff_start, eff_end = _resolve_year_window(
        query,
        start_year,
        end_year,
        memory_inherited=memory_inherited,
        memory_context=memory_context,
        tickers=valid_tickers,
        catalog=catalog,
    )

    return {
        "tickers": valid_tickers,
        "tickers_csv": ",".join(valid_tickers) if valid_tickers else "ALL",
        "gics_sector": effective_sector,
        "section_id": effective_section,
        "start_year": eff_start,
        "end_year": eff_end,
        "memory_inherited": memory_inherited,
        "explicit_tickers": explicit_tickers,
        "unknown_entities": unknown_entities,
    }


@traced("chat.retrieve_citations", record_args=("section_id", "top_k"))
def _retrieve_balanced_citations(
    query: str,
    tickers: list[str],
    gics_sector: str,
    section_id: str,
    start_year: int,
    end_year: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], int]:
    """Evidence-lead pre-pass: one coverage-aware hybrid RAG search over the resolved scope.

    `search_10k_chunks_rag` guarantees at least one passage per `(ticker, fiscal_year)` in a
    ticker-scoped window (see `hybrid_rag.coverage_slices`), which subsumes the old per-ticker
    and boundary-year fan-out (up to 5 parallel full searches) with a single search whose
    coverage query runs concurrently with `VECTOR_SEARCH`.
    """
    result = search_10k_chunks_rag(
        query=query,
        tickers=",".join(tickers) if tickers else "ALL",
        gics_sector=gics_sector,
        section_id=section_id,
        start_year=start_year,
        end_year=end_year,
        top_k=top_k,
    )
    return list(result.get("citations") or []), int(result.get("candidates_searched") or 0)


# ---------------------------------------------------------------------------
# Trace, Identity & Canonical Turn-Response Helpers
# ---------------------------------------------------------------------------


def _track_identity(
    identities_used_by_agent: dict[str, dict[str, str]],
    claims: VerifiedAgentS2SClaims,
) -> None:
    if claims.agent_name not in identities_used_by_agent:
        identities_used_by_agent[claims.agent_name] = {
            "agent_name": claims.agent_name,
            "service_account_email": claims.service_account_email,
            "spiffe_id": claims.spiffe_id,
        }


def _append_trace_step(
    agent_trace: list[dict[str, Any]],
    *,
    agent: str,
    claims: VerifiedAgentS2SClaims,
    tool: str,
    status: str,
    duration_ms: float,
    summary: str,
    model_used: str | None = None,
) -> None:
    """Append a standardized S2S-attested trace step to `agent_trace`."""
    entry: dict[str, Any] = {
        "step": len(agent_trace) + 1,
        "agent": agent,
        "agent_service_account": claims.service_account_email,
        "spiffe_id": claims.spiffe_id,
        "s2s_target_service": claims.target_service,
        "s2s_verified": True,
        "tool": tool,
        "status": status,
        "duration_ms": duration_ms,
        "summary": summary,
    }
    if model_used is not None:
        entry["model_used"] = model_used
    agent_trace.append(entry)


@dataclass
class _TurnContext:
    """Mutable per-turn plumbing shared across guardrails, scope validation, and finalization."""

    turn_start: float
    telemetry_session_id: str
    session_id: str | None
    user_id: str
    memory_ctx: dict[str, Any] = field(default_factory=dict)
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    identities_used_by_agent: dict[str, dict[str, str]] = field(default_factory=dict)
    on_progress: ProgressCallback | None = None

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.turn_start) * 1000.0, 2)

    def emit(self, stage: str, message: str, meta: dict[str, Any] | None = None) -> None:
        if self.on_progress is not None:
            self.on_progress(stage, message, meta)

    def track_identity(self, claims: VerifiedAgentS2SClaims) -> None:
        _track_identity(self.identities_used_by_agent, claims)

    def append_trace(
        self,
        *,
        agent: str,
        claims: VerifiedAgentS2SClaims,
        tool: str,
        status: str,
        duration_ms: float,
        summary: str,
        model_used: str | None = None,
    ) -> None:
        _append_trace_step(
            self.agent_trace,
            agent=agent,
            claims=claims,
            tool=tool,
            status=status,
            duration_ms=duration_ms,
            summary=summary,
            model_used=model_used,
        )


def _finalize_turn(
    ctx: _TurnContext,
    *,
    status: str,
    model_armor_blocked: bool,
    prompt_decision: GuardrailDecision,
    response_decision: GuardrailDecision | None = None,
    execution_mode: str,
    adk_runtime: dict[str, Any] | None = None,
    intent: str,
    intent_source: str = "scope_gate",
    resolved_scope: dict[str, Any],
    answer_markdown: str,
    answer_verified: bool,
    fact_check: dict[str, Any],
    react_execution: dict[str, Any],
    telemetry_tickers: list[str],
    telemetry_report_generated: bool,
    telemetry_fact_check_passed: bool,
    turn_latency_records: list[Any] | None = None,
    turn_latency_summary: Any | None = None,
    memory_persist: dict[str, Any] | None = None,
    structured_report: dict[str, Any] | None = None,
    report_verification: dict[str, Any] | None = None,
    clarification_reason: str | None = None,
    suggested_followups: list[str] | None = None,
    coverage_notices: list[dict[str, Any]] | None = None,
    bundle: _RetrievalBundle | None = None,
    citations_override: list[dict[str, Any]] | None = None,
    evaluation: dict[str, Any] | None = None,
    evaluation_grade: str | None = None,
    evaluation_score: float | None = None,
    emit_complete: bool = False,
) -> dict[str, Any]:
    """Record session telemetry and build the canonical turn response across ALL exit paths."""
    total_ms = ctx.elapsed_ms()
    eff_records = list(turn_latency_records or [])
    eff_citations = (
        list(citations_override)
        if citations_override is not None
        else (list(bundle.citations) if bundle is not None else [])
    )
    variance_reports = list(bundle.variance_reports) if bundle is not None else []
    tel_kwargs: dict[str, Any] = {
        "session_id": ctx.telemetry_session_id,
        "user_id": ctx.user_id,
        "intent": intent,
        "execution_mode": execution_mode,
        "status_outcome": status,
        "tickers_analyzed": telemetry_tickers,
        "gics_sector": str(resolved_scope.get("gics_sector") or "ALL"),
        "start_year": int(resolved_scope.get("start_year") or 2024),
        "end_year": int(resolved_scope.get("end_year") or 2026),
        "report_generated": telemetry_report_generated,
        "variance_reports_count": len(variance_reports),
        "citations_count": len(eff_citations),
        "agent_trace": ctx.agent_trace,
        "turn_latency_records": eff_records,
        "total_latency_ms": total_ms,
        "fact_check_passed": telemetry_fact_check_passed,
    }
    if evaluation_grade is not None:
        tel_kwargs["evaluation_grade"] = evaluation_grade
    if evaluation_score is not None:
        tel_kwargs["evaluation_score"] = evaluation_score
    sess_tel = get_telemetry_sink().record_session_turn_telemetry(**tel_kwargs)

    session_memory_payload = (
        {**ctx.memory_ctx, "persistence": memory_persist}
        if memory_persist is not None
        else ctx.memory_ctx
    )
    context_and_memory = {
        "models": get_model_strategy_metadata(),
        "session_memory": session_memory_payload,
        "turn_context_latency": (
            turn_latency_summary.model_dump(mode="json")
            if turn_latency_summary is not None
            else None
        ),
        "model_call_records": [r.model_dump(mode="json") for r in eff_records],
        "global_context_latency": (
            get_context_latency_tracker().summarize().model_dump(mode="json")
        ),
        "context_cache_runtime": get_context_cache_runtime_stats(),
        "session_telemetry": sess_tel["row"],
    }
    if emit_complete:
        ctx.emit("complete", "Grounded report ready.", {"total_latency_ms": total_ms})

    return {
        "status": status,
        "model_armor_blocked": model_armor_blocked,
        "model_armor": {
            "prompt_inspection": prompt_decision.to_audit_dict(),
            "response_inspection": (
                response_decision.to_audit_dict() if response_decision is not None else None
            ),
        },
        "execution_mode": execution_mode,
        "adk_runtime": (
            adk_runtime if adk_runtime is not None else get_live_runtime_status_metadata()
        ),
        "structured_report": structured_report,
        "report_verification": report_verification,
        "prompt_store": get_prompt_store_status_metadata(),
        "intent": intent,
        "intent_source": intent_source,
        "orchestrator": PRIMARY_ORCHESTRATOR_NAME,
        "resolved_scope": resolved_scope,
        "answer_markdown": answer_markdown,
        "answer_verified": answer_verified,
        "clarification_reason": clarification_reason,
        "suggested_followups": list(suggested_followups or []),
        "coverage_notices": list(coverage_notices or []),
        "sector_coverage": bundle.sector_coverage if bundle is not None else None,
        "missing_tickers": list(bundle.missing_tickers) if bundle is not None else [],
        "variance_reports": variance_reports,
        "sector_benchmarks": list(bundle.sector_benchmarks) if bundle is not None else [],
        "material_variance_flags": list(bundle.material_flags) if bundle is not None else [],
        "material_variance_summary": (
            bundle.material_flags_summary if bundle is not None else None
        ),
        "disclosure_deltas": bundle.disclosure_deltas if bundle is not None else None,
        "citations": eff_citations,
        "candidates_searched": bundle.candidates_searched if bundle is not None else 0,
        "context_cache": None,
        "fact_check": fact_check,
        "evaluation": evaluation,
        "react_execution": react_execution,
        "context_and_memory": context_and_memory,
        "agent_trace": ctx.agent_trace,
        "agent_identities_used": list(ctx.identities_used_by_agent.values()),
        "total_latency_ms": total_ms,
    }


def _early_return_response(
    ctx: _TurnContext,
    *,
    status: str,
    intent: str,
    title: str,
    message: str,
    scope: dict[str, Any],
    prompt_decision: GuardrailDecision,
    coverage_notices: list[dict[str, Any]],
    suggested_followups: list[str],
    reason: str | None,
) -> dict[str, Any]:
    """Build a structured clarification or out-of-corpus response without running LLM synthesis."""
    suggestions_md = ""
    if suggested_followups:
        bullets = "\n".join(f"- *{s}*" for s in suggested_followups)
        suggestions_md = f"\n\n**Try one of these queries:**\n{bullets}"
    answer_markdown = f"### {title}\n\n{message}{suggestions_md}\n"
    return _finalize_turn(
        ctx,
        status=status,
        model_armor_blocked=False,
        prompt_decision=prompt_decision,
        execution_mode=status,
        intent=intent,
        resolved_scope=scope,
        answer_markdown=answer_markdown,
        answer_verified=True,
        clarification_reason=reason,
        suggested_followups=suggested_followups,
        coverage_notices=coverage_notices,
        telemetry_tickers=list(scope.get("tickers") or []),
        telemetry_report_generated=False,
        telemetry_fact_check_passed=True,
        fact_check={
            "passed": True,
            "checked_metrics_count": 0,
            "matched_citations_count": 0,
            "has_required_citations": True,
            "discrepancies": [],
            "model_latency_ms": ctx.elapsed_ms(),
        },
        react_execution={
            "strategy": "scope_validation_gate",
            "execution_mode": status,
            "iterations": 0,
            "steps": [],
            "variance_evaluations": [],
            "cached_10k_explanations": [],
        },
    )


def _run_prompt_guardrail(
    ctx: _TurnContext,
    *,
    query: str,
    gics_sector: str,
    section_id: str,
    start_year: int,
    end_year: int,
    precomputed_decision: GuardrailDecision | None = None,
    precomputed_guard_ms: float | None = None,
) -> tuple[str, GuardrailDecision, dict[str, Any] | None]:
    """Inspect user prompt with Model Armor / SDP; return `(sanitized_query, decision, blocked_response)`."""
    t0 = time.perf_counter()
    if precomputed_decision is not None:
        prompt_decision = precomputed_decision
        elapsed_guard_ms = (
            round(float(precomputed_guard_ms), 2)
            if precomputed_guard_ms is not None
            else round(float(prompt_decision.latency_ms), 2)
        )
    else:
        guardrail_svc = get_model_armor_guardrail()
        prompt_decision = guardrail_svc.inspect_user_prompt(
            query,
            caller_agent=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
        )
        elapsed_guard_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    if prompt_decision.blocked:
        guard_claims = authorize_agent_s2s_hop(
            caller_agent=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
            callee_agent="ModelArmorGuardrail",
        )
        ctx.track_identity(guard_claims)
        cats_str = ", ".join(prompt_decision.matched_categories) or "policy_violation"
        ctx.append_trace(
            agent="ModelArmorGuardrail",
            claims=guard_claims,
            tool="model_armor_sanitize_user_prompt",
            status="blocked",
            duration_ms=elapsed_guard_ms,
            summary=(
                "Blocked user prompt via Google Cloud Model Armor / Compliance Filter "
                f"(categories=[{cats_str}], backend={prompt_decision.backend_used})."
            ),
        )
        blocked_intent = (
            "projection_compliance_block"
            if prompt_decision.projection_blocked
            else "safety_guardrail_block"
        )
        blocked_resp = _finalize_turn(
            ctx,
            status="blocked",
            model_armor_blocked=True,
            prompt_decision=prompt_decision,
            execution_mode="guardrail_blocked",
            intent=blocked_intent,
            resolved_scope={
                "tickers": [],
                "gics_sector": gics_sector,
                "section_id": section_id,
                "start_year": start_year,
                "end_year": end_year,
            },
            answer_markdown=(
                "### Request Blocked by Managed Model Armor & SEC Compliance Guardrails\n\n"
                f"{prompt_decision.block_message}"
            ),
            answer_verified=False,
            telemetry_tickers=[],
            telemetry_report_generated=False,
            telemetry_fact_check_passed=True,
            fact_check={
                "passed": True,
                "checked_metrics_count": 0,
                "matched_citations_count": 0,
                "has_required_citations": True,
                "discrepancies": [],
                "model_latency_ms": elapsed_guard_ms,
            },
            react_execution={
                "strategy": "structured_react_loop",
                "execution_mode": "guardrail_blocked",
                "iterations": 0,
                "steps": [],
                "variance_evaluations": [],
                "cached_10k_explanations": [],
            },
        )
        return query, prompt_decision, blocked_resp

    if prompt_decision.sdp_redacted:
        query = prompt_decision.sanitized_text
        guard_claims = authorize_agent_s2s_hop(
            caller_agent=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
            callee_agent="ModelArmorGuardrail",
        )
        ctx.track_identity(guard_claims)
        ctx.append_trace(
            agent="ModelArmorGuardrail",
            claims=guard_claims,
            tool="cloud_sdp_deidentify_content",
            status="sanitized",
            duration_ms=elapsed_guard_ms,
            summary=(
                "De-identified sensitive PII in user prompt via SDP "
                f"(info_types={list(prompt_decision.sdp_info_types)})."
            ),
        )

    return query, prompt_decision, None


@dataclass
class _RetrievalBundle:
    """Per-turn payloads: RAG pre-pass evidence + deterministic outputs the agent invoked."""

    variance_reports: list[dict[str, Any]] = field(default_factory=list)
    missing_tickers: list[str] = field(default_factory=list)
    sector_benchmarks: list[dict[str, Any]] = field(default_factory=list)
    chunk_health: dict[str, Any] | None = None
    calculated_variances_map: dict[str, Any] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    candidates_searched: int = 0
    material_flags: list[dict[str, Any]] = field(default_factory=list)
    material_flags_summary: dict[str, Any] | None = None
    disclosure_deltas: dict[str, Any] | None = None
    sector_coverage: dict[str, Any] | None = None
    sector_map_summaries: dict[str, Any] | None = None


def _run_rag_prepass(
    *,
    query: str,
    public_scope: dict[str, Any],
    top_k: int,
    rag_claims: VerifiedAgentS2SClaims,
    agent_trace: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Balanced hybrid-RAG evidence leads (the only pre-pass; deterministic engines are tools)."""
    ts = time.perf_counter()
    citations, candidates_searched = _retrieve_balanced_citations(
        query=query,
        tickers=public_scope["tickers"],
        gics_sector=public_scope["gics_sector"],
        section_id=public_scope["section_id"],
        start_year=public_scope["start_year"],
        end_year=public_scope["end_year"],
        top_k=top_k,
    )
    _append_trace_step(
        agent_trace,
        agent="FinancialAnalystAgent",
        claims=rag_claims,
        tool="search_10k_chunks_rag",
        status="completed",
        duration_ms=round((time.perf_counter() - ts) * 1000.0, 2),
        summary=(
            f"Ranked {candidates_searched:,} metadata-filtered 10-K chunks (BigQuery VECTOR_SEARCH + "
            f"Vertex AI Search) → top {len(citations)} grounded citations."
        ),
    )
    return citations, candidates_searched


def _bundle_from_tool_outputs(
    outputs: ToolOutputs,
    *,
    citations: list[dict[str, Any]],
    candidates_searched: int,
) -> _RetrievalBundle:
    """Build the turn bundle from deterministic tool outputs hydrated from ADK session_state."""
    sector_cov = outputs.sector_coverage
    return _RetrievalBundle(
        variance_reports=list(outputs.variance_reports),
        missing_tickers=list(outputs.missing_tickers),
        sector_benchmarks=list(outputs.sector_benchmarks),
        chunk_health=outputs.chunk_health,
        calculated_variances_map=outputs.calculated_variances_map,
        citations=list(citations),
        candidates_searched=(
            int((sector_cov or {}).get("total_chunks_evaluated") or 0) or candidates_searched
        ),
        material_flags=list(outputs.material_variance_flags),
        material_flags_summary=outputs.material_variance_summary,
        disclosure_deltas=outputs.disclosure_deltas,
        sector_coverage=sector_cov,
        sector_map_summaries=outputs.sector_map_summaries,
    )


def _agent_tool_summary(tool: str, bundle: _RetrievalBundle, calls: list[dict[str, Any]]) -> str:
    if tool == "calculate_period_variances":
        tickers = ", ".join(r["ticker"] for r in bundle.variance_reports if r.get("ticker"))
        return (
            "Agent-invoked deterministic Revenue, Operating Income, Net Income, R&D, XBRL "
            f"provenance, and margin bps variances for {len(bundle.variance_reports)} issuer(s)"
            f"{f' ({tickers})' if tickers else ''}."
        )
    if tool == "flag_material_variances":
        summ = bundle.material_flags_summary or {}
        return (
            f"Agent-invoked anomaly detection flagged {summ.get('total_flags', len(bundle.material_flags))} "
            f"material variance(s) ({summ.get('high_severity_count', 0)} high severity); attached "
            f"MD&A explanations to {summ.get('explained_flags', 0)}."
        )
    if tool == "compare_disclosures_across_years":
        d_sum = (bundle.disclosure_deltas or {}).get("summary") or {}
        return (
            f"Agent-invoked theme disclosure comparison for {d_sum.get('companies_analyzed', 0)} "
            f"company(ies) across {d_sum.get('year_pairs', 0)} year pair(s): "
            f"{d_sum.get('added', 0)} added, {d_sum.get('modified', 0)} modified, "
            f"{d_sum.get('removed', 0)} removed."
        )
    if tool == "sector_thematic_map_reduce":
        cov = bundle.sector_coverage or {}
        return (
            f"Agent-invoked full-sector batched map-reduce analyzed {cov.get('issuers_analyzed', 0)}/"
            f"{cov.get('issuers_in_sector', 0)} {cov.get('gics_sector', '')} issuers "
            f"({int(cov.get('total_chunks_evaluated', 0) or 0):,} passages in "
            f"{cov.get('batches_processed', 0)} parallel Flash batches)."
        )
    if tool in ("query_analytics_view", "query_looker_financial_explore"):
        explores = sorted(
            {str((c.get("args") or {}).get("explore_name") or "") for c in calls} - {""}
        )
        return (
            f"Agent-invoked analytics view(s) {explores}: {len(bundle.sector_benchmarks)} sector "
            f"benchmark row(s); chunk health "
            f"{(bundle.chunk_health or {}).get('health_status', 'not requested')}."
        )
    return f"Agent-invoked {tool} ({len(calls)} call(s))."


def _append_agent_tool_trace_steps(
    *,
    agent_trace: list[dict[str, Any]],
    identities_used_by_agent: dict[str, dict[str, str]],
    tool_calls: list[dict[str, Any]],
    bundle: _RetrievalBundle,
    turn_ctx: ToolTurnContext,
) -> list[str]:
    """S2S-attested trace steps for every deterministic tool and cached 10-K lookup the agent invoked."""
    invoked: dict[str, list[dict[str, Any]]] = {}
    cache_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        name = str(call.get("name") or "")
        if name in DETERMINISTIC_TOOL_NAMES:
            invoked.setdefault(name, []).append(call)
        elif name == "query_cached_10k_context":
            cache_calls.append(call)
    for name, calls in invoked.items():
        claims = authorize_agent_tool_invocation(
            agent_name="FinancialAnalystAgent",
            tool_name=name,
            caller=_S2S_ROUTER_IDENTITY_NAME,
        )
        _track_identity(identities_used_by_agent, claims)
        trace_tool = (
            f"{name}({','.join(sorted({str((c.get('args') or {}).get('explore_name') or '') for c in calls}))})"
            if name in ("query_analytics_view", "query_looker_financial_explore")
            else name
        )
        _append_trace_step(
            agent_trace,
            agent="FinancialAnalystAgent",
            claims=claims,
            tool=trace_tool,
            status="completed",
            duration_ms=turn_ctx.tool_duration_ms(name),
            summary=_agent_tool_summary(name, bundle, calls),
        )
    if cache_calls:
        authorize_agent_s2s_hop(
            caller_agent="FinancialAnalystAgent",
            callee_agent="ContextCacheManager",
        )
        c_claims = authorize_agent_tool_invocation(
            agent_name="ContextCacheManager",
            tool_name="query_cached_10k_context",
            caller="FinancialAnalystAgent",
        )
        _track_identity(identities_used_by_agent, c_claims)
        _append_trace_step(
            agent_trace,
            agent="ContextCacheManager",
            claims=c_claims,
            tool="query_cached_10k_context",
            status="completed",
            duration_ms=turn_ctx.tool_duration_ms("query_cached_10k_context"),
            summary=(f"Agent-invoked cached 10-K section lookup ({len(cache_calls)} call(s))."),
        )
    return sorted(invoked)


def _inspect_response_guardrail(
    ctx: _TurnContext,
    *,
    react_result: dict[str, Any],
    answer_markdown: str,
    prompt_decision: GuardrailDecision,
    public_scope: dict[str, Any],
    coverage_notices: list[dict[str, Any]],
    bundle: _RetrievalBundle,
) -> tuple[str, GuardrailDecision, bool, dict[str, Any] | None]:
    """Reconcile ADK/Model Armor response inspection; return `(text, decision, sdp_modified, blocked_resp)`."""
    t_resp_armor = time.perf_counter()
    if react_result.get("model_armor_response_blocked") is True:
        raw_plugin_dec = react_result.get("model_armor_response_decision") or {}
        resp_decision = GuardrailDecision.from_audit_dict(
            {**raw_plugin_dec, "blocked": True, "action": "blocked"},
            text="",
            block_message=str(
                react_result.get("model_armor_response_block_message") or answer_markdown
            ),
        )
    elif react_result.get("adk_response_already_inspected") and isinstance(
        react_result.get("adk_response_decision"), dict
    ):
        resp_decision = GuardrailDecision.from_audit_dict(
            react_result["adk_response_decision"],
            text=answer_markdown,
        )
    else:
        resp_decision = get_model_armor_guardrail().inspect_model_response(
            answer_markdown,
            caller_agent="FinancialAnalystAgent",
        )

    if resp_decision.blocked:
        out_guard_claims = authorize_agent_s2s_hop(
            caller_agent="FinancialAnalystAgent",
            callee_agent="ModelArmorGuardrail",
        )
        ctx.track_identity(out_guard_claims)
        out_guard_ms = round((time.perf_counter() - t_resp_armor) * 1000.0, 2)
        cats_str = ", ".join(resp_decision.matched_categories) or "policy_violation"
        ctx.append_trace(
            agent="ModelArmorGuardrail",
            claims=out_guard_claims,
            tool="model_armor_sanitize_model_response",
            status="blocked",
            duration_ms=out_guard_ms,
            summary=(
                "Withheld model response via Model Armor / SEC Projection Filter "
                f"(categories=[{cats_str}], backend={resp_decision.backend_used})."
            ),
        )
        resp_blocked_intent = (
            "projection_compliance_block"
            if resp_decision.projection_blocked
            else "safety_guardrail_block"
        )
        react_steps = react_result["react_steps"]
        blocked_resp = _finalize_turn(
            ctx,
            status="blocked",
            model_armor_blocked=True,
            prompt_decision=prompt_decision,
            response_decision=resp_decision,
            execution_mode="guardrail_blocked",
            adk_runtime=react_result["adk_runtime"],
            intent=resp_blocked_intent,
            resolved_scope=public_scope,
            answer_markdown=(
                "### Response Withheld by Managed Model Armor & SEC Compliance Guardrails\n\n"
                f"{resp_decision.block_message}"
            ),
            answer_verified=False,
            coverage_notices=coverage_notices,
            bundle=bundle,
            citations_override=[],
            telemetry_tickers=public_scope["tickers"],
            telemetry_report_generated=False,
            telemetry_fact_check_passed=False,
            turn_latency_records=react_result["turn_latency_records"],
            turn_latency_summary=react_result["turn_latency_summary"],
            fact_check={
                "passed": False,
                "checked_metrics_count": 0,
                "matched_citations_count": 0,
                "has_required_citations": False,
                "discrepancies": list(resp_decision.matched_categories),
                "model_latency_ms": out_guard_ms,
            },
            react_execution={
                "strategy": "structured_react_loop",
                "execution_mode": react_result["execution_mode"],
                "iterations": len(react_steps),
                "steps": [s.model_dump(mode="json") for s in react_steps],
                "variance_evaluations": [
                    v.model_dump(mode="json") for v in react_result["variance_evaluations"]
                ],
                "cached_10k_explanations": [
                    e.model_dump(mode="json") for e in react_result["cached_explanations"]
                ],
            },
        )
        return answer_markdown, resp_decision, False, blocked_resp

    sdp_modified = False
    if resp_decision.sdp_redacted:
        if resp_decision.sanitized_text and resp_decision.sanitized_text != answer_markdown:
            answer_markdown = resp_decision.sanitized_text
            sdp_modified = True
        out_guard_claims = authorize_agent_s2s_hop(
            caller_agent="FinancialAnalystAgent",
            callee_agent="ModelArmorGuardrail",
        )
        ctx.track_identity(out_guard_claims)
        ctx.append_trace(
            agent="ModelArmorGuardrail",
            claims=out_guard_claims,
            tool="cloud_sdp_deidentify_content",
            status="sanitized",
            duration_ms=round((time.perf_counter() - t_resp_armor) * 1000.0, 2),
            summary=(
                "De-identified sensitive PII in synthesized model response via SDP "
                f"(info_types={list(resp_decision.sdp_info_types)})."
            ),
        )

    return answer_markdown, resp_decision, sdp_modified, None


def _run_parallel_prepass(
    *,
    memory_mgr: Any,
    session_id: str | None,
    user_id: str,
    query: str,
) -> tuple[dict[str, Any], tuple[GuardrailDecision, float], tuple[EntityCatalog, list[Any]]]:
    """Overlap independent Firestore memory recall, Model Armor prompt check, and catalog fetch."""

    def _run_branch(branch: str) -> Any:
        if branch == "memory":
            return memory_mgr.recall_session_context(session_id, user_id=user_id)
        if branch == "guardrail":
            t_g = time.perf_counter()
            dec = get_model_armor_guardrail().inspect_user_prompt(
                query,
                caller_agent=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
            )
            return dec, (time.perf_counter() - t_g) * 1000.0
        return _get_entity_catalog(allow_fetch=True)

    return tuple(  # type: ignore[return-value]
        run_parallel_with_context(["memory", "guardrail", "catalog"], _run_branch, max_workers=3)
    )


def _resolve_and_validate_scope(
    ctx: _TurnContext,
    *,
    query: str,
    tickers: str,
    gics_sector: str,
    section_id: str,
    start_year: int,
    end_year: int,
    catalog: EntityCatalog,
    prompt_decision: GuardrailDecision,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, dict[str, Any] | None]:
    """Resolve scope, run coverage/clarification gates, and return `(public_scope, pre_notices, provisional_intent, early_resp)`."""
    t1 = time.perf_counter()
    route_claims = authorize_agent_tool_invocation(
        agent_name=_S2S_ROUTER_IDENTITY_NAME,
        tool_name="resolve_query_scope",
        caller=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
    )
    ctx.track_identity(route_claims)
    scope = resolve_query_scope(
        query=query,
        tickers=tickers,
        gics_sector=gics_sector,
        section_id=section_id,
        start_year=start_year,
        end_year=end_year,
        memory_context=ctx.memory_ctx,
        catalog=catalog,
    )
    validation = validate_query_scope_and_coverage(
        query=query,
        scope=scope,
        explicit_tickers=list(scope.get("explicit_tickers") or []),
        unknown_entities=list(scope.get("unknown_entities") or []),
        catalog=catalog,
        memory_context=ctx.memory_ctx,
    )
    pre_notices: list[dict[str, Any]] = list(validation.get("pre_notices") or [])
    public_scope = {
        k: v for k, v in scope.items() if k not in ("explicit_tickers", "unknown_entities")
    }

    if validation["action"] in ("clarification", "out_of_corpus"):
        is_clarification = validation["action"] == "clarification"
        early_status = "clarification" if is_clarification else "out_of_corpus"
        early_intent = "clarification_needed" if is_clarification else "out_of_corpus"
        early_title = (
            "Clarification Needed"
            if is_clarification
            else "Outside Curated SEC 10-K Corpus Coverage"
        )
        ctx.append_trace(
            agent=_S2S_ROUTER_IDENTITY_NAME,
            claims=route_claims,
            model_used=resolve_model_for_task("orchestration"),
            tool="validate_query_scope_and_coverage",
            status=early_status,
            duration_ms=round((time.perf_counter() - t1) * 1000.0, 2),
            summary=(
                f"Scope validation returned '{early_intent}' (reason={validation.get('reason')})."
            ),
        )
        ctx.emit("scope_validation", early_title, {"action": early_status})
        early_resp = _early_return_response(
            ctx,
            status=early_status,
            intent=early_intent,
            title=early_title,
            message=str(validation["message"]),
            scope=public_scope,
            prompt_decision=prompt_decision,
            coverage_notices=pre_notices,
            suggested_followups=list(validation.get("suggested_followups") or []),
            reason=validation.get("reason"),
        )
        return public_scope, pre_notices, early_intent, early_resp

    # Intent is emitted by FinancialAnalystAgent (`AnalystReport.intent`); the router only
    # resolves scope. `provisional_intent` is a structural (non-lexical) routing hint.
    provisional_intent = infer_intent_from_tool_calls(public_scope, [])
    sector_wide_scope = is_sector_wide_scope(public_scope)
    orchestrator_model = resolve_model_for_task("orchestration")
    mem_suffix = " (recalled from session memory)" if public_scope.get("memory_inherited") else ""
    ctx.append_trace(
        agent=_S2S_ROUTER_IDENTITY_NAME,
        claims=route_claims,
        model_used=orchestrator_model,
        tool="resolve_query_scope",
        status="completed",
        duration_ms=round((time.perf_counter() - t1) * 1000.0, 2),
        summary=(
            f"Resolved scope ({orchestrator_model}) · Tickers: {public_scope['tickers_csv']}{mem_suffix} · "
            f"Sector: {public_scope['gics_sector']} · Section: {public_scope['section_id']} · "
            f"FY{public_scope['start_year']}–FY{public_scope['end_year']} · intent and "
            "deterministic tools are selected by FinancialAnalystAgent"
        ),
    )
    ctx.emit(
        "scope_resolved",
        (
            f"Scoped to {public_scope['gics_sector']} sector (FY{public_scope['start_year']}–FY{public_scope['end_year']})"
            if sector_wide_scope
            else f"Scoped to {public_scope['tickers_csv']} (FY{public_scope['start_year']}–FY{public_scope['end_year']})"
        ),
        {
            "intent": None,
            "provisional_intent": provisional_intent,
            "tickers": public_scope["tickers"],
            "gics_sector": public_scope["gics_sector"],
            "sector_wide_scope": sector_wide_scope,
        },
    )
    return public_scope, pre_notices, provisional_intent, None


def _verify_and_finalize_success(
    ctx: _TurnContext,
    *,
    query: str,
    public_scope: dict[str, Any],
    prompt_decision: GuardrailDecision,
    react_result: dict[str, Any],
    bundle: _RetrievalBundle,
    coverage_notices: list[dict[str, Any]],
    intent: str,
    intent_source: str,
    memory_mgr: Any,
) -> dict[str, Any]:
    """Run S2S fact-check attestation, response guardrail, memory persistence, and turn finalization."""
    answer_markdown: str = react_result["answer_markdown"]
    citations: list[dict[str, Any]] = react_result["citations"]
    variance_evaluations = react_result["variance_evaluations"]
    cached_explanations = react_result["cached_explanations"]
    react_steps = react_result["react_steps"]
    eval_report = react_result["evaluation_report"]
    reasoning_model: str = react_result["reasoning_model"]
    evaluation_model: str = react_result["evaluation_model"]
    active_execution_mode: str = react_result["execution_mode"]
    adk_runtime_meta: dict[str, Any] = react_result["adk_runtime"]

    ctx.emit(
        "verifying", "Verifying citations, XBRL provenance, and compliance guardrails...", None
    )
    t5 = time.perf_counter()
    s2s_broker = get_agent_s2s_broker()
    raw_fc_assertion = s2s_broker.mint_s2s_assertion(
        agent_name="FinancialAnalystAgent",
        target_service="fact-check-verification-service",
        caller=_S2S_ROUTER_IDENTITY_NAME,
    )
    fc_claims = s2s_broker.verify_s2s_assertion(
        raw_fc_assertion,
        expected_target_service="fact-check-verification-service",
    )
    ctx.track_identity(fc_claims)
    synth_step = next(
        (s for s in react_steps if s.phase == "format_grounded_output"), react_steps[-1]
    )
    fc_ms = round((time.perf_counter() - t5) * 1000.0 + synth_step.duration_ms, 2)

    answer_markdown, resp_decision, sdp_modified, blocked_resp = _inspect_response_guardrail(
        ctx,
        react_result=react_result,
        answer_markdown=answer_markdown,
        prompt_decision=prompt_decision,
        public_scope=public_scope,
        coverage_notices=coverage_notices,
        bundle=bundle,
    )
    if blocked_resp is not None:
        return blocked_resp

    # Reuse the single FactCheckVerification computed by `react_loop` unless SDP modified the text.
    cached_fc = react_result.get("fact_check_verification")
    if cached_fc is not None and not sdp_modified:
        fact_check = cached_fc.model_copy(update={"model_latency_ms": round(fc_ms, 2)})
    else:
        fact_check = verify_narrative_against_calculations(
            response_text=answer_markdown,
            calculated_variances=bundle.calculated_variances_map,
            retrieved_citations=citations,
            model_latency_ms=fc_ms,
        )

    score_label = (
        f"{eval_report.overall_score:.2f}" if eval_report.overall_score is not None else "pending"
    )
    ctx.append_trace(
        agent="FinancialAnalystAgent",
        claims=fc_claims,
        model_used=f"{reasoning_model} + {evaluation_model}",
        tool="after_model_fact_check_callback",
        status="completed" if fact_check.passed else "warning",
        duration_ms=fc_ms,
        summary=(
            f"Formatted grounded report (`{reasoning_model}`) & verification "
            f"(judge_mode={react_result.get('judge_mode', 'async_background')}, "
            f"grade={eval_report.grade}, score={score_label}) · "
            f"Verified {fact_check.checked_metrics_count} variance figure(s) & "
            f"{fact_check.matched_citations_count} citation tag(s) · Passed: {fact_check.passed}."
        ),
    )

    variance_highlights = [
        ev.material_findings[0] for ev in variance_evaluations if ev.material_findings
    ]
    memory_persist = memory_mgr.record_turn(
        session_id=ctx.session_id,
        user_id=ctx.user_id,
        query=query,
        intent=intent,
        tickers=public_scope["tickers"],
        gics_sector=public_scope["gics_sector"],
        start_year=public_scope["start_year"],
        end_year=public_scope["end_year"],
        variance_highlights=variance_highlights,
        cited_chunk_ids=[c["chunk_id"] for c in citations],
        cached_10k_keys=[e.cache_key for e in cached_explanations],
        answer_markdown=answer_markdown,
    )

    analyzed_tickers = sorted(
        {
            *(public_scope.get("tickers") or []),
            *(r["ticker"] for r in bundle.variance_reports if r.get("ticker")),
            *(c["ticker"] for c in citations if c.get("ticker")),
        }
    )
    return _finalize_turn(
        ctx,
        status="success",
        model_armor_blocked=False,
        prompt_decision=prompt_decision,
        response_decision=resp_decision,
        execution_mode=active_execution_mode,
        adk_runtime=adk_runtime_meta,
        structured_report=adk_runtime_meta.get("structured_report"),
        report_verification=adk_runtime_meta.get("report_verification"),
        intent=intent,
        intent_source=intent_source,
        resolved_scope=public_scope,
        answer_markdown=answer_markdown,
        answer_verified=bool(react_result.get("answer_verified")),
        coverage_notices=coverage_notices,
        bundle=bundle,
        citations_override=citations,
        fact_check=fact_check.model_dump(mode="json"),
        evaluation=eval_report.model_dump(mode="json"),
        evaluation_grade=eval_report.grade,
        evaluation_score=eval_report.overall_score,
        telemetry_tickers=analyzed_tickers,
        telemetry_report_generated=bool(react_result.get("answer_verified")),
        telemetry_fact_check_passed=fact_check.passed,
        turn_latency_records=react_result["turn_latency_records"],
        turn_latency_summary=react_result["turn_latency_summary"],
        memory_persist=memory_persist,
        emit_complete=True,
        react_execution={
            "strategy": "structured_react_loop",
            "execution_mode": active_execution_mode,
            "observed_ttft_ms": adk_runtime_meta.get("observed_ttft_ms"),
            "judge_mode": react_result.get("judge_mode", "async_background"),
            "iterations": len(react_steps),
            "steps": [s.model_dump(mode="json") for s in react_steps],
            "variance_evaluations": [v.model_dump(mode="json") for v in variance_evaluations],
            "cached_10k_explanations": [e.model_dump(mode="json") for e in cached_explanations],
        },
    )


@traced(
    "chat.rag_turn",
    record_args=(
        "tickers",
        "gics_sector",
        "section_id",
        "start_year",
        "end_year",
        "top_k",
        "session_id",
    ),
)
def run_conversational_rag_turn(
    *,
    query: str,
    tickers: str = "ALL",
    gics_sector: str = "ALL",
    section_id: str = "ALL",
    start_year: int = 2024,
    end_year: int = 2026,
    top_k: int = 8,
    session_id: str | None = None,
    user_id: str = "analyst",
    reasoning_llm: BaseLlm | None = None,
    judge_llm: BaseLlm | None = None,
    map_llm: BaseLlm | None = None,
    run_judge_sync: bool = False,
    skip_background_judge: bool = False,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Execute a full source-grounded Hybrid RAG conversational turn with ADK Runner & streaming progress."""
    ctx = _TurnContext(
        turn_start=time.perf_counter(),
        telemetry_session_id=(session_id or "").strip() or f"ephemeral-{uuid.uuid4().hex[:10]}",
        session_id=session_id,
        user_id=user_id,
        on_progress=on_progress,
    )
    ctx.emit("starting", "Checking safety guardrails and session memory...", None)

    # Step 0: Parallel pre-pass (Firestore memory recall, Model Armor prompt check, entity catalog)
    memory_mgr = get_conversation_memory_manager()
    memory_ctx, (raw_prompt_decision, raw_guard_ms), (catalog, all_profiles) = (
        _run_parallel_prepass(
            memory_mgr=memory_mgr,
            session_id=session_id,
            user_id=user_id,
            query=query,
        )
    )
    ctx.memory_ctx = memory_ctx

    query, prompt_decision, blocked_prompt_resp = _run_prompt_guardrail(
        ctx,
        query=query,
        gics_sector=gics_sector,
        section_id=section_id,
        start_year=start_year,
        end_year=end_year,
        precomputed_decision=raw_prompt_decision,
        precomputed_guard_ms=raw_guard_ms,
    )
    if blocked_prompt_resp is not None:
        return blocked_prompt_resp

    # Step 1: Scope resolution, S&P 500 catalog validation, and clarification / out-of-corpus gate
    public_scope, pre_notices, provisional_intent, early_resp = _resolve_and_validate_scope(
        ctx,
        query=query,
        tickers=tickers,
        gics_sector=gics_sector,
        section_id=section_id,
        start_year=start_year,
        end_year=end_year,
        catalog=catalog,
        prompt_decision=prompt_decision,
    )
    if early_resp is not None:
        return early_resp

    # Step 2: Hybrid RAG evidence-lead pre-pass (deterministic engines are agent-invoked tools)
    authorize_agent_s2s_hop(
        caller_agent=_S2S_ROUTER_IDENTITY_NAME,
        callee_agent="FinancialAnalystAgent",
    )
    rag_claims = authorize_agent_tool_invocation(
        agent_name="FinancialAnalystAgent",
        tool_name="search_10k_chunks_rag",
        caller=_S2S_ROUTER_IDENTITY_NAME,
    )
    ctx.track_identity(rag_claims)
    ctx.emit("retrieving", "Retrieving 10-K evidence leads...", None)
    prepass_citations, prepass_candidates = _run_rag_prepass(
        query=query,
        public_scope=public_scope,
        top_k=top_k,
        rag_claims=rag_claims,
        agent_trace=ctx.agent_trace,
    )
    base_coverage_notices = build_coverage_notices(
        variance_reports=[],
        query=query,
        pre_notices=pre_notices,
        profiles=all_profiles,
        tickers=list(public_scope["tickers"]),
        start_year=int(public_scope["start_year"]),
        end_year=int(public_scope["end_year"]),
    )

    # Step 3 & Step 4: Structured ReAct loop; the agent invokes deterministic tools as needed
    with tool_turn_context(
        on_progress=on_progress,
        map_llm=map_llm,
        cached_profiles=all_profiles,
    ) as turn_ctx:
        react_result = StructuredReActExecutor(
            reasoning_llm=reasoning_llm,
            judge_llm=judge_llm,
        ).execute_post_retrieval_reasoning(
            query=query,
            intent=provisional_intent,
            scope=public_scope,
            citations=prepass_citations,
            memory_tokens=int(memory_ctx.get("memory_tokens", 0) or 0),
            memory_prep_latency_ms=float(memory_ctx.get("memory_lookup_latency_ms", 0.0) or 0.0),
            memory_context=memory_ctx,
            session_id=session_id,
            telemetry_session_id=ctx.telemetry_session_id,
            user_id=user_id,
            coverage_notices=base_coverage_notices,
            prompt_already_verified=True,
            run_judge_sync=run_judge_sync,
            skip_background_judge=skip_background_judge,
            on_progress=on_progress,
        )

    bundle = _bundle_from_tool_outputs(
        react_result.get("tool_outputs") or ToolOutputs(),
        citations=react_result["citations"],
        candidates_searched=prepass_candidates,
    )
    coverage_notices: list[dict[str, Any]] = list(
        react_result.get("coverage_notices") or base_coverage_notices
    )
    agent_tool_calls: list[dict[str, Any]] = list(react_result.get("agent_tool_calls") or [])
    intent, intent_source = resolve_turn_intent(
        agent_intent=react_result.get("agent_intent"),
        scope=public_scope,
        tool_calls=agent_tool_calls,
    )

    eval_var_step = react_result["react_steps"][0]
    ctx.append_trace(
        agent="FinancialAnalystAgent",
        claims=rag_claims,
        model_used=react_result["reasoning_model"],
        tool="evaluate_calculated_variances",
        status="completed",
        duration_ms=round(eval_var_step.duration_ms, 2),
        summary=(
            f"ReAct [Thought→Action]: Evaluated {len(react_result['variance_evaluations'])} variance report(s) "
            f"using `{react_result['reasoning_model']}` (context_latency={eval_var_step.context_latency_ms:.2f}ms)."
        ),
    )
    invoked_tools = _append_agent_tool_trace_steps(
        agent_trace=ctx.agent_trace,
        identities_used_by_agent=ctx.identities_used_by_agent,
        tool_calls=agent_tool_calls,
        bundle=bundle,
        turn_ctx=turn_ctx,
    )
    ctx.emit(
        "intent_resolved",
        f"Intent: {intent} ({intent_source}); deterministic tools: {', '.join(invoked_tools) or 'none'}",
        {"intent": intent, "intent_source": intent_source, "agent_invoked_tools": invoked_tools},
    )

    return _verify_and_finalize_success(
        ctx,
        query=query,
        public_scope=public_scope,
        prompt_decision=prompt_decision,
        react_result=react_result,
        bundle=bundle,
        coverage_notices=coverage_notices,
        intent=intent,
        intent_source=intent_source,
        memory_mgr=memory_mgr,
    )


__all__ = [
    "ACTIVE_AGENT_NAMES",
    "PRIMARY_ORCHESTRATOR_NAME",
    "EntityCatalogCache",
    "ProgressCallback",
    "list_curated_companies",
    "resolve_query_scope",
    "run_conversational_rag_turn",
]
