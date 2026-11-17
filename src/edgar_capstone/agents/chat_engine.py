"""Conversational Hybrid RAG orchestration engine for the SEC EDGAR Natural Language Analyst UI."""

from __future__ import annotations

import contextvars
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from google.adk.models.base_llm import BaseLlm

from edgar_capstone.agents.callbacks import verify_narrative_against_calculations
from edgar_capstone.agents.citations import merge_citations
from edgar_capstone.agents.context_memory import (
    get_context_latency_tracker,
    get_conversation_memory_manager,
    get_model_strategy_metadata,
    resolve_model_for_task,
)
from edgar_capstone.agents.disclosure_deltas import (
    compare_disclosures_across_years,
    disclosure_delta_citations,
)
from edgar_capstone.agents.live_runtime import get_live_runtime_status_metadata
from edgar_capstone.agents.material_variances import (
    attach_flag_explanations,
    flag_material_variances,
    flag_material_variances_from_cached_profiles,
)
from edgar_capstone.agents.prompt_store import get_prompt_store_status_metadata
from edgar_capstone.agents.react_loop import StructuredReActExecutor
from edgar_capstone.agents.root_orchestrator import (
    RootOrchestrator,
    classify_query_intent,
)
from edgar_capstone.agents.scope_validation import (
    MAX_CORPUS_YEAR,
    MIN_CORPUS_YEAR,
    SECTOR_ALIASES,
    EntityCatalog,
    build_coverage_notices,
    extract_removed_tickers,
    is_additive_followup,
    validate_query_scope_and_coverage,
)
from edgar_capstone.agents.sector_map_reduce import run_sector_thematic_map_reduce
from edgar_capstone.agents.tools import (
    _parse_ticker_list,
    _profile_to_metric_record,
    calculate_period_variances,
    get_active_gcp_tool_adapters,
    get_context_cache_runtime_stats,
    query_looker_financial_explore,
    search_10k_chunks_rag,
)
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

_YEAR_MENTION_RE = re.compile(r"\b(?:FY\s*)?(202[0-6])\b", re.IGNORECASE)
_ITEM_1A_ONLY_RE = re.compile(r"\b(?:item\s*1a|risk\s+factors?)\b", re.IGNORECASE)
_ITEM_7_ONLY_RE = re.compile(r"\b(?:item\s*7|md&a|management'?s\s+discussion)\b", re.IGNORECASE)

_COMPANY_CATALOG_TTL_SECONDS = 600.0
_CATALOG_LOCK = threading.Lock()
_company_catalog_cache: tuple[float, list[Any], EntityCatalog] | None = None

ProgressCallback = Callable[[str, str, dict[str, Any] | None], None]


def _load_all_profiles_from_bigquery() -> list[Any]:
    """All company-year profiles from BigQuery `financial_metrics` (cached for 10 minutes)."""
    global _company_catalog_cache
    now = time.monotonic()
    with _CATALOG_LOCK:
        if (
            _company_catalog_cache
            and now - _company_catalog_cache[0] < _COMPANY_CATALOG_TTL_SECONDS
        ):
            return _company_catalog_cache[1]
    profiles = get_active_gcp_tool_adapters().bigquery.fetch_profiles(
        tickers=set(), start_year=1900, end_year=2100
    )
    catalog = EntityCatalog(profiles)
    with _CATALOG_LOCK:
        _company_catalog_cache = (time.monotonic(), profiles, catalog)
    return profiles


def _get_entity_catalog(*, allow_fetch: bool = True) -> tuple[EntityCatalog, list[Any]]:
    """Return `(EntityCatalog, profiles)` from the 10-minute cache, loading once if allowed."""
    global _company_catalog_cache
    now = time.monotonic()
    with _CATALOG_LOCK:
        if (
            _company_catalog_cache
            and now - _company_catalog_cache[0] < _COMPANY_CATALOG_TTL_SECONDS
        ):
            return _company_catalog_cache[2], _company_catalog_cache[1]
    if not allow_fetch:
        return EntityCatalog([]), []
    try:
        profiles = _load_all_profiles_from_bigquery()
        with _CATALOG_LOCK:
            if _company_catalog_cache is not None:
                return _company_catalog_cache[2], _company_catalog_cache[1]
        return EntityCatalog(profiles), profiles
    except Exception:
        return EntityCatalog([]), []


def reset_company_catalog_cache() -> None:
    """Clear the cached BigQuery profile/entity catalog (used by tests)."""
    global _company_catalog_cache
    with _CATALOG_LOCK:
        _company_catalog_cache = None


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
    has_prior_mem = (
        isinstance(memory_context, dict)
        and bool(memory_context.get("has_prior_memory"))
        and bool(memory_context.get("remembered_tickers"))
    )
    if has_prior_mem:
        prior_tickers = [str(t) for t in memory_context["remembered_tickers"]]  # type: ignore[index]

    memory_inherited = False
    removed_tickers = extract_removed_tickers(query, catalog) if has_prior_mem else []

    if explicit_tickers:
        effective_tickers = explicit_tickers
    elif removed_tickers and prior_tickers:
        # e.g. "remove Apple" / "without AAPL"
        effective_tickers = [t for t in prior_tickers if t not in removed_tickers]
        memory_inherited = True
    elif detected_tickers and prior_tickers and is_additive_followup(query):
        # e.g. "now add Microsoft" / "compare with AMD" -> MERGE into prior session tickers
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

    # Multi-turn follow-up entity recall when user omits company/sector ("those margins?", "just FY2025")
    broad_corpus_query = bool(
        re.search(
            r"\b(?:across\s+all|chunk\s+health|embedding\s+coverage|vector\s+(?:db|database)\s+health)\b",
            query,
            re.IGNORECASE,
        )
    )
    if (
        not effective_tickers
        and effective_sector == "ALL"
        and not broad_corpus_query
        and not unknown_entities
        and has_prior_mem
    ):
        effective_tickers = list(prior_tickers)
        memory_inherited = True

    # Filter out any invalid explicit tickers ONLY from valid_tickers while keeping unknown_entities
    valid_tickers = (
        [t for t in effective_tickers if catalog.contains_ticker(t)]
        if catalog.tickers
        else list(effective_tickers)
    )

    effective_section = section_id
    if not effective_section or effective_section.strip().upper() in ("ALL", "*"):
        has_1a = bool(_ITEM_1A_ONLY_RE.search(query))
        has_7 = bool(_ITEM_7_ONLY_RE.search(query))
        if has_1a and not has_7:
            effective_section = "item_1a"
        elif has_7 and not has_1a:
            effective_section = "item_7"
        else:
            effective_section = "ALL"

    mentioned_years = sorted({int(y) for y in _YEAR_MENTION_RE.findall(query)})
    eff_start = start_year
    eff_end = end_year
    if mentioned_years:
        if len(mentioned_years) >= 2:
            eff_start = min(mentioned_years)
            eff_end = max(mentioned_years)
        elif re.search(r"\bsince\s+(?:FY\s*)?202[0-6]\b", query, re.IGNORECASE):
            eff_start = mentioned_years[0]
            eff_end = max(end_year, mentioned_years[0])
        elif re.search(r"\b(?:just|only)\s+(?:FY\s*)?202[0-6]\b", query, re.IGNORECASE):
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
    """Run hybrid RAG searches in parallel while guaranteeing multi-ticker AND boundary-year balance."""
    tickers_csv = ",".join(tickers) if tickers else "ALL"

    tasks: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        (
            "primary",
            lambda: search_10k_chunks_rag(
                query=query,
                tickers=tickers_csv,
                gics_sector=gics_sector,
                section_id=section_id,
                start_year=start_year,
                end_year=end_year,
                top_k=max(top_k, 10),
            ),
        )
    ]

    if len(tickers) > 1:
        per_ticker_k = max(2, top_k // len(tickers))
        for sym in tickers[:4]:
            tasks.append(
                (
                    f"ticker:{sym}",
                    lambda s=sym, k=per_ticker_k: search_10k_chunks_rag(
                        query=query,
                        tickers=s,
                        gics_sector="ALL",
                        section_id=section_id,
                        start_year=start_year,
                        end_year=end_year,
                        top_k=k,
                    ),
                )
            )
    elif end_year > start_year:
        for boundary_year in (start_year, end_year):
            tasks.append(
                (
                    f"year:{boundary_year}",
                    lambda yr=boundary_year: search_10k_chunks_rag(
                        query=query,
                        tickers=tickers_csv,
                        gics_sector=gics_sector,
                        section_id=section_id,
                        start_year=yr,
                        end_year=yr,
                        top_k=3,
                    ),
                )
            )

    results_by_tag: dict[str, dict[str, Any]] = {}
    if len(tasks) == 1:
        tag, fn = tasks[0]
        results_by_tag[tag] = fn()
    else:
        with ThreadPoolExecutor(max_workers=min(5, len(tasks))) as pool:
            futs = [(tag, pool.submit(contextvars.copy_context().run, fn)) for tag, fn in tasks]
            for tag, fut in futs:
                results_by_tag[tag] = fut.result()

    primary = results_by_tag["primary"]
    merged_by_id: dict[str, dict[str, Any]] = {}
    candidates_searched = int(primary.get("candidates_searched") or 0)
    for cit in primary.get("citations") or []:
        merged_by_id[cit["chunk_id"]] = cit

    for tag, sub in results_by_tag.items():
        if tag == "primary":
            continue
        for cit in sub.get("citations") or []:
            merged_by_id.setdefault(cit["chunk_id"], cit)

    all_cits = sorted(
        merged_by_id.values(),
        key=lambda c: float(c.get("score") or 0.0),
        reverse=True,
    )

    # Balance across tickers for multi-company peer comparisons
    if len(tickers) > 1 and len(all_cits) > top_k:
        selected: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for sym in tickers:
            sym_hits = [c for c in all_cits if c["ticker"] == sym][:2]
            for hit in sym_hits:
                if hit["chunk_id"] not in used_ids and len(selected) < top_k:
                    selected.append(hit)
                    used_ids.add(hit["chunk_id"])
        for hit in all_cits:
            if hit["chunk_id"] not in used_ids and len(selected) < top_k:
                selected.append(hit)
                used_ids.add(hit["chunk_id"])
        return selected, candidates_searched

    # Balance boundary fiscal years (start_year and end_year) for single-company longitudinal
    # queries so FY2024 filings are never crowded out by FY2025/FY2026 filings (fixes MSFT FY2024!).
    if len(tickers) <= 1 and end_year > start_year and len(all_cits) > top_k:
        selected = []
        used_ids = set()
        for boundary_year in (start_year, end_year):
            yr_hits = [c for c in all_cits if int(c.get("fiscal_year") or 0) == boundary_year][:2]
            for hit in yr_hits:
                if hit["chunk_id"] not in used_ids and len(selected) < top_k:
                    selected.append(hit)
                    used_ids.add(hit["chunk_id"])
        for hit in all_cits:
            if hit["chunk_id"] not in used_ids and len(selected) < top_k:
                selected.append(hit)
                used_ids.add(hit["chunk_id"])
        return selected, candidates_searched

    return all_cits[:top_k], candidates_searched


def _early_return_response(
    *,
    status: str,
    intent: str,
    title: str,
    message: str,
    scope: dict[str, Any],
    prompt_decision: GuardrailDecision,
    memory_ctx: dict[str, Any],
    agent_trace: list[dict[str, Any]],
    identities_used_by_agent: dict[str, dict[str, str]],
    turn_start: float,
    session_id: str | None,
    user_id: str,
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
    total_ms = round((time.perf_counter() - turn_start) * 1000.0, 2)
    sess_tel = get_telemetry_sink().record_session_turn_telemetry(
        session_id=session_id,
        user_id=user_id,
        intent=intent,
        execution_mode=status,
        status_outcome=status,
        tickers_analyzed=list(scope.get("tickers") or []),
        gics_sector=str(scope.get("gics_sector") or "ALL"),
        start_year=int(scope.get("start_year") or 2024),
        end_year=int(scope.get("end_year") or 2026),
        report_generated=False,
        variance_reports_count=0,
        citations_count=0,
        agent_trace=agent_trace,
        turn_latency_records=[],
        total_latency_ms=total_ms,
        fact_check_passed=True,
    )
    return {
        "status": status,
        "model_armor_blocked": False,
        "model_armor": {
            "prompt_inspection": prompt_decision.to_audit_dict(),
            "response_inspection": None,
        },
        "execution_mode": status,
        "adk_runtime": get_live_runtime_status_metadata(),
        "structured_report": None,
        "report_verification": None,
        "prompt_store": get_prompt_store_status_metadata(),
        "intent": intent,
        "orchestrator": RootOrchestrator.name,
        "resolved_scope": scope,
        "answer_markdown": answer_markdown,
        "answer_verified": True,
        "clarification_reason": reason,
        "suggested_followups": suggested_followups,
        "coverage_notices": coverage_notices,
        "sector_coverage": None,
        "missing_tickers": [],
        "variance_reports": [],
        "sector_benchmarks": [],
        "material_variance_flags": [],
        "material_variance_summary": None,
        "disclosure_deltas": None,
        "citations": [],
        "candidates_searched": 0,
        "context_cache": None,
        "fact_check": {
            "passed": True,
            "checked_metrics_count": 0,
            "matched_citations_count": 0,
            "has_required_citations": True,
            "discrepancies": [],
            "model_latency_ms": total_ms,
        },
        "evaluation": None,
        "react_execution": {
            "strategy": "scope_validation_gate",
            "execution_mode": status,
            "iterations": 0,
            "steps": [],
            "variance_evaluations": [],
            "cached_10k_explanations": [],
        },
        "context_and_memory": {
            "models": get_model_strategy_metadata(),
            "session_memory": memory_ctx,
            "turn_context_latency": None,
            "model_call_records": [],
            "global_context_latency": (
                get_context_latency_tracker().summarize().model_dump(mode="json")
            ),
            "context_cache_runtime": get_context_cache_runtime_stats(),
            "session_telemetry": sess_tel["row"],
        },
        "agent_trace": agent_trace,
        "agent_identities_used": list(identities_used_by_agent.values()),
        "total_latency_ms": total_ms,
    }


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
    turn_start = time.perf_counter()
    agent_trace: list[dict[str, Any]] = []
    identities_used_by_agent: dict[str, dict[str, str]] = {}

    def _emit(stage: str, message: str, meta: dict[str, Any] | None = None) -> None:
        if on_progress is not None:
            on_progress(stage, message, meta)

    def _track_agent_identity(claims: VerifiedAgentS2SClaims) -> None:
        if claims.agent_name not in identities_used_by_agent:
            identities_used_by_agent[claims.agent_name] = {
                "agent_name": claims.agent_name,
                "service_account_email": claims.service_account_email,
                "spiffe_id": claims.spiffe_id,
            }

    _emit("starting", "Checking safety guardrails and session memory...", None)

    memory_mgr = get_conversation_memory_manager()
    memory_ctx = memory_mgr.recall_session_context(session_id, user_id=user_id)

    # Step 0: Managed Google Cloud Model Armor, Cloud SDP/DLP & Financial Projection Guardrails
    t0 = time.perf_counter()
    guardrail_svc = get_model_armor_guardrail()
    prompt_decision = guardrail_svc.inspect_user_prompt(
        query,
        caller_agent=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
    )
    if prompt_decision.blocked:
        guard_claims = authorize_agent_s2s_hop(
            caller_agent=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
            callee_agent="ModelArmorGuardrail",
        )
        _track_agent_identity(guard_claims)
        guard_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        cats_str = ", ".join(prompt_decision.matched_categories) or "policy_violation"
        agent_trace.append(
            {
                "step": 1,
                "agent": "ModelArmorGuardrail",
                "agent_service_account": guard_claims.service_account_email,
                "spiffe_id": guard_claims.spiffe_id,
                "s2s_target_service": guard_claims.target_service,
                "s2s_verified": True,
                "tool": "model_armor_sanitize_user_prompt",
                "status": "blocked",
                "duration_ms": guard_ms,
                "summary": (
                    f"Blocked user prompt via Google Cloud Model Armor / Compliance Filter "
                    f"(categories=[{cats_str}], backend={prompt_decision.backend_used})."
                ),
            }
        )
        blocked_intent = (
            "projection_compliance_block"
            if prompt_decision.projection_blocked
            else "safety_guardrail_block"
        )
        blocked_total_ms = round((time.perf_counter() - turn_start) * 1000.0, 2)
        blocked_sess_tel = get_telemetry_sink().record_session_turn_telemetry(
            session_id=session_id,
            user_id=user_id,
            intent=blocked_intent,
            execution_mode="guardrail_blocked",
            status_outcome="blocked",
            tickers_analyzed=[],
            gics_sector=gics_sector,
            start_year=start_year,
            end_year=end_year,
            report_generated=False,
            variance_reports_count=0,
            citations_count=0,
            agent_trace=agent_trace,
            turn_latency_records=[],
            total_latency_ms=blocked_total_ms,
            fact_check_passed=True,
        )
        return {
            "status": "blocked",
            "model_armor_blocked": True,
            "model_armor": prompt_decision.to_audit_dict(),
            "execution_mode": "guardrail_blocked",
            "adk_runtime": get_live_runtime_status_metadata(),
            "prompt_store": get_prompt_store_status_metadata(),
            "intent": blocked_intent,
            "orchestrator": RootOrchestrator.name,
            "resolved_scope": {
                "tickers": [],
                "gics_sector": gics_sector,
                "section_id": section_id,
                "start_year": start_year,
                "end_year": end_year,
            },
            "answer_markdown": (
                "### Request Blocked by Managed Model Armor & SEC Compliance Guardrails\n\n"
                f"{prompt_decision.block_message}"
            ),
            "coverage_notices": [],
            "suggested_followups": [],
            "sector_coverage": None,
            "variance_reports": [],
            "citations": [],
            "context_cache": None,
            "fact_check": {
                "passed": True,
                "checked_metrics_count": 0,
                "matched_citations_count": 0,
                "has_required_citations": True,
                "discrepancies": [],
                "model_latency_ms": guard_ms,
            },
            "react_execution": {
                "strategy": "structured_react_loop",
                "iterations": 0,
                "steps": [],
                "variance_evaluations": [],
                "cached_10k_explanations": [],
            },
            "context_and_memory": {
                "models": get_model_strategy_metadata(),
                "session_memory": memory_ctx,
                "turn_context_latency": None,
                "session_telemetry": blocked_sess_tel["row"],
            },
            "evaluation": None,
            "agent_trace": agent_trace,
            "agent_identities_used": list(identities_used_by_agent.values()),
            "total_latency_ms": blocked_total_ms,
        }

    if prompt_decision.sdp_redacted:
        query = prompt_decision.sanitized_text
        guard_claims = authorize_agent_s2s_hop(
            caller_agent=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
            callee_agent="ModelArmorGuardrail",
        )
        _track_agent_identity(guard_claims)
        agent_trace.append(
            {
                "step": len(agent_trace) + 1,
                "agent": "ModelArmorGuardrail",
                "agent_service_account": guard_claims.service_account_email,
                "spiffe_id": guard_claims.spiffe_id,
                "s2s_target_service": guard_claims.target_service,
                "s2s_verified": True,
                "tool": "cloud_sdp_deidentify_content",
                "status": "sanitized",
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "summary": (
                    f"De-identified sensitive PII in user prompt via SDP "
                    f"(info_types={list(prompt_decision.sdp_info_types)})."
                ),
            }
        )

    # Step 1: Scope Resolution, Full S&P 500 Catalog Validation, Session Memory Recall & Routing
    t1 = time.perf_counter()
    route_claims = authorize_agent_tool_invocation(
        agent_name=RootOrchestrator.name,
        tool_name="classify_query_intent",
        caller=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
    )
    _track_agent_identity(route_claims)
    catalog, all_profiles = _get_entity_catalog(allow_fetch=True)
    scope = resolve_query_scope(
        query=query,
        tickers=tickers,
        gics_sector=gics_sector,
        section_id=section_id,
        start_year=start_year,
        end_year=end_year,
        memory_context=memory_ctx,
        catalog=catalog,
    )

    validation = validate_query_scope_and_coverage(
        query=query,
        scope=scope,
        explicit_tickers=list(scope.get("explicit_tickers") or []),
        unknown_entities=list(scope.get("unknown_entities") or []),
        catalog=catalog,
        memory_context=memory_ctx,
    )
    pre_notices: list[dict[str, Any]] = list(validation.get("pre_notices") or [])

    if validation["action"] in ("clarification", "out_of_corpus"):
        is_clarification = validation["action"] == "clarification"
        early_status = "clarification" if is_clarification else "out_of_corpus"
        early_intent = "clarification_needed" if is_clarification else "out_of_corpus"
        early_title = (
            "Clarification Needed"
            if is_clarification
            else "Outside Curated SEC 10-K Corpus Coverage"
        )
        route_ms = round((time.perf_counter() - t1) * 1000.0, 2)
        agent_trace.append(
            {
                "step": len(agent_trace) + 1,
                "agent": RootOrchestrator.name,
                "model_used": resolve_model_for_task("orchestration"),
                "agent_service_account": route_claims.service_account_email,
                "spiffe_id": route_claims.spiffe_id,
                "s2s_target_service": route_claims.target_service,
                "s2s_verified": True,
                "tool": "validate_query_scope_and_coverage",
                "status": early_status,
                "duration_ms": route_ms,
                "summary": (
                    f"Scope validation returned '{early_intent}' "
                    f"(reason={validation.get('reason')})."
                ),
            }
        )
        _emit("scope_validation", early_title, {"action": early_status})
        return _early_return_response(
            status=early_status,
            intent=early_intent,
            title=early_title,
            message=str(validation["message"]),
            scope={
                k: v for k, v in scope.items() if k not in ("explicit_tickers", "unknown_entities")
            },
            prompt_decision=prompt_decision,
            memory_ctx=memory_ctx,
            agent_trace=agent_trace,
            identities_used_by_agent=identities_used_by_agent,
            turn_start=turn_start,
            session_id=session_id,
            user_id=user_id,
            coverage_notices=pre_notices,
            suggested_followups=list(validation.get("suggested_followups") or []),
            reason=validation.get("reason"),
        )

    # Clean internal keys from returned scope payload
    public_scope = {
        k: v for k, v in scope.items() if k not in ("explicit_tickers", "unknown_entities")
    }
    intent = classify_query_intent(query, tickers=public_scope["tickers"] or None)
    orchestrator_model = resolve_model_for_task("orchestration")
    route_ms = round((time.perf_counter() - t1) * 1000.0, 2)
    mem_suffix = " (recalled from session memory)" if public_scope.get("memory_inherited") else ""
    agent_trace.append(
        {
            "step": len(agent_trace) + 1,
            "agent": RootOrchestrator.name,
            "model_used": orchestrator_model,
            "agent_service_account": route_claims.service_account_email,
            "spiffe_id": route_claims.spiffe_id,
            "s2s_target_service": route_claims.target_service,
            "s2s_verified": True,
            "tool": "classify_query_intent",
            "status": "completed",
            "duration_ms": route_ms,
            "summary": (
                f"Routed to '{intent}' ({orchestrator_model}) · Tickers: {public_scope['tickers_csv']}{mem_suffix} · "
                f"Sector: {public_scope['gics_sector']} · Section: {public_scope['section_id']} · "
                f"FY{public_scope['start_year']}–FY{public_scope['end_year']}"
            ),
        }
    )

    is_sector_wide_thematic = (
        intent == "thematic_trend_tracking"
        and not public_scope["tickers"]
        and public_scope["gics_sector"] not in ("ALL", "", None)
    )
    _emit(
        "scope_resolved",
        (
            f"Scoped to {public_scope['gics_sector']} sector (FY{public_scope['start_year']}–FY{public_scope['end_year']})"
            if is_sector_wide_thematic
            else f"Scoped to {public_scope['tickers_csv']} (FY{public_scope['start_year']}–FY{public_scope['end_year']})"
        ),
        {
            "intent": intent,
            "tickers": public_scope["tickers"],
            "gics_sector": public_scope["gics_sector"],
            "sector_map_reduce": is_sector_wide_thematic,
        },
    )

    # Step 2 & Step 3: Parallel Retrieval (Variances / Benchmarks + Hybrid RAG / Sector Map-Reduce + Flags + Deltas)
    authorize_agent_s2s_hop(
        caller_agent=RootOrchestrator.name,
        callee_agent="FinancialAnalystAgent",
    )
    variance_reports: list[dict[str, Any]] = []
    missing_tickers: list[str] = []
    sector_benchmarks: list[dict[str, Any]] = []
    chunk_health: dict[str, Any] | None = None
    calculated_variances_map: dict[str, Any] = {}
    citations: list[dict[str, Any]] = []
    candidates_searched = 0
    material_flags: list[dict[str, Any]] = []
    material_flags_summary: dict[str, Any] | None = None
    disclosure_deltas: dict[str, Any] | None = None
    sector_coverage: dict[str, Any] | None = None
    sector_map_summaries: dict[str, Any] | None = None

    rag_claims = authorize_agent_tool_invocation(
        agent_name="FinancialAnalystAgent",
        tool_name="search_10k_chunks_rag",
        caller=RootOrchestrator.name,
    )
    _track_agent_identity(rag_claims)

    if is_sector_wide_thematic:
        # Gap 1: Full-sector batched map-reduce covering EVERY issuer in the GICS sector!
        t_sec = time.perf_counter()
        bench_claims = authorize_agent_tool_invocation(
            agent_name="FinancialAnalystAgent",
            tool_name="query_looker_financial_explore",
            caller=RootOrchestrator.name,
        )
        _track_agent_identity(bench_claims)
        delta_claims = authorize_agent_tool_invocation(
            agent_name="FinancialAnalystAgent",
            tool_name="compare_disclosures_across_years",
            caller=RootOrchestrator.name,
        )
        _track_agent_identity(delta_claims)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_bench = pool.submit(
                contextvars.copy_context().run,
                query_looker_financial_explore,
                explore_name="sector_benchmarks",
                tickers="ALL",
                gics_sector=public_scope["gics_sector"],
                start_year=public_scope["start_year"],
                end_year=public_scope["end_year"],
            )
            fut_mr = pool.submit(
                contextvars.copy_context().run,
                run_sector_thematic_map_reduce,
                theme=query,
                gics_sector=public_scope["gics_sector"],
                section_id=public_scope["section_id"],
                start_year=public_scope["start_year"],
                end_year=public_scope["end_year"],
                sector_profiles=all_profiles or None,
                map_llm=map_llm,
                on_progress=on_progress,
            )
            bench_res = fut_bench.result()
            mr_res = fut_mr.result()

        sec_ms = round((time.perf_counter() - t_sec) * 1000.0, 2)
        if bench_res.get("status") == "success":
            sector_benchmarks = list(bench_res.get("rows") or [])
        sector_coverage = mr_res.get("sector_coverage")
        sector_map_summaries = mr_res.get("sector_map_summaries")
        disclosure_deltas = mr_res.get("disclosure_deltas")
        citations = list(mr_res.get("citations") or [])
        candidates_searched = int(
            (sector_coverage or {}).get("total_chunks_evaluated") or len(citations)
        )

        agent_trace.append(
            {
                "step": len(agent_trace) + 1,
                "agent": "FinancialAnalystAgent",
                "agent_service_account": bench_claims.service_account_email,
                "spiffe_id": bench_claims.spiffe_id,
                "s2s_target_service": bench_claims.target_service,
                "s2s_verified": True,
                "tool": "query_looker_financial_explore(sector_benchmarks)",
                "status": "completed",
                "duration_ms": sec_ms,
                "summary": (
                    f"Retrieved {len(sector_benchmarks)} structured sector benchmark row(s) "
                    f"for {public_scope['gics_sector']}."
                ),
            }
        )
        cov = sector_coverage or {}
        agent_trace.append(
            {
                "step": len(agent_trace) + 1,
                "agent": "FinancialAnalystAgent",
                "agent_service_account": delta_claims.service_account_email,
                "spiffe_id": delta_claims.spiffe_id,
                "s2s_target_service": delta_claims.target_service,
                "s2s_verified": True,
                "tool": "sector_thematic_map_reduce",
                "status": "completed",
                "duration_ms": sec_ms,
                "summary": (
                    f"Full-sector batched map-reduce analyzed {cov.get('issuers_analyzed', 0)}/"
                    f"{cov.get('issuers_in_sector', 0)} {public_scope['gics_sector']} issuers "
                    f"({cov.get('total_chunks_evaluated', 0):,} passages in "
                    f"{cov.get('batches_processed', 0)} parallel Flash batches) → "
                    f"{len(citations)} traceable citations."
                ),
            }
        )
    else:
        # Parallelize independent retrieval calls: (metrics/variances OR chunk_health OR benchmarks),
        # balanced hybrid RAG citations, deterministic anomaly flags (+ MD&A explanations), and
        # (for single/peer thematic queries) Disclosure Delta wording comparisons!
        _emit("retrieving", "Fetching financial metrics and 10-K evidence in parallel...", None)
        t_par = time.perf_counter()

        calc_claims: VerifiedAgentS2SClaims | None = None
        flag_claims: VerifiedAgentS2SClaims | None = None
        delta_claims_opt: VerifiedAgentS2SClaims | None = None

        if intent == "chunk_health_monitoring":
            calc_claims = authorize_agent_tool_invocation(
                agent_name="FinancialAnalystAgent",
                tool_name="query_looker_financial_explore",
                caller=RootOrchestrator.name,
            )
            _track_agent_identity(calc_claims)
        elif public_scope["tickers"]:
            calc_claims = authorize_agent_tool_invocation(
                agent_name="FinancialAnalystAgent",
                tool_name="calculate_period_variances",
                caller=RootOrchestrator.name,
            )
            _track_agent_identity(calc_claims)
            flag_claims = authorize_agent_tool_invocation(
                agent_name="FinancialAnalystAgent",
                tool_name="flag_material_variances",
                caller=RootOrchestrator.name,
            )
            _track_agent_identity(flag_claims)
        else:
            calc_claims = authorize_agent_tool_invocation(
                agent_name="FinancialAnalystAgent",
                tool_name="query_looker_financial_explore",
                caller=RootOrchestrator.name,
            )
            _track_agent_identity(calc_claims)

        run_deltas = (
            intent == "thematic_trend_tracking"
            and public_scope["end_year"] > public_scope["start_year"]
        )
        if run_deltas:
            delta_claims_opt = authorize_agent_tool_invocation(
                agent_name="FinancialAnalystAgent",
                tool_name="compare_disclosures_across_years",
                caller=RootOrchestrator.name,
            )
            _track_agent_identity(delta_claims_opt)

        def _run_structured_step() -> tuple[str, dict[str, Any], float]:
            ts = time.perf_counter()
            if intent == "chunk_health_monitoring":
                res = query_looker_financial_explore(
                    explore_name="chunk_health",
                    tickers=public_scope["tickers_csv"],
                    gics_sector=public_scope["gics_sector"],
                    start_year=public_scope["start_year"],
                    end_year=public_scope["end_year"],
                )
                return "chunk_health", res, round((time.perf_counter() - ts) * 1000.0, 2)
            if public_scope["tickers"]:
                res = calculate_period_variances(
                    tickers=public_scope["tickers_csv"],
                    start_year=public_scope["start_year"],
                    end_year=public_scope["end_year"],
                )
                return "variances", res, round((time.perf_counter() - ts) * 1000.0, 2)
            res = query_looker_financial_explore(
                explore_name="sector_benchmarks",
                tickers="ALL",
                gics_sector=public_scope["gics_sector"],
                start_year=public_scope["start_year"],
                end_year=public_scope["end_year"],
            )
            return "benchmarks", res, round((time.perf_counter() - ts) * 1000.0, 2)

        def _run_rag_step() -> tuple[list[dict[str, Any]], int, float]:
            ts = time.perf_counter()
            cits, cand = _retrieve_balanced_citations(
                query=query,
                tickers=public_scope["tickers"],
                gics_sector=public_scope["gics_sector"],
                section_id=public_scope["section_id"],
                start_year=public_scope["start_year"],
                end_year=public_scope["end_year"],
                top_k=top_k,
            )
            return cits, cand, round((time.perf_counter() - ts) * 1000.0, 2)

        def _run_flags_step() -> tuple[
            dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], float
        ]:
            ts = time.perf_counter()
            if all_profiles:
                flag_res = flag_material_variances_from_cached_profiles(
                    all_profiles,
                    tickers=public_scope["tickers_csv"],
                    start_year=public_scope["start_year"],
                    end_year=public_scope["end_year"],
                )
            else:
                flag_res = flag_material_variances(
                    tickers=public_scope["tickers_csv"],
                    start_year=public_scope["start_year"],
                    end_year=public_scope["end_year"],
                )
            exp_flags: list[dict[str, Any]] = []
            exp_cits: list[dict[str, Any]] = []
            if flag_res.get("status") == "success":
                exp_flags, exp_cits = attach_flag_explanations(
                    list(flag_res.get("flags") or []), max_flags=3
                )
            return flag_res, exp_flags, exp_cits, round((time.perf_counter() - ts) * 1000.0, 2)

        def _run_deltas_step() -> tuple[dict[str, Any], float]:
            ts = time.perf_counter()
            res = compare_disclosures_across_years(
                theme=query,
                tickers=public_scope["tickers_csv"],
                gics_sector=public_scope["gics_sector"],
                section_id=public_scope["section_id"],
                start_year=public_scope["start_year"],
                end_year=public_scope["end_year"],
            )
            return res, round((time.perf_counter() - ts) * 1000.0, 2)

        with ThreadPoolExecutor(max_workers=4) as pool:
            fut_struct = pool.submit(contextvars.copy_context().run, _run_structured_step)
            fut_rag = pool.submit(contextvars.copy_context().run, _run_rag_step)
            fut_flags = (
                pool.submit(contextvars.copy_context().run, _run_flags_step)
                if (public_scope["tickers"] and intent != "chunk_health_monitoring")
                else None
            )
            fut_deltas = (
                pool.submit(contextvars.copy_context().run, _run_deltas_step)
                if run_deltas
                else None
            )

            struct_kind, struct_res, calc_ms = fut_struct.result()
            citations, candidates_searched, rag_ms = fut_rag.result()
            flags_out = fut_flags.result() if fut_flags is not None else None
            deltas_out = fut_deltas.result() if fut_deltas is not None else None

        if struct_kind == "chunk_health":
            if struct_res.get("status") == "success":
                chunk_health = struct_res
            agent_trace.append(
                {
                    "step": len(agent_trace) + 1,
                    "agent": "FinancialAnalystAgent",
                    "agent_service_account": calc_claims.service_account_email,  # type: ignore[union-attr]
                    "spiffe_id": calc_claims.spiffe_id,  # type: ignore[union-attr]
                    "s2s_target_service": calc_claims.target_service,  # type: ignore[union-attr]
                    "s2s_verified": True,
                    "tool": "query_looker_financial_explore(chunk_health)",
                    "status": "completed",
                    "duration_ms": calc_ms,
                    "summary": (
                        f"Verified vector index health: {struct_res.get('health_status', 'UNKNOWN')} "
                        f"({struct_res.get('total_chunks', 0):,} chunks)."
                    ),
                }
            )
        elif struct_kind == "variances":
            if struct_res.get("status") == "success":
                variance_reports = list(struct_res.get("variance_reports") or [])
                missing_tickers = list(struct_res.get("missing_tickers") or [])
                for rep in variance_reports:
                    calculated_variances_map[
                        f"{rep['ticker']}_{rep['start_year']}_{rep['end_year']}"
                    ] = rep
            agent_trace.append(
                {
                    "step": len(agent_trace) + 1,
                    "agent": "FinancialAnalystAgent",
                    "agent_service_account": calc_claims.service_account_email,  # type: ignore[union-attr]
                    "spiffe_id": calc_claims.spiffe_id,  # type: ignore[union-attr]
                    "s2s_target_service": calc_claims.target_service,  # type: ignore[union-attr]
                    "s2s_verified": True,
                    "tool": "calculate_period_variances",
                    "status": "completed",
                    "duration_ms": calc_ms,
                    "summary": (
                        f"Computed deterministic Revenue, Operating Income, Net Income, R&D, XBRL "
                        f"provenance, and margin bps variances for {len(variance_reports)} issuer(s) "
                        f"({public_scope['tickers_csv']})."
                    ),
                }
            )
        else:
            if struct_res.get("status") == "success":
                sector_benchmarks = list(struct_res.get("rows") or [])
            agent_trace.append(
                {
                    "step": len(agent_trace) + 1,
                    "agent": "FinancialAnalystAgent",
                    "agent_service_account": calc_claims.service_account_email,  # type: ignore[union-attr]
                    "spiffe_id": calc_claims.spiffe_id,  # type: ignore[union-attr]
                    "s2s_target_service": calc_claims.target_service,  # type: ignore[union-attr]
                    "s2s_verified": True,
                    "tool": "query_looker_financial_explore(sector_benchmarks)",
                    "status": "completed",
                    "duration_ms": calc_ms,
                    "summary": (
                        f"Retrieved {len(sector_benchmarks)} structured sector benchmark row(s) "
                        f"for {public_scope['gics_sector']}."
                    ),
                }
            )

        agent_trace.append(
            {
                "step": len(agent_trace) + 1,
                "agent": "FinancialAnalystAgent",
                "agent_service_account": rag_claims.service_account_email,
                "spiffe_id": rag_claims.spiffe_id,
                "s2s_target_service": rag_claims.target_service,
                "s2s_verified": True,
                "tool": "search_10k_chunks_rag",
                "status": "completed",
                "duration_ms": rag_ms,
                "summary": (
                    f"Ranked {candidates_searched:,} metadata-filtered 10-K chunks (BigQuery VECTOR_SEARCH + "
                    f"Vertex AI Search) → top {len(citations)} grounded citations."
                ),
            }
        )

        if flags_out is not None and flag_claims is not None and variance_reports:
            flag_res, material_flags, explanation_cits, flag_ms = flags_out
            explained_count = len(explanation_cits)
            merge_citations(citations, explanation_cits)
            if flag_res.get("status") == "success":
                material_flags_summary = {
                    "total_flags": flag_res.get("total_flags", 0),
                    "returned_flags": flag_res.get("returned_flags", 0),
                    "high_severity_count": flag_res.get("high_severity_count", 0),
                    "explained_flags": explained_count,
                    "thresholds": flag_res.get("thresholds"),
                }
            agent_trace.append(
                {
                    "step": len(agent_trace) + 1,
                    "agent": "FinancialAnalystAgent",
                    "agent_service_account": flag_claims.service_account_email,
                    "spiffe_id": flag_claims.spiffe_id,
                    "s2s_target_service": flag_claims.target_service,
                    "s2s_verified": True,
                    "tool": "flag_material_variances",
                    "status": "completed" if flag_res.get("status") == "success" else "warning",
                    "duration_ms": flag_ms,
                    "summary": (
                        f"Flagged {flag_res.get('total_flags', 0)} material variance(s) "
                        f"({flag_res.get('high_severity_count', 0)} high severity); attached MD&A "
                        f"explanations to {explained_count}."
                    ),
                }
            )

        if deltas_out is not None and delta_claims_opt is not None:
            disclosure_deltas, delta_ms = deltas_out
            merge_citations(citations, disclosure_delta_citations(disclosure_deltas))
            delta_summary = (disclosure_deltas or {}).get("summary") or {}
            agent_trace.append(
                {
                    "step": len(agent_trace) + 1,
                    "agent": "FinancialAnalystAgent",
                    "agent_service_account": delta_claims_opt.service_account_email,
                    "spiffe_id": delta_claims_opt.spiffe_id,
                    "s2s_target_service": delta_claims_opt.target_service,
                    "s2s_verified": True,
                    "tool": "compare_disclosures_across_years",
                    "status": (
                        "completed"
                        if (disclosure_deltas or {}).get("status") == "success"
                        else "warning"
                    ),
                    "duration_ms": delta_ms,
                    "summary": (
                        f"Compared theme disclosures for {delta_summary.get('companies_analyzed', 0)} "
                        f"company(ies) across {delta_summary.get('year_pairs', 0)} year pair(s): "
                        f"{delta_summary.get('added', 0)} added, {delta_summary.get('modified', 0)} "
                        f"modified, {delta_summary.get('removed', 0)} removed."
                    ),
                }
            )
        _ = time.perf_counter() - t_par

    # Build deterministic coverage notices (missing FY2026 filings, fiscal calendar misalignment, etc.)
    coverage_notices = build_coverage_notices(
        variance_reports=variance_reports,
        missing_tickers=missing_tickers,
        query=query,
        sector_coverage=sector_coverage,
        pre_notices=pre_notices,
    )

    # Step 4 & Step 5: Structured ReAct Execution Loop
    last_cache_claims: VerifiedAgentS2SClaims | None = None

    def _on_cache_s2s_call(_ticker: str, _fy: int, _sec: str) -> None:
        nonlocal last_cache_claims
        authorize_agent_s2s_hop(
            caller_agent="FinancialAnalystAgent",
            callee_agent="ContextCacheManager",
        )
        c_claims = authorize_agent_tool_invocation(
            agent_name="ContextCacheManager",
            tool_name="query_cached_10k_context",
            caller="FinancialAnalystAgent",
        )
        _track_agent_identity(c_claims)
        last_cache_claims = c_claims

    react_result = StructuredReActExecutor(
        reasoning_llm=reasoning_llm,
        judge_llm=judge_llm,
    ).execute_post_retrieval_reasoning(
        query=query,
        intent=intent,
        scope=public_scope,
        variance_reports=variance_reports,
        citations=citations,
        sector_benchmarks=sector_benchmarks,
        chunk_health=chunk_health,
        memory_tokens=int(memory_ctx.get("memory_tokens", 0) or 0),
        memory_prep_latency_ms=float(memory_ctx.get("memory_lookup_latency_ms", 0.0) or 0.0),
        memory_context=memory_ctx,
        session_id=session_id,
        user_id=user_id,
        on_cache_s2s_call=_on_cache_s2s_call,
        material_variance_flags=material_flags,
        disclosure_deltas=disclosure_deltas,
        coverage_notices=coverage_notices,
        sector_map_summaries=sector_map_summaries,
        prompt_already_verified=True,
        run_judge_sync=run_judge_sync,
        skip_background_judge=skip_background_judge,
        on_progress=on_progress,
    )

    answer_markdown: str = react_result["answer_markdown"]
    citations = react_result["citations"]
    variance_evaluations = react_result["variance_evaluations"]
    cached_explanations = react_result["cached_explanations"]
    react_steps = react_result["react_steps"]
    eval_report = react_result["evaluation_report"]
    turn_latency_records = react_result["turn_latency_records"]
    turn_latency_summary = react_result["turn_latency_summary"]
    reasoning_model: str = react_result["reasoning_model"]
    evaluation_model: str = react_result["evaluation_model"]
    active_execution_mode: str = react_result["execution_mode"]
    adk_runtime_meta: dict[str, Any] = react_result["adk_runtime"]

    eval_var_step = react_steps[0]
    agent_trace.append(
        {
            "step": len(agent_trace) + 1,
            "agent": "FinancialAnalystAgent",
            "model_used": reasoning_model,
            "agent_service_account": rag_claims.service_account_email,
            "spiffe_id": rag_claims.spiffe_id,
            "s2s_target_service": rag_claims.target_service,
            "s2s_verified": True,
            "tool": "evaluate_calculated_variances",
            "status": "completed",
            "duration_ms": round(eval_var_step.duration_ms, 2),
            "summary": (
                f"ReAct [Thought→Action]: Evaluated {len(variance_evaluations)} variance report(s) "
                f"using `{reasoning_model}` (context_latency={eval_var_step.context_latency_ms:.2f}ms)."
            ),
        }
    )

    context_cache_info: dict[str, Any] | None = None
    if cached_explanations and last_cache_claims is not None:
        primary_exp = cached_explanations[0]
        cache_hits_count = sum(1 for e in cached_explanations if e.cache_hit)
        total_cached_tokens = sum(e.token_estimate for e in cached_explanations)
        total_lookup_ms = round(sum(e.lookup_latency_ms for e in cached_explanations), 2)
        context_cache_info = {
            "cache_key": primary_exp.cache_key,
            "cache_hit": primary_exp.cache_hit,
            "ticker": primary_exp.ticker,
            "fiscal_year": primary_exp.fiscal_year,
            "section_id": primary_exp.section_id,
            "token_estimate": primary_exp.token_estimate,
            "sec_filing_url": primary_exp.sec_filing_url,
            "sections_queried_count": len(cached_explanations),
            "cache_hits_count": cache_hits_count,
            "total_section_tokens": total_cached_tokens,
            "lookup_latency_ms": total_lookup_ms,
        }
        agent_trace.append(
            {
                "step": len(agent_trace) + 1,
                "agent": "ContextCacheManager",
                "agent_service_account": last_cache_claims.service_account_email,
                "spiffe_id": last_cache_claims.spiffe_id,
                "s2s_target_service": last_cache_claims.target_service,
                "s2s_verified": True,
                "tool": "query_cached_10k_context",
                "status": "completed",
                "duration_ms": total_lookup_ms,
                "summary": (
                    f"ReAct [Query Cached 10-K]: Queried {len(cached_explanations)} cached 10-K section(s) "
                    f"(hits={cache_hits_count}/{len(cached_explanations)}, ~{total_cached_tokens:,} tokens)."
                ),
            }
        )

    # Post-Generation Numeric Fact-Check S2S Attestation
    _emit("verifying", "Verifying citations, XBRL provenance, and compliance guardrails...", None)
    t5 = time.perf_counter()
    s2s_broker = get_agent_s2s_broker()
    raw_fc_assertion = s2s_broker.mint_s2s_assertion(
        agent_name="FinancialAnalystAgent",
        target_service="fact-check-verification-service",
        caller=RootOrchestrator.name,
    )
    fc_claims = s2s_broker.verify_s2s_assertion(
        raw_fc_assertion,
        expected_target_service="fact-check-verification-service",
    )
    _track_agent_identity(fc_claims)
    synth_step = next(
        (s for s in react_steps if s.phase == "format_grounded_output"), react_steps[-1]
    )
    fc_ms = round((time.perf_counter() - t5) * 1000.0 + synth_step.duration_ms, 2)

    # Post-Generation Managed Model Armor, Cloud SDP & Projection Leak Filter
    # Reuse the ADK callback inspection result when already checked, eliminating 2 redundant HTTP calls.
    t_resp_armor = time.perf_counter()
    if react_result.get("model_armor_response_blocked") is True:
        raw_plugin_dec = react_result.get("model_armor_response_decision") or {}
        resp_decision = GuardrailDecision(
            action="blocked",
            blocked=True,
            sdp_redacted=bool(raw_plugin_dec.get("sdp_redacted", False)),
            projection_blocked=bool(raw_plugin_dec.get("projection_blocked", False)),
            original_text="",
            sanitized_text="",
            matched_categories=tuple(
                raw_plugin_dec.get("matched_categories") or ("policy_violation",)
            ),
            sdp_info_types=tuple(raw_plugin_dec.get("sdp_info_types") or ()),
            block_message=str(
                react_result.get("model_armor_response_block_message") or answer_markdown
            ),
            backend_used=str(raw_plugin_dec.get("backend_used") or "gcp_model_armor_v1"),
            latency_ms=float(raw_plugin_dec.get("latency_ms") or 0.0),
            s2s_service_account=str(raw_plugin_dec.get("s2s_service_account") or ""),
            s2s_spiffe_id=str(raw_plugin_dec.get("s2s_spiffe_id") or ""),
        )
    elif react_result.get("adk_response_already_inspected") and isinstance(
        react_result.get("adk_response_decision"), dict
    ):
        prev_dec = react_result["adk_response_decision"]
        resp_decision = GuardrailDecision(
            action=prev_dec.get("action") or "allowed",
            blocked=bool(prev_dec.get("blocked", False)),
            sdp_redacted=bool(prev_dec.get("sdp_redacted", False)),
            projection_blocked=bool(prev_dec.get("projection_blocked", False)),
            original_text=answer_markdown,
            sanitized_text=answer_markdown,
            matched_categories=tuple(prev_dec.get("matched_categories") or ()),
            sdp_info_types=tuple(prev_dec.get("sdp_info_types") or ()),
            block_message=str(prev_dec.get("block_message") or ""),
            backend_used=str(prev_dec.get("backend_used") or "gcp_model_armor_v1"),
            latency_ms=float(prev_dec.get("latency_ms") or 0.0),
            s2s_service_account=str(prev_dec.get("s2s_service_account") or ""),
            s2s_spiffe_id=str(prev_dec.get("s2s_spiffe_id") or ""),
        )
    else:
        resp_decision = guardrail_svc.inspect_model_response(
            answer_markdown,
            caller_agent="FinancialAnalystAgent",
        )

    if resp_decision.blocked:
        out_guard_claims = authorize_agent_s2s_hop(
            caller_agent="FinancialAnalystAgent",
            callee_agent="ModelArmorGuardrail",
        )
        _track_agent_identity(out_guard_claims)
        out_guard_ms = round((time.perf_counter() - t_resp_armor) * 1000.0, 2)
        cats_str = ", ".join(resp_decision.matched_categories) or "policy_violation"
        agent_trace.append(
            {
                "step": len(agent_trace) + 1,
                "agent": "ModelArmorGuardrail",
                "agent_service_account": out_guard_claims.service_account_email,
                "spiffe_id": out_guard_claims.spiffe_id,
                "s2s_target_service": out_guard_claims.target_service,
                "s2s_verified": True,
                "tool": "model_armor_sanitize_model_response",
                "status": "blocked",
                "duration_ms": out_guard_ms,
                "summary": (
                    f"Withheld model response via Model Armor / SEC Projection Filter "
                    f"(categories=[{cats_str}], backend={resp_decision.backend_used})."
                ),
            }
        )
        resp_blocked_intent = (
            "projection_compliance_block"
            if resp_decision.projection_blocked
            else "safety_guardrail_block"
        )
        resp_blocked_ms = round((time.perf_counter() - turn_start) * 1000.0, 2)
        resp_blocked_tel = get_telemetry_sink().record_session_turn_telemetry(
            session_id=session_id,
            user_id=user_id,
            intent=resp_blocked_intent,
            execution_mode="guardrail_blocked",
            status_outcome="blocked",
            tickers_analyzed=public_scope["tickers"],
            gics_sector=public_scope["gics_sector"],
            start_year=public_scope["start_year"],
            end_year=public_scope["end_year"],
            report_generated=False,
            variance_reports_count=len(variance_reports),
            citations_count=0,
            agent_trace=agent_trace,
            turn_latency_records=turn_latency_records,
            total_latency_ms=resp_blocked_ms,
            fact_check_passed=False,
        )
        return {
            "status": "blocked",
            "model_armor_blocked": True,
            "model_armor": {
                "prompt_inspection": prompt_decision.to_audit_dict(),
                "response_inspection": resp_decision.to_audit_dict(),
            },
            "execution_mode": "guardrail_blocked",
            "adk_runtime": adk_runtime_meta,
            "prompt_store": get_prompt_store_status_metadata(),
            "intent": resp_blocked_intent,
            "orchestrator": RootOrchestrator.name,
            "resolved_scope": public_scope,
            "answer_markdown": (
                "### Response Withheld by Managed Model Armor & SEC Compliance Guardrails\n\n"
                f"{resp_decision.block_message}"
            ),
            "coverage_notices": coverage_notices,
            "suggested_followups": [],
            "sector_coverage": sector_coverage,
            "variance_reports": variance_reports,
            "citations": [],
            "context_cache": context_cache_info,
            "fact_check": {
                "passed": False,
                "checked_metrics_count": 0,
                "matched_citations_count": 0,
                "has_required_citations": False,
                "discrepancies": list(resp_decision.matched_categories),
                "model_latency_ms": out_guard_ms,
            },
            "react_execution": {
                "strategy": "structured_react_loop",
                "execution_mode": active_execution_mode,
                "iterations": len(react_steps),
                "steps": [s.model_dump(mode="json") for s in react_steps],
                "variance_evaluations": [v.model_dump(mode="json") for v in variance_evaluations],
                "cached_10k_explanations": [e.model_dump(mode="json") for e in cached_explanations],
            },
            "context_and_memory": {
                "models": get_model_strategy_metadata(),
                "session_memory": memory_ctx,
                "turn_context_latency": turn_latency_summary.model_dump(mode="json"),
                "session_telemetry": resp_blocked_tel["row"],
            },
            "evaluation": None,
            "agent_trace": agent_trace,
            "agent_identities_used": list(identities_used_by_agent.values()),
            "total_latency_ms": resp_blocked_ms,
        }

    if resp_decision.sdp_redacted and resp_decision.sanitized_text != answer_markdown:
        answer_markdown = resp_decision.sanitized_text
        out_guard_claims = authorize_agent_s2s_hop(
            caller_agent="FinancialAnalystAgent",
            callee_agent="ModelArmorGuardrail",
        )
        _track_agent_identity(out_guard_claims)
        agent_trace.append(
            {
                "step": len(agent_trace) + 1,
                "agent": "ModelArmorGuardrail",
                "agent_service_account": out_guard_claims.service_account_email,
                "spiffe_id": out_guard_claims.spiffe_id,
                "s2s_target_service": out_guard_claims.target_service,
                "s2s_verified": True,
                "tool": "cloud_sdp_deidentify_content",
                "status": "sanitized",
                "duration_ms": round((time.perf_counter() - t_resp_armor) * 1000.0, 2),
                "summary": (
                    f"De-identified sensitive PII in synthesized model response via SDP "
                    f"(info_types={list(resp_decision.sdp_info_types)})."
                ),
            }
        )

    fact_check = verify_narrative_against_calculations(
        response_text=answer_markdown,
        calculated_variances=calculated_variances_map,
        retrieved_citations=citations,
        model_latency_ms=fc_ms,
    )
    agent_trace.append(
        {
            "step": len(agent_trace) + 1,
            "agent": "FinancialAnalystAgent",
            "model_used": f"{reasoning_model} + {evaluation_model}",
            "agent_service_account": fc_claims.service_account_email,
            "spiffe_id": fc_claims.spiffe_id,
            "s2s_target_service": fc_claims.target_service,
            "s2s_verified": True,
            "tool": "after_model_fact_check_callback",
            "status": "completed" if fact_check.passed else "warning",
            "duration_ms": fc_ms,
            "summary": (
                f"Formatted grounded report (`{reasoning_model}`) & verification "
                f"(judge_mode={react_result.get('judge_mode', 'async_background')}, "
                f"grade={eval_report.grade}, score={eval_report.overall_score:.2f}) · "
                f"Verified {fact_check.checked_metrics_count} variance figure(s) & "
                f"{fact_check.matched_citations_count} citation tag(s) · Passed: {fact_check.passed}."
            ),
        }
    )

    # Persist turn into durable Firestore-backed ConversationMemoryManager (keyed by user_id + session_id)
    variance_highlights = [
        ev.material_findings[0] for ev in variance_evaluations if ev.material_findings
    ]
    memory_persist = memory_mgr.record_turn(
        session_id=session_id,
        user_id=user_id,
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

    total_latency_ms = round((time.perf_counter() - turn_start) * 1000.0, 2)
    analyzed_tickers = sorted(
        {
            *(public_scope.get("tickers") or []),
            *(r["ticker"] for r in variance_reports if r.get("ticker")),
            *(c["ticker"] for c in citations if c.get("ticker")),
        }
    )
    session_tel = get_telemetry_sink().record_session_turn_telemetry(
        session_id=session_id,
        user_id=user_id,
        intent=intent,
        execution_mode=active_execution_mode,
        status_outcome="success",
        tickers_analyzed=analyzed_tickers,
        gics_sector=public_scope["gics_sector"],
        start_year=public_scope["start_year"],
        end_year=public_scope["end_year"],
        report_generated=bool(react_result.get("answer_verified")),
        variance_reports_count=len(variance_reports),
        citations_count=len(citations),
        agent_trace=agent_trace,
        turn_latency_records=turn_latency_records,
        total_latency_ms=total_latency_ms,
        fact_check_passed=fact_check.passed,
        evaluation_grade=eval_report.grade,
        evaluation_score=eval_report.overall_score,
    )
    _emit("complete", "Grounded report ready.", {"total_latency_ms": total_latency_ms})
    return {
        "status": "success",
        "model_armor_blocked": False,
        "model_armor": {
            "prompt_inspection": prompt_decision.to_audit_dict(),
            "response_inspection": resp_decision.to_audit_dict(),
        },
        "execution_mode": active_execution_mode,
        "adk_runtime": adk_runtime_meta,
        "structured_report": adk_runtime_meta.get("structured_report"),
        "report_verification": adk_runtime_meta.get("report_verification"),
        "prompt_store": get_prompt_store_status_metadata(),
        "intent": intent,
        "orchestrator": RootOrchestrator.name,
        "resolved_scope": public_scope,
        "answer_markdown": answer_markdown,
        "answer_verified": bool(react_result.get("answer_verified")),
        "coverage_notices": coverage_notices,
        "sector_coverage": sector_coverage,
        "missing_tickers": missing_tickers,
        "suggested_followups": [],
        "variance_reports": variance_reports,
        "sector_benchmarks": sector_benchmarks,
        "material_variance_flags": material_flags,
        "material_variance_summary": material_flags_summary,
        "disclosure_deltas": disclosure_deltas,
        "citations": citations,
        "candidates_searched": candidates_searched,
        "context_cache": context_cache_info,
        "fact_check": fact_check.model_dump(mode="json"),
        "evaluation": eval_report.model_dump(mode="json"),
        "react_execution": {
            "strategy": "structured_react_loop",
            "execution_mode": active_execution_mode,
            "observed_ttft_ms": adk_runtime_meta.get("observed_ttft_ms"),
            "judge_mode": react_result.get("judge_mode", "async_background"),
            "iterations": len(react_steps),
            "steps": [s.model_dump(mode="json") for s in react_steps],
            "variance_evaluations": [v.model_dump(mode="json") for v in variance_evaluations],
            "cached_10k_explanations": [e.model_dump(mode="json") for e in cached_explanations],
        },
        "context_and_memory": {
            "models": get_model_strategy_metadata(),
            "session_memory": {
                **memory_ctx,
                "persistence": memory_persist,
            },
            "turn_context_latency": turn_latency_summary.model_dump(mode="json"),
            "model_call_records": [r.model_dump(mode="json") for r in turn_latency_records],
            "global_context_latency": (
                get_context_latency_tracker().summarize().model_dump(mode="json")
            ),
            "context_cache_runtime": get_context_cache_runtime_stats(),
            "session_telemetry": session_tel["row"],
        },
        "agent_trace": agent_trace,
        "agent_identities_used": list(identities_used_by_agent.values()),
        "total_latency_ms": total_latency_ms,
    }
