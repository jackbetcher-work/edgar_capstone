"""Conversational Hybrid RAG orchestration engine for the SEC EDGAR Natural Language Analyst UI."""

from __future__ import annotations

import re
import time
from collections import defaultdict
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
)
from edgar_capstone.agents.prompt_store import get_prompt_store_status_metadata
from edgar_capstone.agents.react_loop import StructuredReActExecutor
from edgar_capstone.agents.root_orchestrator import (
    RootOrchestrator,
    classify_query_intent,
)
from edgar_capstone.agents.tools import (
    _parse_ticker_list,
    _profile_to_metric_record,
    calculate_period_variances,
    get_active_gcp_tool_adapters,
    get_context_cache_runtime_stats,
    load_cached_10k_section,
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

_COMPANY_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bnvidia\b|\bnvda\b", re.IGNORECASE), "NVDA"),
    (re.compile(r"\bapple\b|\baapl\b", re.IGNORECASE), "AAPL"),
    (re.compile(r"\bmicrosoft\b|\bmsft\b", re.IGNORECASE), "MSFT"),
    (re.compile(r"\bbroadcom\b|\bavgo\b", re.IGNORECASE), "AVGO"),
    (re.compile(r"\bsalesforce\b|\bcrm\b", re.IGNORECASE), "CRM"),
    (re.compile(r"\balphabet\b|\bgoogle\b|\bgoogl\b", re.IGNORECASE), "GOOGL"),
    (re.compile(r"\bmeta\s+platforms\b|\bmeta\b|\bfacebook\b", re.IGNORECASE), "META"),
    (re.compile(r"\bamazon\b|\bamzn\b|\baws\b", re.IGNORECASE), "AMZN"),
    (re.compile(r"\btesla\b|\btsla\b", re.IGNORECASE), "TSLA"),
    (re.compile(r"\bwalmart\b|\bwmt\b", re.IGNORECASE), "WMT"),
    (re.compile(r"\bprocter\s*(?:&|and)?\s*gamble\b|\bpg\b", re.IGNORECASE), "PG"),
    (re.compile(r"\bjpmorgan\b|\bjp\s+morgan\b|\bjpm\b", re.IGNORECASE), "JPM"),
    (re.compile(r"\bvisa\b", re.IGNORECASE), "V"),
    (re.compile(r"\bunitedhealth\b|\bunh\b", re.IGNORECASE), "UNH"),
    (re.compile(r"\beli\s+lilly\b|\blilly\b|\blly\b", re.IGNORECASE), "LLY"),
    (re.compile(r"\bjohnson\s*(?:&|and)?\s*johnson\b|\bjnj\b", re.IGNORECASE), "JNJ"),
    (re.compile(r"\bexxon\s*mobil\b|\bexxon\b|\bxom\b", re.IGNORECASE), "XOM"),
    (re.compile(r"\bcaterpillar\b|\bcat\b", re.IGNORECASE), "CAT"),
)

_SECTOR_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\binformation\s+technology\b|\btech\s+companies\b|\btech\s+sector\b|\bsemiconductors?\b",
            re.IGNORECASE,
        ),
        "Information Technology",
    ),
    (
        re.compile(r"\bhealth\s*care\b|\bpharma(?:ceuticals?)?\b|\bbiotech\b", re.IGNORECASE),
        "Health Care",
    ),
    (
        re.compile(r"\bfinancials\b|\bfinancial\s+sector\b|\bbanking\b|\bbanks\b", re.IGNORECASE),
        "Financials",
    ),
    (
        re.compile(r"\bcommunication\s+services\b", re.IGNORECASE),
        "Communication Services",
    ),
    (
        re.compile(r"\bconsumer\s+discretionary\b", re.IGNORECASE),
        "Consumer Discretionary",
    ),
    (
        re.compile(r"\bconsumer\s+staples\b", re.IGNORECASE),
        "Consumer Staples",
    ),
    (
        re.compile(r"\benergy\s+sector\b|\boil\s+and\s+gas\b", re.IGNORECASE),
        "Energy",
    ),
    (
        re.compile(r"\bindustrials\b|\bindustrial\s+sector\b", re.IGNORECASE),
        "Industrials",
    ),
)

_YEAR_MENTION_RE = re.compile(r"\b(?:FY\s*)?(202[0-6])\b", re.IGNORECASE)
_ITEM_1A_ONLY_RE = re.compile(r"\b(?:item\s*1a|risk\s+factors?)\b", re.IGNORECASE)
_ITEM_7_ONLY_RE = re.compile(r"\b(?:item\s*7|md&a|management'?s\s+discussion)\b", re.IGNORECASE)


_COMPANY_CATALOG_TTL_SECONDS = 600.0
_company_catalog_cache: tuple[float, list[Any]] | None = None


def _load_all_profiles_from_bigquery() -> list[Any]:
    """All company-year profiles from BigQuery `financial_metrics` (cached for 10 minutes)."""
    global _company_catalog_cache
    now = time.monotonic()
    if _company_catalog_cache and now - _company_catalog_cache[0] < _COMPANY_CATALOG_TTL_SECONDS:
        return _company_catalog_cache[1]
    profiles = get_active_gcp_tool_adapters().bigquery.fetch_profiles(
        tickers=set(), start_year=1900, end_year=2100
    )
    _company_catalog_cache = (now, profiles)
    return profiles


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
                "latest_revenue": latest.revenue,
                "latest_operating_income": latest.operating_income,
                "latest_net_income": latest.net_income,
                "latest_research_and_development": latest.research_and_development,
                "latest_operating_margin_pct": latest_rec.operating_margin_pct,
                "latest_net_margin_pct": latest_rec.net_margin_pct,
                "sec_filing_url": latest.sec_filing_url,
            }
        )

    return {
        "status": "success",
        "company_count": len(companies),
        "total_profiles": len(profiles),
        "min_year": min(all_years) if all_years else 2020,
        "max_year": max(all_years) if all_years else 2026,
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
) -> dict[str, Any]:
    """Combine explicit UI filter selections, natural-language detection, and session memory recall."""
    explicit_tickers = sorted(_parse_ticker_list(tickers))
    detected_tickers: list[str] = []
    for pattern, symbol in _COMPANY_ALIASES:
        if pattern.search(query) and symbol not in detected_tickers:
            detected_tickers.append(symbol)
    # Exact uppercase standalone 'V' check
    if re.search(r"(?<![A-Za-z0-9\-])V(?![A-Za-z0-9\-])", query) and "V" not in detected_tickers:
        detected_tickers.append("V")

    effective_tickers = explicit_tickers if explicit_tickers else detected_tickers
    memory_inherited = False

    effective_sector = gics_sector
    if not effective_sector or effective_sector.strip().upper() in ("ALL", "*"):
        effective_sector = "ALL"
        if not effective_tickers:
            for pattern, sec_name in _SECTOR_ALIASES:
                if pattern.search(query):
                    effective_sector = sec_name
                    break

    # Multi-turn follow-up entity recall from session memory when user asks a contextual follow-up
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
        and isinstance(memory_context, dict)
        and memory_context.get("has_prior_memory")
        and memory_context.get("remembered_tickers")
    ):
        effective_tickers = list(memory_context["remembered_tickers"])
        memory_inherited = True

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
        elif start_year == 2024 and end_year == 2026:
            # Single fiscal year focus while preserving YoY comparison baseline when helpful
            yr = mentioned_years[0]
            eff_start = max(2020, yr - 1)
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
        "tickers": effective_tickers,
        "tickers_csv": ",".join(effective_tickers) if effective_tickers else "ALL",
        "gics_sector": effective_sector,
        "section_id": effective_section,
        "start_year": eff_start,
        "end_year": eff_end,
        "memory_inherited": memory_inherited,
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
    """Run hybrid RAG search while ensuring multi-company or multi-year comparisons have balanced coverage."""
    tickers_csv = ",".join(tickers) if tickers else "ALL"
    primary = search_10k_chunks_rag(
        query=query,
        tickers=tickers_csv,
        gics_sector=gics_sector,
        section_id=section_id,
        start_year=start_year,
        end_year=end_year,
        top_k=max(top_k, 10),
    )
    merged_by_id: dict[str, dict[str, Any]] = {}
    candidates_searched = int(primary.get("candidates_searched") or 0)

    for cit in primary.get("citations") or []:
        merged_by_id[cit["chunk_id"]] = cit

    # Ensure every explicitly compared ticker has at least 2 grounded citations when available
    if len(tickers) > 1:
        per_ticker_k = max(2, top_k // len(tickers))
        for sym in tickers[:4]:
            sub = search_10k_chunks_rag(
                query=query,
                tickers=sym,
                gics_sector="ALL",
                section_id=section_id,
                start_year=start_year,
                end_year=end_year,
                top_k=per_ticker_k,
            )
            for cit in sub.get("citations") or []:
                merged_by_id.setdefault(cit["chunk_id"], cit)

    # For longitudinal / thematic tracking across multiple years, ensure boundary years are represented
    elif end_year > start_year:
        for boundary_year in (start_year, end_year):
            sub_yr = search_10k_chunks_rag(
                query=query,
                tickers=tickers_csv,
                gics_sector=gics_sector,
                section_id=section_id,
                start_year=boundary_year,
                end_year=boundary_year,
                top_k=3,
            )
            for cit in sub_yr.get("citations") or []:
                merged_by_id.setdefault(cit["chunk_id"], cit)

    # Sort while preserving diversity across tickers and fiscal years
    all_cits = sorted(
        merged_by_id.values(),
        key=lambda c: float(c.get("score") or 0.0),
        reverse=True,
    )

    if len(tickers) > 1 and len(all_cits) > top_k:
        selected: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        # First pass: top 2 per ticker
        for sym in tickers:
            sym_hits = [c for c in all_cits if c["ticker"] == sym][:2]
            for hit in sym_hits:
                if hit["chunk_id"] not in used_ids and len(selected) < top_k:
                    selected.append(hit)
                    used_ids.add(hit["chunk_id"])
        # Fill remaining slots by hybrid relevance score
        for hit in all_cits:
            if hit["chunk_id"] not in used_ids and len(selected) < top_k:
                selected.append(hit)
                used_ids.add(hit["chunk_id"])
        return selected, candidates_searched

    return all_cits[:top_k], candidates_searched


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
) -> dict[str, Any]:
    """Execute a full source-grounded Hybrid RAG conversational turn with ADK Runner / ReAct loop & fact-checking."""
    turn_start = time.perf_counter()
    agent_trace: list[dict[str, Any]] = []
    identities_used_by_agent: dict[str, dict[str, str]] = {}

    def _track_agent_identity(claims: VerifiedAgentS2SClaims) -> None:
        if claims.agent_name not in identities_used_by_agent:
            identities_used_by_agent[claims.agent_name] = {
                "agent_name": claims.agent_name,
                "service_account_email": claims.service_account_email,
                "spiffe_id": claims.spiffe_id,
            }

    memory_mgr = get_conversation_memory_manager()
    memory_ctx = memory_mgr.recall_session_context(session_id)

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

    # Step 1: Scope Resolution, Session Memory Recall & RootOrchestrator Intent Routing
    t1 = time.perf_counter()
    route_claims = authorize_agent_tool_invocation(
        agent_name=RootOrchestrator.name,
        tool_name="classify_query_intent",
        caller=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
    )
    _track_agent_identity(route_claims)
    scope = resolve_query_scope(
        query=query,
        tickers=tickers,
        gics_sector=gics_sector,
        section_id=section_id,
        start_year=start_year,
        end_year=end_year,
        memory_context=memory_ctx,
    )
    intent = classify_query_intent(query, tickers=scope["tickers"] or None)
    orchestrator_model = resolve_model_for_task("orchestration")
    route_ms = round((time.perf_counter() - t1) * 1000.0, 2)
    mem_suffix = " (recalled from session memory)" if scope.get("memory_inherited") else ""
    agent_trace.append(
        {
            "step": 1,
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
                f"Routed to '{intent}' ({orchestrator_model}) · Tickers: {scope['tickers_csv']}{mem_suffix} · "
                f"Sector: {scope['gics_sector']} · Section: {scope['section_id']} · "
                f"FY{scope['start_year']}–FY{scope['end_year']}"
            ),
        }
    )

    # Step 2: Deterministic Calculation Engine / Looker Semantic Layer (RootOrchestrator -> FinancialAnalystAgent S2S hop)
    t2 = time.perf_counter()
    authorize_agent_s2s_hop(
        caller_agent=RootOrchestrator.name,
        callee_agent="FinancialAnalystAgent",
    )
    variance_reports: list[dict[str, Any]] = []
    sector_benchmarks: list[dict[str, Any]] = []
    chunk_health: dict[str, Any] | None = None
    calculated_variances_map: dict[str, Any] = {}

    if intent == "chunk_health_monitoring":
        calc_claims = authorize_agent_tool_invocation(
            agent_name="FinancialAnalystAgent",
            tool_name="query_looker_financial_explore",
            caller=RootOrchestrator.name,
        )
        _track_agent_identity(calc_claims)
        health_res = query_looker_financial_explore(
            explore_name="chunk_health",
            tickers=scope["tickers_csv"],
            gics_sector=scope["gics_sector"],
            start_year=scope["start_year"],
            end_year=scope["end_year"],
        )
        if health_res.get("status") == "success":
            chunk_health = health_res
        calc_ms = round((time.perf_counter() - t2) * 1000.0, 2)
        agent_trace.append(
            {
                "step": 2,
                "agent": "FinancialAnalystAgent",
                "agent_service_account": calc_claims.service_account_email,
                "spiffe_id": calc_claims.spiffe_id,
                "s2s_target_service": calc_claims.target_service,
                "s2s_verified": True,
                "tool": "query_looker_financial_explore(chunk_health)",
                "status": "completed",
                "duration_ms": calc_ms,
                "summary": (
                    f"Verified vector index health: {health_res.get('health_status', 'UNKNOWN')} "
                    f"({health_res.get('total_chunks', 0):,} chunks)."
                ),
            }
        )
    elif scope["tickers"]:
        calc_claims = authorize_agent_tool_invocation(
            agent_name="FinancialAnalystAgent",
            tool_name="calculate_period_variances",
            caller=RootOrchestrator.name,
        )
        _track_agent_identity(calc_claims)
        var_res = calculate_period_variances(
            tickers=scope["tickers_csv"],
            start_year=scope["start_year"],
            end_year=scope["end_year"],
        )
        if var_res.get("status") == "success":
            variance_reports = list(var_res.get("variance_reports") or [])
            for rep in variance_reports:
                calculated_variances_map[
                    f"{rep['ticker']}_{rep['start_year']}_{rep['end_year']}"
                ] = rep
        calc_ms = round((time.perf_counter() - t2) * 1000.0, 2)
        agent_trace.append(
            {
                "step": 2,
                "agent": "FinancialAnalystAgent",
                "agent_service_account": calc_claims.service_account_email,
                "spiffe_id": calc_claims.spiffe_id,
                "s2s_target_service": calc_claims.target_service,
                "s2s_verified": True,
                "tool": "calculate_period_variances",
                "status": "completed",
                "duration_ms": calc_ms,
                "summary": (
                    f"Computed deterministic Revenue, Operating Income, Net Income, R&D, and margin bps variances "
                    f"for {len(variance_reports)} issuer(s) ({scope['tickers_csv']})."
                ),
            }
        )
    else:
        calc_claims = authorize_agent_tool_invocation(
            agent_name="FinancialAnalystAgent",
            tool_name="query_looker_financial_explore",
            caller=RootOrchestrator.name,
        )
        _track_agent_identity(calc_claims)
        bench_res = query_looker_financial_explore(
            explore_name="sector_benchmarks",
            tickers="ALL",
            gics_sector=scope["gics_sector"],
            start_year=scope["start_year"],
            end_year=scope["end_year"],
        )
        if bench_res.get("status") == "success":
            sector_benchmarks = list(bench_res.get("rows") or [])
        calc_ms = round((time.perf_counter() - t2) * 1000.0, 2)
        agent_trace.append(
            {
                "step": 2,
                "agent": "FinancialAnalystAgent",
                "agent_service_account": calc_claims.service_account_email,
                "spiffe_id": calc_claims.spiffe_id,
                "s2s_target_service": calc_claims.target_service,
                "s2s_verified": True,
                "tool": "query_looker_financial_explore(sector_benchmarks)",
                "status": "completed",
                "duration_ms": calc_ms,
                "summary": (
                    f"Retrieved {len(sector_benchmarks)} structured sector benchmark row(s) "
                    f"for {scope['gics_sector']}."
                ),
            }
        )

    # Step 3: Metadata-Filtered Hybrid RAG Search (768-d Dense + Lexical Boost)
    t3 = time.perf_counter()
    rag_claims = authorize_agent_tool_invocation(
        agent_name="FinancialAnalystAgent",
        tool_name="search_10k_chunks_rag",
        caller=RootOrchestrator.name,
    )
    _track_agent_identity(rag_claims)
    citations, candidates_searched = _retrieve_balanced_citations(
        query=query,
        tickers=scope["tickers"],
        gics_sector=scope["gics_sector"],
        section_id=scope["section_id"],
        start_year=scope["start_year"],
        end_year=scope["end_year"],
        top_k=top_k,
    )
    rag_ms = round((time.perf_counter() - t3) * 1000.0, 2)
    agent_trace.append(
        {
            "step": 3,
            "agent": "FinancialAnalystAgent",
            "agent_service_account": rag_claims.service_account_email,
            "spiffe_id": rag_claims.spiffe_id,
            "s2s_target_service": rag_claims.target_service,
            "s2s_verified": True,
            "tool": "search_10k_chunks_rag",
            "status": "completed",
            "duration_ms": rag_ms,
            "summary": (
                f"Ranked {candidates_searched:,} metadata-filtered 10-K chunks (768-d cosine + lexical boost) "
                f"→ top {len(citations)} grounded citations."
            ),
        }
    )

    # If user ran a broad query without naming specific tickers, compute variances for top cited issuers
    if (
        not variance_reports
        and citations
        and intent
        in (
            "longitudinal_single_company",
            "multi_company_peer_comparison",
        )
    ):
        inferred_tickers = []
        for cit in citations:
            if cit["ticker"] not in inferred_tickers:
                inferred_tickers.append(cit["ticker"])
            if len(inferred_tickers) >= 2:
                break
        if inferred_tickers:
            authorize_agent_tool_invocation(
                agent_name="FinancialAnalystAgent",
                tool_name="calculate_period_variances",
                caller=RootOrchestrator.name,
            )
            var_res = calculate_period_variances(
                tickers=",".join(inferred_tickers),
                start_year=scope["start_year"],
                end_year=scope["end_year"],
            )
            if var_res.get("status") == "success":
                variance_reports = list(var_res.get("variance_reports") or [])
                for rep in variance_reports:
                    calculated_variances_map[
                        f"{rep['ticker']}_{rep['start_year']}_{rep['end_year']}"
                    ] = rep

    # Step 3b: Deterministic anomaly flags (explained with MD&A passages) and, for thematic
    # questions, the Disclosure Delta pipeline (year-over-year wording changes). Both become
    # cited grounding facts for synthesis.
    material_flags: list[dict[str, Any]] = []
    material_flags_summary: dict[str, Any] | None = None
    if variance_reports and intent != "chunk_health_monitoring":
        t_flags = time.perf_counter()
        flag_claims = authorize_agent_tool_invocation(
            agent_name="FinancialAnalystAgent",
            tool_name="flag_material_variances",
            caller=RootOrchestrator.name,
        )
        _track_agent_identity(flag_claims)
        flag_res = flag_material_variances(
            tickers=",".join(str(r["ticker"]) for r in variance_reports),
            start_year=scope["start_year"],
            end_year=scope["end_year"],
        )
        explained_count = 0
        if flag_res.get("status") == "success":
            material_flags, explanation_cits = attach_flag_explanations(
                list(flag_res.get("flags") or []), max_flags=3
            )
            explained_count = len(explanation_cits)
            merge_citations(citations, explanation_cits)
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
                "duration_ms": round((time.perf_counter() - t_flags) * 1000.0, 2),
                "summary": (
                    f"Flagged {flag_res.get('total_flags', 0)} material variance(s) "
                    f"({flag_res.get('high_severity_count', 0)} high severity); attached MD&A "
                    f"explanations to {explained_count}."
                ),
            }
        )

    disclosure_deltas: dict[str, Any] | None = None
    if intent == "thematic_trend_tracking" and scope["end_year"] > scope["start_year"]:
        t_deltas = time.perf_counter()
        delta_claims = authorize_agent_tool_invocation(
            agent_name="FinancialAnalystAgent",
            tool_name="compare_disclosures_across_years",
            caller=RootOrchestrator.name,
        )
        _track_agent_identity(delta_claims)
        disclosure_deltas = compare_disclosures_across_years(
            theme=query,
            tickers=scope["tickers_csv"],
            gics_sector=scope["gics_sector"],
            section_id=scope["section_id"],
            start_year=scope["start_year"],
            end_year=scope["end_year"],
        )
        merge_citations(citations, disclosure_delta_citations(disclosure_deltas))
        delta_summary = (disclosure_deltas or {}).get("summary") or {}
        agent_trace.append(
            {
                "step": len(agent_trace) + 1,
                "agent": "FinancialAnalystAgent",
                "agent_service_account": delta_claims.service_account_email,
                "spiffe_id": delta_claims.spiffe_id,
                "s2s_target_service": delta_claims.target_service,
                "s2s_verified": True,
                "tool": "compare_disclosures_across_years",
                "status": (
                    "completed"
                    if (disclosure_deltas or {}).get("status") == "success"
                    else "warning"
                ),
                "duration_ms": round((time.perf_counter() - t_deltas) * 1000.0, 2),
                "summary": (
                    f"Compared theme disclosures for {delta_summary.get('companies_analyzed', 0)} "
                    f"company(ies) across {delta_summary.get('year_pairs', 0)} year pair(s): "
                    f"{delta_summary.get('added', 0)} added, {delta_summary.get('modified', 0)} "
                    f"modified, {delta_summary.get('removed', 0)} removed."
                ),
            }
        )

    # Step 4 & Step 5: Structured ReAct Execution Loop
    # (1. Evaluate calculated variances -> 2. Query cached 10-K context for explanation -> 3. Format grounded output + Flash eval)
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
        scope=scope,
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

    # Record S2S-attested trace step for ReAct Step 1 (Evaluate Variances via FinancialAnalystAgent)
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

    # Populate context_cache_info & S2S-attested trace step for ReAct Step 2 (Query Cached 10-K Context)
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
    elif citations:
        t4 = time.perf_counter()
        authorize_agent_s2s_hop(
            caller_agent="FinancialAnalystAgent",
            callee_agent="ContextCacheManager",
        )
        cache_claims = authorize_agent_tool_invocation(
            agent_name="ContextCacheManager",
            tool_name="load_cached_10k_section",
            caller="FinancialAnalystAgent",
        )
        _track_agent_identity(cache_claims)
        top_cit = citations[0]
        cache_res = load_cached_10k_section(
            ticker=top_cit["ticker"],
            fiscal_year=int(top_cit["fiscal_year"]),
            section_id=top_cit["section_id"],
            max_chars=4000,
        )
        cache_ms = round((time.perf_counter() - t4) * 1000.0, 2)
        if cache_res.get("status") == "success":
            context_cache_info = {
                "cache_key": cache_res["cache_key"],
                "cache_hit": cache_res["cache_hit"],
                "ticker": cache_res["ticker"],
                "fiscal_year": cache_res["fiscal_year"],
                "section_id": cache_res["section_id"],
                "token_estimate": cache_res["token_estimate"],
                "sec_filing_url": cache_res["sec_filing_url"],
                "sections_queried_count": 1,
                "cache_hits_count": 1 if cache_res["cache_hit"] else 0,
                "total_section_tokens": cache_res["token_estimate"],
                "lookup_latency_ms": cache_res.get("lookup_latency_ms", cache_ms),
            }
            agent_trace.append(
                {
                    "step": len(agent_trace) + 1,
                    "agent": "ContextCacheManager",
                    "agent_service_account": cache_claims.service_account_email,
                    "spiffe_id": cache_claims.spiffe_id,
                    "s2s_target_service": cache_claims.target_service,
                    "s2s_verified": True,
                    "tool": "load_cached_10k_section",
                    "status": "completed",
                    "duration_ms": cache_ms,
                    "summary": (
                        f"{'HIT' if cache_res['cache_hit'] else 'WARMED'} cached 10-K context "
                        f"`{cache_res['cache_key']}` (~{cache_res['token_estimate']:,} tokens)."
                    ),
                }
            )

    # Post-Generation Numeric Fact-Check & Gemini 3.5 Flash Evaluation S2S Attestation
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
    else:
        resp_decision = guardrail_svc.inspect_model_response(
            answer_markdown,
            caller_agent="FinancialAnalystAgent",
        )
        raw_report_text = str(react_result.get("raw_structured_report_text") or "")
        if not resp_decision.blocked and raw_report_text:
            # Structured-report verification can drop unsupported sentences before rendering;
            # a projection or MNPI leak in the raw model output must still be blocked & audited.
            raw_decision = guardrail_svc.inspect_model_response(
                raw_report_text,
                caller_agent="FinancialAnalystAgent",
            )
            if raw_decision.blocked:
                resp_decision = raw_decision
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
            tickers_analyzed=scope["tickers"],
            gics_sector=scope["gics_sector"],
            start_year=scope["start_year"],
            end_year=scope["end_year"],
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
            "resolved_scope": scope,
            "answer_markdown": (
                "### Response Withheld by Managed Model Armor & SEC Compliance Guardrails\n\n"
                f"{resp_decision.block_message}"
            ),
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

    if resp_decision.sdp_redacted:
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
                f"Formatted grounded report (`{reasoning_model}`) & evaluated (`{evaluation_model}`: "
                f"grade={eval_report.grade}, score={eval_report.overall_score:.2f}) · "
                f"Verified {fact_check.checked_metrics_count} variance figure(s) & "
                f"{fact_check.matched_citations_count} citation tag(s) · Passed: {fact_check.passed}."
            ),
        }
    )

    # Persist turn into ConversationMemoryManager (with automatic Gemini 3.5 Flash compaction if over threshold)
    variance_highlights = [
        ev.material_findings[0] for ev in variance_evaluations if ev.material_findings
    ]
    memory_persist = memory_mgr.record_turn(
        session_id=session_id,
        user_id=user_id,
        query=query,
        intent=intent,
        tickers=scope["tickers"],
        gics_sector=scope["gics_sector"],
        start_year=scope["start_year"],
        end_year=scope["end_year"],
        variance_highlights=variance_highlights,
        cited_chunk_ids=[c["chunk_id"] for c in citations],
        cached_10k_keys=[e.cache_key for e in cached_explanations],
        answer_markdown=answer_markdown,
    )

    total_latency_ms = round((time.perf_counter() - turn_start) * 1000.0, 2)
    analyzed_tickers = sorted(
        {
            *(scope.get("tickers") or []),
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
        gics_sector=scope["gics_sector"],
        start_year=scope["start_year"],
        end_year=scope["end_year"],
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
    return {
        "status": "success",
        "model_armor_blocked": False,
        "model_armor": {
            "prompt_inspection": prompt_decision.to_audit_dict(),
            "response_inspection": resp_decision.to_audit_dict(),
        },
        "execution_mode": active_execution_mode,
        "adk_runtime": adk_runtime_meta,
        # Verified structured answer (None when the live answer was free text or withheld)
        "structured_report": adk_runtime_meta.get("structured_report"),
        "report_verification": adk_runtime_meta.get("report_verification"),
        "prompt_store": get_prompt_store_status_metadata(),
        "intent": intent,
        "orchestrator": RootOrchestrator.name,
        "resolved_scope": scope,
        "answer_markdown": answer_markdown,
        # False when the live answer failed verification and its narrative was withheld
        "answer_verified": bool(react_result.get("answer_verified")),
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
