"""Conversational Hybrid RAG orchestration engine for the SEC EDGAR Natural Language Analyst UI."""

from __future__ import annotations

from collections import defaultdict
import re
import time
from typing import Any

from edgar_capstone.agents.callbacks import (
    _PROMPT_INJECTION_PATTERNS,
    verify_narrative_against_calculations,
)
from edgar_capstone.agents.root_orchestrator import (
    RootOrchestrator,
    classify_query_intent,
)
from edgar_capstone.agents.tools import (
    _get_dataset_dir,
    _load_all_profiles_cached,
    _parse_ticker_list,
    _profile_to_metric_record,
    calculate_period_variances,
    load_cached_10k_section,
    query_looker_financial_explore,
    search_10k_chunks_rag,
)
from edgar_capstone.security.agent_identity import (
    VerifiedAgentS2SClaims,
    authorize_agent_s2s_hop,
    authorize_agent_tool_invocation,
    get_agent_s2s_broker,
)
from edgar_capstone.security.rbac import CUSTOM_RUNTIME_SA_ACCOUNT_ID

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
_ITEM_1A_ONLY_RE = re.compile(
    r"\b(?:item\s*1a|risk\s+factors?)\b", re.IGNORECASE
)
_ITEM_7_ONLY_RE = re.compile(
    r"\b(?:item\s*7|md&a|management'?s\s+discussion)\b", re.IGNORECASE
)


def _format_usd_compact(value: float | None) -> str:
    """Format raw USD figure into crisp institutional notation (e.g., $215.94B, $842.10M)."""
    if value is None:
        return "N/A"
    sign = "-" if value < 0 else ""
    abs_v = abs(value)
    if abs_v >= 1_000_000_000:
        return f"{sign}${abs_v / 1_000_000_000:,.2f}B"
    if abs_v >= 1_000_000:
        return f"{sign}${abs_v / 1_000_000:,.2f}M"
    return f"{sign}${abs_v:,.0f}"


def _format_signed_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def _format_signed_bps(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f} bps"


def _clean_10k_excerpt(text: str, max_len: int = 340) -> str:
    """Clean escaped Markdown punctuation from 10-K excerpts for crisp analyst synthesis."""
    cleaned = re.sub(r"\\([.\-(),&])", r"\1", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_len:
        return cleaned
    clipped = cleaned[:max_len].rsplit(" ", 1)[0]
    return f"{clipped}..."


def list_curated_companies() -> dict[str, Any]:
    """Return all curated S&P 500 companies, sectors, and fiscal year bounds in the local corpus."""
    dataset_dir = _get_dataset_dir()
    profiles = _load_all_profiles_cached(str(dataset_dir))

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


def resolve_query_scope(
    query: str,
    tickers: str = "ALL",
    gics_sector: str = "ALL",
    section_id: str = "ALL",
    start_year: int = 2024,
    end_year: int = 2026,
) -> dict[str, Any]:
    """Combine explicit UI filter selections with natural-language entity & timeframe detection."""
    explicit_tickers = sorted(_parse_ticker_list(tickers))
    detected_tickers: list[str] = []
    for pattern, symbol in _COMPANY_ALIASES:
        if pattern.search(query) and symbol not in detected_tickers:
            detected_tickers.append(symbol)
    # Exact uppercase standalone 'V' check
    if re.search(r"(?<![A-Za-z0-9\-])V(?![A-Za-z0-9\-])", query) and "V" not in detected_tickers:
        detected_tickers.append("V")

    effective_tickers = explicit_tickers if explicit_tickers else detected_tickers

    effective_sector = gics_sector
    if not effective_sector or effective_sector.strip().upper() in ("ALL", "*"):
        effective_sector = "ALL"
        if not effective_tickers:
            for pattern, sec_name in _SECTOR_ALIASES:
                if pattern.search(query):
                    effective_sector = sec_name
                    break

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

    if eff_start > eff_end:
        eff_start, eff_end = eff_end, eff_start

    return {
        "tickers": effective_tickers,
        "tickers_csv": ",".join(effective_tickers) if effective_tickers else "ALL",
        "gics_sector": effective_sector,
        "section_id": effective_section,
        "start_year": eff_start,
        "end_year": eff_end,
    }


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


def _format_inline_citation(cit: dict[str, Any]) -> str:
    """Format an exact canonical inline citation tag linked to the SEC EDGAR archive URL."""
    ticker = cit["ticker"]
    fy = cit["fiscal_year"]
    sec_label = "Item 1A Risk Factors" if cit["section_id"] == "item_1a" else "Item 7 MD&A"
    chunk_id = cit["chunk_id"]
    url = cit.get("sec_filing_url") or "#source-viewer"
    return f"[{ticker} FY{fy} | {sec_label} | {chunk_id}]({url})"


def _synthesize_grounded_report(
    *,
    query: str,
    intent: str,
    scope: dict[str, Any],
    variance_reports: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    sector_benchmarks: list[dict[str, Any]],
    chunk_health: dict[str, Any] | None,
) -> str:
    """Synthesize a 100% deterministic, source-grounded Markdown report with inline 10-K citations."""
    lines: list[str] = []

    if intent == "chunk_health_monitoring" and chunk_health:
        lines.append("### Vector Corpus & Hybrid Index Diagnostics")
        lines.append(
            f"- **Corpus Status:** `{chunk_health.get('health_status', 'HEALTHY')}` "
            f"(Health Score: **{chunk_health.get('health_score', 1.0)}**)"
        )
        lines.append(
            f"- **Indexed 10-K Chunks:** **{chunk_health.get('total_chunks', 0):,}** "
            f"across 768-dimensional embeddings (`embedding_coverage_ratio`: "
            f"**{float(chunk_health.get('embedding_coverage_ratio') or 1.0) * 100:.1f}%**)"
        )
        by_sec = chunk_health.get("by_section") or {}
        if by_sec:
            lines.append(
                f"- **Section Distribution:** Item 1A Risk Factors (`{by_sec.get('item_1a', 0):,}` chunks) · "
                f"Item 7 MD&A (`{by_sec.get('item_7', 0):,}` chunks)"
            )
        if citations:
            lines.append("\n### Sample Retrieved Grounding Passages")
            for cit in citations[:3]:
                lines.append(
                    f"- **{_clean_10k_excerpt(cit['excerpt'], 240)}** {_format_inline_citation(cit)}"
                )
        return "\n".join(lines)

    # 1. Executive Synthesis Header
    if intent == "multi_company_peer_comparison" and len(variance_reports) >= 2:
        names = ", ".join(f"**{r['company_name']} ({r['ticker']})**" for r in variance_reports)
        lines.append("### Multi-Company Variance & Filing Synthesis")
        lines.append(
            f"Deterministic multi-period analysis comparing {names} across **FY{scope['start_year']}–FY{scope['end_year']}**, "
            f"grounded in SEC Form 10-K Item 7 (MD&A) and Item 1A (Risk Factors) disclosures."
        )
    elif variance_reports:
        rep0 = variance_reports[0]
        lines.append(
            f"### {rep0['company_name']} ({rep0['ticker']}) — FY{rep0['start_year']}–FY{rep0['end_year']} Grounded Variance Synthesis"
        )
        lines.append(
            f"Period-over-period financial performance and SEC Form 10-K narrative drivers for **{rep0['company_name']}** "
            f"(`CIK {rep0['cik']}` · *{rep0['gics_sector']} — {rep0['gics_sub_industry']}*)."
        )
    else:
        sector_desc = (
            f"**{scope['gics_sector']}**"
            if scope["gics_sector"] != "ALL"
            else "**S&P 500 Curated Corpus**"
        )
        lines.append(
            f"### Longitudinal 10-K Thematic Synthesis ({sector_desc}, FY{scope['start_year']}–FY{scope['end_year']})"
        )
        lines.append(
            f"Cross-filing hybrid retrieval combining 768-d semantic similarity and lexical keyword matching "
            f"for query: *\"{query.strip()}\"*."
        )

    # 2. Deterministic Quantitative Findings (100% agreement with calculation engine)
    if variance_reports:
        lines.append("\n### Deterministic Period-over-Period Metrics")
        for rep in variance_reports:
            rev = rep["revenue_variance"]
            op = rep["operating_income_variance"]
            ni = rep["net_income_variance"]
            fy_s, fy_e = rep["start_year"], rep["end_year"]

            cits_for_ticker = [c for c in citations if c["ticker"] == rep["ticker"]]
            primary_cit_tag = (
                f" {_format_inline_citation(cits_for_ticker[0])}" if cits_for_ticker else ""
            )

            cagr_clause = (
                f" ({_format_signed_pct(rev.get('cagr_pct'))} CAGR)"
                if rev.get("cagr_pct") is not None and fy_e > fy_s
                else ""
            )
            lines.append(
                f"- **{rep['ticker']} (FY{fy_s} → FY{fy_e}):** "
                f"Revenue moved from **{_format_usd_compact(rev.get('start_value'))}** to "
                f"**{_format_usd_compact(rev.get('end_value'))}** "
                f"(**{_format_signed_pct(rev.get('percentage_delta'))}**, "
                f"Δ **{_format_usd_compact(rev.get('absolute_delta'))}**{cagr_clause}); "
                f"Operating Income moved **{_format_signed_pct(op.get('percentage_delta'))}** to "
                f"**{_format_usd_compact(op.get('end_value'))}** "
                f"(Operating Margin **{_format_signed_bps(rep.get('operating_margin_bps_delta'))}**); "
                f"Net Income moved **{_format_signed_pct(ni.get('percentage_delta'))}** to "
                f"**{_format_usd_compact(ni.get('end_value'))}** "
                f"(Net Margin **{_format_signed_bps(rep.get('net_margin_bps_delta'))}**).{primary_cit_tag}"
            )

    elif sector_benchmarks:
        lines.append("\n### Sector Quantitative Benchmarks")
        for bench in sector_benchmarks[-3:]:
            lines.append(
                f"- **{bench['gics_sector']} (FY{bench['fiscal_year']} · {bench['company_count']} issuers):** "
                f"Aggregate Revenue **{_format_usd_compact(bench.get('total_sector_revenue'))}**, "
                f"Operating Income **{_format_usd_compact(bench.get('total_sector_operating_income'))}**, "
                f"Net Income **{_format_usd_compact(bench.get('total_sector_net_income'))}**."
            )

    # 3. Grounded 10-K Narrative Drivers & Thematic Shifts
    if citations:
        if intent == "thematic_trend_tracking":
            lines.append("\n### Longitudinal Disclosure Evolution Across Fiscal Periods")
            by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for cit in citations:
                by_year[int(cit["fiscal_year"])].append(cit)
            for fy in sorted(by_year.keys()):
                yr_cits = by_year[fy][:3]
                lines.append(f"#### Fiscal Year {fy} Filing Disclosures")
                for cit in yr_cits:
                    sec_title = cit.get("section_title") or cit["section_id"].upper()
                    lines.append(
                        f"- **{cit['company_name']} ({cit['ticker']}) — {sec_title}:** "
                        f"\"{_clean_10k_excerpt(cit['excerpt'], 290)}\" {_format_inline_citation(cit)}"
                    )
        else:
            item7_cits = [c for c in citations if c["section_id"] == "item_7"]
            item1a_cits = [c for c in citations if c["section_id"] == "item_1a"]

            if item7_cits:
                lines.append("\n### Item 7 (MD&A) Operational & Financial Drivers")
                for cit in item7_cits[:4]:
                    sec_title = cit.get("section_title") or "Item 7. MD&A"
                    lines.append(
                        f"- **{cit['ticker']} FY{cit['fiscal_year']} ({sec_title}):** "
                        f"\"{_clean_10k_excerpt(cit['excerpt'], 300)}\" {_format_inline_citation(cit)}"
                    )

            if item1a_cits:
                lines.append("\n### Item 1A Risk Factor Disclosures & Headwinds")
                for cit in item1a_cits[:4]:
                    sec_title = cit.get("section_title") or "Item 1A. Risk Factors"
                    lines.append(
                        f"- **{cit['ticker']} FY{cit['fiscal_year']} ({sec_title}):** "
                        f"\"{_clean_10k_excerpt(cit['excerpt'], 300)}\" {_format_inline_citation(cit)}"
                    )
    else:
        lines.append(
            "\n### Grounding Status\n"
            "No 10-K filing passages matched the current filter combination. "
            "Widen the fiscal period range or reset the GICS sector / section filter to inspect broader disclosures."
        )

    return "\n".join(lines)


def run_conversational_rag_turn(
    *,
    query: str,
    tickers: str = "ALL",
    gics_sector: str = "ALL",
    section_id: str = "ALL",
    start_year: int = 2024,
    end_year: int = 2026,
    top_k: int = 8,
) -> dict[str, Any]:
    """Execute a full source-grounded Hybrid RAG conversational turn with ReAct trace & fact-checking."""
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

    # Step 0: Model Armor Safety & Prompt-Injection Guardrail
    t0 = time.perf_counter()
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(query):
            guard_claims = authorize_agent_s2s_hop(
                caller_agent=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
                callee_agent="ModelArmorGuardrail",
            )
            _track_agent_identity(guard_claims)
            guard_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            agent_trace.append(
                {
                    "step": 1,
                    "agent": "ModelArmorGuardrail",
                    "agent_service_account": guard_claims.service_account_email,
                    "spiffe_id": guard_claims.spiffe_id,
                    "s2s_target_service": guard_claims.target_service,
                    "s2s_verified": True,
                    "tool": "before_model_safety_and_latency_callback",
                    "status": "blocked",
                    "duration_ms": guard_ms,
                    "summary": "Blocked prompt-injection or grounding-bypass pattern before model execution.",
                }
            )
            return {
                "status": "blocked",
                "model_armor_blocked": True,
                "intent": "safety_guardrail_block",
                "orchestrator": RootOrchestrator.name,
                "resolved_scope": {
                    "tickers": [],
                    "gics_sector": gics_sector,
                    "section_id": section_id,
                    "start_year": start_year,
                    "end_year": end_year,
                },
                "answer_markdown": (
                    "### Request Blocked by Strict Grounding & Safety Guardrails\n\n"
                    "Model Armor detected an attempt to override system instructions or bypass strict SEC 10-K grounding. "
                    "This workspace only synthesizes verified EDGAR Form 10-K disclosures and deterministic financial metrics."
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
                "agent_trace": agent_trace,
                "agent_identities_used": list(identities_used_by_agent.values()),
                "total_latency_ms": round((time.perf_counter() - turn_start) * 1000.0, 2),
            }

    # Step 1: Scope Resolution & RootOrchestrator Intent Routing
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
    )
    intent = classify_query_intent(query, tickers=scope["tickers"] or None)
    route_ms = round((time.perf_counter() - t1) * 1000.0, 2)
    agent_trace.append(
        {
            "step": 1,
            "agent": RootOrchestrator.name,
            "agent_service_account": route_claims.service_account_email,
            "spiffe_id": route_claims.spiffe_id,
            "s2s_target_service": route_claims.target_service,
            "s2s_verified": True,
            "tool": "classify_query_intent",
            "status": "completed",
            "duration_ms": route_ms,
            "summary": (
                f"Routed to '{intent}' · Tickers: {scope['tickers_csv']} · "
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
                    f"Computed deterministic Revenue, Operating Income, Net Income, and margin bps variances "
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

    # If user ran a broad thematic query without naming specific tickers, compute variances for top cited issuers
    # ONLY if quantitative metrics are relevant or if single/peer comparison was detected
    if not variance_reports and citations and intent in (
        "longitudinal_single_company",
        "multi_company_peer_comparison",
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

    # Step 4: Context Cache Warmup for Top Cited 10-K Section (FinancialAnalystAgent -> ContextCacheManager S2S hop)
    context_cache_info: dict[str, Any] | None = None
    if citations:
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
            }
            agent_trace.append(
                {
                    "step": 4,
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

    # Step 5: Grounded Narrative Synthesis & Post-Generation Fact-Check Verification
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
    answer_markdown = _synthesize_grounded_report(
        query=query,
        intent=intent,
        scope=scope,
        variance_reports=variance_reports,
        citations=citations,
        sector_benchmarks=sector_benchmarks,
        chunk_health=chunk_health,
    )
    synth_ms = round((time.perf_counter() - t5) * 1000.0, 2)

    fact_check = verify_narrative_against_calculations(
        response_text=answer_markdown,
        calculated_variances=calculated_variances_map,
        retrieved_citations=citations,
        model_latency_ms=synth_ms,
    )
    agent_trace.append(
        {
            "step": len(agent_trace) + 1,
            "agent": "FinancialAnalystAgent",
            "agent_service_account": fc_claims.service_account_email,
            "spiffe_id": fc_claims.spiffe_id,
            "s2s_target_service": fc_claims.target_service,
            "s2s_verified": True,
            "tool": "after_model_fact_check_callback",
            "status": "completed" if fact_check.passed else "warning",
            "duration_ms": synth_ms,
            "summary": (
                f"Verified {fact_check.checked_metrics_count} deterministic variance figure(s) and "
                f"{fact_check.matched_citations_count} inline 10-K citation tag(s) · "
                f"Passed: {fact_check.passed}."
            ),
        }
    )

    total_latency_ms = round((time.perf_counter() - turn_start) * 1000.0, 2)
    return {
        "status": "success",
        "model_armor_blocked": False,
        "intent": intent,
        "orchestrator": RootOrchestrator.name,
        "resolved_scope": scope,
        "answer_markdown": answer_markdown,
        "variance_reports": variance_reports,
        "sector_benchmarks": sector_benchmarks,
        "citations": citations,
        "candidates_searched": candidates_searched,
        "context_cache": context_cache_info,
        "fact_check": fact_check.model_dump(mode="json"),
        "agent_trace": agent_trace,
        "agent_identities_used": list(identities_used_by_agent.values()),
        "total_latency_ms": total_latency_ms,
    }
