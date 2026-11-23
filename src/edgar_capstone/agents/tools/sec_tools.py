"""ADK tools: MetricFetchTool, AnalyticsViewQueryTool, VarianceCalculationTool, HybridSearchTool, and ContextCacheTool.

Every tool reads live GCP data through the modular client adapters under
`edgar_capstone.agents.adapters`: BigQuery (`financial_metrics`, `filing_chunks`,
`VECTOR_SEARCH`, and sector-wide `ML.DISTANCE` ranking), Vertex AI Search / Discovery Engine,
Vertex AI `text-embedding-005` embeddings, and the Cloud Storage SEC corpus bucket.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from google.adk.tools import FunctionTool, ToolContext

from edgar_capstone.agents.adapters import (
    DEFAULT_BQ_CHUNK_HEALTH_VIEW,
    DEFAULT_BQ_FILING_CHUNKS_TABLE,
    DEFAULT_BQ_FINANCIAL_METRICS_TABLE,
    DEFAULT_BQ_YOY_VARIANCES_VIEW,
    BigQueryAdapter,
    GcpAdapterHttpError,
    GcpEndpointNotReadyError,
    GcpToolAdapters,
    GcpToolConfig,
    GcsCorpusAdapter,
    VertexAISearchAdapter,
    _default_section_title,
    _gcp_adapters,
    _normalize_section_id,
    configure_gcp_tool_adapters,
    get_active_gcp_tool_adapters,
    get_gcp_adapters_status_metadata,
    register_adapter_reset_hook,
    reset_gcp_tool_adapters,
)
from edgar_capstone.agents.core.citations import merge_citations
from edgar_capstone.agents.core.resilience import with_tool_resilience
from edgar_capstone.agents.core.schemas import GroundedCitation
from edgar_capstone.agents.prompts.builders import compact_variance_reports_for_prompt
from edgar_capstone.agents.tools.hybrid_rag import _execute_live_gcp_hybrid_rag
from edgar_capstone.agents.tools.runtime import (
    STATE_ANALYTICS_VIEW_RESULTS,
    STATE_CALCULATED_VARIANCES,
    STATE_CHUNK_HEALTH,
    STATE_VARIANCE_MISSING_TICKERS,
    cached_turn_profiles,
    variance_map,
)
from edgar_capstone.agents.tools.variance_engine import (
    _build_variance_reports_from_profiles,
    _compute_single_metric_variance,
    _numeric_citation_for_profile,
    _profile_to_metric_record,
    _xbrl_concept_for_metric,
)
from edgar_capstone.common.cache import (
    BoundedTTLCache,
    CacheMetrics,
    parse_10k_cache_key,
)
from edgar_capstone.common.domain import (
    SECTION_TITLES,
    canonicalize_gics_sector,
    normalize_edgar_ticker,
)
from edgar_capstone.common.env import env_positive_float, env_positive_int
from edgar_capstone.gcp.defaults import DEFAULT_BIGQUERY_DATASET_ID
from edgar_capstone.gcp.discovery_engine import (
    DEFAULT_VERTEX_AI_SEARCH_COLLECTION,
    DEFAULT_VERTEX_AI_SEARCH_DATASTORE_ID,
    DEFAULT_VERTEX_AI_SEARCH_ENGINE_ID,
    DEFAULT_VERTEX_AI_SEARCH_LOCATION,
)
from edgar_capstone.telemetry import get_telemetry_sink
from edgar_capstone.vector_db.chunker import (
    estimate_token_count,
    extract_sections_from_combined_markdown,
)

logger = logging.getLogger(__name__)

_WORD_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

DEFAULT_10K_SECTION_CACHE_MAX_ENTRIES = 256
DEFAULT_10K_SECTION_CACHE_TTL_SECONDS = 3600.0

_SECTION_CACHE_MAX_ENTRIES = env_positive_int(
    "EDGAR_10K_SECTION_CACHE_MAX_ENTRIES", DEFAULT_10K_SECTION_CACHE_MAX_ENTRIES
)
_SECTION_CACHE_TTL_SECONDS = env_positive_float(
    "EDGAR_10K_SECTION_CACHE_TTL_SECONDS",
    DEFAULT_10K_SECTION_CACHE_TTL_SECONDS,
    minimum=1.0,
)

_SUPPORTED_ANALYTICS_VIEWS = frozenset(
    {
        "yoy_variances",
        "sector_benchmarks",
        "peer_comparison",
        "chunk_health",
        "llm_telemetry",
        "session_analytics",
        "audit_trail",
    }
)
_ANALYTICS_VIEW_ALIASES: dict[str, str] = {
    "llm_telemetry_and_cost": "llm_telemetry",
    "security_audit_logs": "audit_trail",
}


class _TrackedSectionCache(BoundedTTLCache[str, dict[str, Any]]):
    """Thread-safe LRU + TTL section cache extending `BoundedTTLCache` with eviction tracking."""

    def __init__(self, *, max_entries: int, ttl_seconds: float) -> None:
        super().__init__(max_entries=max_entries, ttl_seconds=ttl_seconds)
        self.evictions: int = 0

    def _prune_expired_unlocked(self, now: float) -> None:
        before = len(self._data)
        super()._prune_expired_unlocked(now)
        self.evictions += max(0, before - len(self._data))

    def __getitem__(self, key: str) -> dict[str, Any]:
        with self._lock:
            before = len(self._data)
            try:
                return super().__getitem__(key)
            except KeyError:
                if len(self._data) < before:
                    self.evictions += before - len(self._data)
                raise

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            existed = key in self._data
            before = len(self._data)
            super().__setitem__(key, value)
            expected = before + (0 if existed else 1)
            self.evictions += max(0, expected - len(self._data))


@dataclass(slots=True)
class _ContextCacheRuntimeStats:
    """Thread-safe runtime hit/miss, token-savings, and latency metrics backed by `CacheMetrics`."""

    metrics: CacheMetrics = field(default_factory=CacheMetrics)
    cached_tokens_saved: int = 0
    total_hit_latency_ms: float = 0.0
    total_miss_latency_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_hit(self, *, token_estimate: int, latency_ms: float) -> int:
        saved_delta = int(token_estimate * 0.75)
        self.metrics.record_hit(token_estimate)
        with self._lock:
            self.cached_tokens_saved += saved_delta
            self.total_hit_latency_ms += latency_ms
        return saved_delta

    def record_miss(self, *, token_estimate: int, latency_ms: float) -> None:
        self.metrics.record_miss(token_estimate)
        with self._lock:
            self.total_miss_latency_ms += latency_ms

    def snapshot(self) -> dict[str, Any]:
        base = self.metrics.snapshot()
        with self._lock:
            return {
                **base,
                "cached_tokens_saved": self.cached_tokens_saved,
                "total_hit_latency_ms": self.total_hit_latency_ms,
                "total_miss_latency_ms": self.total_miss_latency_ms,
            }


# Process-local L1 cache of parsed 10-K sections (bounded LRU + TTL)
_SESSION_10K_CONTEXT_CACHE: _TrackedSectionCache = _TrackedSectionCache(
    max_entries=_SECTION_CACHE_MAX_ENTRIES,
    ttl_seconds=_SECTION_CACHE_TTL_SECONDS,
)
# Process-local L1 cache of BigQuery filing_chunks (with pre-computed 768-d embeddings) per section.
_SECTION_CHUNKS_CACHE: _TrackedSectionCache = _TrackedSectionCache(
    max_entries=_SECTION_CACHE_MAX_ENTRIES,
    ttl_seconds=_SECTION_CACHE_TTL_SECONDS,
)
_CONTEXT_CACHE_STATS = _ContextCacheRuntimeStats()

register_adapter_reset_hook(_SECTION_CHUNKS_CACHE.clear)


def _record_gcp_backend_invocation(
    tool_context: ToolContext | None,
    *,
    tool_name: str,
    data_source: str,
    services_hit: list[str],
) -> None:
    """Record which GCP services were invoked on ADK `tool_context.state`."""
    if tool_context is None or not hasattr(tool_context, "state"):
        return
    try:
        entries = list(tool_context.state.get("gcp_backend_calls", []))
        entries.append(
            {
                "tool_name": tool_name,
                "data_source": data_source,
                "services_hit": list(services_hit),
            }
        )
        tool_context.state["gcp_backend_calls"] = entries
    except Exception as exc:
        logger.debug("Skipping gcp_backend_calls telemetry update: %s", exc)


def get_context_cache_runtime_stats() -> dict[str, Any]:
    """Return runtime Context Cache hit/miss counters, cached keys, and latency telemetry."""
    counters = _CONTEXT_CACHE_STATS.snapshot()
    hits = int(counters["hits"])
    misses = int(counters["misses"])
    total = hits + misses
    hit_ms = float(counters["total_hit_latency_ms"])
    miss_ms = float(counters["total_miss_latency_ms"])
    resident = _SESSION_10K_CONTEXT_CACHE.snapshot()
    return {
        "cached_sections_count": len(resident),
        "cached_keys": sorted(resident.keys()),
        "total_cached_tokens_resident": sum(
            int(v.get("token_estimate", 0)) for v in resident.values()
        ),
        "section_cache_max_entries": _SESSION_10K_CONTEXT_CACHE.max_entries,
        "section_cache_ttl_seconds": _SESSION_10K_CONTEXT_CACHE.ttl_seconds,
        "section_cache_evictions": _SESSION_10K_CONTEXT_CACHE.evictions,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 4) if total > 0 else 0.0,
        "cached_tokens_saved": int(counters["cached_tokens_saved"]),
        "mean_hit_lookup_latency_ms": round(hit_ms / hits, 3) if hits > 0 else 0.0,
        "mean_miss_load_latency_ms": round(miss_ms / misses, 3) if misses > 0 else 0.0,
    }


def _parse_ticker_list(tickers_csv: str) -> set[str]:
    """Parse comma- or space-separated ticker string into normalized uppercase symbols."""
    if not tickers_csv or tickers_csv.strip().upper() in ("*", "ALL", ""):
        return set()
    parts = re.split(r"[,;\s]+", tickers_csv.strip())
    return {normalize_edgar_ticker(p) for p in parts if p.strip()}


def _normalize_sector_filter(gics_sector: str | None) -> str | None:
    canon = canonicalize_gics_sector(gics_sector)
    return canon.lower() if canon else None


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2D embedding matrix, guarding against zero-norm vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _update_context_cache_state(
    tool_context: ToolContext | None,
    *,
    cache_hit: bool,
    saved_tokens_delta: int,
    lookup_latency_ms: float,
    cache_key: str,
) -> None:
    if tool_context is None or not hasattr(tool_context, "state"):
        return
    stats = dict(
        tool_context.state.get(
            "context_cache_stats",
            {"hits": 0, "misses": 0, "cached_tokens_saved": 0},
        )
    )
    if cache_hit:
        stats["hits"] = int(stats.get("hits", 0)) + 1
        stats["cached_tokens_saved"] = int(stats.get("cached_tokens_saved", 0)) + saved_tokens_delta
    else:
        stats["misses"] = int(stats.get("misses", 0)) + 1
    stats["last_lookup_latency_ms"] = lookup_latency_ms
    stats["last_cache_key"] = cache_key
    tool_context.state["context_cache_stats"] = stats


# ---------------------------------------------------------------------------
# Analytics View Dispatch Helpers (BigQuery SQL Views & Telemetry Store)
# ---------------------------------------------------------------------------


def _query_telemetry_explore(explore_key: str, ticker_filter: set[str]) -> dict[str, Any]:
    sink = get_telemetry_sink()
    if explore_key == "llm_telemetry":
        rows = sink.query_llm_telemetry_explore(limit=50)
        return {
            "status": "success",
            "explore_name": "llm_telemetry",
            "lookml_model": "sec_edgar_analytics",
            "lookml_explore": "llm_telemetry_and_cost",
            "row_count": len(rows),
            "rows": rows,
        }
    if explore_key == "session_analytics":
        rows = sink.query_session_telemetry(limit=50)
        if ticker_filter:
            rows = [
                r
                for r in rows
                if ticker_filter.intersection(
                    {normalize_edgar_ticker(str(t)) for t in (r.get("tickers_analyzed") or [])}
                )
            ]
        return {
            "status": "success",
            "explore_name": "session_analytics",
            "lookml_model": "sec_edgar_analytics",
            "lookml_explore": "session_analytics",
            "summary": sink.summarize_session_telemetry(),
            "row_count": len(rows),
            "rows": rows,
        }
    rows = sink.query_security_audit_logs(limit=50)
    if ticker_filter:
        rows = [
            r
            for r in rows
            if ticker_filter.intersection(
                {normalize_edgar_ticker(str(t)) for t in (r.get("tickers") or [])}
            )
        ]
    return {
        "status": "success",
        "explore_name": "audit_trail",
        "lookml_model": "sec_edgar_analytics",
        "lookml_explore": "security_audit_logs",
        "row_count": len(rows),
        "rows": rows,
    }


def _query_bigquery_explore(
    explore_key: str,
    *,
    ticker_filter: set[str],
    sector_norm: str | None,
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    adapters = _gcp_adapters()
    if explore_key == "chunk_health":
        return adapters.bigquery.query_chunk_health()
    if explore_key in ("yoy_variances", "peer_comparison"):
        bq_rows = adapters.bigquery.query_yoy_variances(
            tickers=ticker_filter,
            gics_sector=sector_norm,
            start_year=start_year,
            end_year=end_year,
        )
    else:
        bq_rows = adapters.bigquery.query_sector_benchmarks(
            tickers=ticker_filter,
            gics_sector=sector_norm,
            start_year=start_year,
            end_year=end_year,
        )
    return {
        "status": "success",
        "explore_name": explore_key,
        "row_count": len(bq_rows),
        "rows": bq_rows,
    }


# ---------------------------------------------------------------------------
# ADK Tool Implementations
# ---------------------------------------------------------------------------


@with_tool_resilience(tool_name="fetch_financial_metrics")
def fetch_financial_metrics(
    tickers: str,
    start_year: int,
    end_year: int,
    gics_sector: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Retrieves raw structured SEC 10-K financial metrics (Revenue, Operating Income, Net Income, R&D expense, margins, and filing URLs).

    Args:
        tickers: Comma-separated ticker symbols (e.g., 'NVDA,AAPL,MSFT') or 'ALL' for all companies.
        start_year: Earliest fiscal year to retrieve (inclusive, e.g., 2024).
        end_year: Latest fiscal year to retrieve (inclusive, e.g., 2026).
        gics_sector: Optional GICS Sector filter (e.g., 'Information Technology') or 'ALL' for any sector.

    Returns:
        Dictionary with 'status', 'record_count', and 'metrics' list of structured financial records.
    """
    ticker_filter = _parse_ticker_list(tickers)
    sector_norm = _normalize_sector_filter(gics_sector)

    adapters = _gcp_adapters()
    bq_profiles = adapters.bigquery.fetch_profiles(
        tickers=ticker_filter,
        start_year=start_year,
        end_year=end_year,
        gics_sector=sector_norm,
    )
    records = [_profile_to_metric_record(p).model_dump(mode="json") for p in bq_profiles]
    _record_gcp_backend_invocation(
        tool_context,
        tool_name="fetch_financial_metrics",
        data_source="gcp",
        services_hit=["bigquery"],
    )

    if tool_context is not None and hasattr(tool_context, "state"):
        existing = list(tool_context.state.get("fetched_metrics", []))
        existing.extend(records)
        tool_context.state["fetched_metrics"] = existing

    return {
        "status": "success",
        "record_count": len(records),
        "tickers_found": sorted({r["ticker"] for r in records}),
        "fiscal_years": sorted({r["fiscal_year"] for r in records}),
        "metrics": records,
    }


@with_tool_resilience(tool_name="query_analytics_view")
def query_analytics_view(
    explore_name: str,
    tickers: str,
    gics_sector: str,
    start_year: int,
    end_year: int,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Queries BigQuery analytics and telemetry views for YoY variances, sector benchmarks, peer comparisons, chunk health, telemetry/cost, session analytics, or audit trail.

    Args:
        explore_name: Analytics view to query: 'yoy_variances', 'sector_benchmarks', 'peer_comparison', 'chunk_health', 'llm_telemetry', 'session_analytics', or 'audit_trail'.
        tickers: Comma-separated ticker symbols (e.g., 'NVDA,AMD,AVGO') or 'ALL'.
        gics_sector: GICS Sector filter (e.g., 'Information Technology', 'Financials') or 'ALL'.
        start_year: Earliest fiscal year to include (e.g., 2024).
        end_year: Latest fiscal year to include (e.g., 2026).

    Returns:
        Dictionary with 'status', 'explore_name', 'row_count', and 'rows' from the requested analytics view.
    """
    raw_explore_key = explore_name.strip().lower()
    explore_key = _ANALYTICS_VIEW_ALIASES.get(raw_explore_key, raw_explore_key)
    if explore_key not in _SUPPORTED_ANALYTICS_VIEWS:
        return {
            "status": "error",
            "explore_name": explore_name,
            "message": (
                f"Unsupported explore_name '{explore_name}'. "
                "Supported: 'yoy_variances', 'sector_benchmarks', 'peer_comparison', "
                "'chunk_health', 'llm_telemetry', 'session_analytics', 'audit_trail'."
            ),
        }

    ticker_filter = _parse_ticker_list(tickers)
    if explore_key in ("llm_telemetry", "session_analytics", "audit_trail"):
        result = _query_telemetry_explore(explore_key, ticker_filter)
    else:
        result = _query_bigquery_explore(
            explore_key,
            ticker_filter=ticker_filter,
            sector_norm=_normalize_sector_filter(gics_sector),
            start_year=start_year,
            end_year=end_year,
        )
        _record_gcp_backend_invocation(
            tool_context,
            tool_name="query_analytics_view",
            data_source="gcp",
            services_hit=["bigquery"],
        )

    if tool_context is not None and hasattr(tool_context, "state"):
        if explore_key == "chunk_health":
            if result.get("status") == "success":
                tool_context.state[STATE_CHUNK_HEALTH] = result
        else:
            history = list(tool_context.state.get(STATE_ANALYTICS_VIEW_RESULTS) or [])
            history.append(result)
            tool_context.state[STATE_ANALYTICS_VIEW_RESULTS] = history
    return result


# Backward-compatible deprecated alias (prefer `query_analytics_view`).
query_looker_financial_explore = query_analytics_view


@with_tool_resilience(tool_name="calculate_period_variances")
def calculate_period_variances(
    tickers: str,
    start_year: int,
    end_year: int,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Performs deterministic period-over-period variance calculations for Revenue, Operating Income, Net Income, and R&D expense.

    Separates arithmetic calculation from LLM narrative synthesis. Call this tool whenever the
    question asks about reported financial figures (revenue, operating income, net income,
    R&D spend, margins, growth, CAGR, or a peer financial comparison) so narrative claims match
    deterministic numbers 100%. Do NOT call it for purely qualitative questions (risk-factor
    wording, disclosure themes) that do not ask about financial figures.

    Args:
        tickers: Comma-separated ticker symbols (e.g., 'NVDA' or 'NVDA,AAPL').
        start_year: Starting fiscal year for the variance calculation (e.g., 2024).
        end_year: Ending fiscal year for the variance calculation (e.g., 2026).

    Returns:
        Dictionary containing deterministic multi-period and YoY variance reports per company.
    """
    ticker_filter = _parse_ticker_list(tickers)
    if not ticker_filter:
        return {
            "status": "error",
            "message": "Please specify at least one ticker symbol (e.g., 'NVDA' or 'NVDA,AAPL').",
        }

    # Fetch start_year - 1 so the initial year's YoY step delta is populated. Inside a chat
    # turn the cached BigQuery profile catalog is reused (same `financial_metrics` rows, 0 RPCs).
    lo_year = int(start_year) - 1
    cached = cached_turn_profiles()
    if cached:
        profiles = [
            p
            for p in cached
            if normalize_edgar_ticker(p.ticker) in ticker_filter
            and lo_year <= int(p.fiscal_year) <= int(end_year)
        ]
    else:
        profiles = _gcp_adapters().bigquery.fetch_profiles(
            tickers=ticker_filter,
            start_year=lo_year,
            end_year=end_year,
        )
    _record_gcp_backend_invocation(
        tool_context,
        tool_name="calculate_period_variances",
        data_source="gcp",
        services_hit=["bigquery"],
    )

    reports, missing_tickers = _build_variance_reports_from_profiles(
        profiles,
        ticker_filter=ticker_filter,
        start_year=start_year,
        end_year=end_year,
    )

    if tool_context is not None and hasattr(tool_context, "state"):
        calc_state = dict(tool_context.state.get(STATE_CALCULATED_VARIANCES) or {})
        calc_state.update(variance_map(reports))
        tool_context.state[STATE_CALCULATED_VARIANCES] = calc_state
        prior_missing = set(tool_context.state.get(STATE_VARIANCE_MISSING_TICKERS) or [])
        tool_context.state[STATE_VARIANCE_MISSING_TICKERS] = sorted(
            prior_missing | set(missing_tickers)
        )
        # Full XBRL provenance reports live in session_state for claim verification and rendering;
        # return a token-lean digest to the LLM to cut Call 2+ prefill latency (~18 KB/company).
        return {
            "status": "success",
            "company_count": len(reports),
            "missing_tickers": missing_tickers,
            "variance_reports": compact_variance_reports_for_prompt(reports),
        }

    return {
        "status": "success",
        "company_count": len(reports),
        "missing_tickers": missing_tickers,
        "variance_reports": reports,
    }


@with_tool_resilience(tool_name="search_10k_chunks_rag")
def search_10k_chunks_rag(
    query: str,
    tickers: str,
    gics_sector: str,
    section_id: str,
    start_year: int,
    end_year: int,
    top_k: int,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Performs metadata-filtered Hybrid RAG search across SEC 10-K chunks (Item 1A Risk Factors & Item 7 MD&A).

    Combines dense 768-d vector cosine similarity with lexical keyword matching and metadata
    pre-filtering to retrieve grounded 10-K passages with exact section citations and SEC URLs.

    Args:
        query: Natural language search query (e.g., 'AI GPU export control restrictions China H20').
        tickers: Comma-separated ticker symbols (e.g., 'NVDA,AAPL') or 'ALL' for cross-company search.
        gics_sector: GICS Sector filter (e.g., 'Information Technology') or 'ALL'.
        section_id: 10-K section filter: 'item_1a' (Risk Factors), 'item_7' (MD&A), or 'ALL'.
        start_year: Earliest fiscal year to search (inclusive, e.g., 2024).
        end_year: Latest fiscal year to search (inclusive, e.g., 2026).
        top_k: Number of top-ranked grounded citations to return (1 to 25).

    Returns:
        Dictionary with 'status', 'citation_count', and 'citations' list of GroundedCitation records.
    """
    ticker_filter = _parse_ticker_list(tickers)
    sector_norm = _normalize_sector_filter(gics_sector)
    sec_norm = (
        section_id.strip().lower()
        if section_id and section_id.strip().upper() not in ("ALL", "*", "")
        else None
    )
    effective_top_k = max(1, min(int(top_k), 25))

    gcp_citations, cand_count, services_hit = _execute_live_gcp_hybrid_rag(
        _gcp_adapters(),
        query=query,
        ticker_filter=ticker_filter,
        sector_norm=sector_norm,
        sec_norm=sec_norm,
        start_year=start_year,
        end_year=end_year,
        effective_top_k=effective_top_k,
    )
    if tool_context is not None and hasattr(tool_context, "state"):
        existing_cits = list(tool_context.state.get("retrieved_citations", []))
        tool_context.state["retrieved_citations"] = merge_citations(existing_cits, gcp_citations)
    _record_gcp_backend_invocation(
        tool_context,
        tool_name="search_10k_chunks_rag",
        data_source="gcp",
        services_hit=services_hit,
    )
    if not gcp_citations:
        return {
            "status": "success",
            "citation_count": 0,
            "citations": [],
            "message": "No 10-K chunks matched the requested metadata filters.",
        }
    return {
        "status": "success",
        "citation_count": len(gcp_citations),
        "candidates_searched": cand_count,
        "citations": gcp_citations,
    }


@with_tool_resilience(tool_name="load_cached_10k_section")
def load_cached_10k_section(
    ticker: str,
    fiscal_year: int,
    section_id: str,
    max_chars: int,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Loads and caches a company's 10-K section (Item 1A Risk Factors or Item 7 MD&A) via the Context Cache Manager.

    Tracks Vertex AI Context Cache hits/misses, lookup latency, and estimated cached-token cost discounts.

    Args:
        ticker: Company ticker symbol (e.g., 'NVDA', 'AAPL').
        fiscal_year: Reporting fiscal year (e.g., 2026).
        section_id: Section to load: 'item_1a' (Risk Factors) or 'item_7' (MD&A).
        max_chars: Maximum characters of section text to return in the tool response (e.g., 12000).

    Returns:
        Dictionary with 'status', 'cache_hit', 'token_estimate', 'lookup_latency_ms', 'sec_filing_url', and grounded Markdown 'content'.
    """
    t0 = time.perf_counter()
    norm_ticker = normalize_edgar_ticker(ticker)
    sec_key = section_id.strip().lower()
    if sec_key not in ("item_1a", "item_7"):
        return {
            "status": "error",
            "message": f"section_id must be 'item_1a' or 'item_7', got {section_id!r}",
        }

    cache_id = f"{norm_ticker}_{fiscal_year}_{sec_key}"
    entry: dict[str, Any] | None = _SESSION_10K_CONTEXT_CACHE.get(cache_id)
    cache_hit = entry is not None

    if entry is None:
        adapters = _gcp_adapters()
        bq_profiles = adapters.bigquery.fetch_profiles(
            tickers={norm_ticker},
            start_year=fiscal_year,
            end_year=fiscal_year,
        )
        matched_prof = bq_profiles[0] if bq_profiles else None
        if matched_prof is None:
            return {
                "status": "error",
                "message": f"No curated 10-K filing found in BigQuery for {norm_ticker} FY{fiscal_year}.",
            }
        gcs_target = matched_prof.unstructured_gcs_uri or matched_prof.unstructured_blob_path
        md_content = adapters.gcs_corpus.fetch_markdown(gcs_target)
        _record_gcp_backend_invocation(
            tool_context,
            tool_name="load_cached_10k_section",
            data_source="gcp",
            services_hit=["bigquery", "gcs"],
        )

        sections = extract_sections_from_combined_markdown(md_content)
        sec_text = sections.get(sec_key, "")
        token_est = estimate_token_count(sec_text)
        entry = {
            "profile_id": matched_prof.profile_id,
            "ticker": matched_prof.ticker,
            "company_name": matched_prof.company_name,
            "fiscal_year": matched_prof.fiscal_year,
            "section_id": sec_key,
            "sec_filing_url": matched_prof.sec_filing_url,
            "unstructured_gcs_uri": matched_prof.unstructured_gcs_uri,
            "token_estimate": token_est,
            "full_text": sec_text,
            "content_source": "gcs",
        }
        _SESSION_10K_CONTEXT_CACHE[cache_id] = entry

    effective_max = max(500, min(int(max_chars), 50000))
    excerpt = str(entry["full_text"])[:effective_max]
    lookup_latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    saved_tokens_delta = 0
    if cache_hit:
        saved_tokens_delta = _CONTEXT_CACHE_STATS.record_hit(
            token_estimate=int(entry["token_estimate"]),
            latency_ms=lookup_latency_ms,
        )
    else:
        _CONTEXT_CACHE_STATS.record_miss(
            token_estimate=int(entry["token_estimate"]),
            latency_ms=lookup_latency_ms,
        )

    _update_context_cache_state(
        tool_context,
        cache_hit=cache_hit,
        saved_tokens_delta=saved_tokens_delta,
        lookup_latency_ms=lookup_latency_ms,
        cache_key=cache_id,
    )

    return {
        "status": "success",
        "cache_key": cache_id,
        "cache_hit": cache_hit,
        "profile_id": entry["profile_id"],
        "ticker": entry["ticker"],
        "company_name": entry["company_name"],
        "fiscal_year": entry["fiscal_year"],
        "section_id": entry["section_id"],
        "token_estimate": entry["token_estimate"],
        "lookup_latency_ms": lookup_latency_ms,
        "sec_filing_url": entry["sec_filing_url"],
        "unstructured_gcs_uri": entry["unstructured_gcs_uri"],
        "content_source": entry.get("content_source", "unknown"),
        "content": excerpt,
        "truncated": len(str(entry["full_text"])) > effective_max,
    }


def get_full_10k_section(cache_key: str) -> dict[str, Any] | None:
    """Return the full parsed 10-K section entry for a cache key, hydrating the L1 cache on miss.

    Unlike `load_cached_10k_section`, this returns the untruncated `full_text`. It is
    used to register complete sections as remote Vertex AI `CachedContent` resources.
    """
    parsed = parse_10k_cache_key(cache_key)
    if parsed is None:
        return None
    ticker, fiscal_year, section_id = parsed
    norm_ticker = normalize_edgar_ticker(ticker)
    norm_key = f"{norm_ticker}_{fiscal_year}_{section_id}"
    entry = _SESSION_10K_CONTEXT_CACHE.get(norm_key)
    if entry is None:
        res = load_cached_10k_section(
            ticker=norm_ticker,
            fiscal_year=fiscal_year,
            section_id=section_id,
            max_chars=500,
        )
        if res.get("status") != "success":
            return None
        entry = _SESSION_10K_CONTEXT_CACHE.get(norm_key)
    return dict(entry) if entry is not None else None


@with_tool_resilience(tool_name="query_cached_10k_context")
def query_cached_10k_context(
    ticker: str,
    fiscal_year: int,
    section_id: str,
    variance_query: str,
    max_passages: int,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Returns the canonical chunks of one company-year 10-K section most relevant to a question.

    Ranks pre-computed BigQuery `filing_chunks` embeddings against `variance_query` using
    vectorized cosine similarity + lexical term boost, and caches each section's chunks and
    embedding matrix in `_SECTION_CHUNKS_CACHE`. This avoids downloading full GCS Markdown or
    re-embedding hundreds of paragraphs on the critical request path and guarantees exact
    `chunk_id` attribution (`chunk_match='exact'`).

    Args:
        ticker: Company ticker symbol (e.g., 'NVDA', 'AAPL').
        fiscal_year: Reporting fiscal year (e.g., 2026).
        section_id: 10-K section to query: 'item_7' (MD&A) or 'item_1a' (Risk Factors).
        variance_query: Focused question or driver to explain (e.g., 'data center revenue growth Blackwell demand').
        max_passages: Number of passages to return (1 to 10; 2-4 is usually enough).

    Returns:
        Dictionary with 'status', 'passages' (each with 'text', 'chunk_id', 'chunk_match'),
        'section_title', 'sec_filing_url', 'period_of_report', 'accession_number',
        'content_source', and 'matched_citations'.
    """
    t0 = time.perf_counter()
    norm_ticker = normalize_edgar_ticker(ticker)
    sec_key = section_id.strip().lower()
    if sec_key not in ("item_1a", "item_7"):
        return {
            "status": "error",
            "message": f"section_id must be 'item_1a' or 'item_7', got {section_id!r}",
        }

    fy_int = int(fiscal_year)
    cache_key = f"{norm_ticker}_{fy_int}_{sec_key}"
    adapters = _gcp_adapters()
    services_hit: list[str] = []

    cached_chunks_entry = _SECTION_CHUNKS_CACHE.get(cache_key)
    cache_hit = cached_chunks_entry is not None

    if cached_chunks_entry is None:
        section_chunks = adapters.bigquery.fetch_section_chunks(
            ticker=norm_ticker,
            fiscal_year=fy_int,
            section_id=sec_key,
            include_embedding=True,
        )
        services_hit.append("bigquery")
        if not section_chunks:
            return {
                "status": "error",
                "message": (
                    f"No indexed 10-K chunks found in BigQuery for "
                    f"{norm_ticker} FY{fy_int} {sec_key}."
                ),
            }
        raw_embs = [c.get("embedding") for c in section_chunks]
        has_precomputed = all(isinstance(e, list) and len(e) > 0 for e in raw_embs)
        if has_precomputed:
            mat = _l2_normalize_rows(
                np.asarray([[float(v) for v in e] for e in raw_embs], dtype=np.float32)
            )
            emb_source = "bigquery_precomputed"
        else:
            embedded = adapters.embed_texts(
                [str(c.get("chunk_text") or "")[:600] for c in section_chunks],
                task_type="RETRIEVAL_DOCUMENT",
            )
            mat = _l2_normalize_rows(np.asarray(embedded, dtype=np.float32))
            emb_source = "vertex_ai_embedded"
            services_hit.append("vertex_ai_embeddings")

        first = section_chunks[0]
        cached_chunks_entry = {
            "chunks": section_chunks,
            "emb_matrix": mat,
            "emb_source": emb_source,
            "profile_id": str(first.get("profile_id") or f"{norm_ticker}_{fy_int}"),
            "ticker": str(first.get("ticker") or norm_ticker),
            "company_name": str(first.get("company_name") or norm_ticker),
            "sec_filing_url": str(first.get("sec_filing_url") or ""),
            "period_of_report": str(first.get("period_of_report") or "") or None,
            "accession_number": str(first.get("accession_number") or "") or None,
            "token_estimate": sum(
                estimate_token_count(str(c.get("chunk_text") or "")) for c in section_chunks
            ),
        }
        _SECTION_CHUNKS_CACHE[cache_key] = cached_chunks_entry

    section_chunks = cached_chunks_entry["chunks"]
    emb_matrix: np.ndarray = cached_chunks_entry["emb_matrix"]

    q_vec = np.asarray(adapters.embed_query(variance_query), dtype=np.float32)
    q_norm = float(np.linalg.norm(q_vec)) or 1.0
    q_vec = q_vec / q_norm
    if "vertex_ai_embeddings" not in services_hit:
        services_hit.append("vertex_ai_embeddings")

    dense_sims: np.ndarray = emb_matrix @ q_vec
    q_terms = set(_WORD_TOKEN_RE.findall(variance_query.lower()))

    scored: list[tuple[float, dict[str, Any]]] = []
    for idx, chunk in enumerate(section_chunks):
        chunk_text = str(chunk.get("chunk_text") or "").strip()
        if not chunk_text:
            continue
        dense_sim = float(dense_sims[idx]) if idx < len(dense_sims) else 0.0
        lex_boost = 0.0
        if q_terms:
            lower_text = chunk_text.lower()
            hits = sum(1 for term in q_terms if term in lower_text)
            lex_boost = 0.35 * (hits / len(q_terms))
        scored.append((dense_sim + lex_boost, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    eff_limit = max(1, min(int(max_passages), 10))
    top_scored = scored[:eff_limit]

    _record_gcp_backend_invocation(
        tool_context,
        tool_name="query_cached_10k_context",
        data_source="gcp",
        services_hit=services_hit,
    )

    matched_chunk_ids: list[str] = []
    matched_citations: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    top_passages: list[str] = []

    for score_val, chunk in top_scored:
        cid = str(chunk["chunk_id"])
        chunk_text = str(chunk.get("chunk_text") or "").strip()
        top_passages.append(chunk_text)
        passages.append(
            {
                "text": chunk_text,
                "chunk_id": cid,
                "chunk_match": "exact",
                "score": round(float(score_val), 4),
            }
        )
        if cid not in matched_chunk_ids:
            matched_chunk_ids.append(cid)
            resolved_sec = _normalize_section_id(
                str(chunk.get("section_id") or sec_key),
                requested_section=sec_key,
            )
            cit = GroundedCitation.from_row(
                {**chunk, "section_id": resolved_sec},
                score=float(score_val),
                excerpt=chunk_text,
                fallback_ticker=norm_ticker,
                fallback_fiscal_year=fy_int,
                fallback_section_id=resolved_sec,
                clamp_score=True,
            )
            matched_citations.append(cit.model_dump(mode="json"))

    if tool_context is not None and hasattr(tool_context, "state") and matched_citations:
        existing_cits = list(tool_context.state.get("retrieved_citations", []))
        tool_context.state["retrieved_citations"] = merge_citations(
            existing_cits, matched_citations
        )

    total_lookup_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    saved_tokens_delta = 0
    if cache_hit:
        saved_tokens_delta = _CONTEXT_CACHE_STATS.record_hit(
            token_estimate=int(cached_chunks_entry["token_estimate"]),
            latency_ms=total_lookup_ms,
        )
    else:
        _CONTEXT_CACHE_STATS.record_miss(
            token_estimate=int(cached_chunks_entry["token_estimate"]),
            latency_ms=total_lookup_ms,
        )
    _update_context_cache_state(
        tool_context,
        cache_hit=cache_hit,
        saved_tokens_delta=saved_tokens_delta,
        lookup_latency_ms=total_lookup_ms,
        cache_key=cache_key,
    )

    return {
        "status": "success",
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "profile_id": cached_chunks_entry["profile_id"],
        "ticker": cached_chunks_entry["ticker"],
        "company_name": cached_chunks_entry["company_name"],
        "fiscal_year": fy_int,
        "period_of_report": cached_chunks_entry["period_of_report"],
        "accession_number": cached_chunks_entry["accession_number"],
        "section_id": sec_key,
        "section_title": SECTION_TITLES[sec_key],
        "token_estimate": cached_chunks_entry["token_estimate"],
        "lookup_latency_ms": total_lookup_ms,
        "sec_filing_url": cached_chunks_entry["sec_filing_url"],
        "content_source": "bigquery_filing_chunks",
        "embedding_source": cached_chunks_entry["emb_source"],
        "passages": passages,
        "explanations": top_passages,
        "matched_chunk_ids": matched_chunk_ids,
        "matched_citations": matched_citations,
    }


# Explicit ADK FunctionTool instances matching project_reference.md naming
MetricFetchTool = FunctionTool(fetch_financial_metrics)
AnalyticsViewQueryTool = FunctionTool(query_analytics_view)
LookerQueryTool = AnalyticsViewQueryTool  # backward-compatible deprecated alias
VarianceCalculationTool = FunctionTool(calculate_period_variances)
HybridSearchTool = FunctionTool(search_10k_chunks_rag)
ContextCacheTool = FunctionTool(load_cached_10k_section)
QueryCached10KTool = FunctionTool(query_cached_10k_context)

__all__ = [
    "AnalyticsViewQueryTool",
    "BigQueryAdapter",
    "ContextCacheTool",
    "DEFAULT_10K_SECTION_CACHE_MAX_ENTRIES",
    "DEFAULT_10K_SECTION_CACHE_TTL_SECONDS",
    "DEFAULT_BIGQUERY_DATASET_ID",
    "DEFAULT_BQ_CHUNK_HEALTH_VIEW",
    "DEFAULT_BQ_FILING_CHUNKS_TABLE",
    "DEFAULT_BQ_FINANCIAL_METRICS_TABLE",
    "DEFAULT_BQ_YOY_VARIANCES_VIEW",
    "DEFAULT_VERTEX_AI_SEARCH_COLLECTION",
    "DEFAULT_VERTEX_AI_SEARCH_DATASTORE_ID",
    "DEFAULT_VERTEX_AI_SEARCH_ENGINE_ID",
    "DEFAULT_VERTEX_AI_SEARCH_LOCATION",
    "GcpAdapterHttpError",
    "GcpEndpointNotReadyError",
    "GcpToolAdapters",
    "GcpToolConfig",
    "GcsCorpusAdapter",
    "HybridSearchTool",
    "LookerQueryTool",
    "MetricFetchTool",
    "QueryCached10KTool",
    "VarianceCalculationTool",
    "VertexAISearchAdapter",
    "_SECTION_CHUNKS_CACHE",
    "_SESSION_10K_CONTEXT_CACHE",
    "_build_variance_reports_from_profiles",
    "_compute_single_metric_variance",
    "_default_section_title",
    "_execute_live_gcp_hybrid_rag",
    "_gcp_adapters",
    "_normalize_section_id",
    "_numeric_citation_for_profile",
    "_parse_ticker_list",
    "_profile_to_metric_record",
    "_record_gcp_backend_invocation",
    "_xbrl_concept_for_metric",
    "calculate_period_variances",
    "configure_gcp_tool_adapters",
    "fetch_financial_metrics",
    "get_active_gcp_tool_adapters",
    "get_context_cache_runtime_stats",
    "get_full_10k_section",
    "get_gcp_adapters_status_metadata",
    "load_cached_10k_section",
    "parse_10k_cache_key",
    "query_analytics_view",
    "query_cached_10k_context",
    "query_looker_financial_explore",
    "reset_gcp_tool_adapters",
    "search_10k_chunks_rag",
]
