"""ADK tools: MetricFetchTool, LookerQueryTool, VarianceCalculationTool, HybridSearchTool, and ContextCacheTool."""

from __future__ import annotations

from collections import defaultdict
import functools
import json
import logging
import math
import os
from pathlib import Path
import re
from typing import Any

from google.adk.tools import FunctionTool, ToolContext
import numpy as np

from edgar_capstone.agents.resilience import with_tool_resilience
from edgar_capstone.agents.schemas import (
    AnnualStepVariance,
    CompanyVarianceReport,
    GroundedCitation,
    MetricRecord,
    MetricVariance,
)
from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.dataset.pipeline import DEFAULT_OUTPUT_DIR
from edgar_capstone.vector_db.chunker import (
    estimate_token_count,
    extract_sections_from_combined_markdown,
)
from edgar_capstone.vector_db.embedder import generate_deterministic_embedding

logger = logging.getLogger(__name__)

_WORD_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

# In-memory process cache for context caching simulation & fast corpus access
_SESSION_10K_CONTEXT_CACHE: dict[str, dict[str, Any]] = {}


def _get_dataset_dir() -> Path:
    """Resolve the active curated dataset directory."""
    env_dir = os.environ.get("EDGAR_DATASET_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return DEFAULT_OUTPUT_DIR.resolve()


@functools.lru_cache(maxsize=4)
def _load_all_profiles_cached(dataset_dir_str: str) -> list[FinancialDataProfile]:
    """Load and cache all structured `FinancialDataProfile` records from disk."""
    dataset_dir = Path(dataset_dir_str)
    manifest_path = dataset_dir / "manifests" / "bigquery_financials.jsonl"
    profiles: list[FinancialDataProfile] = []

    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            profiles.append(FinancialDataProfile.model_validate_json(line))
    else:
        struct_root = dataset_dir / "structured"
        if struct_root.exists():
            for json_file in sorted(struct_root.rglob("*_financials.json")):
                profiles.append(
                    FinancialDataProfile.model_validate_json(
                        json_file.read_text(encoding="utf-8")
                    )
                )

    profiles.sort(key=lambda p: (p.ticker, p.fiscal_year))
    return profiles


@functools.lru_cache(maxsize=2)
def _load_all_chunks_cached(dataset_dir_str: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load metadata-tagged chunks and their 768-d embedding matrix for fast hybrid search."""
    dataset_dir = Path(dataset_dir_str)
    chunks_path = dataset_dir / "manifests" / "bigquery_filing_chunks.jsonl"
    rows: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []

    if chunks_path.exists():
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            emb = row.get("embedding") or []
            if len(emb) == 768:
                rows.append(row)
                embeddings.append(emb)

    if not embeddings:
        return [], np.zeros((0, 768), dtype=np.float32)

    matrix = np.asarray(embeddings, dtype=np.float32)
    return rows, matrix


def _parse_ticker_list(tickers_csv: str) -> set[str]:
    """Parse comma- or space-separated ticker string into normalized uppercase symbols."""
    if not tickers_csv or tickers_csv.strip().upper() in ("*", "ALL", ""):
        return set()
    parts = re.split(r"[,;\s]+", tickers_csv.strip())
    return {p.strip().upper().replace(".", "-") for p in parts if p.strip()}


def _profile_to_metric_record(p: FinancialDataProfile) -> MetricRecord:
    """Convert a `FinancialDataProfile` into a `MetricRecord` with computed margins."""
    op_margin = (
        round((p.operating_income / abs(p.revenue)) * 100.0, 2)
        if p.revenue and p.operating_income is not None
        else None
    )
    net_margin = (
        round((p.net_income / abs(p.revenue)) * 100.0, 2)
        if p.revenue and p.net_income is not None
        else None
    )
    return MetricRecord(
        profile_id=p.profile_id,
        ticker=p.ticker,
        cik=p.cik,
        company_name=p.company_name,
        gics_sector=p.gics_sector,
        gics_sub_industry=p.gics_sub_industry,
        fiscal_year=p.fiscal_year,
        period_of_report=p.period_of_report,
        filing_date=p.filing_date,
        accession_number=p.accession_number,
        currency=p.currency,
        revenue=p.revenue,
        operating_income=p.operating_income,
        net_income=p.net_income,
        operating_margin_pct=op_margin,
        net_margin_pct=net_margin,
        sec_filing_url=p.sec_filing_url,
        unstructured_gcs_uri=p.unstructured_gcs_uri,
        structured_gcs_uri=p.structured_gcs_uri,
        provenance=p.metric_provenance.model_dump(mode="json"),
    )


def _compute_single_metric_variance(
    metric_name: str,
    start_year: int,
    end_year: int,
    start_val: float | None,
    end_val: float | None,
) -> MetricVariance:
    """Deterministically compute USD delta, percentage delta, and CAGR for a metric."""
    if start_val is None or end_val is None:
        return MetricVariance(
            metric_name=metric_name,  # type: ignore[arg-type]
            start_year=start_year,
            end_year=end_year,
            start_value=start_val,
            end_value=end_val,
            direction="unavailable",
        )

    abs_delta = round(end_val - start_val, 2)
    pct_delta = (
        round(((end_val - start_val) / abs(start_val)) * 100.0, 2)
        if start_val != 0
        else None
    )
    years_span = max(0, end_year - start_year)
    cagr_pct: float | None = None
    if years_span >= 1 and start_val > 0 and end_val > 0:
        cagr_pct = round((((end_val / start_val) ** (1.0 / years_span)) - 1.0) * 100.0, 2)

    if abs_delta > 0:
        direction = "increase"
    elif abs_delta < 0:
        direction = "decrease"
    else:
        direction = "flat"

    return MetricVariance(
        metric_name=metric_name,  # type: ignore[arg-type]
        start_year=start_year,
        end_year=end_year,
        start_value=start_val,
        end_value=end_val,
        absolute_delta=abs_delta,
        percentage_delta=pct_delta,
        cagr_pct=cagr_pct,
        direction=direction,  # type: ignore[arg-type]
    )


@with_tool_resilience(tool_name="fetch_financial_metrics")
def fetch_financial_metrics(
    tickers: str,
    start_year: int,
    end_year: int,
    gics_sector: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Retrieves raw structured SEC 10-K financial metrics (Revenue, Operating Income, Net Income, margins, and filing URLs).

    Args:
        tickers: Comma-separated ticker symbols (e.g., 'NVDA,AAPL,MSFT') or 'ALL' for all companies.
        start_year: Earliest fiscal year to retrieve (inclusive, e.g., 2024).
        end_year: Latest fiscal year to retrieve (inclusive, e.g., 2026).
        gics_sector: Optional GICS Sector filter (e.g., 'Information Technology') or 'ALL' for any sector.

    Returns:
        Dictionary with 'status', 'record_count', and 'metrics' list of structured financial records.
    """
    dataset_dir = _get_dataset_dir()
    profiles = _load_all_profiles_cached(str(dataset_dir))
    ticker_filter = _parse_ticker_list(tickers)
    sector_norm = (
        gics_sector.strip().lower()
        if gics_sector and gics_sector.strip().upper() not in ("ALL", "*", "")
        else None
    )

    records: list[dict[str, Any]] = []
    for p in profiles:
        if ticker_filter and p.ticker.upper().replace(".", "-") not in ticker_filter:
            continue
        if sector_norm and p.gics_sector.lower() != sector_norm:
            continue
        if p.fiscal_year < start_year or p.fiscal_year > end_year:
            continue
        records.append(_profile_to_metric_record(p).model_dump(mode="json"))

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


@with_tool_resilience(tool_name="query_looker_financial_explore")
def query_looker_financial_explore(
    explore_name: str,
    tickers: str,
    gics_sector: str,
    start_year: int,
    end_year: int,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Queries the Looker BI semantic explore for YoY variances, sector benchmarks, peer comparisons, or chunk health.

    Args:
        explore_name: Looker explore to query: 'yoy_variances', 'sector_benchmarks', 'peer_comparison', or 'chunk_health'.
        tickers: Comma-separated ticker symbols (e.g., 'NVDA,AMD,AVGO') or 'ALL'.
        gics_sector: GICS Sector filter (e.g., 'Information Technology', 'Financials') or 'ALL'.
        start_year: Earliest fiscal year to include (e.g., 2024).
        end_year: Latest fiscal year to include (e.g., 2026).

    Returns:
        Dictionary with 'status', 'explore_name', 'row_count', and 'rows' from the requested BI explore.
    """
    dataset_dir = _get_dataset_dir()
    explore_key = explore_name.strip().lower()
    ticker_filter = _parse_ticker_list(tickers)
    sector_norm = (
        gics_sector.strip().lower()
        if gics_sector and gics_sector.strip().upper() not in ("ALL", "*", "")
        else None
    )

    if explore_key == "chunk_health":
        health_path = dataset_dir / "manifests" / "chunk_health_report.json"
        if not health_path.exists():
            return {
                "status": "error",
                "explore_name": explore_key,
                "message": f"Chunk health report not found at {health_path}",
            }
        health_data = json.loads(health_path.read_text(encoding="utf-8"))
        return {
            "status": "success",
            "explore_name": "chunk_health",
            "health_status": health_data.get("health_status"),
            "health_score": health_data.get("health_score"),
            "total_chunks": health_data.get("total_chunks"),
            "embedding_coverage_ratio": health_data.get("embedding_coverage_ratio"),
            "full_section_coverage_ratio": health_data.get("full_section_coverage_ratio"),
            "by_section": health_data.get("by_section", {}),
            "by_sector": health_data.get("by_sector", {}),
        }

    profiles = _load_all_profiles_cached(str(dataset_dir))
    by_ticker: dict[str, list[FinancialDataProfile]] = defaultdict(list)
    for p in profiles:
        if ticker_filter and p.ticker.upper().replace(".", "-") not in ticker_filter:
            continue
        if sector_norm and p.gics_sector.lower() != sector_norm:
            continue
        by_ticker[p.ticker].append(p)

    rows: list[dict[str, Any]] = []

    if explore_key in ("yoy_variances", "peer_comparison"):
        for ticker, t_profiles in sorted(by_ticker.items()):
            t_profiles.sort(key=lambda x: x.fiscal_year)
            prev_prof: FinancialDataProfile | None = None
            for prof in t_profiles:
                if prof.fiscal_year >= start_year and prof.fiscal_year <= end_year:
                    rev_delta = rev_pct = op_delta = op_pct = ni_delta = ni_pct = None
                    if prev_prof and prev_prof.fiscal_year == prof.fiscal_year - 1:
                        if prof.revenue is not None and prev_prof.revenue is not None:
                            rev_delta = round(prof.revenue - prev_prof.revenue, 2)
                            if prev_prof.revenue != 0:
                                rev_pct = round(
                                    (rev_delta / abs(prev_prof.revenue)) * 100.0, 2
                                )
                        if (
                            prof.operating_income is not None
                            and prev_prof.operating_income is not None
                        ):
                            op_delta = round(
                                prof.operating_income - prev_prof.operating_income, 2
                            )
                            if prev_prof.operating_income != 0:
                                op_pct = round(
                                    (op_delta / abs(prev_prof.operating_income)) * 100.0,
                                    2,
                                )
                        if prof.net_income is not None and prev_prof.net_income is not None:
                            ni_delta = round(prof.net_income - prev_prof.net_income, 2)
                            if prev_prof.net_income != 0:
                                ni_pct = round(
                                    (ni_delta / abs(prev_prof.net_income)) * 100.0, 2
                                )

                    rec = _profile_to_metric_record(prof)
                    rows.append(
                        {
                            "profile_id": prof.profile_id,
                            "ticker": prof.ticker,
                            "company_name": prof.company_name,
                            "gics_sector": prof.gics_sector,
                            "fiscal_year": prof.fiscal_year,
                            "prev_fiscal_year": prev_prof.fiscal_year if prev_prof else None,
                            "revenue": prof.revenue,
                            "revenue_yoy_delta": rev_delta,
                            "revenue_yoy_pct": rev_pct,
                            "operating_income": prof.operating_income,
                            "operating_income_yoy_delta": op_delta,
                            "operating_income_yoy_pct": op_pct,
                            "net_income": prof.net_income,
                            "net_income_yoy_delta": ni_delta,
                            "net_income_yoy_pct": ni_pct,
                            "operating_margin_pct": rec.operating_margin_pct,
                            "net_margin_pct": rec.net_margin_pct,
                            "sec_filing_url": prof.sec_filing_url,
                        }
                    )
                prev_prof = prof

    elif explore_key == "sector_benchmarks":
        sector_year_groups: dict[tuple[str, int], list[FinancialDataProfile]] = defaultdict(list)
        for t_profiles in by_ticker.values():
            for prof in t_profiles:
                if start_year <= prof.fiscal_year <= end_year:
                    sector_year_groups[(prof.gics_sector, prof.fiscal_year)].append(prof)

        for (sector_name, fy), group in sorted(sector_year_groups.items()):
            revs = [p.revenue for p in group if p.revenue is not None]
            ops = [p.operating_income for p in group if p.operating_income is not None]
            nis = [p.net_income for p in group if p.net_income is not None]
            rows.append(
                {
                    "gics_sector": sector_name,
                    "fiscal_year": fy,
                    "company_count": len(group),
                    "tickers": sorted(p.ticker for p in group),
                    "total_sector_revenue": round(sum(revs), 2) if revs else None,
                    "avg_company_revenue": round(sum(revs) / len(revs), 2) if revs else None,
                    "total_sector_operating_income": round(sum(ops), 2) if ops else None,
                    "total_sector_net_income": round(sum(nis), 2) if nis else None,
                }
            )
    else:
        return {
            "status": "error",
            "explore_name": explore_name,
            "message": (
                f"Unsupported explore_name '{explore_name}'. "
                "Supported: 'yoy_variances', 'sector_benchmarks', 'peer_comparison', 'chunk_health'."
            ),
        }

    result = {
        "status": "success",
        "explore_name": explore_key,
        "row_count": len(rows),
        "rows": rows,
    }
    if tool_context is not None and hasattr(tool_context, "state"):
        history = list(tool_context.state.get("looker_query_results", []))
        history.append(result)
        tool_context.state["looker_query_results"] = history

    return result


@with_tool_resilience(tool_name="calculate_period_variances")
def calculate_period_variances(
    tickers: str,
    start_year: int,
    end_year: int,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Performs deterministic period-over-period variance calculations for Revenue, Operating Income, and Net Income.

    Separates arithmetic calculation from LLM narrative synthesis. Always call this tool first
    before explaining financial changes so narrative claims match deterministic numbers 100%.

    Args:
        tickers: Comma-separated ticker symbols (e.g., 'NVDA' or 'NVDA,AAPL').
        start_year: Starting fiscal year for the variance calculation (e.g., 2024).
        end_year: Ending fiscal year for the variance calculation (e.g., 2026).

    Returns:
        Dictionary containing deterministic multi-period and YoY variance reports per company.
    """
    dataset_dir = _get_dataset_dir()
    profiles = _load_all_profiles_cached(str(dataset_dir))
    ticker_filter = _parse_ticker_list(tickers)
    if not ticker_filter:
        return {
            "status": "error",
            "message": "Please specify at least one ticker symbol (e.g., 'NVDA' or 'NVDA,AAPL').",
        }

    by_ticker: dict[str, dict[int, FinancialDataProfile]] = defaultdict(dict)
    for p in profiles:
        norm_t = p.ticker.upper().replace(".", "-")
        if norm_t in ticker_filter:
            by_ticker[p.ticker][p.fiscal_year] = p

    reports: list[dict[str, Any]] = []
    missing_tickers = sorted(
        t for t in ticker_filter if t not in {k.upper().replace(".", "-") for k in by_ticker}
    )

    for ticker, fy_map in sorted(by_ticker.items()):
        avail_years = sorted(fy for fy in fy_map if start_year <= fy <= end_year)
        if not avail_years:
            continue

        eff_start = avail_years[0]
        eff_end = avail_years[-1]
        start_prof = fy_map[eff_start]
        end_prof = fy_map[eff_end]

        rev_var = _compute_single_metric_variance(
            "revenue", eff_start, eff_end, start_prof.revenue, end_prof.revenue
        )
        op_var = _compute_single_metric_variance(
            "operating_income",
            eff_start,
            eff_end,
            start_prof.operating_income,
            end_prof.operating_income,
        )
        ni_var = _compute_single_metric_variance(
            "net_income", eff_start, eff_end, start_prof.net_income, end_prof.net_income
        )

        start_rec = _profile_to_metric_record(start_prof)
        end_rec = _profile_to_metric_record(end_prof)

        op_margin_bps = (
            round((end_rec.operating_margin_pct - start_rec.operating_margin_pct) * 100.0, 1)
            if start_rec.operating_margin_pct is not None
            and end_rec.operating_margin_pct is not None
            else None
        )
        net_margin_bps = (
            round((end_rec.net_margin_pct - start_rec.net_margin_pct) * 100.0, 1)
            if start_rec.net_margin_pct is not None and end_rec.net_margin_pct is not None
            else None
        )

        annual_series: list[AnnualStepVariance] = []
        source_urls: dict[int, str] = {}
        for fy in avail_years:
            curr = fy_map[fy]
            source_urls[fy] = curr.sec_filing_url
            prev = fy_map.get(fy - 1)
            rev_yoy_d = rev_yoy_p = op_yoy_d = op_yoy_p = ni_yoy_d = ni_yoy_p = None
            if prev is not None:
                if curr.revenue is not None and prev.revenue is not None:
                    rev_yoy_d = round(curr.revenue - prev.revenue, 2)
                    if prev.revenue != 0:
                        rev_yoy_p = round((rev_yoy_d / abs(prev.revenue)) * 100.0, 2)
                if curr.operating_income is not None and prev.operating_income is not None:
                    op_yoy_d = round(curr.operating_income - prev.operating_income, 2)
                    if prev.operating_income != 0:
                        op_yoy_p = round((op_yoy_d / abs(prev.operating_income)) * 100.0, 2)
                if curr.net_income is not None and prev.net_income is not None:
                    ni_yoy_d = round(curr.net_income - prev.net_income, 2)
                    if prev.net_income != 0:
                        ni_yoy_p = round((ni_yoy_d / abs(prev.net_income)) * 100.0, 2)

            annual_series.append(
                AnnualStepVariance(
                    fiscal_year=fy,
                    prev_fiscal_year=prev.fiscal_year if prev else None,
                    revenue=curr.revenue,
                    revenue_yoy_delta=rev_yoy_d,
                    revenue_yoy_pct=rev_yoy_p,
                    operating_income=curr.operating_income,
                    operating_income_yoy_delta=op_yoy_d,
                    operating_income_yoy_pct=op_yoy_p,
                    net_income=curr.net_income,
                    net_income_yoy_delta=ni_yoy_d,
                    net_income_yoy_pct=ni_yoy_p,
                    sec_filing_url=curr.sec_filing_url,
                )
            )

        report = CompanyVarianceReport(
            ticker=ticker,
            company_name=end_prof.company_name,
            cik=end_prof.cik,
            gics_sector=end_prof.gics_sector,
            gics_sub_industry=end_prof.gics_sub_industry,
            start_year=eff_start,
            end_year=eff_end,
            currency=end_prof.currency,
            revenue_variance=rev_var,
            operating_income_variance=op_var,
            net_income_variance=ni_var,
            operating_margin_bps_delta=op_margin_bps,
            net_margin_bps_delta=net_margin_bps,
            annual_series=annual_series,
            source_urls=source_urls,
        )
        reports.append(report.model_dump(mode="json"))

    if tool_context is not None and hasattr(tool_context, "state"):
        calc_state = dict(tool_context.state.get("calculated_variances", {}))
        for rep in reports:
            calc_state[f"{rep['ticker']}_{rep['start_year']}_{rep['end_year']}"] = rep
        tool_context.state["calculated_variances"] = calc_state

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
    dataset_dir = _get_dataset_dir()
    rows, matrix = _load_all_chunks_cached(str(dataset_dir))
    if not rows:
        return {
            "status": "error",
            "message": "Vector database chunk manifest is empty or not yet generated.",
        }

    ticker_filter = _parse_ticker_list(tickers)
    sector_norm = (
        gics_sector.strip().lower()
        if gics_sector and gics_sector.strip().upper() not in ("ALL", "*", "")
        else None
    )
    sec_norm = (
        section_id.strip().lower()
        if section_id and section_id.strip().upper() not in ("ALL", "*", "")
        else None
    )
    effective_top_k = max(1, min(int(top_k), 25))

    candidate_indices: list[int] = []
    for idx, row in enumerate(rows):
        if ticker_filter and str(row.get("ticker", "")).upper().replace(".", "-") not in ticker_filter:
            continue
        if sector_norm and str(row.get("gics_sector", "")).lower() != sector_norm:
            continue
        if sec_norm and str(row.get("section_id", "")).lower() != sec_norm:
            continue
        fy = int(row.get("fiscal_year", 0))
        if fy < start_year or fy > end_year:
            continue
        candidate_indices.append(idx)

    if not candidate_indices:
        return {
            "status": "success",
            "citation_count": 0,
            "citations": [],
            "message": "No 10-K chunks matched the requested metadata filters.",
        }

    q_vec = np.asarray(generate_deterministic_embedding(query), dtype=np.float32)
    cand_matrix = matrix[candidate_indices]
    dense_scores = cand_matrix @ q_vec

    query_terms = set(_WORD_TOKEN_RE.findall(query.lower()))
    scored: list[tuple[float, int]] = []
    for pos, row_idx in enumerate(candidate_indices):
        row = rows[row_idx]
        text_lower = str(row.get("chunk_text", "")).lower()
        lexical_boost = 0.0
        if query_terms:
            matched_terms = sum(1 for term in query_terms if term in text_lower)
            lexical_boost = 0.35 * (matched_terms / len(query_terms))
        hybrid_score = float(dense_scores[pos]) + lexical_boost
        scored.append((hybrid_score, row_idx))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_hits = scored[:effective_top_k]

    citations: list[dict[str, Any]] = []
    for score, row_idx in top_hits:
        row = rows[row_idx]
        cit = GroundedCitation(
            chunk_id=str(row["chunk_id"]),
            profile_id=str(row["profile_id"]),
            ticker=str(row["ticker"]),
            company_name=str(row["company_name"]),
            gics_sector=str(row.get("gics_sector") or "Unknown"),
            fiscal_year=int(row["fiscal_year"]),
            section_id=row["section_id"],
            section_title=str(row.get("section_title") or row["section_id"]),
            excerpt=str(row["chunk_text"]),
            sec_filing_url=str(row.get("sec_filing_url") or ""),
            unstructured_gcs_uri=str(row.get("unstructured_gcs_uri") or ""),
            score=round(score, 4),
        )
        citations.append(cit.model_dump(mode="json"))

    if tool_context is not None and hasattr(tool_context, "state"):
        existing_cits = list(tool_context.state.get("retrieved_citations", []))
        existing_cits.extend(citations)
        tool_context.state["retrieved_citations"] = existing_cits

    return {
        "status": "success",
        "citation_count": len(citations),
        "candidates_searched": len(candidate_indices),
        "citations": citations,
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

    Tracks Vertex AI Context Cache hits/misses and estimated cached-token cost discounts.

    Args:
        ticker: Company ticker symbol (e.g., 'NVDA', 'AAPL').
        fiscal_year: Reporting fiscal year (e.g., 2026).
        section_id: Section to load: 'item_1a' (Risk Factors) or 'item_7' (MD&A).
        max_chars: Maximum characters of section text to return in the tool response (e.g., 12000).

    Returns:
        Dictionary with 'status', 'cache_hit', 'token_estimate', 'sec_filing_url', and grounded Markdown 'content'.
    """
    dataset_dir = _get_dataset_dir()
    norm_ticker = ticker.strip().upper().replace(".", "-")
    sec_key = section_id.strip().lower()
    if sec_key not in ("item_1a", "item_7"):
        return {
            "status": "error",
            "message": f"section_id must be 'item_1a' or 'item_7', got {section_id!r}",
        }

    cache_id = f"{norm_ticker}_{fiscal_year}_{sec_key}"
    cache_hit = cache_id in _SESSION_10K_CONTEXT_CACHE

    if not cache_hit:
        profiles = _load_all_profiles_cached(str(dataset_dir))
        matched_prof = next(
            (
                p
                for p in profiles
                if p.ticker.upper().replace(".", "-") == norm_ticker
                and p.fiscal_year == fiscal_year
            ),
            None,
        )
        if matched_prof is None:
            return {
                "status": "error",
                "message": f"No curated 10-K filing found for {norm_ticker} FY{fiscal_year}.",
            }
        md_path = dataset_dir / matched_prof.unstructured_blob_path
        if not md_path.exists():
            return {
                "status": "error",
                "message": f"Unstructured 10-K file missing at {md_path}.",
            }
        sections = extract_sections_from_combined_markdown(
            md_path.read_text(encoding="utf-8")
        )
        sec_text = sections.get(sec_key, "")
        token_est = estimate_token_count(sec_text)
        _SESSION_10K_CONTEXT_CACHE[cache_id] = {
            "profile_id": matched_prof.profile_id,
            "ticker": matched_prof.ticker,
            "company_name": matched_prof.company_name,
            "fiscal_year": matched_prof.fiscal_year,
            "section_id": sec_key,
            "sec_filing_url": matched_prof.sec_filing_url,
            "unstructured_gcs_uri": matched_prof.unstructured_gcs_uri,
            "token_estimate": token_est,
            "full_text": sec_text,
        }

    entry = _SESSION_10K_CONTEXT_CACHE[cache_id]
    effective_max = max(500, min(int(max_chars), 50000))
    excerpt = str(entry["full_text"])[:effective_max]

    if tool_context is not None and hasattr(tool_context, "state"):
        stats = dict(
            tool_context.state.get(
                "context_cache_stats",
                {"hits": 0, "misses": 0, "cached_tokens_saved": 0},
            )
        )
        if cache_hit:
            stats["hits"] = int(stats.get("hits", 0)) + 1
            stats["cached_tokens_saved"] = int(stats.get("cached_tokens_saved", 0)) + int(
                entry["token_estimate"] * 0.75
            )
        else:
            stats["misses"] = int(stats.get("misses", 0)) + 1
        tool_context.state["context_cache_stats"] = stats

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
        "sec_filing_url": entry["sec_filing_url"],
        "unstructured_gcs_uri": entry["unstructured_gcs_uri"],
        "content": excerpt,
        "truncated": len(str(entry["full_text"])) > effective_max,
    }


# Explicit ADK FunctionTool instances matching project_reference.md naming
MetricFetchTool = FunctionTool(fetch_financial_metrics)
LookerQueryTool = FunctionTool(query_looker_financial_explore)
VarianceCalculationTool = FunctionTool(calculate_period_variances)
HybridSearchTool = FunctionTool(search_10k_chunks_rag)
ContextCacheTool = FunctionTool(load_cached_10k_section)
