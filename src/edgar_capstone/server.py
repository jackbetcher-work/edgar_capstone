"""FastAPI Cloud Run service entrypoint for the SEC EDGAR Natural Language Analyst."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from edgar_capstone.agents.chat_engine import (
    list_curated_companies,
    run_conversational_rag_turn,
)
from edgar_capstone.agents.root_orchestrator import (
    RootOrchestrator,
    classify_query_intent,
)
from edgar_capstone.agents.tools import (
    _get_dataset_dir,
    _load_all_profiles_cached,
    calculate_period_variances,
    fetch_financial_metrics,
    load_cached_10k_section,
    query_looker_financial_explore,
    search_10k_chunks_rag,
)
from edgar_capstone.security.secrets import (
    DEFAULT_PROJECT_ID,
    MANAGED_SECRETS,
    SecretManagerClient,
)

logger = logging.getLogger(__name__)

_FEEDBACK_AUDIT_LOG: list[dict[str, Any]] = []

app = FastAPI(
    title="SEC EDGAR Natural Language Analyst API",
    version="0.1.0",
    description=(
        "Cloud Run FastAPI backend for deterministic SEC 10-K financial variance "
        "calculations, metadata-filtered Hybrid RAG search, and ADK RootOrchestrator routing."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VarianceRequest(BaseModel):
    tickers: str = Field(..., description="Comma-separated tickers, e.g. 'NVDA,AAPL'")
    start_year: int = Field(2024, ge=2015, le=2035)
    end_year: int = Field(2026, ge=2015, le=2035)


class RagSearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    tickers: str = Field("ALL")
    gics_sector: str = Field("ALL")
    section_id: str = Field("ALL")
    start_year: int = Field(2024, ge=2015, le=2035)
    end_year: int = Field(2026, ge=2015, le=2035)
    top_k: int = Field(5, ge=1, le=25)


class IntentClassifyRequest(BaseModel):
    query: str = Field(..., min_length=2)
    tickers: list[str] | None = None


@app.get("/health")
@app.get("/healthz", include_in_schema=False)
def health() -> dict[str, Any]:
    """Cloud Run external health check and internal probe endpoint."""
    return {
        "status": "ok",
        "service": os.environ.get("K_SERVICE", "sec-edgar-analyst"),
        "project_id": os.environ.get("GCP_PROJECT_ID", DEFAULT_PROJECT_ID),
        "root_agent": RootOrchestrator.name,
        "sub_agents": [sa.name for sa in RootOrchestrator.sub_agents],
    }


@app.get("/api/v1/secrets/status")
def secrets_status() -> dict[str, Any]:
    """Report Secret Manager binding metadata without ever exposing secret values."""
    client = SecretManagerClient()
    bindings = []
    for spec in MANAGED_SECRETS:
        is_mounted_in_env = bool(os.environ.get(spec.env_var))
        bindings.append(
            {
                "env_var": spec.env_var,
                "secret_id": spec.secret_id,
                "resource_name": spec.resource_name(client.project_id),
                "cloud_run_mount": spec.cloud_run_mount_spec(),
                "required": spec.required,
                "mounted_in_env": is_mounted_in_env,
                "value": "***REDACTED***",
            }
        )
    return {
        "status": "ok",
        "project_id": client.project_id,
        "secret_manager_enabled": True,
        "secrets": bindings,
    }


@app.post("/api/v1/variances")
def compute_variances(req: VarianceRequest) -> dict[str, Any]:
    """Compute deterministic period-over-period financial variances."""
    res = calculate_period_variances(
        tickers=req.tickers,
        start_year=req.start_year,
        end_year=req.end_year,
    )
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message", "Invalid request"))
    return res


@app.post("/api/v1/search")
def rag_search(req: RagSearchRequest) -> dict[str, Any]:
    """Run metadata-filtered Hybrid RAG search over SEC 10-K filing chunks."""
    return search_10k_chunks_rag(
        query=req.query,
        tickers=req.tickers,
        gics_sector=req.gics_sector,
        section_id=req.section_id,
        start_year=req.start_year,
        end_year=req.end_year,
        top_k=req.top_k,
    )


@app.post("/api/v1/intent")
def classify_intent(req: IntentClassifyRequest) -> dict[str, Any]:
    """Deterministically classify an analyst query into a RootOrchestrator routing intent."""
    intent = classify_query_intent(req.query, tickers=req.tickers)
    return {
        "status": "success",
        "query": req.query,
        "intent": intent,
        "orchestrator": RootOrchestrator.name,
    }


@app.get("/api/v1/metrics")
def get_metrics(
    tickers: str = "NVDA,AAPL",
    start_year: int = 2024,
    end_year: int = 2026,
    gics_sector: str = "ALL",
) -> dict[str, Any]:
    """Retrieve structured SEC 10-K financial metrics."""
    return fetch_financial_metrics(
        tickers=tickers,
        start_year=start_year,
        end_year=end_year,
        gics_sector=gics_sector,
    )


@app.get("/api/v1/explore/{explore_name}")
def get_looker_explore(
    explore_name: str,
    tickers: str = "ALL",
    gics_sector: str = "ALL",
    start_year: int = 2024,
    end_year: int = 2026,
) -> dict[str, Any]:
    """Query Looker BI semantic explores (yoy_variances, sector_benchmarks, peer_comparison, chunk_health)."""
    return query_looker_financial_explore(
        explore_name=explore_name,
        tickers=tickers,
        gics_sector=gics_sector,
        start_year=start_year,
        end_year=end_year,
    )


class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=2)
    tickers: str = Field("ALL")
    gics_sector: str = Field("ALL")
    section_id: str = Field("ALL")
    start_year: int = Field(2024, ge=2015, le=2035)
    end_year: int = Field(2026, ge=2015, le=2035)
    top_k: int = Field(8, ge=1, le=25)


class SectionLoadRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    fiscal_year: int = Field(..., ge=2015, le=2035)
    section_id: Literal["item_1a", "item_7"] = Field("item_7")
    max_chars: int = Field(32000, ge=500, le=50000)


class FeedbackRequest(BaseModel):
    message_id: str = Field(..., min_length=1)
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=1000)
    query: str = Field(default="", max_length=1000)


@app.get("/api/v1/companies")
def get_companies() -> dict[str, Any]:
    """List all curated S&P 500 issuers, GICS sectors, and available fiscal years."""
    return list_curated_companies()


@app.post("/api/v1/chat")
def conversational_rag_chat(req: ChatQueryRequest) -> dict[str, Any]:
    """Execute a source-grounded conversational Hybrid RAG query with ReAct trace & fact-check verification."""
    return run_conversational_rag_turn(
        query=req.query,
        tickers=req.tickers,
        gics_sector=req.gics_sector,
        section_id=req.section_id,
        start_year=req.start_year,
        end_year=req.end_year,
        top_k=req.top_k,
    )


def _enrich_section_response(
    ticker: str,
    fiscal_year: int,
    section_id: str,
    max_chars: int,
) -> dict[str, Any]:
    res = load_cached_10k_section(
        ticker=ticker,
        fiscal_year=fiscal_year,
        section_id=section_id,
        max_chars=max_chars,
    )
    if res.get("status") == "error":
        raise HTTPException(status_code=404, detail=res.get("message", "Section not found"))

    dataset_dir = _get_dataset_dir()
    profiles = _load_all_profiles_cached(str(dataset_dir))
    norm_t = ticker.strip().upper().replace(".", "-")
    matched_prof = next(
        (
            p
            for p in profiles
            if p.ticker.upper().replace(".", "-") == norm_t and p.fiscal_year == fiscal_year
        ),
        None,
    )
    if matched_prof is not None:
        res["cik"] = matched_prof.cik
        res["accession_number"] = matched_prof.accession_number
        res["filing_date"] = matched_prof.filing_date
        res["period_of_report"] = matched_prof.period_of_report
        res["gics_sector"] = matched_prof.gics_sector
        res["gics_sub_industry"] = matched_prof.gics_sub_industry
    return res


@app.get("/api/v1/section")
def get_cached_section(
    ticker: str = "NVDA",
    fiscal_year: int = 2026,
    section_id: Literal["item_1a", "item_7"] = "item_7",
    max_chars: int = 32000,
) -> dict[str, Any]:
    """Load a cached 10-K filing section (Item 1A Risk Factors or Item 7 MD&A) for the split-pane source viewer."""
    return _enrich_section_response(
        ticker=ticker,
        fiscal_year=fiscal_year,
        section_id=section_id,
        max_chars=max_chars,
    )


@app.post("/api/v1/section")
def post_cached_section(req: SectionLoadRequest) -> dict[str, Any]:
    """Load a cached 10-K filing section via POST payload."""
    return _enrich_section_response(
        ticker=req.ticker,
        fiscal_year=req.fiscal_year,
        section_id=req.section_id,
        max_chars=req.max_chars,
    )


@app.post("/api/v1/feedback")
def submit_analyst_feedback(req: FeedbackRequest) -> dict[str, Any]:
    """Record analyst thumbs-up/down audit feedback and optional comment."""
    entry = {
        "message_id": req.message_id,
        "rating": req.rating,
        "comment": req.comment.strip(),
        "query": req.query.strip(),
        "recorded_at_epoch": round(time.time(), 3),
    }
    _FEEDBACK_AUDIT_LOG.append(entry)
    logger.info(
        "Recorded analyst feedback message_id=%s rating=%s",
        req.message_id,
        req.rating,
    )
    return {
        "status": "success",
        "recorded": entry,
        "total_feedback_count": len(_FEEDBACK_AUDIT_LOG),
    }
