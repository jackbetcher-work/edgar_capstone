"""FastAPI Cloud Run service entrypoint for the SEC EDGAR Natural Language Analyst."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from edgar_capstone.agents.root_orchestrator import (
    RootOrchestrator,
    classify_query_intent,
)
from edgar_capstone.agents.tools import (
    calculate_period_variances,
    fetch_financial_metrics,
    query_looker_financial_explore,
    search_10k_chunks_rag,
)
from edgar_capstone.security.secrets import (
    DEFAULT_PROJECT_ID,
    MANAGED_SECRETS,
    SecretManagerClient,
)

app = FastAPI(
    title="SEC EDGAR Natural Language Analyst API",
    version="0.1.0",
    description=(
        "Cloud Run FastAPI backend for deterministic SEC 10-K financial variance "
        "calculations, metadata-filtered Hybrid RAG search, and ADK RootOrchestrator routing."
    ),
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


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Cloud Run liveness and readiness probe endpoint."""
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
