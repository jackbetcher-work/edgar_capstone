"""Request/response wiring tests for the live GCP tool adapters (BigQuery, Vertex AI Search,
Vector Search, Vertex AI embeddings, and GCS).

These tests inject `httpx.MockTransport` clients so they can assert the exact REST payloads the
adapters send and how they map each API's response shape into tool results. They exercise the
adapter plumbing, not the data: data-dependent tests read live GCP elsewhere in the suite.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import urllib.parse
from collections.abc import AsyncGenerator, Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.auth.exceptions import DefaultCredentialsError
from google.genai import types as genai_types

from edgar_capstone.agents.adapters.registry import (
    GcpToolAdapters,
    GcpToolConfig,
    configure_gcp_tool_adapters,
    get_gcp_adapters_status_metadata,
)
from edgar_capstone.agents.runtime.context_memory import get_conversation_memory_manager
from edgar_capstone.agents.runtime.factories import create_financial_analyst_agent
from edgar_capstone.agents.runtime.latency_tracker import get_context_latency_tracker
from edgar_capstone.agents.runtime.live_runtime import run_adk_agent_stream
from edgar_capstone.agents.runtime.model_config import FINANCIAL_REASONING_MODEL
from edgar_capstone.agents.tools.sec_tools import (
    _SECTION_CHUNKS_CACHE,
    _SESSION_10K_CONTEXT_CACHE,
    calculate_period_variances,
    fetch_financial_metrics,
    load_cached_10k_section,
    query_analytics_view,
    query_cached_10k_context,
    search_10k_chunks_rag,
)
from edgar_capstone.dataset.extractor import format_combined_10k_markdown
from tests.fake_gcp import preserve_gcp_caches
from tests.structured_report_helpers import (
    ScriptedBaseLlm,
    finding_from_citation,
    report_json,
)

_EMBEDDING_DIMS = 768


def _mock_embedding_values(text: str) -> list[float]:
    """Hashed bag-of-words unit vector used as the mocked `text-embedding-005:predict` payload.

    Texts sharing words get a positive cosine similarity, which is all the ranking code needs
    from the mocked response.
    """
    vec = [0.0] * _EMBEDDING_DIMS
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.sha256(token.encode()).hexdigest(), 16) % _EMBEDDING_DIMS] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class _FakeGcpCredentials:
    """Offline mock OAuth2 credentials object for `httpx.MockTransport` tests."""

    def __init__(self) -> None:
        self.valid = True
        self.token = "test-gcp-oauth-bearer"
        self.refresh_calls = 0

    def refresh(self, request: Any) -> None:
        self.refresh_calls += 1
        self.valid = True
        self.token = "test-gcp-oauth-bearer"


class _FailingCredentials:
    """Credentials stub that raises `DefaultCredentialsError` on refresh."""

    valid = False
    token = None

    def refresh(self, request: Any) -> None:
        raise DefaultCredentialsError("No ADC configured in test sandbox")


def _dummy_tool_ctx() -> SimpleNamespace:
    return SimpleNamespace(state={}, actions=SimpleNamespace(escalate=False))


def _bq_profile_schema() -> list[dict[str, Any]]:
    return [
        {"name": "profile_id", "type": "STRING", "mode": "REQUIRED"},
        {"name": "ticker", "type": "STRING", "mode": "REQUIRED"},
        {"name": "cik", "type": "STRING", "mode": "REQUIRED"},
        {"name": "company_name", "type": "STRING", "mode": "REQUIRED"},
        {"name": "gics_sector", "type": "STRING", "mode": "NULLABLE"},
        {"name": "gics_sub_industry", "type": "STRING", "mode": "NULLABLE"},
        {"name": "fiscal_year", "type": "INTEGER", "mode": "REQUIRED"},
        {"name": "period_of_report", "type": "DATE", "mode": "NULLABLE"},
        {"name": "filing_date", "type": "DATE", "mode": "NULLABLE"},
        {"name": "accession_number", "type": "STRING", "mode": "REQUIRED"},
        {"name": "currency", "type": "STRING", "mode": "NULLABLE"},
        {"name": "revenue", "type": "FLOAT", "mode": "NULLABLE"},
        {"name": "operating_income", "type": "FLOAT", "mode": "NULLABLE"},
        {"name": "net_income", "type": "FLOAT", "mode": "NULLABLE"},
        {
            "name": "metric_provenance",
            "type": "RECORD",
            "mode": "NULLABLE",
            "fields": [
                {"name": "revenue", "type": "STRING", "mode": "NULLABLE"},
                {"name": "operating_income", "type": "STRING", "mode": "NULLABLE"},
                {"name": "net_income", "type": "STRING", "mode": "NULLABLE"},
                {"name": "notes", "type": "STRING", "mode": "REPEATED"},
            ],
        },
        {"name": "extracted_sections", "type": "STRING", "mode": "REPEATED"},
        {"name": "unstructured_blob_path", "type": "STRING", "mode": "NULLABLE"},
        {"name": "structured_blob_path", "type": "STRING", "mode": "NULLABLE"},
        {"name": "unstructured_gcs_uri", "type": "STRING", "mode": "NULLABLE"},
        {"name": "structured_gcs_uri", "type": "STRING", "mode": "NULLABLE"},
        {"name": "sec_filing_url", "type": "STRING", "mode": "NULLABLE"},
    ]


def _encode_bq_profile_row(
    *,
    ticker: str,
    fy: int,
    revenue: float,
    operating_income: float,
    net_income: float,
    company_name: str = "NVIDIA CORP",
    sector: str = "Information Technology",
) -> dict[str, Any]:
    return {
        "f": [
            {"v": f"{ticker}_{fy}"},
            {"v": ticker},
            {"v": "0001045810"},
            {"v": company_name},
            {"v": sector},
            {"v": "Semiconductors"},
            {"v": str(fy)},
            {"v": f"{fy}-01-26"},
            {"v": f"{fy}-02-26"},
            {"v": f"0001045810-{str(fy)[-2:]}-000021"},
            {"v": "USD"},
            {"v": str(revenue)},
            {"v": str(operating_income)},
            {"v": str(net_income)},
            {
                "v": {
                    "f": [
                        {"v": "us-gaap:Revenues"},
                        {"v": "us-gaap:OperatingIncomeLoss"},
                        {"v": "us-gaap:NetIncomeLoss"},
                        {"v": [{"v": "XBRL audited 10-K tag"}]},
                    ]
                }
            },
            {"v": [{"v": "item_1a"}, {"v": "item_7"}]},
            {"v": f"unstructured/{ticker}/{fy}_10k.md"},
            {"v": f"structured/{ticker}/{fy}_financials.json"},
            {"v": f"gs://fde-sec-edgar-sandbox-dev-sec-corpus/unstructured/{ticker}/{fy}_10k.md"},
            {
                "v": f"gs://fde-sec-edgar-sandbox-dev-sec-corpus/structured/{ticker}/{fy}_financials.json"
            },
            {"v": f"https://www.sec.gov/Archives/edgar/data/1045810/{ticker}_{fy}_10k.htm"},
        ]
    }


def _sample_nvda_2026_markdown() -> str:
    return format_combined_10k_markdown(
        ticker="NVDA",
        company_name="NVIDIA CORP",
        cik="0001045810",
        gics_sector="Information Technology",
        gics_sub_industry="Semiconductors",
        fiscal_year=2026,
        period_of_report="2026-01-25",
        filing_date="2026-02-25",
        accession_number="0001045810-26-000021",
        sec_filing_url="https://www.sec.gov/Archives/edgar/data/1045810/NVDA_2026_10k.htm",
        item_1a_markdown=(
            "U.S. government export control licensing requirements on H20, Hopper, and Blackwell "
            "AI accelerators shipped to China and D:5 countries impacted international Data Center "
            "shipments and resulted in inventory purchase commitment charges.\n\n"
            "Long wafer foundry and CoWoS advanced packaging lead times create supply concentration risks."
        ),
        item_7_markdown=(
            "Fiscal year 2026 Data Center revenue surged to record levels driven by hyperscale cloud "
            "deployment of Blackwell GB200 NVL72 clusters and Hopper H200 inference systems.\n\n"
            "Operating margin expanded significantly due to high-value accelerated computing platform mix, "
            "partially offset by higher R&D engineering investments."
        ),
    )


@pytest.fixture(autouse=True)
def _cleanup_gcp_adapter_registry() -> Iterator[None]:
    """Ensure injected GCP adapters never leak across tests without wiping warm live caches."""
    with preserve_gcp_caches():
        yield


def test_gcp_tool_config_and_status_metadata_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GcpToolConfig.from_env()` reads the resource env vars; status metadata is always `gcp`."""
    from edgar_capstone.gcp.defaults import resolve_gcp_project_id

    for key in (
        "BIGQUERY_DATASET_ID",
        "VERTEX_AI_SEARCH_ENGINE_ID",
        "VERTEX_AI_SEARCH_DATASTORE_ID",
        "VERTEX_EMBEDDING_DIMENSIONS",
        "SEC_CORPUS_BUCKET",
    ):
        monkeypatch.delenv(key, raising=False)

    default_cfg = GcpToolConfig.from_env()
    assert default_cfg.project_id == (os.environ.get("GCP_PROJECT_ID") or resolve_gcp_project_id())
    assert default_cfg.bq_dataset_id == "sec_edgar"
    assert default_cfg.embedding_dimensions == 768
    meta = get_gcp_adapters_status_metadata()
    assert meta["data_backend"] == "gcp"
    assert meta["project_id"] == default_cfg.project_id
    assert meta["bigquery"]["financial_metrics_table"] == "financial_metrics"
    assert meta["hybrid_retrieval_backends"] == ["bigquery_vector_search", "vertex_ai_search"]
    assert "vertex_vector_search" not in meta

    monkeypatch.setenv("BIGQUERY_DATASET_ID", "sec_edgar_staging")
    monkeypatch.setenv("VERTEX_AI_SEARCH_ENGINE_ID", "engine-x")
    monkeypatch.setenv("VERTEX_EMBEDDING_DIMENSIONS", "256")
    env_cfg = GcpToolConfig.from_env()
    assert env_cfg.bq_dataset_id == "sec_edgar_staging"
    assert env_cfg.vais_engine_id == "engine-x"
    assert env_cfg.embedding_dimensions == 256
    env_meta = get_gcp_adapters_status_metadata()
    assert env_meta["bigquery"]["dataset_id"] == "sec_edgar_staging"
    assert env_meta["vertex_ai_search"]["engine_id"] == "engine-x"

    # An injected adapter bundle's config is what the status metadata reports.
    configure_gcp_tool_adapters(
        GcpToolAdapters(
            GcpToolConfig(project_id="injected-project", bq_dataset_id="injected_ds"),
            credentials=_FakeGcpCredentials(),
        )
    )
    injected_meta = get_gcp_adapters_status_metadata()
    assert injected_meta["data_backend"] == "gcp"
    assert injected_meta["project_id"] == "injected-project"
    assert injected_meta["bigquery"]["dataset_id"] == "injected_ds"


def test_bigquery_adapter_metrics_variances_and_job_polling() -> None:
    """Verify parameterized BigQuery SQL, `jobComplete=False` polling, pagination, and deterministic variance math."""
    captured_requests: list[tuple[str, str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer test-gcp-oauth-bearer"
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        captured_requests.append((request.method, str(request.url), body))

        url_str = str(request.url)
        if request.method == "POST" and url_str.endswith("/queries"):
            # Simulate async job completion + pagination on first query
            return httpx.Response(
                200,
                json={
                    "jobComplete": False,
                    "jobReference": {"jobId": "bq-job-101", "location": "us-central1"},
                },
            )
        if request.method == "GET" and "/queries/bq-job-101" in url_str:
            if "pageToken=page-2" not in url_str:
                return httpx.Response(
                    200,
                    json={
                        "jobComplete": True,
                        "jobReference": {"jobId": "bq-job-101", "location": "us-central1"},
                        "schema": {"fields": _bq_profile_schema()},
                        "rows": [
                            _encode_bq_profile_row(
                                ticker="NVDA",
                                fy=2023,
                                revenue=26_974_000_000.0,
                                operating_income=4_224_000_000.0,
                                net_income=4_368_000_000.0,
                            ),
                            _encode_bq_profile_row(
                                ticker="NVDA",
                                fy=2024,
                                revenue=60_922_000_000.0,
                                operating_income=32_972_000_000.0,
                                net_income=29_760_000_000.0,
                            ),
                        ],
                        "pageToken": "page-2",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "jobComplete": True,
                    "jobReference": {"jobId": "bq-job-101", "location": "us-central1"},
                    "schema": {"fields": _bq_profile_schema()},
                    "rows": [
                        _encode_bq_profile_row(
                            ticker="NVDA",
                            fy=2025,
                            revenue=130_497_000_000.0,
                            operating_income=81_453_000_000.0,
                            net_income=72_880_000_000.0,
                        ),
                        _encode_bq_profile_row(
                            ticker="NVDA",
                            fy=2026,
                            revenue=215_938_000_000.0,
                            operating_income=130_387_000_000.0,
                            net_income=120_067_000_000.0,
                        ),
                    ],
                },
            )
        return httpx.Response(404, text="Unexpected endpoint")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = GcpToolConfig(project_id="iron-radio-514217-q7")
    configure_gcp_tool_adapters(
        GcpToolAdapters(cfg, http_client=client, credentials=_FakeGcpCredentials())
    )

    ctx = _dummy_tool_ctx()
    var_res = calculate_period_variances(
        tickers="NVDA", start_year=2024, end_year=2026, tool_context=ctx
    )
    assert var_res["status"] == "success"
    assert var_res["company_count"] == 1
    rep = var_res["variance_reports"][0]
    assert rep["ticker"] == "NVDA"
    assert rep["revenue_variance"]["start_value"] == 60_922_000_000.0
    assert rep["revenue_variance"]["end_value"] == 215_938_000_000.0
    assert rep["revenue_variance"]["percentage_delta"] == 254.45
    # Verify start_year - 1 (FY2023) was fetched from BigQuery so FY2024 YoY step is populated!
    assert rep["annual_series"][0]["fiscal_year"] == 2024
    assert rep["annual_series"][0]["prev_fiscal_year"] == 2023
    assert rep["annual_series"][0]["revenue_yoy_pct"] == 125.85

    # Verify named SQL parameters sent to BigQuery
    post_method, post_url, post_body = captured_requests[0]
    assert post_method == "POST"
    assert "bigquery.googleapis.com/bigquery/v2/projects/iron-radio-514217-q7/queries" in post_url
    assert post_body["useLegacySql"] is False
    assert post_body["parameterMode"] == "NAMED"
    param_names = {p["name"] for p in post_body["queryParameters"]}
    assert param_names == {"start_year", "end_year", "tickers"}
    assert ctx.state["gcp_backend_calls"][0]["services_hit"] == ["bigquery"]

    # Also test fetch_financial_metrics with nested RECORD metric_provenance decoding
    met_res = fetch_financial_metrics(
        tickers="NVDA",
        start_year=2024,
        end_year=2026,
        gics_sector="Information Technology",
        tool_context=ctx,
    )
    assert met_res["status"] == "success"
    assert met_res["record_count"] == 4
    sample_metric = met_res["metrics"][-1]
    assert sample_metric["provenance"]["revenue"] == "us-gaap:Revenues"
    assert sample_metric["provenance"]["notes"] == ["XBRL audited 10-K tag"]


def test_bigquery_adapter_all_analytics_views() -> None:
    """Verify `query_analytics_view` queries `v_financial_variances_yoy`, `financial_metrics`, and `v_chunk_health_metrics`."""
    executed_sql: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        sql = body["query"]
        executed_sql.append(sql)

        if "v_financial_variances_yoy" in sql:
            return httpx.Response(
                200,
                json={
                    "jobComplete": True,
                    "schema": {
                        "fields": [
                            {"name": "profile_id", "type": "STRING"},
                            {"name": "ticker", "type": "STRING"},
                            {"name": "company_name", "type": "STRING"},
                            {"name": "gics_sector", "type": "STRING"},
                            {"name": "fiscal_year", "type": "INTEGER"},
                            {"name": "prev_fiscal_year", "type": "INTEGER"},
                            {"name": "revenue", "type": "FLOAT"},
                            {"name": "revenue_yoy_delta", "type": "FLOAT"},
                            {"name": "revenue_yoy_pct", "type": "FLOAT"},
                            {"name": "operating_income", "type": "FLOAT"},
                            {"name": "operating_income_yoy_delta", "type": "FLOAT"},
                            {"name": "operating_income_yoy_pct", "type": "FLOAT"},
                            {"name": "net_income", "type": "FLOAT"},
                            {"name": "net_income_yoy_delta", "type": "FLOAT"},
                            {"name": "net_income_yoy_pct", "type": "FLOAT"},
                            {"name": "sec_filing_url", "type": "STRING"},
                        ]
                    },
                    "rows": [
                        {
                            "f": [
                                {"v": "NVDA_2026"},
                                {"v": "NVDA"},
                                {"v": "NVIDIA CORP"},
                                {"v": "Information Technology"},
                                {"v": "2026"},
                                {"v": "2025"},
                                {"v": "215938000000.0"},
                                {"v": "85441000000.0"},
                                {"v": "65.4735"},
                                {"v": "130387000000.0"},
                                {"v": "48934000000.0"},
                                {"v": "60.0764"},
                                {"v": "120067000000.0"},
                                {"v": "47187000000.0"},
                                {"v": "64.7462"},
                                {
                                    "v": "https://www.sec.gov/Archives/edgar/data/1045810/NVDA_2026.htm"
                                },
                            ]
                        }
                    ],
                },
            )
        if "GROUP BY 1, 2" in sql:
            return httpx.Response(
                200,
                json={
                    "jobComplete": True,
                    "schema": {
                        "fields": [
                            {"name": "gics_sector", "type": "STRING"},
                            {"name": "fiscal_year", "type": "INTEGER"},
                            {"name": "company_count", "type": "INTEGER"},
                            {"name": "tickers", "type": "STRING", "mode": "REPEATED"},
                            {"name": "total_sector_revenue", "type": "FLOAT"},
                            {"name": "avg_company_revenue", "type": "FLOAT"},
                            {"name": "total_sector_operating_income", "type": "FLOAT"},
                            {"name": "total_sector_net_income", "type": "FLOAT"},
                        ]
                    },
                    "rows": [
                        {
                            "f": [
                                {"v": "Information Technology"},
                                {"v": "2026"},
                                {"v": "2"},
                                {"v": [{"v": "AAPL"}, {"v": "NVDA"}]},
                                {"v": "630000000000.0"},
                                {"v": "315000000000.0"},
                                {"v": "260000000000.0"},
                                {"v": "225000000000.0"},
                            ]
                        }
                    ],
                },
            )
        if "v_chunk_health_metrics" in sql:
            fields = [
                {"name": "gics_sector", "type": "STRING"},
                {"name": "section_id", "type": "STRING"},
                {"name": "total_chunks", "type": "INTEGER"},
                {"name": "distinct_profiles", "type": "INTEGER"},
                {"name": "distinct_tickers", "type": "INTEGER"},
                {"name": "valid_embedding_count", "type": "INTEGER"},
                {"name": "embedding_coverage_ratio", "type": "FLOAT"},
                {"name": "filing_coverage_ratio", "type": "FLOAT"},
                {"name": "avg_char_count", "type": "FLOAT"},
                {"name": "avg_token_estimate", "type": "FLOAT"},
                {"name": "min_token_estimate", "type": "INTEGER"},
                {"name": "max_token_estimate", "type": "INTEGER"},
                {"name": "undersized_chunk_count", "type": "INTEGER"},
                {"name": "oversized_chunk_count", "type": "INTEGER"},
                {"name": "missing_metadata_count", "type": "INTEGER"},
                {"name": "health_score", "type": "FLOAT"},
                {"name": "health_status", "type": "STRING"},
            ]
            return httpx.Response(
                200,
                json={
                    "jobComplete": True,
                    "schema": {"fields": fields},
                    "rows": [
                        {
                            "f": [
                                {"v": "ALL_SECTORS"},
                                {"v": "ALL_SECTIONS"},
                                {"v": "12180"},
                                {"v": "114"},
                                {"v": "18"},
                                {"v": "12180"},
                                {"v": "1.0"},
                                {"v": "1.0"},
                                {"v": "1715.4"},
                                {"v": "395.3"},
                                {"v": "54"},
                                {"v": "1188"},
                                {"v": "0"},
                                {"v": "0"},
                                {"v": "0"},
                                {"v": "100.0"},
                                {"v": "HEALTHY"},
                            ]
                        },
                        {
                            "f": [
                                {"v": "Information Technology"},
                                {"v": "ALL_SECTIONS"},
                                {"v": "4200"},
                                {"v": "30"},
                                {"v": "5"},
                                {"v": "4200"},
                                {"v": "1.0"},
                                {"v": "1.0"},
                                {"v": "1720.0"},
                                {"v": "398.0"},
                                {"v": "60"},
                                {"v": "1100"},
                                {"v": "0"},
                                {"v": "0"},
                                {"v": "0"},
                                {"v": "100.0"},
                                {"v": "HEALTHY"},
                            ]
                        },
                        {
                            "f": [
                                {"v": "Information Technology"},
                                {"v": "item_1a"},
                                {"v": "1900"},
                                {"v": "30"},
                                {"v": "5"},
                                {"v": "1900"},
                                {"v": "1.0"},
                                {"v": "1.0"},
                                {"v": "1800.0"},
                                {"v": "405.0"},
                                {"v": "60"},
                                {"v": "950"},
                                {"v": "0"},
                                {"v": "0"},
                                {"v": "0"},
                                {"v": "100.0"},
                                {"v": "HEALTHY"},
                            ]
                        },
                        {
                            "f": [
                                {"v": "Information Technology"},
                                {"v": "item_7"},
                                {"v": "2300"},
                                {"v": "30"},
                                {"v": "5"},
                                {"v": "2300"},
                                {"v": "1.0"},
                                {"v": "1.0"},
                                {"v": "1653.9"},
                                {"v": "392.2"},
                                {"v": "65"},
                                {"v": "1100"},
                                {"v": "0"},
                                {"v": "0"},
                                {"v": "0"},
                                {"v": "100.0"},
                                {"v": "HEALTHY"},
                            ]
                        },
                    ],
                },
            )
        return httpx.Response(400, text="Unexpected SQL")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    configure_gcp_tool_adapters(
        GcpToolAdapters(
            GcpToolConfig(),
            http_client=client,
            credentials=_FakeGcpCredentials(),
        )
    )

    yoy = query_analytics_view(
        explore_name="yoy_variances",
        tickers="NVDA",
        gics_sector="Information Technology",
        start_year=2025,
        end_year=2026,
    )
    assert yoy["status"] == "success"
    assert yoy["row_count"] == 1
    assert yoy["rows"][0]["revenue_yoy_pct"] == 65.47
    assert yoy["rows"][0]["operating_margin_pct"] == 60.38

    bench = query_analytics_view(
        explore_name="sector_benchmarks",
        tickers="ALL",
        gics_sector="Information Technology",
        start_year=2026,
        end_year=2026,
    )
    assert bench["status"] == "success"
    assert bench["rows"][0]["tickers"] == ["AAPL", "NVDA"]
    assert bench["rows"][0]["company_count"] == 2

    health = query_analytics_view(
        explore_name="chunk_health",
        tickers="ALL",
        gics_sector="ALL",
        start_year=2024,
        end_year=2026,
    )
    assert health["status"] == "success"
    assert health["health_status"] == "HEALTHY"
    assert health["total_chunks"] == 12180
    assert "Information Technology" in health["by_sector"]
    assert set(health["by_section"].keys()) == {"item_1a", "item_7"}
    assert len(executed_sql) == 3


def test_hybrid_rag_fuses_bigquery_vector_search_and_vertex_ai_search() -> None:
    """Verify `search_10k_chunks_rag` runs Vertex Embeddings + BigQuery VECTOR_SEARCH in parallel with Vertex AI Search (`:search`) and caches servingConfig fallback."""
    hit_log: list[str] = []
    q_vec = _mock_embedding_values("China export controls H20 Blackwell GPU")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content.decode("utf-8")) if request.content else {}

        # 1. Vertex AI Embeddings (`text-embedding-005:predict`)
        if ":predict" in url and "text-embedding-005" in url:
            hit_log.append("vertex_embeddings")
            assert body["instances"][0]["task_type"] == "RETRIEVAL_QUERY"
            assert body["parameters"]["outputDimensionality"] == 768
            return httpx.Response(
                200,
                json={"predictions": [{"embeddings": {"values": q_vec}}]},
            )

        # 2. Vertex AI Search / Discovery Engine (simulate 404 on engine -> cached fallback to datastore!)
        if "discoveryengine.googleapis.com" in url:
            if "/engines/sec-edgar-10k-hybrid-search/" in url:
                hit_log.append("vais_engine_404")
                return httpx.Response(404, text="Engine not yet created")
            if "/dataStores/sec-edgar-10k-corpus-ds/" in url:
                hit_log.append("vais_datastore_search")
                assert 'ticker: ANY("NVDA")' in body["filter"]
                assert 'gics_sector: ANY("Information Technology")' in body["filter"]
                assert 'section_id: ANY("item_1a")' in body["filter"]
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "document": {
                                    "id": "NVDA_2026_item_1a",
                                    "structData": {
                                        "profile_id": "NVDA_2026",
                                        "ticker": "NVDA",
                                        "company_name": "NVIDIA CORP",
                                        "gics_sector": "Information Technology",
                                        "fiscal_year": 2026,
                                        "section_id": "item_1a",
                                        "section_title": "Item 1A. Risk Factors",
                                        "sec_filing_url": "https://www.sec.gov/Archives/edgar/data/1045810/NVDA_2026.htm",
                                    },
                                    "content": {
                                        "uri": "gs://fde-sec-edgar-sandbox-dev-sec-corpus/unstructured/NVDA/2026_item_1a.md"
                                    },
                                    "derivedStructData": {
                                        "extractive_segments": [
                                            {
                                                "content": (
                                                    "U.S. government export control licensing requirements on H20 "
                                                    "and Blackwell AI accelerators shipped to China restricted revenue."
                                                ),
                                                "relevanceScore": 0.94,
                                            }
                                        ],
                                    },
                                }
                            }
                        ]
                    },
                )

        # 3. BigQuery VECTOR_SEARCH (returns two chunks with full text and metadata)
        if "bigquery.googleapis.com" in url:
            sql = body["query"]
            chunk_fields = [
                {"name": "chunk_id", "type": "STRING"},
                {"name": "profile_id", "type": "STRING"},
                {"name": "ticker", "type": "STRING"},
                {"name": "company_name", "type": "STRING"},
                {"name": "gics_sector", "type": "STRING"},
                {"name": "fiscal_year", "type": "INTEGER"},
                {"name": "section_id", "type": "STRING"},
                {"name": "section_title", "type": "STRING"},
                {"name": "chunk_index", "type": "INTEGER"},
                {"name": "chunk_text", "type": "STRING"},
                {"name": "sec_filing_url", "type": "STRING"},
                {"name": "unstructured_gcs_uri", "type": "STRING"},
                {"name": "period_of_report", "type": "STRING"},
                {"name": "filing_date", "type": "STRING"},
                {"name": "accession_number", "type": "STRING"},
            ]
            if "VECTOR_SEARCH" in sql:
                hit_log.append("bq_vector_search")
                return httpx.Response(
                    200,
                    json={
                        "jobComplete": True,
                        "schema": {
                            "fields": [*chunk_fields, {"name": "distance", "type": "FLOAT"}]
                        },
                        "rows": [
                            {
                                "f": [
                                    {"v": "NVDA_2026_item_1a_0000"},
                                    {"v": "NVDA_2026"},
                                    {"v": "NVDA"},
                                    {"v": "NVIDIA CORP"},
                                    {"v": "Information Technology"},
                                    {"v": "2026"},
                                    {"v": "item_1a"},
                                    {"v": "Item 1A. Risk Factors — Export Controls"},
                                    {"v": "0"},
                                    {
                                        "v": (
                                            "U.S. government export control licensing requirements on H20 "
                                            "and Blackwell AI accelerators shipped to China restricted revenue "
                                            "and resulted in excess inventory charges in FY2026."
                                        )
                                    },
                                    {
                                        "v": "https://www.sec.gov/Archives/edgar/data/1045810/NVDA_2026.htm"
                                    },
                                    {
                                        "v": "gs://fde-sec-edgar-sandbox-dev-sec-corpus/unstructured/NVDA/2026_10k.md"
                                    },
                                    {"v": "2026-01-25"},
                                    {"v": "2026-02-25"},
                                    {"v": "0001045810-26-000021"},
                                    {"v": "0.10"},
                                ]
                            },
                            {
                                "f": [
                                    {"v": "NVDA_2025_item_1a_0002"},
                                    {"v": "NVDA_2025"},
                                    {"v": "NVDA"},
                                    {"v": "NVIDIA CORP"},
                                    {"v": "Information Technology"},
                                    {"v": "2025"},
                                    {"v": "item_1a"},
                                    {"v": "Item 1A. Risk Factors — International Trade"},
                                    {"v": "2"},
                                    {
                                        "v": (
                                            "Expanded export licensing rules in FY2025 constrained "
                                            "Data Center GPU shipments to China and D:5 regions."
                                        )
                                    },
                                    {
                                        "v": "https://www.sec.gov/Archives/edgar/data/1045810/NVDA_2025.htm"
                                    },
                                    {
                                        "v": "gs://fde-sec-edgar-sandbox-dev-sec-corpus/unstructured/NVDA/2025_10k.md"
                                    },
                                    {"v": "2025-01-26"},
                                    {"v": "2025-02-26"},
                                    {"v": "0001045810-25-000023"},
                                    {"v": "0.19"},
                                ]
                            },
                        ],
                    },
                )

        return httpx.Response(500, text=f"Unhandled URL: {url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    configure_gcp_tool_adapters(
        GcpToolAdapters(
            GcpToolConfig(),
            http_client=client,
            credentials=_FakeGcpCredentials(),
        )
    )

    ctx = _dummy_tool_ctx()
    res = search_10k_chunks_rag(
        query="China export controls H20 Blackwell GPU",
        tickers="NVDA",
        gics_sector="information technology",  # lowercase input -> canonicalized!
        section_id="item_1a",
        start_year=2024,
        end_year=2026,
        top_k=5,
        tool_context=ctx,
    )

    assert res["status"] == "success"
    assert res["citation_count"] >= 2
    assert res["citations"][0]["chunk_id"] == "NVDA_2026_item_1a_0000"
    assert res["citations"][0]["period_of_report"] == "2026-01-25"
    assert res["citations"][0]["accession_number"] == "0001045810-26-000021"
    assert "Blackwell" in res["citations"][0]["excerpt"]
    chunk_ids = {c["chunk_id"] for c in res["citations"]}
    assert "NVDA_2025_item_1a_0002" in chunk_ids

    # Parallel execution: compare sets rather than fixed sequential order
    assert set(hit_log) == {
        "vertex_embeddings",
        "vais_engine_404",
        "vais_datastore_search",
        "bq_vector_search",
    }
    assert set(ctx.state["gcp_backend_calls"][0]["services_hit"]) == {
        "vertex_ai_embeddings",
        "vertex_ai_search",
        "bigquery_vector_search",
    }

    # Second search reuses the cached Discovery Engine datastore URL (no repeated 404!)
    hit_log.clear()
    res2 = search_10k_chunks_rag(
        query="China export controls licensing fees",
        tickers="NVDA",
        gics_sector="information technology",
        section_id="item_1a",
        start_year=2024,
        end_year=2026,
        top_k=5,
    )
    assert res2["status"] == "success"
    assert "vais_engine_404" not in hit_log
    assert "vais_datastore_search" in hit_log


def test_context_cache_miss_loads_from_bigquery_and_gcs_and_queries_cached_section() -> None:
    """Verify `load_cached_10k_section` & `query_cached_10k_context` read from BigQuery + GCS on miss and hit L1 chunk cache on repeat."""
    _SESSION_10K_CONTEXT_CACHE.pop("NVDA_2026_item_7", None)
    _SECTION_CHUNKS_CACHE.pop("NVDA_2026_item_7", None)
    hit_log: list[str] = []
    md_doc = _sample_nvda_2026_markdown()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content.decode("utf-8")) if request.content else {}

        if "bigquery.googleapis.com" in url:
            sql = body["query"]
            if "FROM `iron-radio-514217-q7.sec_edgar.financial_metrics`" in sql:
                hit_log.append("bq_fetch_profile")
                return httpx.Response(
                    200,
                    json={
                        "jobComplete": True,
                        "schema": {"fields": _bq_profile_schema()},
                        "rows": [
                            _encode_bq_profile_row(
                                ticker="NVDA",
                                fy=2026,
                                revenue=215_938_000_000.0,
                                operating_income=130_387_000_000.0,
                                net_income=120_067_000_000.0,
                            )
                        ],
                    },
                )
            if "FROM `iron-radio-514217-q7.sec_edgar.filing_chunks`" in sql:
                hit_log.append("bq_fetch_section_chunks")
                return httpx.Response(
                    200,
                    json={
                        "jobComplete": True,
                        "schema": {
                            "fields": [
                                {"name": "chunk_id", "type": "STRING"},
                                {"name": "profile_id", "type": "STRING"},
                                {"name": "ticker", "type": "STRING"},
                                {"name": "company_name", "type": "STRING"},
                                {"name": "gics_sector", "type": "STRING"},
                                {"name": "fiscal_year", "type": "INTEGER"},
                                {"name": "section_id", "type": "STRING"},
                                {"name": "section_title", "type": "STRING"},
                                {"name": "chunk_index", "type": "INTEGER"},
                                {"name": "chunk_text", "type": "STRING"},
                                {"name": "sec_filing_url", "type": "STRING"},
                                {"name": "unstructured_gcs_uri", "type": "STRING"},
                            ]
                        },
                        "rows": [
                            {
                                "f": [
                                    {"v": "NVDA_2026_item_7_0000"},
                                    {"v": "NVDA_2026"},
                                    {"v": "NVDA"},
                                    {"v": "NVIDIA CORP"},
                                    {"v": "Information Technology"},
                                    {"v": "2026"},
                                    {"v": "item_7"},
                                    {"v": "Item 7. MD&A — Data Center"},
                                    {"v": "0"},
                                    {
                                        "v": (
                                            "Fiscal year 2026 Data Center revenue surged to record levels driven "
                                            "by hyperscale cloud deployment of Blackwell GB200 NVL72 clusters."
                                        )
                                    },
                                    {
                                        "v": "https://www.sec.gov/Archives/edgar/data/1045810/NVDA_2026_10k.htm"
                                    },
                                    {
                                        "v": "gs://fde-sec-edgar-sandbox-dev-sec-corpus/unstructured/NVDA/2026_10k.md"
                                    },
                                ]
                            }
                        ],
                    },
                )

        if "storage.googleapis.com/storage/v1/b/" in url:
            hit_log.append("gcs_download_10k_md")
            expected_obj = urllib.parse.quote("unstructured/NVDA/2026_10k.md", safe="")
            assert expected_obj in url
            assert "alt=media" in url
            return httpx.Response(200, text=md_doc)

        if ":predict" in url and "text-embedding-005" in url:
            hit_log.append("vertex_embeddings")
            preds = [
                {"embeddings": {"values": _mock_embedding_values(inst["content"])}}
                for inst in body["instances"]
            ]
            return httpx.Response(200, json={"predictions": preds})

        return httpx.Response(404, text=f"Unhandled URL: {url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    configure_gcp_tool_adapters(
        GcpToolAdapters(
            GcpToolConfig(),
            http_client=client,
            credentials=_FakeGcpCredentials(),
        )
    )

    ctx = _dummy_tool_ctx()
    miss_res = load_cached_10k_section(
        ticker="NVDA",
        fiscal_year=2026,
        section_id="item_7",
        max_chars=12000,
        tool_context=ctx,
    )
    assert miss_res["status"] == "success"
    assert miss_res["cache_hit"] is False
    assert "Blackwell GB200 NVL72" in miss_res["content"]

    # Cold `_SECTION_CHUNKS_CACHE` miss fetches chunks from BigQuery and reports `cache_hit=False`
    query_res = query_cached_10k_context(
        ticker="NVDA",
        fiscal_year=2026,
        section_id="item_7",
        variance_query="Blackwell GB200 Data Center revenue growth",
        max_passages=2,
        tool_context=ctx,
    )
    assert query_res["status"] == "success"
    assert query_res["cache_hit"] is False
    assert query_res["matched_chunk_ids"] == ["NVDA_2026_item_7_0000"]
    assert "bq_fetch_profile" in hit_log
    assert "gcs_download_10k_md" in hit_log
    assert "bq_fetch_section_chunks" in hit_log
    assert "vertex_embeddings" in hit_log

    # Second call for the same section hits `_SECTION_CHUNKS_CACHE` with no extra BigQuery query
    bq_chunk_fetches = hit_log.count("bq_fetch_section_chunks")
    query_res_2 = query_cached_10k_context(
        ticker="NVDA",
        fiscal_year=2026,
        section_id="item_7",
        variance_query="Blackwell GB200 Data Center revenue growth",
        max_passages=2,
        tool_context=ctx,
    )
    assert query_res_2["status"] == "success"
    assert query_res_2["cache_hit"] is True
    assert query_res_2["matched_chunk_ids"] == ["NVDA_2026_item_7_0000"]
    assert hit_log.count("bq_fetch_section_chunks") == bq_chunk_fetches


class _LiveToolCallingAnalystLlm(ScriptedBaseLlm):
    """ADK `BaseLlm` that invokes `calculate_period_variances` + `search_10k_chunks_rag` and synthesizes grounded output."""

    model: str = FINANCIAL_REASONING_MODEL

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        assert stream is True
        fn_responses: dict[str, dict[str, Any]] = {}
        for content in llm_request.contents or []:
            for part in content.parts or []:
                fn_resp = getattr(part, "function_response", None)
                if fn_resp is not None and fn_resp.name:
                    fn_responses[fn_resp.name] = dict(fn_resp.response or {})

        if "calculate_period_variances" not in fn_responses:
            yield LlmResponse(
                model_version=self.model,
                partial=False,
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part.from_function_call(
                            name="calculate_period_variances",
                            args={"tickers": "NVDA", "start_year": 2024, "end_year": 2026},
                        ),
                        genai_types.Part.from_function_call(
                            name="search_10k_chunks_rag",
                            args={
                                "query": "Blackwell Data Center export control risks",
                                "tickers": "NVDA",
                                "gics_sector": "Information Technology",
                                "section_id": "ALL",
                                "start_year": 2024,
                                "end_year": 2026,
                                "top_k": 3,
                            },
                        ),
                    ],
                ),
            )
            return

        cit = fn_responses["search_10k_chunks_rag"]["citations"][0]
        final_text = report_json(
            headline="NVDA Live GCP Grounded Analysis",
            key_metrics=[("NVDA", "revenue")],
            findings=[finding_from_citation(cit, "Data Center demand supported revenue growth.")],
        )
        await asyncio.sleep(0.005)
        yield LlmResponse(
            model_version=self.model,
            partial=True,
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=final_text[:40])]
            ),
        )
        yield LlmResponse(
            model_version=self.model,
            partial=False,
            content=genai_types.Content(role="model", parts=[genai_types.Part(text=final_text)]),
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=1800,
                candidates_token_count=75,
                thoughts_token_count=20,
            ),
        )


def test_end_to_end_adk_agent_runner_hits_bigquery_vertex_search_and_gcs() -> None:
    """Verify `FinancialAnalystAgent` and `load_cached_10k_section` hit BigQuery `VECTOR_SEARCH`, Vertex AI Search, embeddings, and GCS through the ADK Runner."""
    get_context_latency_tracker().reset()
    get_conversation_memory_manager().reset()
    _SESSION_10K_CONTEXT_CACHE.pop("NVDA_2026_item_1a", None)

    gcp_services_contacted: set[str] = set()
    q_vec = _mock_embedding_values("Blackwell Data Center export control risks")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content.decode("utf-8")) if request.content else {}

        if "bigquery.googleapis.com" in url:
            gcp_services_contacted.add("bigquery")
            sql = body["query"]
            if "financial_metrics" in sql:
                return httpx.Response(
                    200,
                    json={
                        "jobComplete": True,
                        "schema": {"fields": _bq_profile_schema()},
                        "rows": [
                            _encode_bq_profile_row(
                                ticker="NVDA",
                                fy=2024,
                                revenue=60_922_000_000.0,
                                operating_income=32_972_000_000.0,
                                net_income=29_760_000_000.0,
                            ),
                            _encode_bq_profile_row(
                                ticker="NVDA",
                                fy=2026,
                                revenue=215_938_000_000.0,
                                operating_income=130_387_000_000.0,
                                net_income=120_067_000_000.0,
                            ),
                        ],
                    },
                )
            if "VECTOR_SEARCH" in sql:
                return httpx.Response(
                    200,
                    json={
                        "jobComplete": True,
                        "schema": {
                            "fields": [
                                {"name": "chunk_id", "type": "STRING"},
                                {"name": "profile_id", "type": "STRING"},
                                {"name": "ticker", "type": "STRING"},
                                {"name": "company_name", "type": "STRING"},
                                {"name": "gics_sector", "type": "STRING"},
                                {"name": "fiscal_year", "type": "INTEGER"},
                                {"name": "section_id", "type": "STRING"},
                                {"name": "section_title", "type": "STRING"},
                                {"name": "chunk_index", "type": "INTEGER"},
                                {"name": "chunk_text", "type": "STRING"},
                                {"name": "sec_filing_url", "type": "STRING"},
                                {"name": "unstructured_gcs_uri", "type": "STRING"},
                                {"name": "distance", "type": "FLOAT"},
                            ]
                        },
                        "rows": [
                            {
                                "f": [
                                    {"v": "NVDA_2026_item_7_0001"},
                                    {"v": "NVDA_2026"},
                                    {"v": "NVDA"},
                                    {"v": "NVIDIA CORP"},
                                    {"v": "Information Technology"},
                                    {"v": "2026"},
                                    {"v": "item_7"},
                                    {"v": "Item 7. MD&A"},
                                    {"v": "1"},
                                    {
                                        "v": "Blackwell architecture ramped across cloud hyperscalers in FY2026."
                                    },
                                    {
                                        "v": "https://www.sec.gov/Archives/edgar/data/1045810/NVDA_2026.htm"
                                    },
                                    {
                                        "v": "gs://fde-sec-edgar-sandbox-dev-sec-corpus/unstructured/NVDA/2026_10k.md"
                                    },
                                    {"v": "0.08"},
                                ]
                            }
                        ],
                    },
                )

        if ":predict" in url and "text-embedding-005" in url:
            gcp_services_contacted.add("vertex_ai_embeddings")
            return httpx.Response(200, json={"predictions": [{"embeddings": {"values": q_vec}}]})

        if "discoveryengine.googleapis.com" in url:
            gcp_services_contacted.add("vertex_ai_search")
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "document": {
                                "id": "NVDA_2026_10K",
                                "structData": {
                                    "profile_id": "NVDA_2026",
                                    "ticker": "NVDA",
                                    "company_name": "NVIDIA CORP",
                                    "gics_sector": "Information Technology",
                                    "fiscal_year": 2026,
                                    "sec_filing_url": "https://www.sec.gov/Archives/edgar/data/1045810/NVDA_2026.htm",
                                },
                                "derivedStructData": {
                                    "extractive_segments": [
                                        {
                                            "content": "Export control licensing impacted China H20 shipments.",
                                            "relevanceScore": 0.91,
                                        }
                                    ]
                                },
                            }
                        }
                    ]
                },
            )

        if "storage.googleapis.com" in url:
            gcp_services_contacted.add("gcs")
            return httpx.Response(200, text=_sample_nvda_2026_markdown())

        return httpx.Response(404, text=f"Unhandled URL: {url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = GcpToolConfig()
    configure_gcp_tool_adapters(
        GcpToolAdapters(cfg, http_client=client, credentials=_FakeGcpCredentials())
    )

    analyst = create_financial_analyst_agent(model=_LiveToolCallingAnalystLlm())
    stream_res = run_adk_agent_stream(
        agent=analyst,
        prompt_text="Explain NVIDIA's FY2024 to FY2026 revenue variance and AI risks.",
        session_id="adk-live-gcp-tools-session",
        use_adk_app=False,
        backend="vertex_ai",
    )

    from edgar_capstone.agents.evaluation.structured_report import (
        parse_analyst_report,
        render_analyst_report_markdown,
        verify_analyst_report,
    )

    report = stream_res.session_state["financial_analyst_report"]
    assert report["headline"] == "NVDA Live GCP Grounded Analysis"
    assert report["findings"][0]["chunk_ids"][0] in {
        "NVDA_2026_item_7_0001",
        "NVDA_2026_item_7_vais_0000",
    }
    # Deterministic claim verification renders exact calculated figures from tool state
    parsed = parse_analyst_report(report)
    assert parsed is not None
    calc_reports = [
        v for v in stream_res.session_state["calculated_variances"].values() if isinstance(v, dict)
    ]
    verified = verify_analyst_report(
        parsed,
        variance_reports=calc_reports,
        citations=stream_res.session_state["retrieved_citations"],
    )
    assert verified.verification.usable is True
    rendered = render_analyst_report_markdown(
        verified,
        variance_reports=calc_reports,
        citations=stream_res.session_state["retrieved_citations"],
    )
    assert "+254.45%" in rendered
    assert {fc["name"] for fc in stream_res.function_calls} == {
        "calculate_period_variances",
        "search_10k_chunks_rag",
    }
    assert gcp_services_contacted == {
        "bigquery",
        "vertex_ai_embeddings",
        "vertex_ai_search",
    }

    # Also verify `load_cached_10k_section` hits BigQuery + GCS on a cache miss
    cache_res = load_cached_10k_section(
        ticker="NVDA", fiscal_year=2026, section_id="item_1a", max_chars=4000
    )
    assert cache_res["status"] == "success"
    assert "gcs" in gcp_services_contacted


def test_http_429_rate_limit_retries_and_missing_credentials_error() -> None:
    """Verify HTTP 429 Retry-After backoff recovery and that missing ADC is reported as an error."""
    attempts = 0

    def rate_limited_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0.01"},
                json={"error": {"code": 429, "message": "Quota exceeded"}},
            )
        return httpx.Response(
            200,
            json={
                "jobComplete": True,
                "schema": {"fields": _bq_profile_schema()},
                "rows": [
                    _encode_bq_profile_row(
                        ticker="NVDA",
                        fy=2026,
                        revenue=215_938_000_000.0,
                        operating_income=130_387_000_000.0,
                        net_income=120_067_000_000.0,
                    )
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(rate_limited_handler))
    configure_gcp_tool_adapters(
        GcpToolAdapters(
            GcpToolConfig(),
            http_client=client,
            credentials=_FakeGcpCredentials(),
        )
    )

    ctx = _dummy_tool_ctx()
    res = fetch_financial_metrics(
        tickers="NVDA",
        start_year=2026,
        end_year=2026,
        gics_sector="ALL",
        tool_context=ctx,
    )
    assert res["status"] == "success"
    assert attempts == 2
    assert ctx.state["tool_telemetry"][0]["attempts"] == 2

    # Missing ADC credentials surface as a tool error; there is no local data to fall back to
    configure_gcp_tool_adapters(
        GcpToolAdapters(
            GcpToolConfig(),
            http_client=client,
            credentials=_FailingCredentials(),
        )
    )
    attempts_before = attempts
    strict_err = fetch_financial_metrics(
        tickers="NVDA", start_year=2026, end_year=2026, gics_sector="ALL"
    )
    assert strict_err["status"] == "error"
    assert "DefaultCredentialsError" in strict_err["error_type"]
    assert attempts == attempts_before  # no request is sent without a bearer token


def test_bigquery_vector_search_sector_top_per_ticker() -> None:
    """Single-query BigQuery ML.DISTANCE + QUALIFY returns top chunks per issuer-year for sector map-reduce."""
    from edgar_capstone.agents.adapters.bigquery import BigQueryAdapter
    from edgar_capstone.agents.adapters.registry import GcpToolConfig

    captured_sql: list[str] = []
    captured_params: list[list[dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_sql.append(body["query"])
        captured_params.append(body.get("queryParameters") or [])
        return httpx.Response(
            200,
            json={
                "jobComplete": True,
                "schema": {
                    "fields": [
                        {"name": "chunk_id", "type": "STRING"},
                        {"name": "profile_id", "type": "STRING"},
                        {"name": "ticker", "type": "STRING"},
                        {"name": "company_name", "type": "STRING"},
                        {"name": "gics_sector", "type": "STRING"},
                        {"name": "fiscal_year", "type": "INTEGER"},
                        {"name": "section_id", "type": "STRING"},
                        {"name": "section_title", "type": "STRING"},
                        {"name": "chunk_index", "type": "INTEGER"},
                        {"name": "chunk_text", "type": "STRING"},
                        {"name": "sec_filing_url", "type": "STRING"},
                        {"name": "unstructured_gcs_uri", "type": "STRING"},
                        {"name": "period_of_report", "type": "STRING"},
                        {"name": "filing_date", "type": "STRING"},
                        {"name": "accession_number", "type": "STRING"},
                        {"name": "distance", "type": "FLOAT"},
                    ]
                },
                "rows": [
                    {
                        "f": [
                            {"v": "NVDA_2026_item_1a_0003"},
                            {"v": "NVDA_2026"},
                            {"v": "NVDA"},
                            {"v": "NVIDIA CORP"},
                            {"v": "Information Technology"},
                            {"v": "2026"},
                            {"v": "item_1a"},
                            {"v": "Item 1A. Risk Factors"},
                            {"v": "3"},
                            {"v": "Generative AI and export control risks expanded in FY2026."},
                            {"v": "https://www.sec.gov/nvda2026.htm"},
                            {"v": "gs://bucket/nvda/2026.md"},
                            {"v": "2026-01-25"},
                            {"v": "2026-02-25"},
                            {"v": "0001045810-26-000021"},
                            {"v": "0.14"},
                        ]
                    },
                    {
                        "f": [
                            {"v": "AAPL_2025_item_1a_0001"},
                            {"v": "AAPL_2025"},
                            {"v": "AAPL"},
                            {"v": "Apple Inc."},
                            {"v": "Information Technology"},
                            {"v": "2025"},
                            {"v": "item_1a"},
                            {"v": "Item 1A. Risk Factors"},
                            {"v": "1"},
                            {
                                "v": "Generative AI features introduce regulatory and reputational risk."
                            },
                            {"v": "https://www.sec.gov/aapl2025.htm"},
                            {"v": "gs://bucket/aapl/2025.md"},
                            {"v": "2025-09-27"},
                            {"v": "2025-10-31"},
                            {"v": "0000320193-25-000079"},
                            {"v": "0.18"},
                        ]
                    },
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bq = BigQueryAdapter(
        GcpToolConfig(project_id="iron-radio-514217-q7"),
        http_client=client,
        credentials=_FakeGcpCredentials(),
    )
    rows = bq.vector_search_sector_top_per_ticker(
        query_embedding=[0.1] * 8,
        gics_sector="information technology",
        section_id="item_1a",
        start_year=2024,
        end_year=2026,
        per_company_year=2,
    )
    assert len(captured_sql) == 1
    assert "ML.DISTANCE" in captured_sql[0]
    assert "QUALIFY ROW_NUMBER() OVER (" in captured_sql[0]
    assert "PARTITION BY ticker, fiscal_year" in captured_sql[0]
    assert len(rows) == 2
    assert rows[0]["ticker"] == "NVDA"
    assert rows[0]["period_of_report"] == "2026-01-25"
    assert rows[0]["accession_number"] == "0001045810-26-000021"
    assert rows[1]["ticker"] == "AAPL"


def test_vertex_ai_search_passage_extraction_fallback_and_errors() -> None:
    """Covers `extractive_answers`, HTML-stripped `snippets`, `chunk_text`, `doc_id` parsing, filters, and errors."""
    from edgar_capstone.agents.adapters.registry import (
        GcpAdapterHttpError,
        GcpEndpointNotReadyError,
    )
    from edgar_capstone.agents.adapters.vertex_search import VertexAISearchAdapter

    # 1. No engine_id or datastore_id configured -> raises GcpEndpointNotReadyError
    empty_cfg = GcpToolConfig(
        project_id="iron-radio-514217-q7", vais_engine_id="", vais_datastore_id=""
    )
    empty_adapter = VertexAISearchAdapter(empty_cfg, credentials=_FakeGcpCredentials())
    with pytest.raises(GcpEndpointNotReadyError):
        empty_adapter.search_documents(
            "AI risks",
            tickers=set(),
            gics_sector=None,
            section_id=None,
            start_year=2024,
            end_year=2026,
        )

    # 2. 404 fallback across engine -> datastore -> non-404 / exhausted 404 raises GcpAdapterHttpError
    def _all_404(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "not found"}})

    fail_adapter = VertexAISearchAdapter(
        GcpToolConfig(
            project_id="iron-radio-514217-q7",
            vais_engine_id="eng-missing",
            vais_datastore_id="ds-missing",
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(_all_404)),
        credentials=_FakeGcpCredentials(),
    )
    with pytest.raises(GcpAdapterHttpError):
        fail_adapter.search_documents(
            "AI",
            tickers=set(),
            gics_sector=None,
            section_id=None,
            start_year=2024,
            end_year=2026,
        )

    # 3. Passage extraction: doc_id fallback, extractive_answers, HTML snippet stripping, chunk_text, & filters
    def _mixed_results(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    # Missing both ticker/fy and unparseable id -> skipped
                    {"document": {"id": "malformed", "structData": {}, "derivedStructData": {}}},
                    # Out of year range -> skipped
                    {
                        "document": {
                            "id": "NVDA_2019",
                            "structData": {"ticker": "NVDA", "fiscal_year": 2019},
                        }
                    },
                    # Wrong ticker -> skipped
                    {
                        "document": {
                            "id": "AAPL_2025",
                            "structData": {"ticker": "AAPL", "fiscal_year": 2025},
                        }
                    },
                    # Wrong sector -> skipped
                    {
                        "document": {
                            "id": "NVDA_2025",
                            "structData": {
                                "ticker": "NVDA",
                                "fiscal_year": 2025,
                                "gics_sector": "Energy",
                            },
                        }
                    },
                    # Parsed via doc_id fallback (`NVDA_2026_10K`) with answers + HTML snippet + chunk_text
                    {
                        "document": {
                            "id": "NVDA_2026_10K",
                            "structData": {
                                "gics_sector": "Information Technology",
                                "chunk_text": "Item 1A Risk factors include foundry supply concentration.",
                            },
                            "derivedStructData": {
                                "extractive_answers": [
                                    {
                                        "content": "Export control risks restrict China AI shipments.",
                                        "section_id": "item_1a",
                                    }
                                ],
                                "snippets": [
                                    {
                                        "snippet": "<b>Regulatory</b> risks and <em>export</em> litigation adversely affect demand."
                                    }
                                ],
                            },
                        }
                    },
                ]
            },
        )

    ok_adapter = VertexAISearchAdapter(
        GcpToolConfig(project_id="iron-radio-514217-q7"),
        http_client=httpx.Client(transport=httpx.MockTransport(_mixed_results)),
        credentials=_FakeGcpCredentials(),
    )
    passages = ok_adapter.search_documents(
        "export risks",
        tickers={"NVDA"},
        gics_sector="Information Technology",
        section_id="item_1a",
        start_year=2024,
        end_year=2026,
    )
    assert len(passages) == 3
    texts = [p["chunk_text"] for p in passages]
    assert any("Export control risks" in t for t in texts)
    assert any("Regulatory risks and export litigation" in t and "<b>" not in t for t in texts)
    assert any("foundry supply concentration" in t for t in texts)
    assert all(p["ticker"] == "NVDA" and p["fiscal_year"] == 2026 for p in passages)


def test_gcp_tool_adapters_embed_texts_error_translation_and_context_manager() -> None:
    """Covers `embed_texts` empty list, 429/503/timeout/runtime error translation, and context manager."""
    from unittest.mock import MagicMock

    from edgar_capstone.agents.core.resilience import (
        RateLimitExceededError,
        ToolExecutionError,
        UpstreamTimeoutError,
    )

    fake_embedder = MagicMock()
    with GcpToolAdapters(
        GcpToolConfig(project_id="iron-radio-514217-q7"),
        embedder=fake_embedder,
        credentials=_FakeGcpCredentials(),
    ) as adapters:
        assert adapters.embed_texts([]) == []

        fake_embedder.embed_texts.side_effect = httpx.ReadTimeout("timed out")
        with pytest.raises(UpstreamTimeoutError):
            adapters.embed_texts(["q1"])

        fake_embedder.embed_texts.side_effect = RuntimeError("Vertex HTTP 429 quota exceeded")
        with pytest.raises(RateLimitExceededError):
            adapters.embed_texts(["q2"])

        fake_embedder.embed_texts.side_effect = RuntimeError("Vertex HTTP 503 unavailable")
        with pytest.raises(UpstreamTimeoutError):
            adapters.embed_texts(["q3"])

        fake_embedder.embed_texts.side_effect = RuntimeError("Vertex HTTP 400 invalid argument")
        with pytest.raises(ToolExecutionError):
            adapters.embed_texts(["q4"])

    fake_embedder.close.assert_called_once()
