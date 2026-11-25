"""Comprehensive tests for live GCP SDK tool adapters:
BigQuery (`google-cloud-bigquery`), Vertex AI Search (`google-cloud-discoveryengine`),
Vertex AI Embeddings (`google-genai`), and Cloud Storage (`google-cloud-storage`).

Uses deterministic typed SDK fakes (no network traffic) to exercise the exact production
adapter code paths, error translation, hybrid RAG fusion, context caching, and ADK tools.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from google.api_core import exceptions as gcp_exceptions
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import discoveryengine_v1 as de
from google.genai import errors as genai_errors
from google.protobuf.struct_pb2 import Struct

from edgar_capstone.agents.adapters.bigquery import BigQueryAdapter
from edgar_capstone.agents.adapters.gcs import GcsCorpusAdapter
from edgar_capstone.agents.adapters.registry import (
    GcpAdapterHttpError,
    GcpEndpointNotReadyError,
    GcpToolAdapters,
    GcpToolConfig,
    configure_gcp_tool_adapters,
    get_gcp_adapters_status_metadata,
)
from edgar_capstone.agents.adapters.vertex_search import VertexAISearchAdapter
from edgar_capstone.agents.core.resilience import (
    RateLimitExceededError,
    ToolExecutionError,
    UpstreamTimeoutError,
)
from edgar_capstone.agents.tools import (
    calculate_period_variances,
    load_cached_10k_section,
    query_analytics_view,
    search_10k_chunks_rag,
)
from edgar_capstone.gcp.errors import GcpHttpError, GcpRateLimitError, GcpTimeoutError
from tests.fake_gcp import (
    FakeGcpCredentials,
    create_fake_gcp_adapters,
    preserve_gcp_caches,
)


def _struct(data: dict[str, Any]) -> Struct:
    st = Struct()
    st.update(data)
    return st


@pytest.fixture(autouse=True)
def _restore_gcp_adapters_after_test():
    """Ensure injected GCP adapters never leak across tests without wiping warm live caches."""
    with preserve_gcp_caches():
        yield


def test_gcp_tool_config_and_status_metadata_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GcpToolConfig.from_env()` reads resource env vars; status metadata is always `gcp`."""
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

    configure_gcp_tool_adapters(
        GcpToolAdapters(
            GcpToolConfig(project_id="injected-project", bq_dataset_id="injected_ds"),
            credentials=FakeGcpCredentials(),
        )
    )
    injected_meta = get_gcp_adapters_status_metadata()
    assert injected_meta["data_backend"] == "gcp"
    assert injected_meta["project_id"] == "injected-project"
    assert injected_meta["bigquery"]["dataset_id"] == "injected_ds"


def test_bigquery_adapter_metrics_variances_and_type_normalization() -> None:
    """Verify parameterized BigQuery SDK queries, TIMESTAMP/DATE normalization, and variance math."""
    captured_queries: list[tuple[str, list[Any]]] = []

    class _FakeJob:
        def result(self, **_kw: Any) -> list[dict[str, Any]]:
            return [
                {
                    "profile_id": "NVDA_2024",
                    "ticker": "NVDA",
                    "cik": "0001045810",
                    "company_name": "NVIDIA CORP",
                    "gics_sector": "Information Technology",
                    "gics_sub_industry": "Semiconductors",
                    "fiscal_year": 2024,
                    "period_of_report": date(2024, 1, 28),
                    "filing_date": date(2024, 2, 21),
                    "accession_number": "0001045810-24-000029",
                    "currency": "USD",
                    "revenue": 60922000000.0,
                    "operating_income": 32972000000.0,
                    "net_income": 29760000000.0,
                    "research_and_development": 8675000000.0,
                    "metric_provenance": {
                        "revenue": "us-gaap:Revenues",
                        "operating_income": "us-gaap:OperatingIncomeLoss",
                        "net_income": "us-gaap:NetIncomeLoss",
                        "research_and_development": "us-gaap:ResearchAndDevelopmentExpense",
                        "notes": [],
                    },
                    "extracted_sections": ["item_1a", "item_7"],
                    "unstructured_blob_path": "unstructured/NVDA/2024_10k.md",
                    "structured_blob_path": "structured/NVDA/2024_financials.json",
                    "unstructured_gcs_uri": "gs://bucket/unstructured/NVDA/2024_10k.md",
                    "structured_gcs_uri": "gs://bucket/structured/NVDA/2024_financials.json",
                    "sec_filing_url": "https://www.sec.gov/Archives/edgar/data/1045810/2024.htm",
                    "ingested_at": datetime(2024, 3, 1, 12, 0, 0, tzinfo=UTC),
                },
                {
                    "profile_id": "NVDA_2026",
                    "ticker": "NVDA",
                    "cik": "0001045810",
                    "company_name": "NVIDIA CORP",
                    "gics_sector": "Information Technology",
                    "gics_sub_industry": "Semiconductors",
                    "fiscal_year": 2026,
                    "period_of_report": date(2026, 1, 25),
                    "filing_date": date(2026, 2, 25),
                    "accession_number": "0001045810-26-000021",
                    "currency": "USD",
                    "revenue": 215938000000.0,
                    "operating_income": 130387000000.0,
                    "net_income": 120067000000.0,
                    "research_and_development": 18497000000.0,
                    "metric_provenance": {
                        "revenue": "us-gaap:Revenues",
                        "operating_income": "us-gaap:OperatingIncomeLoss",
                        "net_income": "us-gaap:NetIncomeLoss",
                        "research_and_development": "us-gaap:ResearchAndDevelopmentExpense",
                        "notes": [],
                    },
                    "extracted_sections": ["item_1a", "item_7"],
                    "unstructured_blob_path": "unstructured/NVDA/2026_10k.md",
                    "structured_blob_path": "structured/NVDA/2026_financials.json",
                    "unstructured_gcs_uri": "gs://bucket/unstructured/NVDA/2026_10k.md",
                    "structured_gcs_uri": "gs://bucket/structured/NVDA/2026_financials.json",
                    "sec_filing_url": "https://www.sec.gov/Archives/edgar/data/1045810/2026.htm",
                    "ingested_at": datetime(2026, 2, 26, 8, 30, 0, tzinfo=UTC),
                },
            ]

    class _FakeBqClient:
        def query(self, sql: str, *, job_config: Any = None, **_kw: Any) -> _FakeJob:
            captured_queries.append((sql, list(getattr(job_config, "query_parameters", ()) or ())))
            return _FakeJob()

    cfg = GcpToolConfig(project_id="iron-radio-514217-q7", bq_dataset_id="sec_edgar")
    configure_gcp_tool_adapters(GcpToolAdapters(cfg, bq_client=_FakeBqClient()))

    res = calculate_period_variances(tickers="NVDA", start_year=2024, end_year=2026)
    assert res["status"] == "success"
    assert res["company_count"] == 1
    assert len(captured_queries) == 1
    sql, params = captured_queries[0]
    assert "`iron-radio-514217-q7.sec_edgar.financial_metrics`" in sql
    param_names = {p.name for p in params}
    assert {"start_year", "end_year", "tickers"} <= param_names

    # Verify normalized ISO-8601 TIMESTAMP (not scientific notation epoch!) and variance math
    profiles = BigQueryAdapter(cfg, bq_client=_FakeBqClient()).fetch_profiles(
        tickers={"NVDA"}, start_year=2024, end_year=2026
    )
    assert profiles[0].ingested_at == "2024-03-01T12:00:00Z"
    assert profiles[1].filing_date == "2026-02-25"
    nvda_rev = res["variance_reports"][0]["revenue_variance"]
    assert nvda_rev["percentage_delta"] == pytest.approx(254.45, abs=0.05)


def test_bigquery_adapter_all_analytics_views() -> None:
    """Verify `query_analytics_view` across YoY variances, sector benchmarks, and chunk health."""
    adapters, _log = create_fake_gcp_adapters()
    configure_gcp_tool_adapters(adapters)

    yoy = query_analytics_view(
        explore_name="yoy_variances",
        tickers="NVDA",
        gics_sector="ALL",
        start_year=2024,
        end_year=2026,
    )
    assert yoy["status"] == "success"
    assert yoy["row_count"] >= 2
    assert any(r["fiscal_year"] == 2026 and r["revenue_yoy_pct"] is not None for r in yoy["rows"])

    bench = query_analytics_view(
        explore_name="sector_benchmarks",
        tickers="ALL",
        gics_sector="Information Technology",
        start_year=2024,
        end_year=2026,
    )
    assert bench["status"] == "success"
    assert bench["row_count"] >= 2
    assert all(r["gics_sector"] == "Information Technology" for r in bench["rows"])

    health = query_analytics_view(
        explore_name="chunk_health",
        tickers="ALL",
        gics_sector="ALL",
        start_year=2024,
        end_year=2026,
    )
    assert health["status"] == "success"
    assert health["health_status"] == "HEALTHY"
    assert health["total_chunks"] > 0


def test_hybrid_rag_fuses_bigquery_vector_search_and_vertex_ai_search() -> None:
    """Verify `search_10k_chunks_rag` fuses BigQuery `VECTOR_SEARCH` and Discovery Engine passages."""
    adapters, log = create_fake_gcp_adapters()
    configure_gcp_tool_adapters(adapters)

    res = search_10k_chunks_rag(
        query="China export control license requirements on Hopper and Blackwell GPUs",
        tickers="NVDA",
        gics_sector="ALL",
        section_id="item_1a",
        start_year=2024,
        end_year=2026,
        top_k=6,
    )
    assert res["status"] == "success"
    assert res["citation_count"] > 0
    assert any("genai:embed_content" in target for _, target in log)
    assert any("bigquery:query" in target for _, target in log)
    assert any("discoveryengine:search" in target for _, target in log)
    assert any("export control" in str(r.get("excerpt", "")).lower() for r in res["citations"])


def test_context_cache_miss_loads_from_bigquery_and_gcs_and_queries_cached_section() -> None:
    """Verify `load_cached_10k_section` hydrates from BigQuery & GCS corpus markdown."""
    adapters, log = create_fake_gcp_adapters()
    configure_gcp_tool_adapters(adapters)

    res = load_cached_10k_section(
        ticker="NVDA",
        fiscal_year=2026,
        section_id="item_7",
        max_chars=12000,
    )
    assert res["status"] == "success"
    assert res["ticker"] == "NVDA"
    assert res["fiscal_year"] == 2026
    assert res["content_source"] == "gcs"
    assert "Blackwell" in res["content"]
    assert any("bigquery:query" in target for _, target in log)
    assert any(target.startswith("gcs://") for _, target in log)

    # Also verify GcsCorpusAdapter directly reads markdown and translates NotFound -> GcpAdapterHttpError
    md = adapters.gcs_corpus.fetch_markdown("unstructured/NVDA/2026_10k.md")
    assert "Blackwell" in md
    with pytest.raises(GcpAdapterHttpError) as exc_404:
        adapters.gcs_corpus.fetch_markdown("unstructured/NOPE/2099_10k.md")
    assert exc_404.value.status_code == 404


def test_end_to_end_adk_agent_runner_hits_bigquery_vertex_search_and_gcs() -> None:
    """Verify a full conversational RAG turn routes through the typed SDK adapters."""
    from edgar_capstone.agents.orchestration.chat_engine import run_conversational_rag_turn
    from tests.structured_report_helpers import ScriptedAnalystLlm

    adapters, log = create_fake_gcp_adapters()
    configure_gcp_tool_adapters(adapters)

    llm = ScriptedAnalystLlm(
        tool_calls=[
            (
                "calculate_period_variances",
                {"tickers": "NVDA", "start_year": 2024, "end_year": 2026},
            ),
            (
                "search_10k_chunks_rag",
                {
                    "query": "export controls Blackwell GPU",
                    "tickers": "NVDA",
                    "gics_sector": "ALL",
                    "section_id": "item_1a",
                    "start_year": 2024,
                    "end_year": 2026,
                    "top_k": 5,
                },
            ),
        ],
        headline="NVDA revenue expanded +254.45% from FY2024 to FY2026 alongside export control risks.",
        intent="longitudinal_single_company",
        key_metrics=[("NVDA", "revenue")],
    )
    turn = run_conversational_rag_turn(
        query="Analyze NVIDIA revenue growth and export control risks from FY2024 to FY2026",
        tickers="NVDA",
        start_year=2024,
        end_year=2026,
        reasoning_llm=llm,
        skip_background_judge=True,
    )
    assert turn["status"] == "success"
    assert any("bigquery:query" in target for _, target in log)
    assert any("discoveryengine:search" in target for _, target in log)


def test_http_429_rate_limit_retries_and_missing_credentials_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 429 TooManyRequests retries with backoff and DefaultCredentialsError propagates."""
    monkeypatch.setattr("edgar_capstone.agents.core.resilience.time.sleep", lambda _s: None)
    attempts = {"n": 0}

    class _FlakyBq:
        def query(self, sql: str, **_kw: Any) -> Any:
            _ = sql
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise gcp_exceptions.TooManyRequests(
                    "quota", response=SimpleNamespace(status_code=429, headers={"Retry-After": "0"})
                )
            return SimpleNamespace(result=lambda **_rkw: [])

    cfg = GcpToolConfig(project_id="iron-radio-514217-q7")
    configure_gcp_tool_adapters(GcpToolAdapters(cfg, bq_client=_FlakyBq()))

    # with_tool_resilience retries the RateLimitExceededError and succeeds on attempt 2
    res = calculate_period_variances(tickers="NVDA", start_year=2024, end_year=2026)
    assert attempts["n"] == 2
    assert res["status"] == "success"
    assert res["company_count"] == 0
    assert res["missing_tickers"] == ["NVDA"]

    # Missing ADC propagates clearly from BigQueryAdapter
    class _MissingAdcBq:
        def query(self, sql: str, **_kw: Any) -> Any:
            _ = sql
            raise DefaultCredentialsError("ADC not configured")

    with pytest.raises(DefaultCredentialsError):
        BigQueryAdapter(cfg, bq_client=_MissingAdcBq()).run_query("SELECT 1")


def test_bigquery_vector_search_sector_top_per_ticker() -> None:
    """Verify `vector_search_sector_top_per_ticker` uses ML.DISTANCE + QUALIFY ROW_NUMBER."""
    seen_sql: list[str] = []

    class _SectorBq:
        def query(self, sql: str, **_kw: Any) -> Any:
            seen_sql.append(sql)
            return SimpleNamespace(
                result=lambda **_rkw: [
                    {
                        "chunk_id": "NVDA_2026_item_1a_0001",
                        "profile_id": "NVDA_2026",
                        "ticker": "NVDA",
                        "company_name": "NVIDIA CORP",
                        "gics_sector": "Information Technology",
                        "fiscal_year": 2026,
                        "period_of_report": date(2026, 1, 25),
                        "filing_date": date(2026, 2, 25),
                        "accession_number": "0001045810-26-000021",
                        "section_id": "item_1a",
                        "section_title": "Item 1A. Risk Factors",
                        "chunk_index": 1,
                        "chunk_text": "Export control risks.",
                        "sec_filing_url": "https://www.sec.gov/nvda-2026.htm",
                        "unstructured_gcs_uri": "gs://bucket/NVDA/2026.md",
                        "distance": 0.12,
                    },
                    {
                        "chunk_id": "AAPL_2026_item_1a_0001",
                        "profile_id": "AAPL_2026",
                        "ticker": "AAPL",
                        "company_name": "Apple Inc.",
                        "gics_sector": "Information Technology",
                        "fiscal_year": 2026,
                        "period_of_report": date(2026, 9, 26),
                        "filing_date": date(2026, 11, 1),
                        "accession_number": "0000320193-26-000011",
                        "section_id": "item_1a",
                        "section_title": "Item 1A. Risk Factors",
                        "chunk_index": 1,
                        "chunk_text": "Tariff risks.",
                        "sec_filing_url": "https://www.sec.gov/aapl-2026.htm",
                        "unstructured_gcs_uri": "gs://bucket/AAPL/2026.md",
                        "distance": 0.25,
                    },
                ]
            )

    adapter = BigQueryAdapter(
        GcpToolConfig(project_id="iron-radio-514217-q7"),
        bq_client=_SectorBq(),
    )
    rows = adapter.vector_search_sector_top_per_ticker(
        [0.1] * 8,
        gics_sector="Information Technology",
        start_year=2024,
        end_year=2026,
        section_id="item_1a",
        per_company_year=2,
    )
    assert len(seen_sql) == 1 and "ML.DISTANCE" in seen_sql[0]
    assert len(rows) == 2
    assert rows[0]["period_of_report"] == "2026-01-25"
    assert rows[0]["accession_number"] == "0001045810-26-000021"
    assert rows[1]["ticker"] == "AAPL"


def test_vertex_ai_search_passage_extraction_fallback_and_errors() -> None:
    """Covers filter escaping, engine->datastore 404 fallback, proto Struct parsing, HTML stripping, and errors."""
    # Filter escaping handles backslashes and quotes safely
    expr = VertexAISearchAdapter.build_filter_expression(
        tickers={'NV"DA', r"MS\FT"},
        gics_sector="Information Technology",
        section_id="item_1a",
        start_year=2024,
        end_year=2026,
    )
    assert r"\"" in expr and r"\\" in expr

    # 1. No engine_id or datastore_id configured -> raises GcpEndpointNotReadyError
    empty_cfg = GcpToolConfig(
        project_id="iron-radio-514217-q7", vais_engine_id="", vais_datastore_id=""
    )
    empty_adapter = VertexAISearchAdapter(empty_cfg, credentials=FakeGcpCredentials())
    with pytest.raises(GcpEndpointNotReadyError):
        empty_adapter.search_documents(
            "AI risks",
            tickers=set(),
            gics_sector=None,
            section_id=None,
            start_year=2024,
            end_year=2026,
        )

    # 2. 404 fallback across engine -> datastore -> exhausted raises GcpAdapterHttpError
    tried_configs: list[str] = []

    class _All404Search:
        def search(self, *, request: de.SearchRequest, timeout: float | None = None) -> Any:
            _ = timeout
            tried_configs.append(request.serving_config)
            raise gcp_exceptions.NotFound("servingConfig not found")

    fail_adapter = VertexAISearchAdapter(
        GcpToolConfig(
            project_id="iron-radio-514217-q7",
            vais_engine_id="eng-missing",
            vais_datastore_id="ds-missing",
        ),
        search_client=_All404Search(),
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
    assert len(tried_configs) == 2
    assert "/engines/eng-missing/" in tried_configs[0]
    assert "/dataStores/ds-missing/" in tried_configs[1]

    # 3. Passage extraction: doc_id fallback, extractive_answers, HTML snippet stripping, chunk_text
    class _MixedSearch:
        def search(
            self, *, request: de.SearchRequest, timeout: float | None = None
        ) -> de.SearchResponse:
            _ = (request, timeout)
            return de.SearchResponse(
                results=[
                    de.SearchResponse.SearchResult(document=de.Document(id="malformed")),
                    de.SearchResponse.SearchResult(
                        document=de.Document(
                            id="NVDA_2019",
                            struct_data=_struct({"ticker": "NVDA", "fiscal_year": 2019}),
                        )
                    ),
                    de.SearchResponse.SearchResult(
                        document=de.Document(
                            id="AAPL_2025",
                            struct_data=_struct({"ticker": "AAPL", "fiscal_year": 2025}),
                        )
                    ),
                    de.SearchResponse.SearchResult(
                        document=de.Document(
                            id="NVDA_2025",
                            struct_data=_struct(
                                {"ticker": "NVDA", "fiscal_year": 2025, "gics_sector": "Energy"}
                            ),
                        )
                    ),
                    de.SearchResponse.SearchResult(
                        document=de.Document(
                            id="NVDA_2026_10K",
                            struct_data=_struct(
                                {
                                    "gics_sector": "Information Technology",
                                    "chunk_text": "Item 1A Risk factors include foundry supply concentration.",
                                }
                            ),
                            derived_struct_data=_struct(
                                {
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
                                }
                            ),
                        )
                    ),
                ]
            )

    ok_adapter = VertexAISearchAdapter(
        GcpToolConfig(project_id="iron-radio-514217-q7"),
        search_client=_MixedSearch(),
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
    """Covers `embed_texts` empty list, typed SDK/APIError translation, and context manager."""
    fake_embedder = MagicMock()
    with GcpToolAdapters(
        GcpToolConfig(project_id="iron-radio-514217-q7"),
        embedder=fake_embedder,
        credentials=FakeGcpCredentials(),
    ) as adapters:
        assert adapters.embed_texts([]) == []

        fake_embedder.embed_texts.side_effect = GcpTimeoutError("timed out")
        with pytest.raises(UpstreamTimeoutError):
            adapters.embed_texts(["q1"])

        fake_embedder.embed_texts.side_effect = GcpRateLimitError("quota", retry_after_seconds=2.0)
        with pytest.raises(RateLimitExceededError) as exc_rl:
            adapters.embed_texts(["q2"])
        assert exc_rl.value.retry_after_seconds == 2.0

        fake_embedder.embed_texts.side_effect = genai_errors.ServerError(
            503, {"error": {"code": 503, "message": "unavailable"}}
        )
        with pytest.raises(UpstreamTimeoutError):
            adapters.embed_texts(["q3"])

        fake_embedder.embed_texts.side_effect = GcpHttpError(400, "embed", "invalid argument")
        with pytest.raises(ToolExecutionError):
            adapters.embed_texts(["q4"])

    fake_embedder.close.assert_called_once()


def test_gcs_corpus_adapter_uri_parsing() -> None:
    adapter = GcsCorpusAdapter(GcpToolConfig(sec_corpus_bucket="default-bkt"))
    assert adapter.parse_gcs_uri("gs://custom-bkt/unstructured/NVDA/2026_10k.md") == (
        "custom-bkt",
        "unstructured/NVDA/2026_10k.md",
    )
    assert adapter.parse_gcs_uri("/unstructured/NVDA/2026_10k.md") == (
        "default-bkt",
        "unstructured/NVDA/2026_10k.md",
    )
    with pytest.raises(ValueError, match="Invalid GCS URI"):
        adapter.parse_gcs_uri("gs://bucket-only")
