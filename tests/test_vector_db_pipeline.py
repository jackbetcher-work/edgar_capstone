"""Automated unit tests for the metadata-tagged RAG vector DB and chunk health monitoring."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from edgar_capstone.dataset.extractor import format_combined_10k_markdown
from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.vector_db.chunker import (
    chunk_filing_document,
    estimate_token_count,
    extract_sections_from_combined_markdown,
    split_section_into_passages,
)
from edgar_capstone.vector_db.embedder import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    VertexAIEmbedder,
    generate_deterministic_embedding,
    populate_chunk_embeddings,
)
from edgar_capstone.vector_db.models import FilingChunk
from edgar_capstone.vector_db.monitoring import (
    CloudMonitoringHealthPublisher,
    build_cloud_monitoring_timeseries,
    evaluate_chunk_health,
    format_structured_health_log,
)
from edgar_capstone.vector_db.pipeline import (
    VectorDBBuildConfig,
    run_vector_db_pipeline,
)
from scripts.build_vector_db import parse_args


def _sample_profile(
    ticker: str = "NVDA",
    fiscal_year: int = 2026,
    gics_sector: str = "Information Technology",
) -> FinancialDataProfile:
    return FinancialDataProfile(
        profile_id=f"{ticker}_{fiscal_year}",
        ticker=ticker,
        cik="0001045810",
        company_name="NVIDIA CORP",
        gics_sector=gics_sector,
        gics_sub_industry="Semiconductors",
        form_type="10-K",
        fiscal_year=fiscal_year,
        period_of_report=f"{fiscal_year}-01-25",
        filing_date=f"{fiscal_year}-02-25",
        accession_number=f"0001045810-{str(fiscal_year)[-2:]}-000021",
        revenue=215938000000.0,
        operating_income=130497000000.0,
        net_income=115200000000.0,
        extracted_sections=["item_1a", "item_7"],
        unstructured_blob_path=f"unstructured/{ticker}/{fiscal_year}_10k.md",
        structured_blob_path=f"structured/{ticker}/{fiscal_year}_financials.json",
        unstructured_gcs_uri=f"gs://test-bucket/unstructured/{ticker}/{fiscal_year}_10k.md",
        structured_gcs_uri=f"gs://test-bucket/structured/{ticker}/{fiscal_year}_financials.json",
        sec_filing_url="https://www.sec.gov/Archives/edgar/data/1045810/nvda-20260125.htm",
    )


def _sample_markdown(
    profile: FinancialDataProfile,
    *,
    include_item_1a: bool = True,
    include_item_7: bool = True,
) -> str:
    item_1a = (
        "**Risks Related to Our Industry and Markets**\n\n"
        "Export controls targeting advanced AI semiconductors (including H20 and H200 GPUs) "
        "in China and D:5 countries have materially impacted our Data Center revenue and "
        "resulted in inventory charges during fiscal year 2026.\n\n"
        "**Supply Chain & Foundry Concentration**\n\n"
        "We rely on TSMC and CoWoS advanced packaging capacity in Taiwan and South Korea. "
        "Long manufacturing lead times and substrate constraints can cause supply-demand mismatches."
        if include_item_1a
        else None
    )
    item_7 = (
        "**Fiscal Year 2026 Overview**\n\n"
        "Data Center revenue increased 65% year-over-year driven by Blackwell GB200 and GB300 "
        "AI supercomputing infrastructure deployments across cloud service providers.\n\n"
        "**Results of Operations Table**\n\n"
        "| Segment | FY2026 ($M) | FY2025 ($M) | YoY % |\n"
        "| --- | --- | --- | --- |\n"
        "| Data Center | $195,000 | $115,000 | +70% |\n"
        "| Gaming | $12,500 | $10,400 | +20% |\n"
        if include_item_7
        else None
    )
    return format_combined_10k_markdown(
        ticker=profile.ticker,
        company_name=profile.company_name,
        cik=profile.cik,
        gics_sector=profile.gics_sector,
        gics_sub_industry=profile.gics_sub_industry,
        fiscal_year=profile.fiscal_year,
        period_of_report=profile.period_of_report,
        filing_date=profile.filing_date,
        accession_number=profile.accession_number,
        sec_filing_url=profile.sec_filing_url,
        item_1a_markdown=item_1a,
        item_7_markdown=item_7,
    )


def test_section_extraction_and_heading_breadcrumbs() -> None:
    prof = _sample_profile()
    md = _sample_markdown(prof)

    sections = extract_sections_from_combined_markdown(md)
    assert set(sections.keys()) == {"item_1a", "item_7"}
    assert "Export controls targeting advanced AI" in sections["item_1a"]
    assert "| Data Center | $195,000 |" in sections["item_7"]

    chunks = chunk_filing_document(
        prof,
        md,
        target_chars=300,
        max_chars=800,
        min_chars=80,
        overlap_chars=50,
    )
    assert len(chunks) >= 4
    sec_ids = {c.section_id for c in chunks}
    assert sec_ids == {"item_1a", "item_7"}

    # Check subsection heading breadcrumbs on chunks
    titles = [c.section_title for c in chunks]
    assert any("Risks Related to Our Industry and Markets" in t for t in titles)
    assert any("Fiscal Year 2026 Overview" in t or "Results of Operations Table" in t for t in titles)

    # Check metadata & schema alignment with terraform/schemas/filing_chunks.json
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "terraform"
        / "schemas"
        / "filing_chunks.json"
    )
    bq_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    expected_cols = {col["name"] for col in bq_schema}
    assert set(chunks[0].to_bigquery_record().keys()) == expected_cols


def test_missing_section_placeholder_is_skipped() -> None:
    prof = _sample_profile()
    md_missing_1a = _sample_markdown(prof, include_item_1a=False, include_item_7=True)
    sections = extract_sections_from_combined_markdown(md_missing_1a)
    assert "item_1a" not in sections
    assert "item_7" in sections


def test_vertex_vector_search_record_format() -> None:
    prof = _sample_profile()
    md = _sample_markdown(prof)
    chunks = chunk_filing_document(prof, md)
    chunks[0].embedding = generate_deterministic_embedding(chunks[0].chunk_text)

    vx_rec = chunks[0].to_vertex_vector_search_record()
    assert vx_rec["id"] == chunks[0].chunk_id
    assert len(vx_rec["embedding"]) == DEFAULT_EMBEDDING_DIMENSIONS
    assert vx_rec["crowding_tag"] == "NVDA"

    restrict_map = {r["namespace"]: r["allow"] for r in vx_rec["restricts"]}
    assert restrict_map["ticker"] == ["NVDA"]
    assert restrict_map["gics_sector"] == ["Information Technology"]
    assert restrict_map["section_id"] == ["item_1a"]
    assert restrict_map["profile_id"] == ["NVDA_2026"]

    num_restricts = {r["namespace"]: r["value_int"] for r in vx_rec["numeric_restricts"]}
    assert num_restricts["fiscal_year"] == 2026


def test_deterministic_and_vertex_ai_embeddings() -> None:
    v1 = generate_deterministic_embedding(
        "AI GPU export control restrictions in China impacted Data Center revenue."
    )
    v2 = generate_deterministic_embedding(
        "US export control restrictions on China AI GPU chips reduced Data Center sales."
    )
    v3 = generate_deterministic_embedding(
        "Retail store lease obligations and consumer packaged soap pricing."
    )

    assert len(v1) == DEFAULT_EMBEDDING_DIMENSIONS
    norm_v1 = math.sqrt(sum(x * x for x in v1))
    assert math.isclose(norm_v1, 1.0, rel_tol=1e-3)

    sim_related = float(np.dot(v1, v2))
    sim_unrelated = float(np.dot(v1, v3))
    assert sim_related > sim_unrelated
    assert sim_related > 0.25

    # Test VertexAIEmbedder with MockTransport
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        n = len(body["instances"])
        preds = [
            {"embeddings": {"values": [0.1] * DEFAULT_EMBEDDING_DIMENSIONS}}
            for _ in range(n)
        ]
        return httpx.Response(200, json={"predictions": preds})

    mock_creds = MagicMock()
    mock_creds.valid = True
    mock_creds.token = "fake-oauth-token"

    embedder = VertexAIEmbedder(
        project_id="iron-radio-514217-q7",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        credentials=mock_creds,
    )
    prof = _sample_profile()
    chunks = chunk_filing_document(prof, _sample_markdown(prof))
    populated, backend = populate_chunk_embeddings(
        chunks,
        project_id="iron-radio-514217-q7",
        backend="vertex_ai",
        vertex_embedder=embedder,
    )
    assert backend == "vertex_ai"
    assert all(len(c.embedding) == DEFAULT_EMBEDDING_DIMENSIONS for c in populated)


def test_chunk_health_evaluation_and_cloud_monitoring_export() -> None:
    prof_nvda = _sample_profile("NVDA", 2026)
    prof_aapl = _sample_profile("AAPL", 2026)

    chunks_nvda = chunk_filing_document(prof_nvda, _sample_markdown(prof_nvda))
    chunks_aapl = chunk_filing_document(prof_aapl, _sample_markdown(prof_aapl))
    all_chunks = chunks_nvda + chunks_aapl
    populate_chunk_embeddings(
        all_chunks,
        project_id="iron-radio-514217-q7",
        backend="deterministic",
    )

    healthy_report = evaluate_chunk_health(
        all_chunks,
        profiles=[prof_nvda, prof_aapl],
    )
    assert healthy_report.health_status == "HEALTHY"
    assert healthy_report.health_score == 100.0
    assert healthy_report.embedding_coverage_ratio == 1.0
    assert healthy_report.full_section_coverage_ratio == 1.0
    assert healthy_report.anomalies.total_anomalies == 0

    # Inject synthetic anomalies (missing embedding, wrong dim, undersized)
    degraded_chunks = [c.model_copy(deep=True) for c in all_chunks]
    degraded_chunks[0].embedding = []
    if len(degraded_chunks) > 1:
        degraded_chunks[1].embedding = [0.5, 0.5]  # wrong dimension
        degraded_chunks[1].char_count = 20  # undersized
        degraded_chunks[1].gics_sector = "Unknown"  # missing metadata

    prof_unindexed = _sample_profile("MSFT", 2026)
    anomaly_report = evaluate_chunk_health(
        degraded_chunks,
        profiles=[prof_nvda, prof_aapl, prof_unindexed],
    )
    assert anomaly_report.anomalies.missing_embedding == 1
    assert anomaly_report.anomalies.dimension_mismatch == 1
    assert anomaly_report.anomalies.undersized_chunk == 1
    assert anomaly_report.anomalies.missing_metadata == 1
    assert anomaly_report.anomalies.missing_section_filings == 1
    assert anomaly_report.missing_profile_ids == ["MSFT_2026"]
    assert anomaly_report.health_score < 90.0

    # Verify Cloud Monitoring TimeSeries generation & publishing
    ts_list = build_cloud_monitoring_timeseries(healthy_report)
    metric_types = {ts["metric"]["type"] for ts in ts_list}
    assert "custom.googleapis.com/sec_edgar/chunk_health/health_score" in metric_types
    assert "custom.googleapis.com/sec_edgar/chunk_health/embedding_coverage_ratio" in metric_types
    assert "custom.googleapis.com/sec_edgar/chunk_health/filing_coverage_ratio" in metric_types
    assert "custom.googleapis.com/sec_edgar/chunk_health/total_chunks" in metric_types
    assert "custom.googleapis.com/sec_edgar/chunk_health/token_count_mean" in metric_types
    assert "custom.googleapis.com/sec_edgar/chunk_health/anomaly_chunk_count" in metric_types

    structured_log = format_structured_health_log(healthy_report)
    assert structured_log["event_type"] == "sec_edgar_chunk_health_evaluation"
    assert structured_log["severity"] == "INFO"

    captured_requests: list[dict] = []

    def monitoring_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={})

    mock_creds = MagicMock()
    mock_creds.valid = True
    mock_creds.token = "fake-monitoring-token"

    publisher = CloudMonitoringHealthPublisher(
        project_id="iron-radio-514217-q7",
        http_client=httpx.Client(transport=httpx.MockTransport(monitoring_handler)),
        credentials=mock_creds,
    )
    written_count = publisher.publish_report(healthy_report)
    assert written_count == len(ts_list)
    assert healthy_report.published_to_cloud_monitoring is True
    assert len(captured_requests) == 1


def test_end_to_end_vector_db_pipeline_local(tmp_path: Path) -> None:
    prof = _sample_profile("NVDA", 2026)
    md = _sample_markdown(prof)

    md_path = tmp_path / prof.unstructured_blob_path
    json_path = tmp_path / prof.structured_blob_path
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(prof.model_dump(mode="json")), encoding="utf-8")

    cfg = VectorDBBuildConfig(
        output_dir=tmp_path,
        local_only=True,
        target_chunk_chars=400,
        max_chunk_chars=900,
        min_chunk_chars=80,
    )
    summary, report = run_vector_db_pipeline(cfg)

    assert summary.total_profiles_processed == 1
    assert summary.total_chunks_generated >= 2
    assert summary.health_status == "HEALTHY"
    assert report.health_score == 100.0

    bq_jsonl = Path(summary.bigquery_manifest_path)
    vx_jsonl = Path(summary.vertex_vector_manifest_path)
    health_json = Path(summary.health_report_path)

    assert bq_jsonl.exists()
    assert vx_jsonl.exists()
    assert health_json.exists()

    bq_lines = bq_jsonl.read_text(encoding="utf-8").strip().splitlines()
    vx_lines = vx_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(bq_lines) == summary.total_chunks_generated
    assert len(vx_lines) == summary.total_chunks_generated

    # Second run should reuse cached embeddings
    summary_cached, _ = run_vector_db_pipeline(cfg)
    assert summary_cached.embedding_backend == "cached"


def test_cli_parse_args_and_terraform_monitoring_definitions() -> None:
    args = parse_args(["--tickers", "NVDA", "AAPL", "--local-only", "--target-chars", "1800"])
    assert args.tickers == ["NVDA", "AAPL"]
    assert args.local_only is True
    assert args.target_chars == 1800
    assert estimate_token_count("NVIDIA Blackwell data center revenue grew significantly.") > 5

    tf_dir = Path(__file__).resolve().parents[1] / "terraform"
    monitoring_tf = (tf_dir / "monitoring.tf").read_text(encoding="utf-8")
    bigquery_tf = (tf_dir / "bigquery.tf").read_text(encoding="utf-8")
    vector_tf = (tf_dir / "vector_search.tf").read_text(encoding="utf-8")

    assert 'resource "google_monitoring_dashboard" "chunk_health_dashboard"' in monitoring_tf
    assert 'resource "google_bigquery_table" "v_chunk_health_metrics"' in bigquery_tf
    assert 'resource "google_bigquery_table" "v_filing_chunk_coverage"' in bigquery_tf
    assert 'resource "google_vertex_ai_index" "sec_10k_chunks"' in vector_tf
    assert 'resource "google_discovery_engine_data_store" "sec_10k_corpus"' in vector_tf
