"""Tests for the metadata-tagged RAG vector DB, chunker, embedder, and chunk health monitoring.

One semantic-similarity test (`@pytest.mark.live`) calls live Vertex AI `text-embedding-005` with ADC.
All format, chunking, retry, checkpointing, health monitoring, and staged pipeline tests run offline
against typed `google-genai` fakes over the real `VertexAIEmbedder`.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from edgar_capstone.dataset.extractor import format_combined_10k_markdown
from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.vector_db.chunker import (
    chunk_filing_document,
    estimate_token_count,
    extract_sections_from_combined_markdown,
)
from edgar_capstone.vector_db.embedder import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    VertexAIEmbedder,
    populate_chunk_embeddings,
)
from edgar_capstone.vector_db.monitoring import (
    build_cloud_monitoring_timeseries,
    evaluate_chunk_health,
    format_structured_health_log,
)
from edgar_capstone.vector_db.pipeline import (
    VectorDBBuildConfig,
    run_vector_db_pipeline,
)
from scripts.build_vector_db import parse_args

LIVE_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "iron-radio-514217-q7")
LIVE_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")


@pytest.fixture(scope="module")
def live_embedder() -> VertexAIEmbedder:
    """Live Vertex AI `text-embedding-005` client authenticated with ADC."""
    return VertexAIEmbedder(project_id=LIVE_PROJECT_ID, location=LIVE_LOCATION)


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
    assert any(
        "Fiscal Year 2026 Overview" in t or "Results of Operations Table" in t for t in titles
    )

    # Check metadata & schema alignment with terraform/schemas/filing_chunks.json
    schema_path = (
        Path(__file__).resolve().parents[1] / "terraform" / "schemas" / "filing_chunks.json"
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


@pytest.mark.live
def test_live_vertex_ai_embeddings_capture_semantic_similarity(
    live_embedder: VertexAIEmbedder,
) -> None:
    v1, v2, v3 = live_embedder.embed_texts(
        [
            "AI GPU export control restrictions in China impacted Data Center revenue.",
            "US export control restrictions on China AI GPU chips reduced Data Center sales.",
            "Retail store lease obligations and consumer packaged soap pricing.",
        ]
    )

    assert len(v1) == DEFAULT_EMBEDDING_DIMENSIONS
    norm_v1 = math.sqrt(sum(x * x for x in v1))
    assert math.isclose(norm_v1, 1.0, rel_tol=1e-2)

    sim_related = float(np.dot(v1, v2))
    sim_unrelated = float(np.dot(v1, v3))
    assert sim_related > sim_unrelated
    assert sim_related > 0.7


def test_populate_chunk_embeddings_fails_loudly_without_vertex_ai() -> None:
    """No synthetic fallback: a Vertex AI failure must abort ingestion."""
    prof = _sample_profile()
    chunks = chunk_filing_document(prof, _sample_markdown(prof))
    with pytest.raises(RuntimeError, match="HTTP 403"):
        populate_chunk_embeddings(
            chunks,
            project_id="p",
            vertex_embedder=_mock_vertex_embedder([], fail_after=0),
        )
    assert all(not c.embedding for c in chunks)


def test_chunk_health_evaluation_and_cloud_monitoring_export() -> None:
    prof_nvda = _sample_profile("NVDA", 2026)
    prof_aapl = _sample_profile("AAPL", 2026)

    chunks_nvda = chunk_filing_document(prof_nvda, _sample_markdown(prof_nvda))
    chunks_aapl = chunk_filing_document(prof_aapl, _sample_markdown(prof_aapl))
    all_chunks = chunks_nvda + chunks_aapl
    populated, source = populate_chunk_embeddings(
        all_chunks,
        project_id=LIVE_PROJECT_ID,
        location=LIVE_LOCATION,
        vertex_embedder=_mock_vertex_embedder([]),
    )
    assert source == "vertex_ai"
    assert all(c.embedding_model == "text-embedding-005" for c in populated)

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
    assert "custom.googleapis.com/sec_edgar/chunk_health/missing_section_filings" in metric_types

    structured_log = format_structured_health_log(healthy_report)
    assert structured_log["event_type"] == "sec_edgar_chunk_health_evaluation"
    assert structured_log["severity"] == "INFO"
    assert structured_log["missing_section_filings"] == 0
    assert structured_log["missing_profile_ids"] == []
    assert structured_log["incomplete_section_profile_ids"] == []

    anomaly_log = format_structured_health_log(anomaly_report, max_profile_ids=1)
    assert anomaly_log["missing_section_filings"] == 1
    assert anomaly_log["missing_profile_ids"] == ["MSFT_2026"]
    assert anomaly_log["missing_profile_ids_truncated"] == 0

    assert len(ts_list) >= 10
    structured_log = format_structured_health_log(healthy_report)
    assert structured_log["event_type"] == "sec_edgar_chunk_health_evaluation"
    assert structured_log["health_status"] == "HEALTHY"


def test_end_to_end_vector_db_pipeline_staged_locally(tmp_path: Path) -> None:
    """Stage manifests locally (no GCS/BQ/Monitoring publish) with mocked Vertex AI vectors."""
    prof = _sample_profile("NVDA", 2026)
    md = _sample_markdown(prof)

    md_path = tmp_path / prof.unstructured_blob_path
    json_path = tmp_path / prof.structured_blob_path
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(prof.model_dump(mode="json")), encoding="utf-8")

    cfg = VectorDBBuildConfig(
        project_id=LIVE_PROJECT_ID,
        location=LIVE_LOCATION,
        output_dir=tmp_path,
        local_only=True,
        target_chunk_chars=400,
        max_chunk_chars=900,
        min_chunk_chars=80,
    )
    seen: list[int] = []
    summary, report = run_vector_db_pipeline(cfg, vertex_embedder=_mock_vertex_embedder(seen))

    assert summary.embedding_backend == "vertex_ai"
    assert sum(seen) == summary.total_chunks_generated
    assert summary.uploaded_to_gcs is False
    assert summary.total_profiles_processed == 1
    assert summary.total_chunks_generated >= 2
    assert summary.health_status == "HEALTHY"
    assert report.health_score == 100.0

    bq_jsonl = Path(summary.bigquery_manifest_path)
    health_json = Path(summary.health_report_path)

    assert bq_jsonl.exists()
    assert health_json.exists()

    bq_lines = bq_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(bq_lines) == summary.total_chunks_generated

    first_row = json.loads(bq_lines[0])
    assert len(first_row["embedding"]) == DEFAULT_EMBEDDING_DIMENSIONS

    # Second run reuses the cached Vertex AI vectors; a failing embedder proves no call is made.
    summary_cached, _ = run_vector_db_pipeline(
        cfg, vertex_embedder=_mock_vertex_embedder([], fail_after=0)
    )
    assert summary_cached.embedding_backend == "cached"


def test_cli_parse_args_and_token_estimation() -> None:
    args = parse_args(["--tickers", "NVDA", "AAPL", "--local-only", "--target-chars", "1800"])
    assert args.tickers == ["NVDA", "AAPL"]
    assert args.local_only is True
    assert args.target_chars == 1800
    assert not hasattr(args, "embedding_backend")
    with pytest.raises(SystemExit):
        parse_args(["--embedding-backend", "deterministic"])
    assert estimate_token_count("") == 0
    assert estimate_token_count("NVIDIA Blackwell data center revenue grew significantly.") > 5


def _mock_vertex_embedder(
    requests_seen: list[int],
    *,
    reject_over: int | None = None,
    fail_after: int | None = None,
    **kwargs: object,
) -> VertexAIEmbedder:
    """Build a VertexAIEmbedder backed by a typed fake ``google-genai`` client."""
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types

    class _FakeModels:
        def embed_content(
            self, *, model: str, contents: list[str], config: Any = None
        ) -> genai_types.EmbedContentResponse:
            _ = (model, config)
            n = len(contents)
            if reject_over is not None and n > reject_over:
                raise genai_errors.ClientError(
                    400,
                    {
                        "error": {
                            "code": 400,
                            "message": "Unable to submit request because the input token count is 25000 but the model supports up to 20000.",
                        }
                    },
                )
            if fail_after is not None and len(requests_seen) >= fail_after:
                raise genai_errors.ClientError(403, {"error": {"code": 403, "message": "quota"}})
            requests_seen.append(n)
            return genai_types.EmbedContentResponse(
                embeddings=[
                    genai_types.ContentEmbedding(
                        values=[0.123456789] * DEFAULT_EMBEDDING_DIMENSIONS
                    )
                    for _ in range(n)
                ]
            )

    class _FakeGenAi:
        models = _FakeModels()

    return VertexAIEmbedder(
        project_id="iron-radio-514217-q7",
        genai_client=_FakeGenAi(),
        **kwargs,  # type: ignore[arg-type]
    )


def test_vertex_embedder_batches_by_size_splits_oversized_and_runs_concurrently() -> None:
    texts = [f"text {i} " + "x" * 1000 for i in range(40)]

    seen: list[int] = []
    embedder = _mock_vertex_embedder(seen, batch_size=50, max_batch_chars=10_000)
    assert len(embedder.embed_texts(texts)) == 40
    assert max(seen) <= 9  # ~1,008 chars each -> at most 9 per 10,000-char batch
    assert sum(seen) == 40

    seen = []
    embedder = _mock_vertex_embedder(seen, reject_over=4, batch_size=16, max_batch_chars=10**9)
    assert len(embedder.embed_texts(texts)) == 40
    assert max(seen) <= 4 and sum(seen) == 40

    seen = []
    embedder = _mock_vertex_embedder(seen, batch_size=3, max_workers=4)
    order: list[int] = []
    vecs = embedder.embed_texts(texts, on_batch_complete=lambda s, e, v: order.append(s))
    assert len(vecs) == 40 and sum(seen) == 40
    assert sorted(order) == list(range(0, 40, 3))


def test_populate_chunk_embeddings_checkpoints_and_resumes(tmp_path: Path) -> None:
    prof = _sample_profile()
    chunks = chunk_filing_document(prof, _sample_markdown(prof))
    assert len(chunks) >= 2
    ckpt = tmp_path / "ckpt.jsonl"

    seen: list[int] = []
    failing = _mock_vertex_embedder(seen, fail_after=1, batch_size=1)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        populate_chunk_embeddings(
            chunks,
            project_id="p",
            vertex_embedder=failing,
            checkpoint_path=ckpt,
        )
    assert len(ckpt.read_text(encoding="utf-8").splitlines()) == 1

    fresh = chunk_filing_document(prof, _sample_markdown(prof))
    seen = []
    ok = _mock_vertex_embedder(seen, batch_size=1)
    populated, backend = populate_chunk_embeddings(
        fresh,
        project_id="p",
        vertex_embedder=ok,
        checkpoint_path=ckpt,
    )
    assert backend == "vertex_ai"
    assert sum(seen) == len(fresh) - 1  # the checkpointed chunk was not re-embedded
    assert all(c.embedding[0] == round(0.123456789, 6) for c in populated)

    again, backend = populate_chunk_embeddings(
        chunk_filing_document(prof, _sample_markdown(prof)),
        project_id="p",
        vertex_embedder=_mock_vertex_embedder([], fail_after=0),
        checkpoint_path=ckpt,
    )
    assert backend == "vertex_ai"
    assert all(len(c.embedding) == DEFAULT_EMBEDDING_DIMENSIONS for c in again)


def test_vector_db_upload_builds_storage_client_when_not_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `build-vector-db` previously passed `client=None` to `ensure_gcs_bucket`."""
    from edgar_capstone.vector_db import pipeline as vdb_pipeline

    uploaded: list[str] = []

    class _Blob:
        def __init__(self, name: str) -> None:
            self.name = name

        def upload_from_filename(self, filename: str, content_type: str | None = None) -> None:
            uploaded.append(self.name)

    class _Bucket:
        location = "US-CENTRAL1"

        def blob(self, name: str) -> _Blob:
            return _Blob(name)

    class _Client:
        def __init__(self, project: str | None = None) -> None:
            self.project = project

        def lookup_bucket(self, name: str) -> _Bucket:
            return _Bucket()

    monkeypatch.setattr(vdb_pipeline.storage, "Client", _Client)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "bigquery_filing_chunks.jsonl").write_text("{}\n", encoding="utf-8")

    count = vdb_pipeline.upload_vector_db_artifacts_to_gcs(
        output_dir=tmp_path,
        project_id="unit-test-project",
        bucket_name="fde-sec-edgar-sandbox-dev-sec-corpus",
    )
    assert count == 1
    assert uploaded == ["manifests/bigquery_filing_chunks.jsonl"]


def test_chunker_oversized_tables_paragraphs_and_markdown_subheadings() -> None:
    """Covers `_detect_subheading`, `_split_long_text`, `_split_oversized_block`, and `split_section_into_passages`."""
    from edgar_capstone.vector_db.chunker import (
        _detect_subheading,
        _split_long_text,
        _split_oversized_block,
        split_section_into_passages,
    )

    assert _detect_subheading("### Supply Chain\\. & Risks\\-2026.") == "Supply Chain. & Risks-2026"
    assert _detect_subheading("Line one\nLine two") is None
    assert _detect_subheading("**Note:**") is None

    # `_split_long_text`: short text, multi-sentence split, single giant run-on word/sentence
    assert _split_long_text("   ", 50) == []
    assert _split_long_text("Short sentence.", 50) == ["Short sentence."]
    multi_sent = (
        "First financial sentence describes revenue growth. "
        "Second sentence details Blackwell infrastructure deployments. "
        + "A" * 95
        + ". Third short closing sentence."
    )
    pieces = _split_long_text(multi_sent, max_chars=70)
    assert len(pieces) >= 4
    assert all(len(p) <= 70 for p in pieces)

    # `_split_oversized_block`: large Markdown table repeats header on every sub-block
    table_lines = [
        "| Segment | FY2026 | FY2025 |",
        "| --- | --- | --- |",
        *[f"| Region_{i} | ${100 + i}M | ${90 + i}M |" for i in range(10)],
    ]
    table_parts = _split_oversized_block("\n".join(table_lines), max_chars=140)
    assert len(table_parts) >= 3
    for part in table_parts:
        assert part.startswith("| Segment | FY2026 | FY2025 |\n| --- | --- | --- |")

    # Oversized single table row splits cleanly while keeping header prefix when budget allows
    wide_row_table = "\n".join(
        [
            "| Col | Note |",
            "| --- | --- |",
            "| Row1 | ok |",
            "| Row2 | " + "Export restriction disclosure detail. " * 6 + "|",
        ]
    )
    wide_parts = _split_oversized_block(wide_row_table, max_chars=120)
    assert len(wide_parts) >= 3
    assert any("Row1" in p for p in wide_parts)
    assert any(p.startswith("| Col | Note |") and "Export restriction" in p for p in wide_parts)

    # Table whose header alone exceeds max_chars falls back to line splitting without header duplication
    huge_hdr = "\n".join(
        [
            "| " + "H" * 90 + " |",
            "| --- |",
            "| small row |",
            "| another row |",
        ]
    )
    assert _split_oversized_block(huge_hdr, max_chars=80)

    # Multi-line non-table block & giant single paragraph fallback
    non_table = "\n".join(
        f"Bullet line {i} covering export controls and compliance." for i in range(8)
    )
    assert len(_split_oversized_block(non_table, max_chars=110)) >= 3
    single_para = "Demand remained strong. " * 20
    assert len(_split_oversized_block(single_para, max_chars=100)) >= 4

    # `split_section_into_passages` empty input + markdown ### heading + overlap carry-over
    assert split_section_into_passages("   \n\n  ", section_id="item_1a") == []
    section_md = (
        "### Overview\n\n"
        "Paragraph one discusses Data Center revenue expansion in detail across hyperscalers.\n\n"
        "Short bridge paragraph for overlap.\n\n"
        "Paragraph two discusses gross margin dynamics and Blackwell ramp manufacturing costs.\n\n"
        "### New Risk Topic\n\n"
        "Export licensing rules introduced inventory charges in fiscal year 2026."
    )
    passages = split_section_into_passages(
        section_md,
        section_id="item_7",
        target_chars=140,
        max_chars=240,
        min_chars=40,
        overlap_chars=80,
    )
    assert len(passages) >= 2
    assert any("Overview" in title for title, _ in passages)


def test_monitoring_edge_cases_embedder_retries_and_lazy_package_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers `_is_valid_finite_embedding`, `_compute_group_breakdown`, embedder retries, and `vector_db.__getattr__`."""
    import time

    import edgar_capstone.vector_db as vdb_pkg
    from edgar_capstone.vector_db.monitoring import (
        _compute_group_breakdown,
        _has_complete_metadata,
        _is_valid_finite_embedding,
    )

    # Lazy package exports
    assert vdb_pkg.chunk_filing_document is chunk_filing_document
    assert "VertexAIEmbedder" in dir(vdb_pkg)
    with pytest.raises(AttributeError):
        _ = vdb_pkg.NonExistentSymbol

    # Monitoring finite/zero-norm embedding & metadata checks + unflagged breakdown
    assert _is_valid_finite_embedding([], 4) == (False, False, False)
    assert _is_valid_finite_embedding([1.0, 2.0], 4) == (True, False, False)
    assert _is_valid_finite_embedding([float("nan"), 0.0, 0.0, 1.0], 4) == (True, True, False)
    assert _is_valid_finite_embedding([0.0, 0.0, 0.0, 0.0], 4) == (True, True, False)
    assert _is_valid_finite_embedding([1.0, 0.0, 0.0, 0.0], 4) == (True, True, True)

    empty_bd = _compute_group_breakdown("empty", [], expected_dim=4)
    assert empty_bd.total_chunks == 0

    prof = _sample_profile()
    chunks = chunk_filing_document(prof, _sample_markdown(prof))
    chunks[0].embedding = [0.0] * DEFAULT_EMBEDDING_DIMENSIONS  # zero norm -> non_finite
    chunks[0].char_count = 9500  # oversized
    chunks[0].sec_filing_url = ""  # incomplete metadata
    assert _has_complete_metadata(chunks[0]) is False
    bad_ticker_chunk = chunks[0].model_copy(update={"ticker": ""})
    assert _has_complete_metadata(bad_ticker_chunk) is False
    bd = _compute_group_breakdown("item_1a", [chunks[0]], expected_dim=DEFAULT_EMBEDDING_DIMENSIONS)
    assert bd.oversized_chunk_count == 1 and bd.missing_metadata_count == 1
    rep = evaluate_chunk_health([chunks[0]], profiles=[prof])
    assert rep.anomalies.non_finite_embedding == 1 and rep.anomalies.oversized_chunk == 1

    # Embedder retry on transient timeout, HTTP 429, and mismatched count error
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types

    calls = {"n": 0}

    class _RetryingModels:
        def embed_content(
            self, *, model: str, contents: list[str], config: Any = None
        ) -> genai_types.EmbedContentResponse:
            _ = (model, contents, config)
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("transient network drop")
            if calls["n"] == 2:
                raise genai_errors.ClientError(
                    429, {"error": {"code": 429, "message": "rate limited"}}
                )
            return genai_types.EmbedContentResponse(
                embeddings=[genai_types.ContentEmbedding(values=[0.25] * 4)]
            )

    class _RetryingGenAi:
        models = _RetryingModels()

    emb = VertexAIEmbedder(
        project_id="p",
        dimensions=4,
        max_retries=3,
        genai_client=_RetryingGenAi(),
    )
    vecs = emb.embed_texts(["hello"], titles=["Title"])
    assert vecs == [[0.25, 0.25, 0.25, 0.25]]
    assert calls["n"] == 3

    # Mismatched prediction count raises RuntimeError
    class _EmptyModels:
        def embed_content(self, **_kw: Any) -> genai_types.EmbedContentResponse:
            return genai_types.EmbedContentResponse(embeddings=[])

    class _EmptyGenAi:
        models = _EmptyModels()

    bad_count_emb = VertexAIEmbedder(
        project_id="p",
        dimensions=4,
        max_retries=1,
        genai_client=_EmptyGenAi(),
    )
    with pytest.raises(RuntimeError, match="returned 0 embeddings"):
        bad_count_emb.embed_texts(["one"])


def test_evaluate_chunk_health_from_bigquery() -> None:
    """Verifies BigQuery-backed chunk health evaluation and structured health log formatting."""
    from edgar_capstone.vector_db.monitoring import evaluate_chunk_health_from_bigquery

    class _FakeBQClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def table_ref(self, table_id: str) -> str:
            return f"`iron-radio-514217-q7.sec_edgar.{table_id}`"

        def run_query(self, sql: str, **_kwargs: object) -> list[dict[str, object]]:
            self.queries.append(sql)
            if "v_chunk_health_metrics" in sql:
                return [
                    {
                        "gics_sector": "ALL_SECTORS",
                        "section_id": "ALL_SECTIONS",
                        "total_chunks": 200,
                        "distinct_profiles": 9,
                        "distinct_tickers": 5,
                        "expected_profiles": 10,
                        "profiles_with_both_sections": 8,
                        "valid_embedding_count": 200,
                        "missing_embedding_count": 0,
                        "dimension_mismatch_count": 0,
                        "embedding_coverage_ratio": 1.0,
                        "avg_char_count": 1700.0,
                        "avg_token_estimate": 395.0,
                        "min_token_estimate": 50,
                        "max_token_estimate": 900,
                        "p50_token_estimate": 400.0,
                        "p95_token_estimate": 550.0,
                        "undersized_chunk_count": 0,
                        "oversized_chunk_count": 0,
                        "missing_metadata_count": 0,
                    },
                    {
                        "gics_sector": "ALL_SECTORS",
                        "section_id": "item_1a",
                        "total_chunks": 90,
                        "distinct_profiles": 9,
                        "distinct_tickers": 5,
                        "valid_embedding_count": 90,
                        "embedding_coverage_ratio": 1.0,
                        "avg_char_count": 1750.0,
                        "avg_token_estimate": 400.0,
                        "min_token_estimate": 60,
                        "max_token_estimate": 850,
                        "p50_token_estimate": 410.0,
                        "p95_token_estimate": 540.0,
                        "undersized_chunk_count": 0,
                        "oversized_chunk_count": 0,
                        "missing_metadata_count": 0,
                    },
                    {
                        "gics_sector": "Information Technology",
                        "section_id": "ALL_SECTIONS",
                        "total_chunks": 200,
                        "distinct_profiles": 9,
                        "distinct_tickers": 5,
                        "valid_embedding_count": 200,
                        "embedding_coverage_ratio": 1.0,
                        "avg_char_count": 1700.0,
                        "avg_token_estimate": 395.0,
                        "min_token_estimate": 50,
                        "max_token_estimate": 900,
                        "p50_token_estimate": 400.0,
                        "p95_token_estimate": 550.0,
                        "undersized_chunk_count": 0,
                        "oversized_chunk_count": 0,
                        "missing_metadata_count": 0,
                    },
                ]
            return [
                {"profile_id": "FDX_2020", "filing_chunk_status": "UNINDEXED"},
                {"profile_id": "IBM_2025", "filing_chunk_status": "PARTIAL_SECTIONS"},
            ]

    report = evaluate_chunk_health_from_bigquery(
        project_id="iron-radio-514217-q7",
        bq_client=_FakeBQClient(),  # type: ignore[arg-type]
    )
    assert report.total_profiles_expected == 10
    assert report.total_profiles_indexed == 9
    assert report.profiles_with_both_sections == 8
    assert report.full_section_coverage_ratio == 0.8
    assert report.anomalies.missing_section_filings == 2
    assert report.missing_profile_ids == ["FDX_2020"]
    assert report.incomplete_section_profile_ids == ["IBM_2025"]
    assert "item_1a" in report.by_section
    assert "Information Technology" in report.by_sector

    log_entry = format_structured_health_log(report)
    assert log_entry["event_type"] == "sec_edgar_chunk_health_evaluation"
    assert log_entry["missing_section_filings"] == 2
    assert log_entry["missing_profile_ids"] == ["FDX_2020"]
    assert len(build_cloud_monitoring_timeseries(report)) >= 10
