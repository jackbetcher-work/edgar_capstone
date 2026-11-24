"""End-to-end pipeline for creating, populating, and monitoring the SEC 10-K RAG vector DB."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import google.auth.exceptions
import httpx

from edgar_capstone.common.domain import normalize_edgar_ticker, sha256_text
from edgar_capstone.common.jsonl import iter_jsonl, write_json
from edgar_capstone.dataset.cloud_publish import run_bigquery_load_job
from edgar_capstone.dataset.gcs_storage import (
    DEFAULT_RETRY,
    GCSAccessError,
    content_type_for_path,
    ensure_gcs_bucket,
    normalize_bucket_name,
    upload_files_to_bucket,
)
from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.dataset.paths import resolve_curated_dataset_dir
from edgar_capstone.gcp.defaults import (
    DEFAULT_BIGQUERY_DATASET_ID,
    DEFAULT_BUCKET_NAME,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
)
from edgar_capstone.vector_db.chunker import chunk_filing_document
from edgar_capstone.vector_db.embedder import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_BATCH_CHARS,
    VertexAIEmbedder,
    populate_chunk_embeddings,
)
from edgar_capstone.vector_db.models import (
    ChunkHealthReport,
    FilingChunk,
    VectorDBBuildSummary,
)
from edgar_capstone.vector_db.monitoring import (
    CloudMonitoringHealthPublisher,
    evaluate_chunk_health,
)

try:
    from google.cloud import storage
except ImportError:  # pragma: no cover - optional `[ingest]` extra
    storage = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


VECTOR_DB_ARTIFACT_PATHS: tuple[str, ...] = (
    "manifests/bigquery_filing_chunks.jsonl",
    "vector_search/embeddings/chunks_embeddings.json",
    "manifests/chunk_health_report.json",
    "manifests/vector_db_catalog.json",
)
# Small files rewritten after GCS/BigQuery/Monitoring sync so their published flags persist.
_VECTOR_DB_STATUS_PATHS = [
    "manifests/chunk_health_report.json",
    "manifests/vector_db_catalog.json",
]


@dataclass(slots=True)
class VectorDBBuildConfig:
    """Configuration for chunking, embedding, populating, and monitoring the RAG vector DB."""

    project_id: str = DEFAULT_PROJECT_ID
    bucket_name: str = DEFAULT_BUCKET_NAME
    dataset_id: str = DEFAULT_BIGQUERY_DATASET_ID
    table_id: str = "filing_chunks"
    location: str = DEFAULT_LOCATION
    output_dir: Path = field(default_factory=resolve_curated_dataset_dir)
    tickers: list[str] | None = None
    sector_filter: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    embedding_batch_size: int = 50
    embedding_max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS
    embedding_max_workers: int = 8
    bigquery_load_timeout_seconds: float = 1800.0
    target_chunk_chars: int = 2200
    max_chunk_chars: int = 3600
    min_chunk_chars: int = 150
    overlap_chunk_chars: int = 200
    skip_existing_embeddings: bool = True
    # Stage manifests and the health report locally (still embedding with Vertex AI) without
    # publishing to GCS, BigQuery, or Cloud Monitoring.
    local_only: bool = False
    load_bigquery: bool = True
    publish_monitoring_metrics: bool = True
    create_bucket_if_missing: bool = True


def _load_filtered_profiles(
    output_dir: Path,
    bucket_name: str,
    *,
    tickers: list[str] | None = None,
    sector_filter: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[FinancialDataProfile]:
    """Load staged `FinancialDataProfile` records matching optional scope filters."""
    struct_root = output_dir / "structured"
    if not struct_root.exists():
        return []

    ticker_set = (
        {norm for raw in tickers for t in raw.split(",") if (norm := normalize_edgar_ticker(t))}
        if tickers
        else None
    )
    sector_norm = sector_filter.strip().lower() if sector_filter else None

    profiles: list[FinancialDataProfile] = []
    for json_path in sorted(struct_root.rglob("*_financials.json")):
        profile = FinancialDataProfile.model_validate_json(json_path.read_text(encoding="utf-8"))
        profile.unstructured_gcs_uri = f"gs://{bucket_name}/{profile.unstructured_blob_path}"
        profile.structured_gcs_uri = f"gs://{bucket_name}/{profile.structured_blob_path}"

        if ticker_set and normalize_edgar_ticker(profile.ticker) not in ticker_set:
            continue
        if sector_norm and profile.gics_sector.lower() != sector_norm:
            continue
        if start_year is not None and profile.fiscal_year < start_year:
            continue
        if end_year is not None and profile.fiscal_year > end_year:
            continue
        profiles.append(profile)

    profiles.sort(key=lambda p: (p.ticker, p.fiscal_year))
    return profiles


def _load_cached_embeddings(
    bq_chunks_jsonl_path: Path,
) -> dict[str, tuple[str, list[float]]]:
    """Stream previously generated `chunk_id -> (sha256(chunk_text), embedding)` mappings.

    Uses line-by-line streaming (`iter_jsonl(..., skip_corrupt=True)`) to avoid loading
    multi-GB manifests into RAM at once and to isolate any single corrupt trailing line.
    """
    if not bq_chunks_jsonl_path.exists():
        return {}

    cached: dict[str, tuple[str, list[float]]] = {}
    try:
        for row in iter_jsonl(bq_chunks_jsonl_path, skip_corrupt=True):
            cid = row.get("chunk_id")
            text = row.get("chunk_text", "")
            emb = row.get("embedding", [])
            if cid and text and isinstance(emb, list) and emb:
                try:
                    cached[str(cid)] = (sha256_text(str(text)), [float(v) for v in emb])
                except (TypeError, ValueError) as exc:
                    logger.warning("Skipping invalid embedding row %r: %s", cid, exc)
    except OSError as exc:
        logger.warning(
            "Ignoring unreadable cached chunk manifest %s: %s", bq_chunks_jsonl_path, exc
        )

    return cached


def _write_status_artifacts(
    output_dir: Path,
    health_report: ChunkHealthReport,
    summary: VectorDBBuildSummary,
) -> tuple[Path, Path]:
    """Write only the small status files (`chunk_health_report.json` and `vector_db_catalog.json`)."""
    manifests_dir = output_dir / "manifests"
    health_path = manifests_dir / "chunk_health_report.json"
    catalog_path = manifests_dir / "vector_db_catalog.json"
    write_json(health_path, health_report.model_dump(mode="json"), indent=2)
    write_json(catalog_path, summary.model_dump(mode="json"), indent=2)
    return health_path, catalog_path


def _write_vector_db_artifacts(
    output_dir: Path,
    chunks: list[FilingChunk],
    health_report: ChunkHealthReport,
    summary: VectorDBBuildSummary,
) -> tuple[Path, Path, Path, Path]:
    """Write BigQuery chunks JSONL, Vertex AI Vector Search JSONL, health report, and catalog."""
    manifests_dir = output_dir / "manifests"
    vector_dir = output_dir / "vector_search" / "embeddings"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)

    bq_chunks_path = manifests_dir / "bigquery_filing_chunks.jsonl"
    vertex_chunks_path = vector_dir / "chunks_embeddings.json"

    with (
        bq_chunks_path.open("w", encoding="utf-8") as bq_f,
        vertex_chunks_path.open("w", encoding="utf-8") as vx_f,
    ):
        for chunk in chunks:
            bq_f.write(json.dumps(chunk.to_bigquery_record()) + "\n")
            vx_f.write(json.dumps(chunk.to_vertex_vector_search_record()) + "\n")

    health_path, catalog_path = _write_status_artifacts(output_dir, health_report, summary)
    return bq_chunks_path, vertex_chunks_path, health_path, catalog_path


def _vector_artifact_content_type(local_path: Path, blob_path: str) -> str:
    """Resolve HTTP Content-Type for vector DB artifacts (Vertex batch JSON is NDJSON)."""
    if blob_path.endswith("chunks_embeddings.json"):
        return "application/x-ndjson; charset=utf-8"
    return content_type_for_path(local_path)


def upload_vector_db_artifacts_to_gcs(
    *,
    output_dir: Path,
    project_id: str,
    bucket_name: str,
    location: str = DEFAULT_LOCATION,
    create_bucket_if_missing: bool = True,
    storage_client: Any | None = None,
    rel_paths: list[str] | None = None,
) -> int:
    """Upload the vector database manifests and Vertex AI Vector Search files to GCS.

    `rel_paths` limits the upload to a subset of `VECTOR_DB_ARTIFACT_PATHS`.
    """
    if storage_client is not None:
        client = storage_client
    else:
        if storage is None:
            raise GCSAccessError(
                "google-cloud-storage is not installed. Install the optional ingestion "
                "dependencies with: pip install -e '.[ingest]'"
            )
        try:
            client = storage.Client(project=project_id)
        except google.auth.exceptions.DefaultCredentialsError as exc:
            raise GCSAccessError(
                f"Google Cloud Application Default Credentials not found for project "
                f"'{project_id}': {exc}\n"
                "Run `gcloud auth application-default login`, then re-run "
                "`build-vector-db --upload-only`."
            ) from exc

    bucket = ensure_gcs_bucket(
        client,
        project_id=project_id,
        bucket_name=bucket_name,
        location=location,
        create_if_missing=create_bucket_if_missing,
    )
    if rel_paths is None:
        rel_paths = list(VECTOR_DB_ARTIFACT_PATHS)

    items = [(local_file, rel) for rel in rel_paths if (local_file := output_dir / rel).exists()]
    uploaded = upload_files_to_bucket(
        bucket,
        items,
        max_workers=1,
        timeout=300.0,
        retry=DEFAULT_RETRY,
        content_type_fn=_vector_artifact_content_type,
    )

    logger.info(
        "Uploaded %d vector DB artifacts to gs://%s/",
        uploaded,
        normalize_bucket_name(bucket_name),
    )
    return uploaded


def trigger_bigquery_chunks_load_job(
    *,
    project_id: str,
    dataset_id: str,
    table_id: str,
    source_gcs_uri: str,
    location: str = DEFAULT_LOCATION,
    quota_project_id: str | None = None,
    http_client: httpx.Client | None = None,
    credentials: Any | None = None,
    poll_timeout_seconds: float = 120.0,
) -> str:
    """Trigger a BigQuery `WRITE_TRUNCATE` load job from `source_gcs_uri` into `filing_chunks`.

    Thin wrapper over `run_bigquery_load_job`: raises `RuntimeError` on a failed job, and logs a
    warning (returning the job id) if the job is still running after `poll_timeout_seconds`.
    """
    result = run_bigquery_load_job(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        source_uris=[source_gcs_uri],
        location=location,
        write_disposition="WRITE_TRUNCATE",
        job_id_prefix="load_filing_chunks",
        poll_timeout_seconds=poll_timeout_seconds,
        raise_on_timeout=False,
        quota_project_id=quota_project_id,
        http_client=http_client,
        credentials=credentials,
    )
    return result.job_id


def run_vector_db_pipeline(
    config: VectorDBBuildConfig,
    *,
    storage_client: Any | None = None,
    vertex_embedder: VertexAIEmbedder | None = None,
    monitoring_publisher: CloudMonitoringHealthPublisher | None = None,
    http_client: httpx.Client | None = None,
) -> tuple[VectorDBBuildSummary, ChunkHealthReport]:
    """Execute end-to-end 10-K chunking, embedding, manifest writing, GCS/BQ sync, and monitoring."""
    bucket_name = normalize_bucket_name(config.bucket_name)
    output_dir = config.output_dir.resolve()
    warnings: list[str] = []

    profiles = _load_filtered_profiles(
        output_dir,
        bucket_name,
        tickers=config.tickers,
        sector_filter=config.sector_filter,
        start_year=config.start_year,
        end_year=config.end_year,
    )
    if not profiles:
        raise RuntimeError(
            f"No staged 10-K financial profiles found in {output_dir / 'structured'}. "
            "Run scripts/build_sec_dataset.py first."
        )

    logger.info(
        "Chunking %d curated 10-K filings from %s (target=%d chars, max=%d chars)...",
        len(profiles),
        output_dir,
        config.target_chunk_chars,
        config.max_chunk_chars,
    )

    all_chunks: list[FilingChunk] = []
    for prof in profiles:
        md_path = output_dir / prof.unstructured_blob_path
        if not md_path.exists():
            msg = f"[{prof.profile_id}] Missing unstructured Markdown file at {md_path}"
            logger.warning(msg)
            warnings.append(msg)
            continue

        md_text = md_path.read_text(encoding="utf-8")
        doc_chunks = chunk_filing_document(
            prof,
            md_text,
            embedding_model=config.embedding_model,
            target_chars=config.target_chunk_chars,
            max_chars=config.max_chunk_chars,
            min_chars=config.min_chunk_chars,
            overlap_chars=config.overlap_chunk_chars,
        )
        if not doc_chunks:
            msg = f"[{prof.profile_id}] Produced 0 chunks from {md_path}"
            logger.warning(msg)
            warnings.append(msg)
        all_chunks.extend(doc_chunks)

    bq_manifest_path = output_dir / "manifests" / "bigquery_filing_chunks.jsonl"
    cached_embeddings = (
        _load_cached_embeddings(bq_manifest_path) if config.skip_existing_embeddings else None
    )

    checkpoint_path = output_dir / "vector_search" / ".embedding_checkpoint.jsonl"
    all_chunks, embedding_source = populate_chunk_embeddings(
        all_chunks,
        project_id=config.project_id,
        location=config.location,
        model_name=config.embedding_model,
        dimensions=config.embedding_dimensions,
        batch_size=config.embedding_batch_size,
        existing_embeddings=cached_embeddings,
        vertex_embedder=vertex_embedder,
        max_workers=config.embedding_max_workers,
        max_batch_chars=config.embedding_max_batch_chars,
        checkpoint_path=checkpoint_path,
    )
    cached_embeddings = None

    health_report = evaluate_chunk_health(
        all_chunks,
        profiles=profiles,
        project_id=config.project_id,
        dataset_id=config.dataset_id,
        table_id=config.table_id,
        expected_embedding_dim=config.embedding_dimensions,
        embedding_model=config.embedding_model,
    )

    item_1a_count = sum(1 for c in all_chunks if c.section_id == "item_1a")
    item_7_count = sum(1 for c in all_chunks if c.section_id == "item_7")

    vertex_manifest_path = output_dir / "vector_search" / "embeddings" / "chunks_embeddings.json"
    health_report_path = output_dir / "manifests" / "chunk_health_report.json"

    summary = VectorDBBuildSummary(
        project_id=config.project_id,
        bucket_name=bucket_name,
        dataset_id=config.dataset_id,
        table_id=config.table_id,
        location=config.location,
        embedding_model=config.embedding_model,
        embedding_backend=embedding_source,
        embedding_dimensions=config.embedding_dimensions,
        total_profiles_processed=len(profiles),
        total_chunks_generated=len(all_chunks),
        item_1a_chunks=item_1a_count,
        item_7_chunks=item_7_count,
        bigquery_manifest_path=str(bq_manifest_path),
        vertex_vector_manifest_path=str(vertex_manifest_path),
        health_report_path=str(health_report_path),
        health_score=health_report.health_score,
        health_status=health_report.health_status,
        warnings=warnings,
    )

    _write_vector_db_artifacts(output_dir, all_chunks, health_report, summary)
    # The manifests now hold every embedding, so the resume checkpoint is no longer needed.
    checkpoint_path.unlink(missing_ok=True)

    if not config.local_only:
        upload_vector_db_artifacts_to_gcs(
            output_dir=output_dir,
            project_id=config.project_id,
            bucket_name=bucket_name,
            location=config.location,
            create_bucket_if_missing=config.create_bucket_if_missing,
            storage_client=storage_client,
        )
        summary.uploaded_to_gcs = True

        if config.load_bigquery:
            source_uri = f"gs://{bucket_name}/manifests/bigquery_filing_chunks.jsonl"
            trigger_bigquery_chunks_load_job(
                project_id=config.project_id,
                dataset_id=config.dataset_id,
                table_id=config.table_id,
                source_gcs_uri=source_uri,
                location=config.location,
                quota_project_id=config.project_id,
                http_client=http_client,
                poll_timeout_seconds=config.bigquery_load_timeout_seconds,
            )
            summary.loaded_to_bigquery = True

        if config.publish_monitoring_metrics:
            owns_publisher = monitoring_publisher is None
            publisher = monitoring_publisher or CloudMonitoringHealthPublisher(
                project_id=config.project_id,
                http_client=http_client,
                write_cloud_logging=True,
            )
            try:
                publisher.publish_report(health_report)
            finally:
                if owns_publisher:
                    publisher.close()
            summary.published_monitoring_metrics = True

        # Write ONLY the tiny catalog & health report files so published flags are persisted
        # without rewriting multi-GB `bigquery_filing_chunks.jsonl` and `chunks_embeddings.json`.
        _write_status_artifacts(output_dir, health_report, summary)
        upload_vector_db_artifacts_to_gcs(
            output_dir=output_dir,
            project_id=config.project_id,
            bucket_name=bucket_name,
            location=config.location,
            create_bucket_if_missing=False,
            storage_client=storage_client,
            rel_paths=_VECTOR_DB_STATUS_PATHS,
        )

    return summary, health_report
