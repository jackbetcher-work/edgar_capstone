"""End-to-end pipeline for creating, populating, and monitoring the SEC 10-K RAG vector DB."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Literal
import uuid

import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.cloud import storage
import httpx

from edgar_capstone.dataset.gcs_storage import (
    DEFAULT_BUCKET_NAME,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    ensure_gcs_bucket,
    normalize_bucket_name,
)
from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.dataset.pipeline import DEFAULT_OUTPUT_DIR
from edgar_capstone.vector_db.chunker import chunk_filing_document
from edgar_capstone.vector_db.embedder import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
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

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VectorDBBuildConfig:
    """Configuration for chunking, embedding, populating, and monitoring the RAG vector DB."""

    project_id: str = DEFAULT_PROJECT_ID
    bucket_name: str = DEFAULT_BUCKET_NAME
    dataset_id: str = "sec_edgar"
    table_id: str = "filing_chunks"
    location: str = DEFAULT_LOCATION
    output_dir: Path = DEFAULT_OUTPUT_DIR
    tickers: list[str] | None = None
    sector_filter: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    embedding_backend: Literal["vertex_ai", "deterministic", "auto"] = "auto"
    embedding_batch_size: int = 50
    target_chunk_chars: int = 2200
    max_chunk_chars: int = 3600
    min_chunk_chars: int = 150
    overlap_chunk_chars: int = 200
    skip_existing_embeddings: bool = True
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
        {t.strip().upper() for raw in tickers for t in raw.split(",") if t.strip()}
        if tickers
        else None
    )
    sector_norm = sector_filter.strip().lower() if sector_filter else None

    profiles: list[FinancialDataProfile] = []
    for json_path in sorted(struct_root.rglob("*_financials.json")):
        profile = FinancialDataProfile.model_validate_json(
            json_path.read_text(encoding="utf-8")
        )
        profile.unstructured_gcs_uri = f"gs://{bucket_name}/{profile.unstructured_blob_path}"
        profile.structured_gcs_uri = f"gs://{bucket_name}/{profile.structured_blob_path}"

        if ticker_set and profile.ticker.upper() not in ticker_set:
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
    """Load previously generated `chunk_id -> (sha256(chunk_text), embedding)` mappings."""
    if not bq_chunks_jsonl_path.exists():
        return {}

    cached: dict[str, tuple[str, list[float]]] = {}
    try:
        for line in bq_chunks_jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = row.get("chunk_id")
            text = row.get("chunk_text", "")
            emb = row.get("embedding", [])
            if cid and text and isinstance(emb, list) and emb:
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                cached[str(cid)] = (text_hash, [float(v) for v in emb])
    except Exception as exc:
        logger.warning("Ignoring unreadable cached chunk manifest %s: %s", bq_chunks_jsonl_path, exc)

    return cached


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
    health_path = manifests_dir / "chunk_health_report.json"
    catalog_path = manifests_dir / "vector_db_catalog.json"

    with bq_chunks_path.open("w", encoding="utf-8") as bq_f, vertex_chunks_path.open(
        "w", encoding="utf-8"
    ) as vx_f:
        for chunk in chunks:
            bq_f.write(json.dumps(chunk.to_bigquery_record()) + "\n")
            vx_f.write(json.dumps(chunk.to_vertex_vector_search_record()) + "\n")

    health_path.write_text(
        json.dumps(health_report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    catalog_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )

    return bq_chunks_path, vertex_chunks_path, health_path, catalog_path


def upload_vector_db_artifacts_to_gcs(
    *,
    output_dir: Path,
    project_id: str,
    bucket_name: str,
    location: str = DEFAULT_LOCATION,
    create_bucket_if_missing: bool = True,
    storage_client: storage.Client | None = None,
) -> int:
    """Upload the vector database manifests and Vertex AI Vector Search files to GCS."""
    bucket = ensure_gcs_bucket(
        project_id=project_id,
        bucket_name=bucket_name,
        location=location,
        create_if_missing=create_bucket_if_missing,
        client=storage_client,
    )
    rel_paths = [
        "manifests/bigquery_filing_chunks.jsonl",
        "vector_search/embeddings/chunks_embeddings.json",
        "manifests/chunk_health_report.json",
        "manifests/vector_db_catalog.json",
    ]
    uploaded = 0
    for rel in rel_paths:
        local_file = output_dir / rel
        if not local_file.exists():
            continue
        blob = bucket.blob(rel)
        content_type = (
            "application/x-ndjson"
            if rel.endswith(".jsonl") or rel.endswith("chunks_embeddings.json")
            else "application/json; charset=utf-8"
        )
        blob.upload_from_filename(str(local_file), content_type=content_type)
        uploaded += 1

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
    http_client: httpx.Client | None = None,
    credentials: Any | None = None,
    poll_timeout_seconds: float = 120.0,
) -> str:
    """Trigger a BigQuery `WRITE_TRUNCATE` load job from `source_gcs_uri` into `filing_chunks`."""
    if credentials is None:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    if not credentials.valid:
        credentials.refresh(GoogleAuthRequest())
    token = str(credentials.token)

    job_id = f"load_filing_chunks_{uuid.uuid4().hex[:12]}"
    insert_url = f"https://bigquery.googleapis.com/bigquery/v2/projects/{project_id}/jobs"
    payload = {
        "jobReference": {
            "projectId": project_id,
            "jobId": job_id,
            "location": location,
        },
        "configuration": {
            "load": {
                "sourceUris": [source_gcs_uri],
                "destinationTable": {
                    "projectId": project_id,
                    "datasetId": dataset_id,
                    "tableId": table_id,
                },
                "sourceFormat": "NEWLINE_DELIMITED_JSON",
                "writeDisposition": "WRITE_TRUNCATE",
                "autodetect": False,
            }
        },
    }

    client = http_client or httpx.Client(timeout=60.0)
    close_client = http_client is None
    try:
        resp = client.post(
            insert_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"BigQuery load job submission failed (HTTP {resp.status_code}): {resp.text[:400]}"
            )

        status_url = (
            f"https://bigquery.googleapis.com/bigquery/v2/projects/{project_id}/"
            f"jobs/{job_id}?location={location}"
        )
        deadline = time.monotonic() + poll_timeout_seconds
        while time.monotonic() < deadline:
            poll_resp = client.get(
                status_url,
                headers={"Authorization": f"Bearer {token}"},
            )
            if poll_resp.status_code == 200:
                job_data = poll_resp.json()
                state = job_data.get("status", {}).get("state")
                if state == "DONE":
                    err = job_data.get("status", {}).get("errorResult")
                    if err:
                        raise RuntimeError(f"BigQuery load job {job_id} failed: {err}")
                    return job_id
            time.sleep(1.5)

        return job_id
    finally:
        if close_client:
            client.close()


def run_vector_db_pipeline(
    config: VectorDBBuildConfig,
    *,
    storage_client: storage.Client | None = None,
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
        _load_cached_embeddings(bq_manifest_path)
        if config.skip_existing_embeddings
        else None
    )

    requested_backend: Literal["vertex_ai", "deterministic", "auto"] = (
        "deterministic"
        if (config.local_only and config.embedding_backend == "auto")
        else config.embedding_backend
    )

    all_chunks, resolved_backend = populate_chunk_embeddings(
        all_chunks,
        project_id=config.project_id,
        location=config.location,
        model_name=config.embedding_model,
        dimensions=config.embedding_dimensions,
        backend=requested_backend,
        batch_size=config.embedding_batch_size,
        existing_embeddings=cached_embeddings,
        vertex_embedder=vertex_embedder,
    )

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
        embedding_backend=resolved_backend,
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
                http_client=http_client,
            )
            summary.loaded_to_bigquery = True

        if config.publish_monitoring_metrics:
            publisher = monitoring_publisher or CloudMonitoringHealthPublisher(
                project_id=config.project_id,
                http_client=http_client,
            )
            publisher.publish_report(health_report)
            summary.published_monitoring_metrics = True

        # Re-write local & GCS catalog/health files so published flags are persisted
        _write_vector_db_artifacts(output_dir, all_chunks, health_report, summary)
        upload_vector_db_artifacts_to_gcs(
            output_dir=output_dir,
            project_id=config.project_id,
            bucket_name=bucket_name,
            location=config.location,
            create_bucket_if_missing=False,
            storage_client=storage_client,
        )

    return summary, health_report
