#!/usr/bin/env python3
"""Create, populate, and evaluate health metrics for the SEC 10-K RAG vector database.

Produces:
  1. Metadata-tagged BigQuery VECTOR_SEARCH chunks manifest:
       data/curated_dataset/manifests/bigquery_filing_chunks.jsonl
       -> gs://<bucket>/manifests/bigquery_filing_chunks.jsonl
       -> BigQuery table `sec_edgar.filing_chunks`
  2. Vertex AI Vector Search (Matching Engine) embeddings + metadata restricts:
       data/curated_dataset/vector_search/embeddings/chunks_embeddings.json
       -> gs://<bucket>/vector_search/embeddings/chunks_embeddings.json
  3. Chunk Health Telemetry Report & Google Cloud Monitoring TimeSeries:
       data/curated_dataset/manifests/chunk_health_report.json
       -> custom.googleapis.com/sec_edgar/chunk_health/*

Examples:
  # Local staging (chunk + embed all 114 filings + compute chunk health report)
  .venv/bin/python scripts/build_vector_db.py --local-only

  # End-to-end cloud population (Vertex AI embeddings -> GCS -> BigQuery -> Cloud Monitoring)
  .venv/bin/python scripts/build_vector_db.py

  # Specific tickers or sector
  .venv/bin/python scripts/build_vector_db.py --tickers NVDA AAPL MSFT --local-only
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edgar_capstone.dataset.gcs_storage import (  # noqa: E402
    DEFAULT_BUCKET_NAME,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    GCSAccessError,
)
from edgar_capstone.dataset.pipeline import DEFAULT_OUTPUT_DIR  # noqa: E402
from edgar_capstone.vector_db.embedder import (  # noqa: E402
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
)
from edgar_capstone.vector_db.pipeline import (  # noqa: E402
    VectorDBBuildConfig,
    run_vector_db_pipeline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for building and populating the RAG vector database."""
    parser = argparse.ArgumentParser(
        description=(
            "Chunk curated SEC 10-K Markdown filings (Item 1A & Item 7), generate dense "
            "vector embeddings with metadata tags for BigQuery and Vertex AI Vector Search, "
            "and publish chunk health metrics to Google Cloud Monitoring."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Optional list of tickers to process (e.g., NVDA AAPL MSFT).",
    )
    parser.add_argument(
        "--sector",
        type=str,
        default=None,
        help="Optional GICS Sector filter (e.g., 'Information Technology').",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Optional earliest fiscal year filter (inclusive).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Optional latest fiscal year filter (inclusive).",
    )

    # Google Cloud & BigQuery settings
    parser.add_argument(
        "--project-id",
        type=str,
        default=os.environ.get("EDGAR_GCP_PROJECT_ID", DEFAULT_PROJECT_ID),
        help="Google Cloud project ID.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=os.environ.get("EDGAR_GCS_BUCKET", DEFAULT_BUCKET_NAME),
        help="Target GCS corpus bucket name.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=os.environ.get("EDGAR_BQ_DATASET", "sec_edgar"),
        help="Target BigQuery dataset ID.",
    )
    parser.add_argument(
        "--table-id",
        type=str,
        default="filing_chunks",
        help="Target BigQuery chunks table ID.",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=os.environ.get("EDGAR_GCP_REGION", DEFAULT_LOCATION),
        help="Google Cloud region for Vertex AI and BigQuery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local staging directory containing curated 10-K profiles and Markdown.",
    )

    # Chunking & Embedding settings
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Vertex AI text embedding model identifier.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSIONS,
        help="Output embedding vector dimensionality.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("auto", "vertex_ai", "deterministic"),
        default="auto",
        help="Embedding generation backend ('auto' uses Vertex AI when online, deterministic locally).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for Vertex AI embedding requests.",
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=2200,
        help="Target character length per 10-K chunk (~550 tokens).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3600,
        help="Maximum character length per 10-K chunk (~900 tokens).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=150,
        help="Minimum character length per 10-K chunk.",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=200,
        help="Target character overlap between adjacent chunks within a section.",
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Re-compute embeddings even if cached chunk text hashes match.",
    )

    # Execution & Cloud sync flags
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Stage JSONL manifests and chunk health report locally without calling GCS/BQ/Monitoring.",
    )
    parser.add_argument(
        "--no-load-bigquery",
        action="store_true",
        help="Skip triggering a direct BigQuery load job after uploading manifests to GCS.",
    )
    parser.add_argument(
        "--no-publish-metrics",
        action="store_true",
        help="Skip publishing custom chunk health time series to Google Cloud Monitoring.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config = VectorDBBuildConfig(
        project_id=args.project_id,
        bucket_name=args.bucket,
        dataset_id=args.dataset_id,
        table_id=args.table_id,
        location=args.location,
        output_dir=args.output_dir,
        tickers=args.tickers,
        sector_filter=args.sector,
        start_year=args.start_year,
        end_year=args.end_year,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        embedding_backend=args.embedding_backend,
        embedding_batch_size=args.batch_size,
        target_chunk_chars=args.target_chars,
        max_chunk_chars=args.max_chars,
        min_chunk_chars=args.min_chars,
        overlap_chunk_chars=args.overlap_chars,
        skip_existing_embeddings=not args.force_reembed,
        local_only=args.local_only,
        load_bigquery=not args.no_load_bigquery,
        publish_monitoring_metrics=not args.no_publish_metrics,
    )

    try:
        summary, health = run_vector_db_pipeline(config)
    except GCSAccessError as exc:
        logging.error("GCS access or authentication error:\n%s", exc)
        return 2
    except Exception as exc:
        logging.error("Vector database pipeline failed: %s", exc)
        return 1

    print("\n" + "=" * 78)
    print("SEC EDGAR 10-K METADATA-TAGGED VECTOR DB & CHUNK HEALTH SUMMARY")
    print("=" * 78)
    print(f"Project / Dataset : {summary.project_id}.{summary.dataset_id}.{summary.table_id}")
    print(f"GCS Corpus Bucket : gs://{summary.bucket_name}/")
    print(
        f"Embedding Config  : {summary.embedding_model} ({summary.embedding_dimensions}-d, "
        f"backend={summary.embedding_backend})"
    )
    print(f"Filings Indexed   : {health.total_profiles_indexed}/{health.total_profiles_expected} "
          f"({health.full_section_coverage_ratio * 100:.1f}% full Item 1A + Item 7 coverage)")
    print(
        f"Total Chunks      : {summary.total_chunks_generated:,} "
        f"(Item 1A: {summary.item_1a_chunks:,} | Item 7: {summary.item_7_chunks:,})"
    )
    print(
        f"Embedding Health  : {health.valid_embedding_count:,}/{health.total_chunks:,} valid "
        f"({health.embedding_coverage_ratio * 100:.2f}%)"
    )
    print(
        f"Chunk Size Stats  : avg={health.avg_char_count:.0f} chars (~{health.avg_token_estimate:.0f} tokens), "
        f"p50={health.p50_token_estimate:.0f} tok, p95={health.p95_token_estimate:.0f} tok"
    )
    print(
        f"Anomalies         : {health.anomalies.total_anomalies} "
        f"(missing_emb={health.anomalies.missing_embedding}, "
        f"dim_mismatch={health.anomalies.dimension_mismatch}, "
        f"undersized={health.anomalies.undersized_chunk}, "
        f"oversized={health.anomalies.oversized_chunk}, "
        f"missing_meta={health.anomalies.missing_metadata})"
    )
    print(f"DB Health Status  : {health.health_status} (Score: {health.health_score:.2f} / 100.0)")
    print(f"BQ Manifest       : {summary.bigquery_manifest_path}")
    print(f"Vertex AI Manifest: {summary.vertex_vector_manifest_path}")
    print(f"Health Report     : {summary.health_report_path}")
    print(
        f"Cloud Sync        : GCS={'YES' if summary.uploaded_to_gcs else 'LOCAL_ONLY'} | "
        f"BigQuery={'YES' if summary.loaded_to_bigquery else 'NO'} | "
        f"CloudMonitoring={'YES' if summary.published_monitoring_metrics else 'NO'}"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
