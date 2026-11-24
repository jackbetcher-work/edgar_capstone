#!/usr/bin/env python3
"""Evaluate live BigQuery chunk health views and publish gauges to Cloud Monitoring & Cloud Logging.

Queries `sec_edgar.v_chunk_health_metrics` and `sec_edgar.v_filing_chunk_coverage` in BigQuery,
builds a `ChunkHealthReport`, and pushes fresh time-series gauges to Cloud Monitoring
(`custom.googleapis.com/sec_edgar/chunk_health/*`) plus a structured evaluation log entry to
Cloud Logging (`event_type="sec_edgar_chunk_health_evaluation"`).

Designed for both ad-hoc operator runs (`make publish-chunk-health`) and scheduled Cloud Run Job
heartbeats (`sec-edgar-analyst-chunk-health-publisher`).
"""

from __future__ import annotations

import argparse
import json
import logging
import os

try:
    from scripts._cli import add_gcp_args, add_logging_args, configure_script_logging
except ImportError:  # pragma: no cover - direct script execution without editable install
    from _cli import (  # type: ignore[import-not-found,no-redef]
        add_gcp_args,
        add_logging_args,
        configure_script_logging,
    )

from edgar_capstone.gcp.defaults import (
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    resolve_bq_dataset_id,
)
from edgar_capstone.vector_db.embedder import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
)
from edgar_capstone.vector_db.monitoring import (
    CloudMonitoringHealthPublisher,
    evaluate_chunk_health_from_bigquery,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for publishing live BigQuery chunk health metrics."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate live BigQuery SEC 10-K chunk health views and publish gauges to "
            "Google Cloud Monitoring and structured evaluation logs to Cloud Logging."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_gcp_args(
        parser,
        include_project=True,
        include_dataset=True,
        include_location=True,
        project_default=(
            os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT_ID")
            or DEFAULT_PROJECT_ID
        ),
        dataset_default=resolve_bq_dataset_id(),
        location_default=(
            os.environ.get("GOOGLE_CLOUD_LOCATION")
            or os.environ.get("GCP_REGION")
            or DEFAULT_LOCATION
        ),
        project_help="Google Cloud project ID.",
        dataset_help="BigQuery dataset ID containing chunk health views.",
        location_help="BigQuery dataset location.",
    )
    parser.add_argument(
        "--table-id",
        type=str,
        default="filing_chunks",
        help="Target BigQuery chunks table ID label for published metrics.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model label recorded on coverage time series.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSIONS,
        help="Expected vector dimensionality.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate BigQuery views and print the report without writing metrics or logs.",
    )
    parser.add_argument(
        "--no-cloud-logging",
        action="store_true",
        help="Skip writing the structured evaluation entry to the Cloud Logging REST API.",
    )
    parser.add_argument(
        "--min-section-coverage",
        type=float,
        default=0.0,
        help="Optional quality gate (0.0-1.0): exit non-zero if dual-section coverage is lower.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the evaluated ChunkHealthReport as JSON.",
    )
    add_logging_args(parser, use_verbose=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    configure_script_logging(
        verbose=args.verbose,
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        report = evaluate_chunk_health_from_bigquery(
            project_id=args.project_id,
            dataset_id=args.dataset_id,
            table_id=args.table_id,
            location=args.location,
            expected_embedding_dim=args.embedding_dimensions,
            embedding_model=args.embedding_model,
        )
        if not args.dry_run:
            publisher = CloudMonitoringHealthPublisher(
                project_id=args.project_id,
                write_cloud_logging=not args.no_cloud_logging,
            )
            try:
                publisher.publish_report(report)
            finally:
                publisher.close()
    except Exception:
        logging.exception("Chunk health evaluation/publishing failed.")
        return 1

    if args.json:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        print("=" * 72)
        print("SEC EDGAR LIVE CHUNK HEALTH EVALUATION")
        print("=" * 72)
        print(f"Project / Table   : {report.project_id}.{report.dataset_id}.{report.table_id}")
        print(
            f"Health Status     : {report.health_status} (Score: {report.health_score:.2f} / 100.0)"
        )
        print(
            f"Dual-Section Cov  : {report.profiles_with_both_sections}/{report.total_profiles_expected} "
            f"({report.full_section_coverage_ratio * 100:.2f}%)"
        )
        print(
            f"Total Chunks      : {report.total_chunks:,} "
            f"(valid embeddings: {report.valid_embedding_count:,}, "
            f"{report.embedding_coverage_ratio * 100:.2f}%)"
        )
        print(
            f"Anomalies         : total={report.anomalies.total_anomalies} "
            f"(missing_section_filings={report.anomalies.missing_section_filings}, "
            f"missing_emb={report.anomalies.missing_embedding}, "
            f"dim_mismatch={report.anomalies.dimension_mismatch}, "
            f"undersized={report.anomalies.undersized_chunk}, "
            f"oversized={report.anomalies.oversized_chunk}, "
            f"missing_meta={report.anomalies.missing_metadata})"
        )
        if report.missing_profile_ids or report.incomplete_section_profile_ids:
            sample = (report.missing_profile_ids + report.incomplete_section_profile_ids)[:10]
            print(f"Missing Profiles  : {', '.join(sample)}")
        print(
            f"Published         : {'DRY_RUN' if args.dry_run else f'YES ({report.published_timeseries_count} timeSeries)'}"
        )
        print("=" * 72)

    if (
        args.min_section_coverage > 0.0
        and report.full_section_coverage_ratio < args.min_section_coverage
    ):
        logging.error(
            "Dual-section filing coverage %.4f is below --min-section-coverage %.4f.",
            report.full_section_coverage_ratio,
            args.min_section_coverage,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
