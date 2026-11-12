#!/usr/bin/env python3
"""Build and upload the static curated SEC 10-K dataset to Google Cloud Storage.

Creates Data Profiles consisting of:
  1. Unstructured 10-K text (Item 1A Risk Factors + Item 7 MD&A in Markdown):
       gs://<bucket>/unstructured/{TICKER}/{FY}_10k.md
  2. Structured financial metrics JSON (Revenue, Operating Income, Net Income):
       gs://<bucket>/structured/{TICKER}/{FY}_financials.json
  3. Downstream ingestion manifests:
       gs://<bucket>/manifests/vertex_ai_search_metadata.jsonl
       gs://<bucket>/manifests/bigquery_financials.jsonl
       gs://<bucket>/manifests/dataset_catalog.json

Examples:
  # Default curated S&P 500 starter set (18 companies, 2020-present) -> GCS
  .venv/bin/python scripts/build_sec_dataset.py

  # Full S&P 500 (2020-present) -> GCS
  .venv/bin/python scripts/build_sec_dataset.py --sp500

  # Specific companies staged locally first, then synced to GCS once authenticated
  .venv/bin/python scripts/build_sec_dataset.py --tickers NVDA AAPL MSFT --local-only
  .venv/bin/python scripts/build_sec_dataset.py --upload-only
"""

from __future__ import annotations

import argparse
from datetime import date
import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edgar_capstone.dataset.extractor import DEFAULT_SEC_USER_AGENT  # noqa: E402
from edgar_capstone.dataset.gcs_storage import (  # noqa: E402
    DEFAULT_BUCKET_NAME,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    GCSAccessError,
)
from edgar_capstone.dataset.pipeline import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DatasetBuildConfig,
    run_dataset_pipeline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Curate SEC EDGAR 10-K unstructured Markdown (Item 1A & Item 7) and "
            "structured financial JSON Data Profiles (Revenue, Operating Income, Net Income) "
            "and upload them to Google Cloud Storage."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Scope & company selection
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument(
        "--sp500",
        action="store_true",
        help="Process the full S&P 500 constituent list (~503 companies) instead of the starter set.",
    )
    scope_group.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Explicit list of tickers (space- or comma-separated, e.g., NVDA AAPL MSFT).",
    )
    parser.add_argument(
        "--sector",
        type=str,
        default=None,
        help="Optional GICS Sector filter (e.g., 'Information Technology').",
    )
    parser.add_argument(
        "--refresh-sp500",
        action="store_true",
        help="Refresh data/sp500_constituents.csv from upstream before running.",
    )

    # Fiscal period bounds
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="Earliest 10-K fiscal year to include (inclusive).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=date.today().year,
        help="Latest 10-K fiscal year to include (inclusive).",
    )

    # Google Cloud Storage settings
    parser.add_argument(
        "--project-id",
        type=str,
        default=os.environ.get("EDGAR_GCP_PROJECT_ID", DEFAULT_PROJECT_ID),
        help="Google Cloud project ID (override via --project-id or EDGAR_GCP_PROJECT_ID).",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=os.environ.get("EDGAR_GCS_BUCKET", DEFAULT_BUCKET_NAME),
        help="Target GCS bucket name (with or without gs:// prefix).",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=os.environ.get("EDGAR_GCP_REGION", DEFAULT_LOCATION),
        help="GCS bucket location when creating the bucket if missing.",
    )
    parser.add_argument(
        "--no-create-bucket",
        action="store_true",
        help="Fail instead of creating the GCS bucket if it does not already exist.",
    )

    # Execution & staging controls
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get("EDGAR_DATASET_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))
        ),
        help="Local staging directory mirroring the GCS object layout.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--local-only",
        action="store_true",
        help="Extract and stage all Data Profiles and manifests locally without uploading to GCS.",
    )
    mode_group.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip SEC EDGAR downloads and upload existing staged Data Profiles in --output-dir to GCS.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract filings even if already staged locally.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=os.environ.get("SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT),
        help="SEC EDGAR User-Agent header ('Organization Name email@domain.com').",
    )
    parser.add_argument(
        "--max-upload-workers",
        type=int,
        default=8,
        help="Maximum concurrent threads when uploading files to GCS.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    args = parser.parse_args(argv)
    if args.start_year > args.end_year:
        parser.error("--start-year must be less than or equal to --end-year.")
    return args


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = DatasetBuildConfig(
        project_id=args.project_id,
        bucket_name=args.bucket,
        location=args.location,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        full_sp500=args.sp500,
        tickers=args.tickers,
        sector_filter=args.sector,
        refresh_sp500=args.refresh_sp500,
        sec_user_agent=args.user_agent,
        skip_existing=not args.force,
        local_only=args.local_only,
        upload_only=args.upload_only,
        create_bucket_if_missing=not args.no_create_bucket,
        max_upload_workers=args.max_upload_workers,
    )

    try:
        summary = run_dataset_pipeline(config)
    except GCSAccessError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 2
    except Exception as exc:
        logging.getLogger(__name__).exception("Dataset pipeline failed: %s", exc)
        return 1

    print("\n" + "=" * 72)
    print("SEC 10-K Curated Dataset Summary")
    print("=" * 72)
    print(f"  GCP Project ID     : {summary.project_id}")
    print(f"  Target GCS Bucket  : gs://{summary.bucket_name} ({summary.location})")
    print(f"  Local Staging Dir  : {config.output_dir.resolve()}")
    print(f"  Fiscal Year Range  : {summary.start_year} - {summary.end_year}")
    print(
        f"  Companies Processed: {len(summary.processed_tickers)} / {len(summary.requested_tickers)}"
    )
    print(f"  Total Data Profiles: {summary.total_profiles}")
    if summary.uploaded_to_gcs:
        print(f"  GCS Upload Status  : Uploaded ({summary.uploaded_blob_count} objects)")
    else:
        print("  GCS Upload Status  : Skipped (--local-only)")
    if summary.failed_tickers:
        print(f"  Failed Tickers     : {', '.join(sorted(summary.failed_tickers))}")
    if summary.warnings:
        print(f"  Extraction Warnings: {len(summary.warnings)} (see dataset_catalog.json)")
    print("=" * 72)
    return 1 if summary.failed_tickers and not summary.processed_tickers else 0


if __name__ == "__main__":
    raise SystemExit(main())
