#!/usr/bin/env python3
"""Backfill the Research & Development metric onto the staged SEC 10-K corpus.

New extractions (`build-sec-dataset`) read R&D from each filing's income statement. This
script adds it to profiles that were staged before R&D existed, without re-parsing every
10-K: it looks up the as-filed XBRL facts in SEC's `companyfacts` API (one request per
company), matched by accession number and fiscal period. Issuers that don't report R&D get
`null`, not 0.

It then rewrites `manifests/bigquery_financials.jsonl` and
`manifests/vertex_ai_search_metadata.jsonl` and, unless `--local-only`, uploads the scanned
structured JSON files plus those manifests to GCS. Load BigQuery afterwards with
`make publish-financials`.

Examples:
  # Backfill the whole corpus and publish the structured JSON + manifests to GCS
  .venv/bin/python scripts/backfill_rd_metric.py

  # Two tickers, local files only
  .venv/bin/python scripts/backfill_rd_metric.py --tickers NVDA AAPL --local-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edgar_capstone.dataset.extractor import DEFAULT_SEC_USER_AGENT  # noqa: E402
from edgar_capstone.dataset.gcs_storage import (  # noqa: E402
    DEFAULT_BUCKET_NAME,
    DEFAULT_PROJECT_ID,
    content_type_for_path,
    normalize_bucket_name,
)
from edgar_capstone.dataset.models import DatasetBuildSummary  # noqa: E402
from edgar_capstone.dataset.paths import resolve_curated_dataset_dir  # noqa: E402
from edgar_capstone.dataset.pipeline import (  # noqa: E402
    _collect_all_staged_profiles,
    _write_manifests,
)
from edgar_capstone.dataset.xbrl_facts import (  # noqa: E402
    DEFAULT_REQUESTS_PER_SECOND,
    backfill_research_and_development,
)

MANIFESTS_TO_PUBLISH = (
    "manifests/bigquery_financials.jsonl",
    "manifests/vertex_ai_search_metadata.jsonl",
    "manifests/dataset_catalog.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill R&D expense onto staged 10-K profiles from SEC XBRL companyfacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Curated dataset directory (default: resolved data/curated_dataset).",
    )
    parser.add_argument("--tickers", nargs="*", default=None, help="Limit to these tickers.")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("SEC_CORPUS_BUCKET") or DEFAULT_BUCKET_NAME,
        help="GCS corpus bucket (default: $SEC_CORPUS_BUCKET or %(default)s).",
    )
    parser.add_argument(
        "--project-id",
        default=os.environ.get("GCP_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or DEFAULT_PROJECT_ID,
        help="Google Cloud project for GCS uploads.",
    )
    parser.add_argument(
        "--sec-user-agent",
        default=os.environ.get("SEC_USER_AGENT") or DEFAULT_SEC_USER_AGENT,
        help="User-Agent sent to SEC EDGAR (SEC requires a contact identity).",
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=DEFAULT_REQUESTS_PER_SECOND,
        help="SEC request rate cap (SEC allows at most 10/s).",
    )
    parser.add_argument(
        "--overwrite-extracted",
        action="store_true",
        help="Also replace R&D values that were read from the filing during extraction.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Update local files and manifests only; skip the GCS upload.",
    )
    parser.add_argument("--max-upload-workers", type=int, default=16)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _load_catalog(output_dir: Path, profiles_count: int, args: argparse.Namespace):
    catalog_path = output_dir / "manifests" / "dataset_catalog.json"
    if catalog_path.exists():
        summary = DatasetBuildSummary.model_validate_json(catalog_path.read_text(encoding="utf-8"))
    else:
        summary = DatasetBuildSummary(
            project_id=args.project_id,
            bucket_name=normalize_bucket_name(args.bucket),
            location="us-central1",
            start_year=2020,
            end_year=2026,
            requested_tickers=[],
            processed_tickers=[],
        )
    summary.total_profiles = profiles_count
    return summary


def _upload(output_dir: Path, blob_paths: list[str], args: argparse.Namespace) -> int:
    from google.cloud import storage

    client = storage.Client(project=args.project_id)
    bucket = client.bucket(normalize_bucket_name(args.bucket))

    def _one(rel: str) -> str:
        local = output_dir / rel
        bucket.blob(rel).upload_from_filename(
            str(local), content_type=content_type_for_path(local), timeout=300
        )
        return rel

    with ThreadPoolExecutor(max_workers=max(1, args.max_upload_workers)) as pool:
        return len(list(pool.map(_one, blob_paths)))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    output_dir = (args.output_dir or resolve_curated_dataset_dir()).resolve()
    bucket_name = normalize_bucket_name(args.bucket)

    result = backfill_research_and_development(
        output_dir,
        tickers=args.tickers,
        user_agent=args.sec_user_agent,
        requests_per_second=args.requests_per_second,
        overwrite_extracted=args.overwrite_extracted,
    )

    profiles = _collect_all_staged_profiles(output_dir, bucket_name)
    catalog = _load_catalog(output_dir, len(profiles), args)
    _write_manifests(output_dir, profiles, catalog)

    uploaded = 0
    if not args.local_only:
        # Upload every scanned profile, not just this run's changes, so a previous
        # `--local-only` run still gets published.
        uploaded = _upload(output_dir, [*result.scanned_paths, *MANIFESTS_TO_PUBLISH], args)

    print(
        json.dumps(
            {
                "profiles_scanned": result.profiles_scanned,
                "profiles_with_rd": result.profiles_with_value,
                "profiles_without_rd": result.profiles_without_value,
                "profiles_updated": result.profiles_updated,
                "companies_queried": result.companies_queried,
                "companies_without_xbrl_facts": result.companies_without_facts,
                "companies_failed": result.companies_failed,
                "gcs_objects_uploaded": uploaded,
            },
            indent=2,
        )
    )
    return 1 if result.companies_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
