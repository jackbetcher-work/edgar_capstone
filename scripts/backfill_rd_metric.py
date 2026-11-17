#!/usr/bin/env python3
"""Backfill the Research & Development metric onto the staged SEC 10-K corpus.

New extractions (`build-sec-dataset`) read R&D from each filing's income statement. This
script adds it to profiles that were staged before R&D existed, without re-parsing every
10-K: it looks up the as-filed XBRL facts in SEC's `companyfacts` API (one request per
company), matched by accession number and fiscal period. Issuers that don't report R&D get
`null`, not 0.

It then rewrites `manifests/bigquery_financials.jsonl` and
`manifests/vertex_ai_search_metadata.jsonl` and, unless `--local-only`, uploads the scanned
structured JSON files (preserving GCS custom object metadata!) plus those manifests to GCS.
Load BigQuery afterwards with `make publish-financials`.

Examples:
  # Backfill the whole corpus and publish the structured JSON + manifests to GCS
  .venv/bin/python scripts/backfill_rd_metric.py

  # Two tickers, local files only
  .venv/bin/python scripts/backfill_rd_metric.py --tickers NVDA AAPL --local-only
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

try:
    from scripts._cli import (
        add_gcp_args,
        add_logging_args,
        configure_script_logging,
        parse_ticker_args,
    )
except ImportError:  # pragma: no cover - direct script execution without editable install
    from _cli import (  # type: ignore[import-not-found,no-redef]
        add_gcp_args,
        add_logging_args,
        configure_script_logging,
        parse_ticker_args,
    )

from edgar_capstone.dataset.gcs_storage import (
    DEFAULT_RETRY,
    create_storage_client,
    normalize_bucket_name,
    profile_blob_metadata,
    upload_files_to_bucket,
)
from edgar_capstone.dataset.models import DatasetBuildSummary, FinancialDataProfile
from edgar_capstone.dataset.paths import resolve_curated_dataset_dir
from edgar_capstone.dataset.pipeline import write_manifests
from edgar_capstone.dataset.xbrl_facts import (
    DEFAULT_REQUESTS_PER_SECOND,
    backfill_research_and_development,
)
from edgar_capstone.gcp.defaults import (
    resolve_corpus_bucket,
    resolve_gcp_location,
    resolve_gcp_project_id,
    resolve_sec_user_agent,
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
    parser.add_argument("--tickers", nargs="+", default=None, help="Limit to these tickers.")
    add_gcp_args(
        parser,
        include_project=True,
        include_bucket=True,
        project_default=resolve_gcp_project_id(),
        bucket_default=resolve_corpus_bucket(),
        project_help="Google Cloud project for GCS uploads.",
        bucket_help="GCS corpus bucket (default: $SEC_CORPUS_BUCKET or %(default)s).",
    )
    parser.add_argument(
        "--sec-user-agent",
        default=resolve_sec_user_agent(),
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
    add_logging_args(parser)
    return parser.parse_args(argv)


def _load_catalog(
    output_dir: Path, profiles_count: int, args: argparse.Namespace
) -> DatasetBuildSummary:
    catalog_path = output_dir / "manifests" / "dataset_catalog.json"
    if catalog_path.exists():
        summary = DatasetBuildSummary.model_validate_json(catalog_path.read_text(encoding="utf-8"))
    else:
        summary = DatasetBuildSummary(
            project_id=args.project_id,
            bucket_name=normalize_bucket_name(args.bucket),
            location=resolve_gcp_location(allow_global=False),
            start_year=2020,
            end_year=date.today().year,
            requested_tickers=[],
            processed_tickers=[],
        )
    summary.total_profiles = profiles_count
    return summary


def _sync_profile_uris(profiles: list[FinancialDataProfile], bucket_name: str) -> None:
    """Normalize in-memory GCS URIs on already-loaded profiles without re-parsing disk."""
    for prof in profiles:
        prof.unstructured_gcs_uri = f"gs://{bucket_name}/{prof.unstructured_blob_path}"
        prof.structured_gcs_uri = f"gs://{bucket_name}/{prof.structured_blob_path}"


def _upload(
    output_dir: Path,
    blob_paths: list[str],
    args: argparse.Namespace,
    *,
    profiles_by_blob: dict[str, FinancialDataProfile],
) -> int:
    """Upload updated JSON profiles and manifests to GCS preserving `blob.metadata` + retries."""
    client = create_storage_client(project_id=args.project_id)
    bucket = client.bucket(normalize_bucket_name(args.bucket))

    items: list[tuple[Path, str, dict[str, str] | None]] = []
    for rel in blob_paths:
        local_path = output_dir / rel
        if not local_path.exists():
            continue
        prof = profiles_by_blob.get(rel)
        metadata = profile_blob_metadata(prof) if prof is not None else None
        items.append((local_path, rel, metadata))

    return upload_files_to_bucket(
        bucket,
        items,
        max_workers=max(1, args.max_upload_workers),
        timeout=300.0,
        retry=DEFAULT_RETRY,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_script_logging(
        level=args.log_level,
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt=None,
    )
    output_dir = (args.output_dir or resolve_curated_dataset_dir()).resolve()
    bucket_name = normalize_bucket_name(args.bucket)
    normalized_tickers = parse_ticker_args(args.tickers)

    result = backfill_research_and_development(
        output_dir,
        tickers=normalized_tickers,
        user_agent=args.sec_user_agent,
        requests_per_second=args.requests_per_second,
        overwrite_extracted=args.overwrite_extracted,
    )

    # Reuse the already-validated profiles from the backfill scan (avoids re-parsing ~3,000 files).
    profiles = result.all_profiles
    _sync_profile_uris(profiles, bucket_name)
    catalog = _load_catalog(output_dir, len(profiles), args)
    write_manifests(output_dir, profiles, catalog)

    uploaded = 0
    if not args.local_only:
        profiles_by_blob = {p.structured_blob_path: p for p in profiles}
        # Upload every scanned profile, not just this run's changes, so a previous
        # `--local-only` run still gets published with full GCS object metadata.
        uploaded = _upload(
            output_dir,
            [*result.scanned_paths, *MANIFESTS_TO_PUBLISH],
            args,
            profiles_by_blob=profiles_by_blob,
        )

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
