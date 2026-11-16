#!/usr/bin/env python3
"""Hydrate gitignored local fallback corpus artifacts from the GCS SEC corpus bucket.

Downloads the heavyweight artifacts that are excluded from git and container builds
(`manifests/bigquery_filing_chunks.jsonl`, `unstructured/<TICKER>/<FY>_10k.md`, ...)
into the local curated dataset directory so `EDGAR_DATA_BACKEND=local` tools serve
real 10-K chunks and sections instead of synthesized fallbacks.

Examples:
  # Hydrate default scopes (chunks manifest + unstructured Markdown) into data/curated_dataset
  .venv/bin/python scripts/hydrate_corpus.py

  # Preview what would be downloaded for two tickers
  .venv/bin/python scripts/hydrate_corpus.py --scopes unstructured --tickers NVDA AAPL --dry-run

  # Mirror everything (manifests, structured, unstructured, vector_search) and overwrite
  .venv/bin/python scripts/hydrate_corpus.py --scopes all --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edgar_capstone.dataset.corpus_hydration import (  # noqa: E402
    DEFAULT_HYDRATION_SCOPES,
    HYDRATION_SCOPES,
    CorpusHydrationConfig,
    CorpusHydrationError,
    hydrate_local_corpus,
)
from edgar_capstone.dataset.gcs_storage import (  # noqa: E402
    DEFAULT_BUCKET_NAME,
    DEFAULT_PROJECT_ID,
)
from edgar_capstone.dataset.paths import resolve_curated_dataset_dir  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download gitignored SEC 10-K corpus artifacts from GCS for offline/local fallback use."
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("SEC_CORPUS_BUCKET") or DEFAULT_BUCKET_NAME,
        help="Source GCS corpus bucket (default: $SEC_CORPUS_BUCKET or %(default)s).",
    )
    parser.add_argument(
        "--project-id",
        default=os.environ.get("GCP_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or DEFAULT_PROJECT_ID,
        help="GCP project used for the Storage client.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=resolve_curated_dataset_dir(),
        help="Destination curated dataset directory (default: %(default)s).",
    )
    parser.add_argument(
        "--scopes",
        default=",".join(DEFAULT_HYDRATION_SCOPES),
        help=(
            "Comma-separated scopes to hydrate: "
            f"{', '.join(sorted(HYDRATION_SCOPES))}, or 'all' (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Restrict unstructured/structured downloads to these tickers (e.g. NVDA AAPL).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download files that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="List planned downloads without writing files.")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel download workers.")
    parser.add_argument("--json", action="store_true", help="Emit the hydration result as JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    try:
        config = CorpusHydrationConfig(
            mode="required",
            bucket_name=args.bucket,
            project_id=args.project_id,
            dataset_dir=args.dataset_dir,
            scopes=args.scopes,
            tickers=args.tickers or (),
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            max_workers=args.max_workers,
        )
    except CorpusHydrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    result = hydrate_local_corpus(config)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("=" * 72)
        print("SEC 10-K Corpus Hydration")
        print("=" * 72)
        print(f"Source            : gs://{result.bucket}/")
        print(f"Destination       : {result.dataset_dir}")
        print(f"Scopes            : {', '.join(result.scopes)}")
        if result.tickers:
            print(f"Tickers           : {', '.join(result.tickers)}")
        print(f"Status            : {result.status.upper()}")
        print(f"Objects considered: {result.objects_considered}")
        print(f"Downloaded        : {result.downloaded_objects} ({result.bytes_downloaded:,} bytes)")
        print(f"Skipped (present) : {result.skipped_existing}")
        if result.planned_objects:
            print("Planned (sample)  :")
            for name in result.planned_objects:
                print(f"  - {name}")
        for err in result.errors:
            print(f"ERROR: {err}", file=sys.stderr)
        print(f"Duration          : {result.duration_ms:.1f} ms")

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
