#!/usr/bin/env python3
"""Validate `evals/golden_dataset.jsonl` and load it into BigQuery `sec_edgar.golden_eval_set`.

The load is WRITE_TRUNCATE, so the table always mirrors the committed file. Each row carries
`dataset_version` (a content hash) that `scripts/run_evals.py` also stamps on every
`eval_results` row.

    python scripts/load_golden_dataset.py --project-id iron-radio-514217-q7
    python scripts/load_golden_dataset.py --dry-run      # validate only
"""

from __future__ import annotations

import argparse
import json
import sys

from edgar_capstone.dataset.gcs_storage import DEFAULT_LOCATION, DEFAULT_PROJECT_ID
from edgar_capstone.evals.bigquery_io import EvalBigQueryClient, EvalBigQueryError
from edgar_capstone.evals.golden import (
    DEFAULT_GOLDEN_DATASET_PATH,
    GoldenDatasetError,
    dataset_fingerprint,
    load_golden_dataset,
    validate_golden_dataset,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", default=str(DEFAULT_GOLDEN_DATASET_PATH))
    p.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    p.add_argument("--bq-dataset", default="sec_edgar")
    p.add_argument("--table", default="golden_eval_set")
    p.add_argument("--location", default=DEFAULT_LOCATION)
    p.add_argument("--dry-run", action="store_true", help="Validate the file without loading.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        items = load_golden_dataset(args.dataset)
        counts = validate_golden_dataset(items)
    except GoldenDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    version = dataset_fingerprint(items)
    print(f"Validated {len(items)} golden items (version {version}): {json.dumps(counts)}")
    if args.dry_run:
        return 0

    client = EvalBigQueryClient(
        project_id=args.project_id, dataset_id=args.bq_dataset, location=args.location
    )
    try:
        result = client.load_golden_dataset(items, table_id=args.table)
    except EvalBigQueryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(f"Loaded {result['rows_loaded']} rows into {result['table']} (job {result['job_id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
