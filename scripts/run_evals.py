#!/usr/bin/env python3
"""Run the golden eval regression suite (`make eval-all`).

Examples:
    python scripts/run_evals.py                              # live Gemini + Flash judge, live GCP data
    python scripts/run_evals.py --only nvda_longitudinal_fy24_fy26 --only safety_prompt_injection
    python scripts/run_evals.py --dataset bigquery           # read sec_edgar.golden_eval_set
    python scripts/run_evals.py --write-bigquery             # stream the run to sec_edgar.eval_results

Exit codes: 0 all gates passed · 1 regression (a gate failed) · 2 invalid golden dataset ·
3 BigQuery read/write failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from edgar_capstone.dataset.gcs_storage import DEFAULT_LOCATION, DEFAULT_PROJECT_ID
from edgar_capstone.evals.golden import (
    DEFAULT_GOLDEN_DATASET_PATH,
    REPO_ROOT,
    GoldenDatasetError,
    load_golden_dataset,
    validate_golden_dataset,
)
from edgar_capstone.evals.runner import (
    DEFAULT_THRESHOLDS_PATH,
    EvalThresholds,
    format_report,
    run_golden_evals,
    write_report_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--dataset",
        default=str(DEFAULT_GOLDEN_DATASET_PATH.relative_to(REPO_ROOT)),
        help="Golden JSONL path, or 'bigquery' to read sec_edgar.golden_eval_set.",
    )
    p.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS_PATH))
    p.add_argument("--output-dir", default=str(REPO_ROOT / "build" / "evals"))
    p.add_argument("--only", action="append", default=[], help="Run only these item ids.")
    p.add_argument("--category", action="append", default=[], help="Run only these categories.")
    p.add_argument(
        "--write-bigquery",
        action="store_true",
        help="Stream the run summary to BigQuery sec_edgar.eval_results (fails the run if it can't).",
    )
    p.add_argument("--project-id", default=os.environ.get("GCP_PROJECT_ID", DEFAULT_PROJECT_ID))
    p.add_argument("--bq-dataset", default=os.environ.get("BIGQUERY_DATASET_ID", "sec_edgar"))
    p.add_argument("--location", default=DEFAULT_LOCATION)
    p.add_argument("--quiet", action="store_true", help="Suppress per-item progress lines.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from edgar_capstone.evals.bigquery_io import EvalBigQueryClient, EvalBigQueryError

    bq = EvalBigQueryClient(
        project_id=args.project_id, dataset_id=args.bq_dataset, location=args.location
    )
    try:
        if args.dataset.lower() == "bigquery":
            items = bq.read_golden_dataset()
            source = f"bigquery:{args.project_id}.{args.bq_dataset}.golden_eval_set"
        else:
            items = load_golden_dataset(args.dataset)
            source = args.dataset
        thresholds = EvalThresholds.load(args.thresholds)
        counts = validate_golden_dataset(items, min_scored_items=thresholds.min_scored_items)
    except GoldenDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except EvalBigQueryError as exc:
        print(f"ERROR reading golden set from BigQuery: {exc}", file=sys.stderr)
        return 3

    selected = [
        i
        for i in items
        if (not args.only or i.id in args.only)
        and (not args.category or i.category in args.category)
    ]
    unknown = set(args.only) - {i.id for i in items}
    if unknown or not selected:
        print(f"ERROR: no golden items selected (unknown ids: {sorted(unknown)})", file=sys.stderr)
        return 2

    print(
        f"Running {len(selected)}/{len(items)} golden items live "
        f"(dataset: {source}; categories: {json.dumps(counts)})",
        flush=True,
    )
    report = run_golden_evals(
        selected,
        thresholds=thresholds,
        dataset_source=source,
        progress=None if args.quiet else (lambda line: print(line, flush=True)),
    )
    print(format_report(report))
    report_path = write_report_json(report, Path(args.output_dir))
    print(f"Report: {report_path}")

    if args.write_bigquery:
        try:
            written = bq.stream_eval_result(report.to_bigquery_row())
            print(f"Streamed eval_results row {written['run_id']} -> {written['table']}")
        except Exception as exc:  # noqa: BLE001 - credentials, IAM, schema drift
            print(f"ERROR writing eval_results to BigQuery: {exc}", file=sys.stderr)
            return 3

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
