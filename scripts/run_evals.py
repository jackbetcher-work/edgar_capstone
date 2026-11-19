#!/usr/bin/env python3
"""Run the golden eval regression suite (`make eval-all`).

Examples:
    python scripts/run_evals.py                              # live Gemini + Flash judge, concurrent pool (default 4)
    python scripts/run_evals.py --concurrency 1              # sequential run; enforces the p95_ttft_ms gate
    python scripts/run_evals.py --only nvda_longitudinal_fy24_fy26 --only safety_prompt_injection
    python scripts/run_evals.py --dataset bigquery           # read sec_edgar.golden_eval_set
    python scripts/run_evals.py --write-bigquery             # stream the run to sec_edgar.eval_results

Note on concurrency & TTFT:
    Running multiple golden items concurrently (`--concurrency > 1`) speeds up wall-clock
    execution (~4-5x) by overlapping sync GCP pre-pass I/O and multiplexing ADK Gemini
    streams on the shared event loop, but concurrent traffic can inflate `p95_ttft_ms`.
    The `p95_ttft_ms` gate is automatically skipped whenever more than one item runs
    concurrently (the measured value is still recorded); pass `--concurrency 1` to enforce it.

Exit codes: 0 all gates passed · 1 regression (a gate failed) · 2 invalid golden dataset / args ·
3 BigQuery read/write failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts._cli import add_gcp_args
except ImportError:  # pragma: no cover - direct script execution without editable install
    from _cli import add_gcp_args  # type: ignore[import-not-found,no-redef]

from edgar_capstone.evals.golden import (
    DEFAULT_GOLDEN_DATASET_PATH,
    REPO_ROOT,
    GoldenDatasetError,
    load_golden_dataset,
    validate_golden_dataset,
)
from edgar_capstone.evals.runner import (
    DEFAULT_EVAL_CONCURRENCY,
    DEFAULT_THRESHOLDS_PATH,
    EvalThresholds,
    format_report,
    run_golden_evals,
    write_report_json,
)
from edgar_capstone.gcp.defaults import (
    resolve_bq_dataset_id,
    resolve_gcp_location,
    resolve_gcp_project_id,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--dataset",
        default=str(DEFAULT_GOLDEN_DATASET_PATH.relative_to(REPO_ROOT)),
        help="Golden JSONL path, or 'bigquery' to read sec_edgar.golden_eval_set.",
    )
    p.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "build" / "evals")
    p.add_argument("--only", action="append", default=[], help="Run only these item ids.")
    p.add_argument("--category", action="append", default=[], help="Run only these categories.")
    p.add_argument(
        "-j",
        "--concurrency",
        type=int,
        default=DEFAULT_EVAL_CONCURRENCY,
        help=(
            f"Max golden items to evaluate concurrently (default: {DEFAULT_EVAL_CONCURRENCY}). "
            "When > 1, items execute in a bounded worker pool and the p95_ttft_ms gate is "
            "skipped because concurrent requests can inflate TTFT; use --concurrency 1 to enforce it."
        ),
    )
    p.add_argument(
        "--write-bigquery",
        action="store_true",
        help="Stream the run summary to BigQuery sec_edgar.eval_results (fails the run if it can't).",
    )
    add_gcp_args(
        p,
        include_project=True,
        include_dataset=True,
        include_location=True,
        dataset_flag="--bq-dataset",
        project_default=resolve_gcp_project_id(),
        dataset_default=resolve_bq_dataset_id(),
        location_default=resolve_gcp_location(allow_global=False),
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-item progress lines.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.concurrency < 1:
        print(f"ERROR: --concurrency must be >= 1 (got {args.concurrency})", file=sys.stderr)
        return 2

    from edgar_capstone.evals.bigquery_io import EvalBigQueryClient, EvalBigQueryError

    with EvalBigQueryClient(
        project_id=args.project_id, dataset_id=args.bq_dataset, location=args.location
    ) as bq:
        try:
            if args.dataset.lower() == "bigquery":
                items = bq.read_golden_dataset()
                source = f"bigquery:{args.project_id}.{args.bq_dataset}.golden_eval_set"
            else:
                items = load_golden_dataset(args.dataset)
                source = args.dataset
            thresholds = EvalThresholds.load(args.thresholds)
            counts = validate_golden_dataset(items, min_scored_items=thresholds.min_scored_items)
        except (GoldenDatasetError, FileNotFoundError, ValueError) as exc:
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
            print(
                f"ERROR: no golden items selected (unknown ids: {sorted(unknown)})", file=sys.stderr
            )
            return 2

        effective_workers = min(args.concurrency, len(selected))
        ttft_note = (
            "; p95_ttft_ms gate skipped under concurrency (use --concurrency 1 to enforce)"
            if effective_workers > 1
            else ""
        )
        print(
            f"Running {len(selected)}/{len(items)} golden items live "
            f"(workers: {effective_workers}{ttft_note}; dataset: {source}; "
            f"categories: {json.dumps(counts)})",
            flush=True,
        )
        report = run_golden_evals(
            selected,
            thresholds=thresholds,
            dataset_source=source,
            progress=None if args.quiet else (lambda line: print(line, flush=True)),
            max_workers=args.concurrency,
        )
        print(format_report(report))
        report_path = write_report_json(report, args.output_dir)
        print(f"Report: {report_path}")

        if args.write_bigquery:
            try:
                written = bq.stream_eval_result(report.to_bigquery_row())
                print(f"Streamed eval_results row {written['run_id']} -> {written['table']}")
            except Exception as exc:
                print(f"ERROR writing eval_results to BigQuery: {exc}", file=sys.stderr)
                return 3

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
