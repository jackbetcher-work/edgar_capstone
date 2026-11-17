#!/usr/bin/env python3
"""Recompute the numeric fields of `expected_variances` in the golden dataset from the local,
deterministic calculation engine (`calculate_period_variances` over the curated manifest).

Use this to bootstrap new items: write the query, tickers, fiscal_years, and which metrics to
check, leave the numbers null, then run this script and REVIEW THE DIFF. Running it to make a
failing eval pass would hide a real numeric regression.

    python scripts/refresh_golden_expectations.py            # rewrite in place
    python scripts/refresh_golden_expectations.py --check    # exit 1 if any value would change
"""

from __future__ import annotations

import argparse
import sys

from edgar_capstone.evals.runner import configure_eval_environment


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", default=None)
    p.add_argument("--check", action="store_true", help="Report drift without rewriting.")
    args = p.parse_args(argv)

    configure_eval_environment("deterministic")
    from edgar_capstone.evals.golden import (
        DEFAULT_GOLDEN_DATASET_PATH,
        dump_golden_dataset,
        load_golden_dataset,
        refresh_expected_values,
    )

    path = args.dataset or DEFAULT_GOLDEN_DATASET_PATH
    items = load_golden_dataset(path)
    refreshed = [refresh_expected_values(i) for i in items]
    changed = [
        new.id
        for old, new in zip(items, refreshed, strict=True)
        if old.expected_variances != new.expected_variances
    ]
    if args.check:
        print(f"{len(changed)} item(s) would change: {changed}" if changed else "No drift.")
        return 1 if changed else 0
    dump_golden_dataset(refreshed, path)
    print(f"Refreshed {len(changed)} item(s) in {path}: {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
