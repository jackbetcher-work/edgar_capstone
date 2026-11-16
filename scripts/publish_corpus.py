#!/usr/bin/env python3
"""Publish the staged SEC 10-K corpus manifests from GCS into BigQuery and Vertex AI Search.

Subcommands:
  financial-metrics  Reload native BigQuery `sec_edgar.financial_metrics` (WRITE_TRUNCATE) from
                     gs://<bucket>/manifests/bigquery_financials.jsonl with the explicit schema in
                     terraform/schemas/financial_metrics.json.
  vertex-search      Import gs://<bucket>/manifests/vertex_ai_search_metadata.jsonl into the
                     Vertex AI Search data store (documents:import), wait for the long-running
                     operation, then run a smoke search query.
  all                Run both steps in order.

Examples:
  # Reload financial_metrics and wait for completion
  .venv/bin/python scripts/publish_corpus.py financial-metrics

  # Incremental Vertex AI Search import + smoke query
  .venv/bin/python scripts/publish_corpus.py vertex-search --smoke-query "Agilent risk factors"

  # Full reconciliation (deletes documents absent from the manifest), fire-and-forget
  .venv/bin/python scripts/publish_corpus.py vertex-search --reconciliation-mode FULL --no-wait

  # Both steps
  .venv/bin/python scripts/publish_corpus.py all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edgar_capstone.dataset.cloud_publish import (  # noqa: E402
    DEFAULT_BIGQUERY_DATASET_ID,
    DEFAULT_BIGQUERY_LOCATION,
    DEFAULT_FINANCIAL_METRICS_TABLE_ID,
    DEFAULT_SMOKE_QUERY,
    DEFAULT_VERTEX_SEARCH_DATASTORE_ID,
    RECONCILIATION_MODES,
    CloudPublishError,
    import_vertex_ai_search_documents,
    reload_financial_metrics_table,
    search_vertex_ai_datastore,
)
from edgar_capstone.dataset.gcs_storage import (  # noqa: E402
    DEFAULT_BUCKET_NAME,
    DEFAULT_PROJECT_ID,
)

SUBCOMMANDS = ("financial-metrics", "vertex-search", "all")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-id",
        # Deliberately not $GCP_PROJECT_ID: shells often carry a different sandbox project there.
        default=os.environ.get("GOOGLE_CLOUD_PROJECT") or DEFAULT_PROJECT_ID,
        help="GCP project (default: $GOOGLE_CLOUD_PROJECT or %(default)s).",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("SEC_CORPUS_BUCKET") or DEFAULT_BUCKET_NAME,
        help="GCS corpus bucket holding manifests/ (default: $SEC_CORPUS_BUCKET or %(default)s).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Max seconds to wait for the job/operation (default: 600 for BigQuery, 3600 for Vertex AI Search).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit the job/operation and return without polling for completion.",
    )
    parser.add_argument("--json", action="store_true", help="Emit results as JSON.")


def _add_financial_metrics_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("BigQuery financial_metrics")
    group.add_argument("--dataset-id", default=DEFAULT_BIGQUERY_DATASET_ID)
    group.add_argument("--table-id", default=DEFAULT_FINANCIAL_METRICS_TABLE_ID)
    group.add_argument(
        "--bq-location",
        default=os.environ.get("BIGQUERY_LOCATION") or DEFAULT_BIGQUERY_LOCATION,
        help="BigQuery dataset location (default: $BIGQUERY_LOCATION or %(default)s).",
    )
    group.add_argument(
        "--source-uri",
        default=None,
        help="Override the source URI (default: gs://<bucket>/manifests/bigquery_financials.jsonl).",
    )
    group.add_argument(
        "--schema-path",
        type=Path,
        default=None,
        help="Schema JSON (relative paths resolve from the repo root; default: terraform/schemas/financial_metrics.json).",
    )
    group.add_argument(
        "--no-schema",
        action="store_true",
        help="Do not send an explicit schema; rely on the existing table schema.",
    )
    group.add_argument(
        "--write-disposition",
        choices=("WRITE_TRUNCATE", "WRITE_APPEND"),
        default="WRITE_TRUNCATE",
        help="WRITE_APPEND also sends schemaUpdateOptions=[ALLOW_FIELD_ADDITION] (default: %(default)s).",
    )
    group.add_argument(
        "--ignore-unknown-values",
        action="store_true",
        help="Skip JSON fields that are not in the schema instead of failing the load.",
    )


def _add_vertex_search_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Vertex AI Search")
    group.add_argument(
        "--data-store-id",
        default=os.environ.get("VERTEX_AI_SEARCH_DATASTORE_ID")
        or DEFAULT_VERTEX_SEARCH_DATASTORE_ID,
        help="Data store id (default: $VERTEX_AI_SEARCH_DATASTORE_ID or %(default)s).",
    )
    group.add_argument(
        "--reconciliation-mode",
        type=str.upper,
        choices=RECONCILIATION_MODES,
        default="INCREMENTAL",
        help="INCREMENTAL upserts; FULL also deletes documents missing from the manifest.",
    )
    group.add_argument(
        "--smoke-query",
        default=DEFAULT_SMOKE_QUERY,
        help="Post-import search query used as a smoke check (default: %(default)r).",
    )
    group.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the post-import smoke search query.",
    )
    group.add_argument("--smoke-page-size", type=int, default=5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish staged SEC 10-K corpus manifests to BigQuery and Vertex AI Search."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="{" + ",".join(SUBCOMMANDS) + "}"
    )

    fm = subparsers.add_parser(
        "financial-metrics", help="Reload BigQuery sec_edgar.financial_metrics from GCS."
    )
    _add_common_args(fm)
    _add_financial_metrics_args(fm)

    vs = subparsers.add_parser(
        "vertex-search", help="Import documents into the Vertex AI Search data store."
    )
    _add_common_args(vs)
    _add_vertex_search_args(vs)

    both = subparsers.add_parser("all", help="Run financial-metrics then vertex-search.")
    _add_common_args(both)
    _add_financial_metrics_args(both)
    _add_vertex_search_args(both)

    return parser.parse_args(argv)


def run_financial_metrics(args: argparse.Namespace) -> dict[str, Any]:
    result = reload_financial_metrics_table(
        project_id=args.project_id,
        bucket_name=args.bucket,
        dataset_id=args.dataset_id,
        table_id=args.table_id,
        location=args.bq_location,
        source_uri=args.source_uri,
        schema_path=args.schema_path,
        use_schema_file=not args.no_schema,
        write_disposition=args.write_disposition,
        ignore_unknown_values=args.ignore_unknown_values,
        wait=not args.no_wait,
        timeout_seconds=args.timeout_seconds or 600.0,
    )
    return result.to_dict()


def run_vertex_search(args: argparse.Namespace) -> dict[str, Any]:
    result = import_vertex_ai_search_documents(
        project_id=args.project_id,
        bucket_name=args.bucket,
        data_store_id=args.data_store_id,
        reconciliation_mode=args.reconciliation_mode,
        wait=not args.no_wait,
        timeout_seconds=args.timeout_seconds or 3600.0,
    )
    summary = result.to_dict()
    summary.pop("metadata", None)
    if result.done and not args.skip_smoke and args.smoke_query:
        summary["smoke_query"] = args.smoke_query
        summary["smoke_hits"] = search_vertex_ai_datastore(
            args.smoke_query,
            project_id=args.project_id,
            data_store_id=args.data_store_id,
            page_size=args.smoke_page_size,
        )
    return summary


def _print_summary(title: str, summary: dict[str, Any]) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    for key, value in summary.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value) if value else "-"
        print(f"{key:<22}: {value}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    results: dict[str, dict[str, Any]] = {}
    exit_code = 0
    try:
        if args.command in ("financial-metrics", "all"):
            results["financial_metrics"] = run_financial_metrics(args)
        if args.command in ("vertex-search", "all"):
            vs_summary = run_vertex_search(args)
            results["vertex_search"] = vs_summary
            if vs_summary.get("failure_count"):
                print(
                    f"WARNING: {vs_summary['failure_count']} documents failed to import; "
                    f"see {vs_summary.get('error_gcs_prefix')}",
                    file=sys.stderr,
                )
                exit_code = 1
            if vs_summary.get("smoke_query") and not vs_summary.get("smoke_hits"):
                print(
                    f"WARNING: smoke query {vs_summary['smoke_query']!r} returned no hits "
                    "(indexing can lag the import by several minutes).",
                    file=sys.stderr,
                )
    except (CloudPublishError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        exit_code = 1

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        if "financial_metrics" in results:
            _print_summary("BigQuery financial_metrics reload", results["financial_metrics"])
        if "vertex_search" in results:
            _print_summary("Vertex AI Search documents:import", results["vertex_search"])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
