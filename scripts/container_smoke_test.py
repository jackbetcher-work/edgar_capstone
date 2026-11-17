#!/usr/bin/env python3
"""Smoke test for the packaged SEC EDGAR Analyst container (or any running instance).

The image carries no SEC corpus: every data request reads live GCP (BigQuery `sec_edgar`,
Vertex AI Search, Vertex AI embeddings, GCS). The default checks therefore need no GCP
access and verify only that the image starts and is wired correctly:

  1. `GET /health` -> HTTP 200, `status == "ok"`, and a `data_backend` block naming the
     live GCP project, BigQuery dataset, and corpus bucket.
  2. `GET /`       -> the compiled Terminal UI (`frontend/dist/index.html`).

`--live-data` adds one live read, for runs that have Application Default Credentials:

  3. `POST /api/v1/auth/sso/dev-token` + `GET /api/v1/companies` -> issuers served from
     BigQuery `sec_edgar.financial_metrics`.

Examples:
  # Against a running container (`make docker-smoke` does this for you)
  python scripts/container_smoke_test.py --base-url http://127.0.0.1:8080 --wait-seconds 60

  # Inside the image, in-process (Cloud Build / GitHub Actions after `docker build`)
  docker run --rm --entrypoint python IMAGE -m scripts.container_smoke_test --in-process

  # Also read live BigQuery data (needs ADC, e.g. `--network=cloudbuild` in Cloud Build)
  python scripts/container_smoke_test.py --in-process --live-data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import AbstractContextManager
from typing import Any

import httpx

try:
    from scripts._cli import ensure_src_on_path
except ImportError:  # pragma: no cover - direct script execution without editable install
    from _cli import ensure_src_on_path  # type: ignore[import-not-found,no-redef]

REQUIRED_DATA_BACKEND_KEYS = ("project_id", "bigquery", "vertex_ai_search", "gcs_corpus_bucket")


class SmokeTestFailure(AssertionError):
    """Raised when a smoke check fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def _json(resp: Any, label: str) -> dict[str, Any]:
    _require(
        resp.status_code == 200,
        f"{label} returned HTTP {resp.status_code}: {resp.text[:400]}",
    )
    payload = resp.json()
    _require(isinstance(payload, dict), f"{label} did not return a JSON object")
    return payload


def wait_for_health(client: Any, wait_seconds: float) -> dict[str, Any]:
    """Poll `/health` until it returns HTTP 200 or the wait budget is exhausted."""
    deadline = time.monotonic() + max(0.0, wait_seconds)
    last_error = "no attempt made"
    while True:
        try:
            resp = client.get("/health")
            if resp.status_code == 200:
                return resp.json()
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        if time.monotonic() >= deadline:
            raise SmokeTestFailure(f"/health never became ready: {last_error}")
        time.sleep(1.0)


def check_health(client: Any, *, wait_seconds: float = 0.0) -> dict[str, Any]:
    """Assert `/health` is ok and reports the live GCP `data_backend` configuration."""
    health = wait_for_health(client, wait_seconds)
    _require(health.get("status") == "ok", f"/health status is {health.get('status')!r}")
    backend = health.get("data_backend")
    _require(isinstance(backend, dict) and bool(backend), "/health is missing 'data_backend'")
    missing = [key for key in REQUIRED_DATA_BACKEND_KEYS if not backend.get(key)]
    _require(not missing, f"/health data_backend is missing {missing}: {backend}")
    bigquery = backend.get("bigquery") or {}
    _require(bool(bigquery.get("dataset_id")), "/health data_backend.bigquery has no dataset_id")
    return {
        "data_backend": backend.get("data_backend"),
        "project_id": backend.get("project_id"),
        "bigquery_dataset": bigquery.get("dataset_id"),
        "gcs_corpus_bucket": backend.get("gcs_corpus_bucket"),
    }


def check_ui(client: Any) -> dict[str, Any]:
    """Assert `/` serves the compiled single-page Terminal UI."""
    resp = client.get("/")
    _require(resp.status_code == 200, f"GET / returned HTTP {resp.status_code}: {resp.text[:200]}")
    content_type = resp.headers.get("content-type", "")
    _require("text/html" in content_type, f"GET / content-type is {content_type!r}, not HTML")
    body = resp.text
    _require(
        '<div id="root"' in body and "<script" in body,
        "GET / did not return the compiled Terminal UI index.html",
    )
    return {"content_type": content_type, "bytes": len(body)}


def check_live_data(client: Any) -> dict[str, Any]:
    """Read the live issuer catalog from BigQuery through the authenticated API."""
    token_payload = _json(
        client.post("/api/v1/auth/sso/dev-token", json={}),
        "POST /api/v1/auth/sso/dev-token",
    )
    token = token_payload.get("access_token")
    _require(bool(token), "dev-token response did not include an access_token")
    companies = _json(
        client.get("/api/v1/companies", headers={"Authorization": f"Bearer {token}"}),
        "GET /api/v1/companies",
    )
    company_list = companies.get("companies") or []
    _require(
        companies.get("company_count", 0) > 0 and bool(company_list),
        f"GET /api/v1/companies returned no live issuers: {json.dumps(companies)[:400]}",
    )
    return {
        "company_count": companies.get("company_count"),
        "sample_tickers": [c.get("ticker") for c in company_list[:5]],
    }


def run_smoke_checks(
    client: Any,
    *,
    wait_seconds: float = 0.0,
    check_frontend: bool = True,
    live_data: bool = False,
) -> dict[str, Any]:
    """Execute the smoke checks against an httpx-compatible client and return a report."""
    report: dict[str, Any] = {"checks": {}}
    report["checks"]["health"] = check_health(client, wait_seconds=wait_seconds)
    if check_frontend:
        report["checks"]["ui"] = check_ui(client)
    if live_data:
        report["checks"]["live_companies"] = check_live_data(client)
    report["status"] = "passed"
    return report


def _build_client(args: argparse.Namespace) -> AbstractContextManager[Any]:
    if args.in_process:
        ensure_src_on_path()
        from fastapi.testclient import TestClient

        from edgar_capstone.server import app

        # Entering the TestClient context runs the FastAPI lifespan.
        return TestClient(app)
    return httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEC EDGAR Analyst container smoke test.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Running service URL.")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Exercise the FastAPI app in-process via TestClient instead of over HTTP.",
    )
    parser.add_argument(
        "--live-data",
        action="store_true",
        help=(
            "Also read the live issuer catalog from BigQuery (needs GCP Application Default "
            "Credentials with read access to the sec_edgar dataset)."
        ),
    )
    parser.add_argument(
        "--skip-ui",
        action="store_true",
        help="Skip the GET / Terminal UI check (e.g. a source checkout without frontend/dist).",
    )
    parser.add_argument(
        "--wait-seconds", type=float, default=0.0, help="Seconds to wait for /health."
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Per-request timeout (seconds)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with _build_client(args) as client:
            report = run_smoke_checks(
                client,
                wait_seconds=args.wait_seconds,
                check_frontend=not args.skip_ui,
                live_data=args.live_data,
            )
    except SmokeTestFailure as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
