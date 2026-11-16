#!/usr/bin/env python3
"""Offline smoke test for the packaged SEC EDGAR Analyst container (or any running instance).

Verifies that the corpus packaged into (or hydrated inside) the image is discoverable
and that the core deterministic APIs work without BigQuery / Vertex AI Search access:

  1. `GET  /health`                      -> service up, corpus layout reported
  2. `POST /api/v1/auth/sso/dev-token`   -> local Workspace/Okta dev JWT (non-strict mode)
  3. `GET  /api/v1/companies`            -> curated issuers resolved from the baked manifest
  4. `POST /api/v1/variances`            -> deterministic period-over-period math
  5. `POST /api/v1/search`               -> metadata-filtered hybrid RAG citations
  6. `GET  /api/v1/section`              -> cached Item 7 MD&A section text

Examples:
  # Against a running container (`make docker-smoke` does this for you)
  python scripts/container_smoke_test.py --base-url http://127.0.0.1:8080 --wait-seconds 60

  # Inside the image without networking (used by Cloud Build after `docker build`)
  docker run --rm -e EDGAR_DATA_BACKEND=local --entrypoint python IMAGE \\
      scripts/container_smoke_test.py --in-process --expect-corpus baked

  # Assert the real chunks manifest + 10-K Markdown were hydrated from GCS
  python scripts/container_smoke_test.py --expect-corpus hydrated
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
try:
    # Prefer the installed package (the container's /opt/venv site-packages, exactly what
    # uvicorn imports) so in-process checks exercise the same path resolution as production.
    import edgar_capstone  # noqa: F401
except ImportError:
    if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

import httpx  # noqa: E402

EXPECT_CHOICES = ("any", "baked", "hydrated")


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


def run_smoke_checks(
    client: Any,
    *,
    expect_corpus: str = "any",
    wait_seconds: float = 0.0,
) -> dict[str, Any]:
    """Execute the smoke checks against an httpx-compatible client and return a report."""
    if expect_corpus not in EXPECT_CHOICES:
        raise ValueError(f"expect_corpus must be one of {EXPECT_CHOICES}")

    report: dict[str, Any] = {"expect_corpus": expect_corpus, "checks": {}}

    health = wait_for_health(client, wait_seconds)
    _require(health.get("status") == "ok", f"/health status is {health.get('status')!r}")
    corpus = health.get("corpus") or {}
    _require(bool(corpus), "/health is missing the 'corpus' packaging block")
    report["corpus"] = {
        k: corpus.get(k)
        for k in (
            "dataset_dir",
            "profile_source",
            "chunk_source",
            "section_source",
        )
    }
    report["corpus"]["hydration_status"] = (
        (corpus.get("hydration") or {}).get("last_result") or {}
    ).get("status")
    if expect_corpus in ("baked", "hydrated"):
        _require(
            corpus.get("profile_source") != "missing",
            f"No curated financial profiles found in {corpus.get('dataset_dir')}",
        )
    if expect_corpus == "hydrated":
        _require(
            corpus.get("chunk_source") == "chunks_manifest",
            "Expected hydrated manifests/bigquery_filing_chunks.jsonl but chunks are synthesized",
        )
        _require(
            corpus.get("section_source") == "markdown_files",
            "Expected hydrated unstructured/*.md 10-K Markdown but sections are synthesized",
        )
    report["checks"]["health"] = "ok"

    token_payload = _json(
        client.post("/api/v1/auth/sso/dev-token", json={}),
        "POST /api/v1/auth/sso/dev-token",
    )
    token = token_payload.get("access_token")
    _require(bool(token), "dev-token response did not include an access_token")
    headers = {"Authorization": f"Bearer {token}"}
    report["checks"]["dev_token"] = "ok"

    companies = _json(client.get("/api/v1/companies", headers=headers), "GET /api/v1/companies")
    company_list = companies.get("companies") or []
    _require(
        companies.get("company_count", 0) > 0 and company_list, "No curated companies resolved"
    )
    target = next((c for c in company_list if c.get("ticker") == "NVDA"), company_list[0])
    ticker = str(target["ticker"])
    years = sorted(int(y) for y in target.get("fiscal_years") or [])
    _require(bool(years), f"{ticker} has no fiscal years")
    end_year = years[-1]
    start_year = years[-2] if len(years) > 1 else end_year
    report["checks"]["companies"] = {
        "company_count": companies.get("company_count"),
        "probe_ticker": ticker,
    }

    variances = _json(
        client.post(
            "/api/v1/variances",
            json={"tickers": ticker, "start_year": start_year, "end_year": end_year},
            headers=headers,
        ),
        "POST /api/v1/variances",
    )
    _require(variances.get("status") == "success", f"variances status={variances.get('status')!r}")
    report["checks"]["variances"] = "ok"

    search = _json(
        client.post(
            "/api/v1/search",
            json={
                "query": "risk factors revenue growth outlook",
                "tickers": ticker,
                "start_year": start_year,
                "end_year": end_year,
                "top_k": 3,
            },
            headers=headers,
        ),
        "POST /api/v1/search",
    )
    _require(search.get("status") == "success", f"search status={search.get('status')!r}")
    _require(search.get("citation_count", 0) > 0, f"search returned no citations for {ticker}")
    report["checks"]["search"] = {"citation_count": search.get("citation_count")}

    section = _json(
        client.get(
            "/api/v1/section",
            params={
                "ticker": ticker,
                "fiscal_year": end_year,
                "section_id": "item_7",
                "max_chars": 2000,
            },
            headers=headers,
        ),
        "GET /api/v1/section",
    )
    _require(section.get("status") == "success", f"section status={section.get('status')!r}")
    _require(bool(str(section.get("content") or "").strip()), "section content is empty")
    report["checks"]["section"] = {"ticker": ticker, "fiscal_year": end_year}

    report["status"] = "passed"
    return report


def _build_client(args: argparse.Namespace) -> AbstractContextManager[Any]:
    if args.in_process:
        from fastapi.testclient import TestClient

        from edgar_capstone.server import app

        # Entering the TestClient context runs the FastAPI lifespan (incl. corpus hydration).
        return TestClient(app)
    return nullcontext(httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEC EDGAR Analyst container smoke test.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Running service URL.")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Exercise the FastAPI app in-process via TestClient instead of over HTTP.",
    )
    parser.add_argument(
        "--expect-corpus",
        choices=EXPECT_CHOICES,
        default="baked",
        help=(
            "'baked' requires curated financial profiles; 'hydrated' additionally requires "
            "the real chunks manifest and 10-K Markdown; 'any' skips corpus assertions."
        ),
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
                expect_corpus=args.expect_corpus,
                wait_seconds=args.wait_seconds,
            )
            if not args.in_process:
                client.close()
    except SmokeTestFailure as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
