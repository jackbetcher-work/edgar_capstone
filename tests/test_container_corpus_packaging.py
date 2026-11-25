"""Tests for container packaging, lazy import hygiene, container smoke test, and staging paths.

The image ships no SEC corpus: the runtime reads live BigQuery / Vertex AI Search / GCS data.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from edgar_capstone.dataset import paths as corpus_paths

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Ingestion staging path resolution (build-sec-dataset / build-vector-db / backfill-rd)
# ---------------------------------------------------------------------------


def test_resolve_data_root_prefers_env_then_source_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EDGAR_DATA_ROOT", raising=False)
    monkeypatch.delenv("EDGAR_DATASET_DIR", raising=False)
    assert corpus_paths.resolve_data_root() == REPO_ROOT / "data"
    assert corpus_paths.resolve_curated_dataset_dir() == REPO_ROOT / "data" / "curated_dataset"

    monkeypatch.setenv("EDGAR_DATA_ROOT", str(tmp_path))
    assert corpus_paths.resolve_data_root() == tmp_path.resolve()
    assert (
        corpus_paths.resolve_sp500_constituents_csv()
        == tmp_path.resolve() / "sp500_constituents.csv"
    )

    explicit = tmp_path / "explicit_dataset"
    monkeypatch.setenv("EDGAR_DATASET_DIR", str(explicit))
    assert corpus_paths.resolve_curated_dataset_dir() == explicit.resolve()


def test_resolve_data_root_falls_back_to_cwd_for_site_packages_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate `/opt/venv/.../site-packages` where `parents[3]/data` does not exist."""
    monkeypatch.delenv("EDGAR_DATA_ROOT", raising=False)
    fake_site_root = tmp_path / "opt" / "venv" / "lib" / "data"
    monkeypatch.setattr(corpus_paths, "_source_tree_data_root", lambda: fake_site_root)
    app_dir = tmp_path / "app"
    (app_dir / "data").mkdir(parents=True)
    monkeypatch.chdir(app_dir)
    assert corpus_paths.resolve_data_root() == (app_dir / "data").resolve()


# ---------------------------------------------------------------------------
# Container smoke test (in-process)
# ---------------------------------------------------------------------------


def test_container_smoke_script_checks_health_and_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from edgar_capstone import server
    from scripts.container_smoke_test import run_smoke_checks

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        '<!doctype html><html><body><div id="root"></div>'
        '<script type="module" src="/assets/index.js"></script></body></html>',
        encoding="utf-8",
    )
    monkeypatch.setenv("FRONTEND_DIST_DIR", str(dist))

    with TestClient(server.app) as client:
        report = run_smoke_checks(client)

    assert report["status"] == "passed"
    health = report["checks"]["health"]
    assert health["data_backend"] == "gcp"
    assert health["bigquery_dataset"] == "sec_edgar"
    assert health["gcs_corpus_bucket"]
    assert "text/html" in report["checks"]["ui"]["content_type"]
    assert "live_companies" not in report["checks"]


@pytest.mark.live
def test_container_smoke_script_live_data_reads_bigquery() -> None:
    """`--live-data` lists issuers from live BigQuery `sec_edgar.financial_metrics`."""
    from edgar_capstone import server
    from scripts.container_smoke_test import run_smoke_checks

    with TestClient(server.app) as client:
        report = run_smoke_checks(client, check_frontend=False, live_data=True)

    live = report["checks"]["live_companies"]
    assert live["company_count"] > 0
    assert all(isinstance(t, str) and t for t in live["sample_tickers"])


def test_container_smoke_script_fails_without_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from edgar_capstone import server
    from edgar_capstone.api.routers import spa as spa_router
    from scripts.container_smoke_test import SmokeTestFailure, check_ui

    monkeypatch.setenv("FRONTEND_DIST_DIR", str(tmp_path / "missing"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(spa_router, "_resolve_frontend_dist_dir", lambda: None)
    with TestClient(server.app) as client, pytest.raises(SmokeTestFailure, match="GET /"):
        check_ui(client)


def test_container_smoke_cli_flags() -> None:
    from scripts.container_smoke_test import parse_args

    args = parse_args(["--in-process", "--live-data", "--wait-seconds", "5"])
    assert args.in_process is True
    assert args.live_data is True
    assert args.skip_ui is False
    assert args.wait_seconds == 5.0
    with pytest.raises(SystemExit):
        parse_args(["--expect-corpus", "baked"])


# ---------------------------------------------------------------------------
# Packaging & security hygiene
# ---------------------------------------------------------------------------


def test_dockerfile_dockerignore_and_gitignore_exclude_corpus_and_credentials() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    runtime_stage = dockerfile.split("AS runtime", 1)[1]
    assert "COPY --chown=appuser:appgroup data/" not in dockerfile
    for asset in ("prompts/ ./prompts/", "/frontend/dist ./frontend/dist"):
        assert asset in runtime_stage
    assert "COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv" in runtime_stage
    assert runtime_stage.index("COPY --from=builder") < runtime_stage.index("USER appuser")

    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for pattern in ("data", "evals", "tests", ".env", "*credentials*.json", "gha-creds-*.json"):
        assert pattern in dockerignore

    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "gha-creds-*.json" in gitignore

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'container-smoke-test = "scripts.container_smoke_test:main"' in pyproject


def test_runtime_dependencies_exclude_dev_and_ingest_packages() -> None:
    """Keep ingestion-only & dev SDKs out of runtime `requirements.txt` and core deps."""
    req_lines = [
        line.strip().lower()
        for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for forbidden_pkg in (
        "pytest",
        "ruff",
        "pandas",
        "edgartools",
        "google-cloud-service-usage",
    ):
        assert not any(line.startswith(forbidden_pkg) for line in req_lines), (
            f"{forbidden_pkg} should not be in runtime requirements.txt"
        )

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime_deps = [d.lower() for d in pyproject["project"]["dependencies"]]
    ingest_deps = [d.lower() for d in pyproject["project"]["optional-dependencies"]["ingest"]]
    for pkg in ("edgartools", "pandas", "google-cloud-service-usage"):
        assert not any(d.startswith(pkg) for d in runtime_deps)
        assert any(d.startswith(pkg) for d in ingest_deps)
    for runtime_sdk in (
        "google-cloud-bigquery",
        "google-cloud-storage",
        "google-cloud-firestore",
        "google-cloud-modelarmor",
        "google-cloud-discoveryengine",
        "pyjwt",
    ):
        assert any(d.startswith(runtime_sdk) for d in runtime_deps)


def _run_isolated_import(code: str) -> dict[str, Any]:
    env = dict(os.environ)
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing}" if existing else src_path
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_lazy_imports_avoid_heavy_sdks_on_cold_start() -> None:
    """Run all 4 cold-start import isolation checks concurrently so runtime stays low."""
    checks: list[tuple[str, str, list[str]]] = [
        (
            "agents",
            "import edgar_capstone.agents",
            ["google.adk", "google.genai", "edgar", "pandas", "google.cloud.storage"],
        ),
        (
            "models",
            "import edgar_capstone.dataset.models, edgar_capstone.vector_db.models",
            ["edgar", "pandas", "google.cloud.storage"],
        ),
        (
            "security",
            "import edgar_capstone.security; from edgar_capstone.security.secrets import scan_text_for_exposed_secrets",
            ["fastapi", "google.adk", "edgar_capstone.telemetry"],
        ),
        (
            "server",
            "import edgar_capstone.server",
            ["edgar", "pandas", "google.cloud.storage"],
        ),
    ]

    def _run_check(entry: tuple[str, str, list[str]]) -> tuple[str, list[str]]:
        label, stmt, forbidden = entry
        snippet = (
            f"import json, sys; {stmt}; "
            f"bad = [m for m in {forbidden!r} if m in sys.modules or any(k.startswith(m + '.') for k in sys.modules)]; "
            "print(json.dumps({'bad': bad}))"
        )
        return label, list(_run_isolated_import(snippet)["bad"])

    with ThreadPoolExecutor(max_workers=len(checks)) as pool:
        results = list(pool.map(_run_check, checks))

    for label, bad in results:
        assert bad == [], f"{label} eagerly imported forbidden heavy SDK(s): {bad}"
