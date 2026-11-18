"""Tests for container packaging, the container smoke test, and ingestion staging paths.

The image ships no SEC corpus: the runtime reads live BigQuery / Vertex AI Search / GCS data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from edgar_capstone.dataset import paths as corpus_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
BUCKET = "unit-test-sec-corpus"


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
    monkeypatch.setattr(corpus_paths, "CONTAINER_DATA_ROOT", tmp_path / "missing_app_data")
    app_dir = tmp_path / "app"
    (app_dir / "data").mkdir(parents=True)
    monkeypatch.chdir(app_dir)
    assert corpus_paths.resolve_data_root() == (app_dir / "data").resolve()


def test_runtime_no_longer_ships_corpus_hydration() -> None:
    assert not (REPO_ROOT / "src" / "edgar_capstone" / "dataset" / "corpus_hydration.py").exists()
    assert not (REPO_ROOT / "scripts" / "hydrate_corpus.py").exists()


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


def test_container_smoke_script_live_data_reads_bigquery(monkeypatch: pytest.MonkeyPatch) -> None:
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
    from scripts.container_smoke_test import SmokeTestFailure, check_ui

    monkeypatch.setenv("FRONTEND_DIST_DIR", str(tmp_path / "missing"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server, "_resolve_frontend_dist_dir", lambda: None)
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
# Packaging & CI wiring
# ---------------------------------------------------------------------------


def test_dockerfile_dockerignore_and_makefile_ship_no_corpus() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    runtime_stage = dockerfile.split("AS runtime", 1)[1]
    for removed in (
        "curated_dataset",
        "EDGAR_DATASET_DIR",
        "EDGAR_DATA_ROOT",
        "EDGAR_CORPUS_HYDRATION",
        "corpus_hydration",
        "describe_local_corpus",
        "sp500_constituents.csv",
        "COPY --chown=appuser:appgroup data/",
    ):
        assert removed not in dockerfile, removed
    for asset in ("prompts/ ./prompts/", "/frontend/dist ./frontend/dist"):
        assert asset in runtime_stage
    # The un-deployed LookML semantic layer was removed; it must not ship in the image.
    assert "looker/" not in dockerfile
    assert "COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv" in runtime_stage
    assert runtime_stage.index("COPY --from=builder") < runtime_stage.index("USER appuser")

    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for pattern in ("data", "evals", "tests", ".env", "*credentials*.json", "gha-creds-*.json"):
        assert pattern in dockerignore
    assert not any("HYDRATION" in line for line in dockerignore)

    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "gha-creds-*.json" in gitignore

    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("smoke:", "docker-build:", "docker-smoke:"):
        assert f"\n{target}" in makefile
    for removed in (
        "hydrate-corpus",
        "docker-smoke-hydrated",
        "HYDRATION",
        "OFFLINE_ENV",
        "EVAL_MODE",
        "--mode",
        "--expect-corpus",
        "EDGAR_DATA_BACKEND",
        "EDGAR_EXECUTION_MODE",
        "deterministic",
    ):
        assert removed not in makefile, removed

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "hydrate_corpus" not in pyproject
    assert 'container-smoke-test = "scripts.container_smoke_test:main"' in pyproject


def test_makefile_pins_live_gcp_env_for_dev_test_and_eval_targets() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "CORPUS_PROJECT_ID ?= iron-radio-514217-q7" in makefile
    live_env = makefile.split("LIVE_GCP_ENV :=", 1)[1].split("\n\n", 1)[0]
    for assignment in (
        "GCP_PROJECT_ID=$(CORPUS_PROJECT_ID)",
        "GOOGLE_CLOUD_PROJECT=$(CORPUS_PROJECT_ID)",
        "GOOGLE_CLOUD_LOCATION=$(GCP_LOCATION)",
        "GOOGLE_GENAI_USE_VERTEXAI=true",
        "EDGAR_GEMINI_LOCATION=$(GEMINI_LOCATION)",
        "LIVE_TEST_ENV := $(LIVE_GCP_ENV) EDGAR_TELEMETRY_BACKEND=local "
        "EDGAR_FEEDBACK_BACKEND=local",
    ):
        assert assignment in live_env
    assert "GCP_LOCATION ?= us-central1" in makefile
    assert "GEMINI_LOCATION ?= global" in makefile

    def recipe(target: str) -> str:
        return makefile.split(f"\n{target}:", 1)[1].split("\n\n", 1)[0]

    for target in ("dev", "backend", "start"):
        assert "$(LIVE_GCP_ENV) $(PYTHON) -m uvicorn" in recipe(target), target
    assert "$(LIVE_TEST_ENV) $(PYTHON) -m pytest" in recipe("test")
    assert "$(LIVE_TEST_ENV) $(PYTHON) scripts/run_evals.py --dataset" in recipe("eval-all")
    assert "$(LIVE_TEST_ENV) $(PYTHON) scripts/container_smoke_test.py --in-process" in recipe(
        "smoke"
    )
    assert "$(DOCKER_LIVE_TEST_ENV)" in recipe("docker-smoke")


def test_ci_pipelines_run_container_smoke_test_without_offline_modes() -> None:
    cloudbuild_text = (REPO_ROOT / "cloudbuild.yaml").read_text(encoding="utf-8")
    cloudbuild = yaml.safe_load(cloudbuild_text)
    build_step = next(s for s in cloudbuild["steps"] if s["id"] == "build")
    build_script = build_step["args"][-1]
    assert "-m scripts.container_smoke_test --in-process --live-data" in build_script
    assert "--network=cloudbuild" in build_script
    assert build_script.index("scripts.container_smoke_test") < build_script.index("docker push")

    workflow_text = (REPO_ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    assert "-m scripts.container_smoke_test --in-process" in workflow_text

    for text in (cloudbuild_text, workflow_text):
        for removed in (
            "--expect-corpus",
            "EDGAR_DATA_BACKEND",
            "EDGAR_EXECUTION_MODE",
            "EDGAR_MODEL_ARMOR_MODE=disabled",
            "EDGAR_CORPUS_HYDRATION",
        ):
            assert removed not in text, removed


# ---------------------------------------------------------------------------
# Ingestion upload regression
# ---------------------------------------------------------------------------


def test_vector_db_upload_builds_storage_client_when_not_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `build-vector-db` previously passed `client=None` to `ensure_gcs_bucket`."""
    from edgar_capstone.vector_db import pipeline as vdb_pipeline

    uploaded: list[str] = []

    class _Blob:
        def __init__(self, name: str) -> None:
            self.name = name

        def upload_from_filename(self, filename: str, content_type: str | None = None) -> None:
            uploaded.append(self.name)

    class _Bucket:
        location = "US-CENTRAL1"

        def blob(self, name: str) -> _Blob:
            return _Blob(name)

    class _Client:
        def __init__(self, project: str | None = None) -> None:
            self.project = project

        def lookup_bucket(self, name: str) -> _Bucket:
            return _Bucket()

    monkeypatch.setattr(vdb_pipeline.storage, "Client", _Client)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "bigquery_filing_chunks.jsonl").write_text("{}\n", encoding="utf-8")

    count = vdb_pipeline.upload_vector_db_artifacts_to_gcs(
        output_dir=tmp_path,
        project_id="unit-test-project",
        bucket_name=BUCKET,
    )
    assert count == 1
    assert uploaded == ["manifests/bigquery_filing_chunks.jsonl"]
