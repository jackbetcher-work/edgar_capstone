"""Tests for container corpus packaging: path resolution, GCS hydration, startup wiring & CI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from edgar_capstone.dataset import corpus_hydration as ch
from edgar_capstone.dataset import paths as corpus_paths
from edgar_capstone.dataset.corpus_hydration import (
    CHUNKS_MANIFEST_REL,
    CorpusHydrationConfig,
    CorpusHydrationError,
    describe_local_corpus,
    hydrate_local_corpus,
    parse_hydration_mode,
    parse_hydration_scopes,
    parse_ticker_filter,
    run_startup_corpus_hydration,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUCKET = "unit-test-sec-corpus"


# ---------------------------------------------------------------------------
# In-memory fake of the google-cloud-storage client surface used by hydration
# ---------------------------------------------------------------------------


@dataclass
class _FakeBlob:
    name: str
    data: bytes
    fail: bool = False

    @property
    def size(self) -> int:
        return len(self.data)

    def download_to_filename(self, filename: str, timeout: float | None = None) -> None:
        if self.fail:
            raise OSError(f"simulated download failure for {self.name}")
        Path(filename).write_bytes(self.data)


class _FakeBucket:
    def __init__(self, objects: dict[str, _FakeBlob]) -> None:
        self._objects = objects

    def get_blob(self, name: str) -> _FakeBlob | None:
        return self._objects.get(name)

    def blob(self, name: str) -> _FakeBlob:
        return self._objects[name]


class _FakeStorageClient:
    def __init__(self, objects: dict[str, bytes], failing: set[str] | None = None) -> None:
        failing = failing or set()
        self.objects = {
            name: _FakeBlob(name, data, fail=name in failing) for name, data in objects.items()
        }
        self.list_calls: list[str] = []

    def bucket(self, name: str) -> _FakeBucket:
        assert name == BUCKET
        return _FakeBucket(self.objects)

    def list_blobs(self, bucket_name: str, prefix: str = "") -> list[_FakeBlob]:
        assert bucket_name == BUCKET
        self.list_calls.append(prefix)
        return [b for n, b in sorted(self.objects.items()) if n.startswith(prefix)]


def _corpus_objects() -> dict[str, bytes]:
    return {
        CHUNKS_MANIFEST_REL: b'{"chunk_id": "NVDA_2025_item_7_0001"}\n',
        "manifests/bigquery_financials.jsonl": b'{"profile_id": "NVDA_2025"}\n',
        "unstructured/NVDA/2025_10k.md": b"# NVDA FY2025\n## Item 7\nRevenue grew.\n",
        "unstructured/AAPL/2025_10k.md": b"# AAPL FY2025\n## Item 7\nServices grew.\n",
        "unstructured/": b"",  # directory placeholder object
        "structured/NVDA/2025_financials.json": b"{}",
    }


def _config(tmp_path: Path, **overrides: Any) -> CorpusHydrationConfig:
    params: dict[str, Any] = {
        "mode": "required",
        "bucket_name": f"gs://{BUCKET}/",
        "project_id": "unit-test-project",
        "dataset_dir": tmp_path / "curated_dataset",
    }
    params.update(overrides)
    return CorpusHydrationConfig(**params)


@pytest.fixture(autouse=True)
def _reset_hydration_state() -> Any:
    ch.reset_corpus_hydration_state()
    yield
    ch.reset_corpus_hydration_state()


# ---------------------------------------------------------------------------
# Path resolution (non-editable container installs)
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


def test_tools_dataset_dir_uses_shared_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from edgar_capstone.agents.tools import _get_dataset_dir

    monkeypatch.delenv("EDGAR_DATASET_DIR", raising=False)
    monkeypatch.setenv("EDGAR_DATA_ROOT", str(tmp_path))
    assert _get_dataset_dir() == tmp_path.resolve() / "curated_dataset"


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_parse_hydration_mode_scopes_and_tickers() -> None:
    assert parse_hydration_mode(None) == "off"
    assert parse_hydration_mode("false") == "off"
    assert parse_hydration_mode("TRUE") == "auto"
    assert parse_hydration_mode("best-effort") == "auto"
    assert parse_hydration_mode("strict") == "required"
    with pytest.raises(CorpusHydrationError):
        parse_hydration_mode("sometimes")

    assert parse_hydration_scopes(None) == ("chunks", "unstructured")
    assert parse_hydration_scopes("unstructured, chunks unstructured") == ("unstructured", "chunks")
    assert parse_hydration_scopes("all") == (
        "manifests",
        "unstructured",
        "structured",
        "vector_search",
    )
    with pytest.raises(CorpusHydrationError):
        parse_hydration_scopes("chunks,bogus")

    assert parse_ticker_filter("nvda, brk.b") == frozenset({"NVDA", "BRK-B"})
    assert parse_ticker_filter(["NVDA,AAPL", "msft"]) == frozenset({"NVDA", "AAPL", "MSFT"})
    assert parse_ticker_filter("ALL") == frozenset()


def test_config_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGAR_CORPUS_HYDRATION", "required")
    monkeypatch.setenv("EDGAR_CORPUS_HYDRATION_SCOPES", "unstructured")
    monkeypatch.setenv("EDGAR_CORPUS_HYDRATION_TICKERS", "NVDA,AAPL")
    monkeypatch.setenv("EDGAR_CORPUS_HYDRATION_OVERWRITE", "yes")
    monkeypatch.setenv("EDGAR_CORPUS_HYDRATION_MAX_WORKERS", "3")
    monkeypatch.setenv("SEC_CORPUS_BUCKET", f"gs://{BUCKET}")
    monkeypatch.setenv("GCP_PROJECT_ID", "unit-test-project")
    monkeypatch.setenv("EDGAR_DATASET_DIR", str(tmp_path))

    cfg = CorpusHydrationConfig.from_env()
    assert cfg.mode == "required"
    assert cfg.bucket_name == BUCKET
    assert cfg.project_id == "unit-test-project"
    assert cfg.dataset_dir == tmp_path.resolve()
    assert cfg.scopes == ("unstructured",)
    assert cfg.tickers == frozenset({"NVDA", "AAPL"})
    assert cfg.overwrite is True
    assert cfg.max_workers == 3
    assert cfg.to_public_dict()["source_uri"] == f"gs://{BUCKET}/"


# ---------------------------------------------------------------------------
# Hydration behavior
# ---------------------------------------------------------------------------


def test_hydrate_downloads_missing_artifacts_and_skips_existing(tmp_path: Path) -> None:
    client = _FakeStorageClient(_corpus_objects())
    cfg = _config(tmp_path)
    dataset_dir = cfg.dataset_dir
    existing = dataset_dir / "unstructured" / "AAPL" / "2025_10k.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("LOCAL COPY", encoding="utf-8")

    result = hydrate_local_corpus(cfg, storage_client=client)

    assert result.status == "hydrated", result.errors
    assert result.ok
    assert result.downloaded_objects == 2  # chunks manifest + NVDA markdown
    assert result.skipped_existing == 1
    assert (dataset_dir / CHUNKS_MANIFEST_REL).read_bytes() == _corpus_objects()[
        CHUNKS_MANIFEST_REL
    ]
    assert (dataset_dir / "unstructured" / "NVDA" / "2025_10k.md").is_file()
    assert existing.read_text(encoding="utf-8") == "LOCAL COPY"
    # Scopes not requested are not touched.
    assert not (dataset_dir / "structured").exists()
    assert not (dataset_dir / "manifests" / "bigquery_financials.jsonl").exists()
    # No temp/partial files remain.
    assert not list(dataset_dir.rglob("*.part"))

    second = hydrate_local_corpus(cfg, storage_client=client)
    assert second.status == "up_to_date"
    assert second.downloaded_objects == 0

    overwrite = hydrate_local_corpus(_config(tmp_path, overwrite=True), storage_client=client)
    assert overwrite.downloaded_objects == 3
    assert existing.read_bytes() == _corpus_objects()["unstructured/AAPL/2025_10k.md"]


def test_hydrate_ticker_filter_and_dry_run(tmp_path: Path) -> None:
    client = _FakeStorageClient(_corpus_objects())
    cfg = _config(tmp_path, scopes="unstructured,structured", tickers=["nvda"], dry_run=True)

    result = hydrate_local_corpus(cfg, storage_client=client)

    assert result.status == "dry_run"
    assert result.ok
    assert result.planned_objects == [
        "structured/NVDA/2025_financials.json",
        "unstructured/NVDA/2025_10k.md",
    ]
    assert set(client.list_calls) == {"unstructured/NVDA/", "structured/NVDA/"}
    assert not cfg.dataset_dir.exists()


def test_hydrate_reports_missing_objects_failures_and_traversal(tmp_path: Path) -> None:
    objects = _corpus_objects()
    objects.pop(CHUNKS_MANIFEST_REL)
    objects["unstructured/../../escape.md"] = b"nope"
    client = _FakeStorageClient(objects, failing={"unstructured/AAPL/2025_10k.md"})

    result = hydrate_local_corpus(_config(tmp_path), storage_client=client)

    assert result.status == "partial"
    assert not result.ok
    assert result.downloaded_objects == 1  # NVDA markdown only
    joined = "\n".join(result.errors)
    assert f"Missing GCS object gs://{BUCKET}/{CHUNKS_MANIFEST_REL}" in joined
    assert "simulated download failure" in joined
    assert "escapes dataset dir" in joined
    assert not (tmp_path / "escape.md").exists()


def test_hydrate_handles_client_initialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(project_id: str) -> Any:
        raise RuntimeError("no application default credentials")

    monkeypatch.setattr(ch, "_build_storage_client", _boom)
    result = hydrate_local_corpus(_config(tmp_path))
    assert result.status == "failed"
    assert "no application default credentials" in result.errors[0]


def test_startup_hydration_modes(tmp_path: Path) -> None:
    client = _FakeStorageClient({})  # empty bucket -> chunks manifest missing

    disabled = run_startup_corpus_hydration(_config(tmp_path, mode="off"), storage_client=client)
    assert disabled.status == "disabled"
    assert client.list_calls == []

    best_effort = run_startup_corpus_hydration(
        _config(tmp_path, mode="auto"), storage_client=client
    )
    assert best_effort.status == "failed"
    assert ch.get_last_corpus_hydration_result() is best_effort

    with pytest.raises(CorpusHydrationError, match="Required corpus hydration"):
        run_startup_corpus_hydration(_config(tmp_path, mode="required"), storage_client=client)


def test_describe_local_corpus_reports_fallback_sources(tmp_path: Path) -> None:
    empty = describe_local_corpus(tmp_path)
    assert empty["profile_source"] == "missing"
    assert empty["chunk_source"] == "synthesized_from_profiles"
    assert empty["section_source"] == "synthesized_from_profiles"

    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "bigquery_financials.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / CHUNKS_MANIFEST_REL).write_text("{}\n", encoding="utf-8")
    (tmp_path / "unstructured" / "NVDA").mkdir(parents=True)
    (tmp_path / "unstructured" / "NVDA" / "2025_10k.md").write_text("# x", encoding="utf-8")
    full = describe_local_corpus(tmp_path)
    assert full["profile_source"] == "financials_manifest"
    assert full["chunk_source"] == "chunks_manifest"
    assert full["section_source"] == "markdown_files"


# ---------------------------------------------------------------------------
# FastAPI startup wiring & offline smoke test
# ---------------------------------------------------------------------------


def test_server_lifespan_hydrates_and_health_reports_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from edgar_capstone import server
    from edgar_capstone.agents import tools

    dataset_dir = tmp_path / "curated_dataset"
    monkeypatch.setenv("EDGAR_DATASET_DIR", str(dataset_dir))
    monkeypatch.setenv("EDGAR_CORPUS_HYDRATION", "auto")
    monkeypatch.setenv("SEC_CORPUS_BUCKET", BUCKET)
    fake_client = _FakeStorageClient(_corpus_objects())
    monkeypatch.setattr(ch, "_build_storage_client", lambda project_id: fake_client)

    try:
        with TestClient(server.app) as client:
            assert (dataset_dir / CHUNKS_MANIFEST_REL).is_file()
            body = client.get("/health").json()
    finally:
        tools._load_all_chunks_cached.cache_clear()
        tools._load_all_profiles_cached.cache_clear()

    corpus = body["corpus"]
    assert corpus["dataset_dir"] == str(dataset_dir.resolve())
    assert corpus["chunk_source"] == "chunks_manifest"
    assert corpus["section_source"] == "markdown_files"
    assert corpus["hydration"]["mode"] == "auto"
    assert corpus["hydration"]["last_result"]["status"] == "hydrated"
    assert corpus["hydration"]["last_result"]["downloaded_objects"] == 3


def test_container_smoke_script_passes_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    from edgar_capstone import server
    from scripts.container_smoke_test import run_smoke_checks

    monkeypatch.delenv("EDGAR_DATASET_DIR", raising=False)
    monkeypatch.setenv("EDGAR_CORPUS_HYDRATION", "off")
    monkeypatch.setenv("EDGAR_DATA_BACKEND", "local")
    monkeypatch.setenv("EDGAR_EXECUTION_MODE", "deterministic")
    monkeypatch.setenv("EDGAR_TELEMETRY_BACKEND", "local")
    monkeypatch.setenv("EDGAR_FEEDBACK_BACKEND", "local")

    with TestClient(server.app) as client:
        report = run_smoke_checks(client, expect_corpus="baked")

    assert report["status"] == "passed"
    assert report["corpus"]["profile_source"] == "financials_manifest"
    assert report["checks"]["search"]["citation_count"] > 0


# ---------------------------------------------------------------------------
# Packaging & CI wiring
# ---------------------------------------------------------------------------


def test_dockerfile_dockerignore_and_makefile_package_corpus() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'EDGAR_DATASET_DIR="/app/data/curated_dataset"' in dockerfile
    assert 'EDGAR_CORPUS_HYDRATION="off"' in dockerfile
    assert "chown -R appuser:appgroup /app/data" in dockerfile
    assert "describe_local_corpus" in dockerfile
    assert dockerfile.index("chown -R appuser:appgroup /app/data") < dockerfile.index(
        "USER appuser"
    )

    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for pattern in (
        "data/curated_dataset/unstructured",
        "data/curated_dataset/manifests/bigquery_filing_chunks.jsonl",
        "data/curated_dataset/vector_search",
    ):
        assert pattern in dockerignore
    # Baked manifests copied by the Dockerfile must exist in git and not be docker-ignored.
    for line in dockerfile.splitlines():
        if line.startswith("COPY") and "data/curated_dataset/manifests/" in line:
            src = line.split()[2]
            assert src not in dockerignore
            assert (REPO_ROOT / src).is_file(), f"Baked corpus file missing from repo: {src}"

    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    for target in (
        "hydrate-corpus:",
        "smoke:",
        "docker-build:",
        "docker-smoke:",
        "docker-smoke-hydrated:",
    ):
        assert target in makefile
    assert "EDGAR_CORPUS_HYDRATION=required" in makefile

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'hydrate-sec-corpus = "scripts.hydrate_corpus:main"' in pyproject
    assert 'container-smoke-test = "scripts.container_smoke_test:main"' in pyproject


def test_ci_pipelines_run_offline_container_smoke_test() -> None:
    cloudbuild = yaml.safe_load((REPO_ROOT / "cloudbuild.yaml").read_text(encoding="utf-8"))
    build_step = next(s for s in cloudbuild["steps"] if s["id"] == "build")
    build_script = build_step["args"][-1]
    assert "scripts/container_smoke_test.py --in-process --expect-corpus baked" in build_script
    assert build_script.index("container_smoke_test.py") < build_script.index("docker push")

    workflow_text = (REPO_ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    assert "scripts/container_smoke_test.py --in-process --expect-corpus baked" in workflow_text


def test_hydrate_cli_dry_run_uses_fake_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import hydrate_corpus

    fake_client = _FakeStorageClient(_corpus_objects())
    monkeypatch.setattr(ch, "_build_storage_client", lambda project_id: fake_client)
    exit_code = hydrate_corpus.main(
        [
            "--bucket",
            BUCKET,
            "--dataset-dir",
            str(tmp_path),
            "--scopes",
            "unstructured",
            "--tickers",
            "NVDA",
            "--dry-run",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert '"status": "dry_run"' in out
    assert "unstructured/NVDA/2025_10k.md" in out
    assert not any(tmp_path.rglob("*.md"))


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
