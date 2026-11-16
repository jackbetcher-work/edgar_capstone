"""Hydrate gitignored local fallback corpus artifacts from the GCS SEC corpus bucket.

The production container only bakes lightweight, git-tracked manifests
(`bigquery_financials.jsonl`, `chunk_health_report.json`, catalogs). The heavyweight
artifacts that the local (`EDGAR_DATA_BACKEND=local` / `auto` fallback) tool paths
read are gitignored and therefore absent from CI checkouts and container builds:

* `manifests/bigquery_filing_chunks.jsonl` (~100 MB of 768-d embedded chunks)
* `unstructured/<TICKER>/<FY>_10k.md` (cleaned Item 1A / Item 7 Markdown)

Without them the local tools still work but synthesize deterministic placeholder
chunks and 10-K sections. For offline container smoke tests (no BigQuery /
Vertex AI Search access) this module optionally downloads the real artifacts from
`gs://<SEC_CORPUS_BUCKET>/` into the active curated dataset directory on startup.

Runtime controls (all optional):

* `EDGAR_CORPUS_HYDRATION` — `off` (default) | `auto` (best effort, never blocks
  startup) | `required` (fail startup if hydration fails).
* `EDGAR_CORPUS_HYDRATION_SCOPES` — comma-separated scopes (default
  `chunks,unstructured`); see `HYDRATION_SCOPES`, or `all`.
* `EDGAR_CORPUS_HYDRATION_TICKERS` — restrict `unstructured` / `structured`
  downloads to specific tickers (e.g. `NVDA,AAPL`).
* `EDGAR_CORPUS_HYDRATION_OVERWRITE` — re-download files that already exist locally.
* `EDGAR_CORPUS_HYDRATION_MAX_WORKERS` — parallel download workers (default 8).
* `SEC_CORPUS_BUCKET`, `GCP_PROJECT_ID` / `GOOGLE_CLOUD_PROJECT` — GCS source.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from edgar_capstone.dataset.gcs_storage import (
    DEFAULT_BUCKET_NAME,
    DEFAULT_PROJECT_ID,
    normalize_bucket_name,
)
from edgar_capstone.dataset.paths import resolve_curated_dataset_dir

logger = logging.getLogger(__name__)

HydrationMode = Literal["off", "auto", "required"]
HydrationStatus = Literal["disabled", "hydrated", "up_to_date", "dry_run", "partial", "failed"]

FINANCIALS_MANIFEST_REL = "manifests/bigquery_financials.jsonl"
CHUNKS_MANIFEST_REL = "manifests/bigquery_filing_chunks.jsonl"
SEARCH_METADATA_MANIFEST_REL = "manifests/vertex_ai_search_metadata.jsonl"

# scope name -> (kind, GCS object name or prefix relative to the bucket root)
HYDRATION_SCOPES: dict[str, tuple[Literal["object", "prefix"], str]] = {
    "chunks": ("object", CHUNKS_MANIFEST_REL),
    "search_metadata": ("object", SEARCH_METADATA_MANIFEST_REL),
    "manifests": ("prefix", "manifests/"),
    "unstructured": ("prefix", "unstructured/"),
    "structured": ("prefix", "structured/"),
    "vector_search": ("prefix", "vector_search/"),
}
_TICKER_SCOPED_PREFIXES = frozenset({"unstructured", "structured"})
ALL_SCOPE_ALIASES = ("manifests", "unstructured", "structured", "vector_search")
DEFAULT_HYDRATION_SCOPES: tuple[str, ...] = ("chunks", "unstructured")

_OFF_VALUES = {"", "0", "off", "false", "no", "disabled", "none"}
_AUTO_VALUES = {"1", "on", "true", "yes", "auto", "best_effort", "best-effort"}
_REQUIRED_VALUES = {"required", "strict", "require", "fail_closed"}
_TRUTHY = {"1", "true", "yes", "on"}
_DRY_RUN_SAMPLE_LIMIT = 25


class CorpusHydrationError(RuntimeError):
    """Raised when `required` corpus hydration fails or a config value is invalid."""


def parse_hydration_mode(raw_value: str | None) -> HydrationMode:
    """Normalize an `EDGAR_CORPUS_HYDRATION` value into `off` | `auto` | `required`."""
    normalized = (raw_value or "").strip().lower()
    if normalized in _OFF_VALUES:
        return "off"
    if normalized in _AUTO_VALUES:
        return "auto"
    if normalized in _REQUIRED_VALUES:
        return "required"
    raise CorpusHydrationError(
        f"Unsupported EDGAR_CORPUS_HYDRATION={raw_value!r}; expected 'off', 'auto', or 'required'."
    )


def parse_hydration_scopes(raw_value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Parse comma/space separated scopes (or `all`) into a validated, de-duplicated tuple."""
    if raw_value is None:
        return DEFAULT_HYDRATION_SCOPES
    if isinstance(raw_value, str):
        tokens = [t for t in re.split(r"[,;\s]+", raw_value.strip().lower()) if t]
    else:
        tokens = [str(t).strip().lower() for t in raw_value if str(t).strip()]
    if not tokens:
        return DEFAULT_HYDRATION_SCOPES

    resolved: list[str] = []
    for token in tokens:
        expanded = ALL_SCOPE_ALIASES if token == "all" else (token,)
        for scope in expanded:
            if scope not in HYDRATION_SCOPES:
                raise CorpusHydrationError(
                    f"Unknown corpus hydration scope {scope!r}; expected one of "
                    f"{sorted(HYDRATION_SCOPES)} or 'all'."
                )
            if scope not in resolved:
                resolved.append(scope)
    return tuple(resolved)


def parse_ticker_filter(raw_value: str | list[str] | tuple[str, ...] | None) -> frozenset[str]:
    """Parse a ticker filter into normalized uppercase EDGAR symbols (`BRK.B` -> `BRK-B`)."""
    if raw_value is None:
        return frozenset()
    raw_tokens = [raw_value] if isinstance(raw_value, str) else [str(t) for t in raw_value]
    tokens = [t for raw in raw_tokens for t in re.split(r"[,;\s]+", raw.strip())]
    normalized = {
        str(t).strip().upper().replace(".", "-")
        for t in tokens
        if str(t).strip() and str(t).strip().upper() not in ("*", "ALL")
    }
    return frozenset(normalized)


@dataclass(slots=True)
class CorpusHydrationConfig:
    """Resolved settings for downloading local fallback corpus artifacts from GCS."""

    mode: HydrationMode = "off"
    bucket_name: str = DEFAULT_BUCKET_NAME
    project_id: str = DEFAULT_PROJECT_ID
    dataset_dir: Path = field(default_factory=resolve_curated_dataset_dir)
    scopes: tuple[str, ...] = DEFAULT_HYDRATION_SCOPES
    tickers: frozenset[str] = frozenset()
    overwrite: bool = False
    dry_run: bool = False
    max_workers: int = 8
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        self.bucket_name = normalize_bucket_name(self.bucket_name)
        self.dataset_dir = Path(self.dataset_dir).expanduser().resolve()
        self.scopes = parse_hydration_scopes(self.scopes)
        self.tickers = parse_ticker_filter(
            self.tickers if isinstance(self.tickers, str) else sorted(self.tickers)
        )
        self.max_workers = max(1, int(self.max_workers))

    @classmethod
    def from_env(cls) -> CorpusHydrationConfig:
        """Build a config from `EDGAR_CORPUS_HYDRATION*` and GCP environment variables."""
        raw_workers = os.environ.get("EDGAR_CORPUS_HYDRATION_MAX_WORKERS", "").strip()
        return cls(
            mode=parse_hydration_mode(os.environ.get("EDGAR_CORPUS_HYDRATION")),
            bucket_name=(os.environ.get("SEC_CORPUS_BUCKET") or DEFAULT_BUCKET_NAME).strip(),
            project_id=(
                os.environ.get("GCP_PROJECT_ID")
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
                or DEFAULT_PROJECT_ID
            ).strip(),
            dataset_dir=resolve_curated_dataset_dir(),
            scopes=parse_hydration_scopes(os.environ.get("EDGAR_CORPUS_HYDRATION_SCOPES")),
            tickers=parse_ticker_filter(os.environ.get("EDGAR_CORPUS_HYDRATION_TICKERS")),
            overwrite=os.environ.get("EDGAR_CORPUS_HYDRATION_OVERWRITE", "").strip().lower()
            in _TRUTHY,
            max_workers=int(raw_workers) if raw_workers.isdigit() else 8,
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Secret-free summary for `/health` and CLI output."""
        return {
            "mode": self.mode,
            "bucket": self.bucket_name,
            "source_uri": f"gs://{self.bucket_name}/",
            "project_id": self.project_id,
            "dataset_dir": str(self.dataset_dir),
            "scopes": list(self.scopes),
            "tickers": sorted(self.tickers),
            "overwrite": self.overwrite,
        }


@dataclass(slots=True)
class CorpusHydrationResult:
    """Outcome of a corpus hydration run."""

    mode: HydrationMode
    status: HydrationStatus
    bucket: str
    dataset_dir: str
    scopes: list[str]
    tickers: list[str] = field(default_factory=list)
    objects_considered: int = 0
    downloaded_objects: int = 0
    skipped_existing: int = 0
    bytes_downloaded: int = 0
    planned_objects: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    completed_at: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors and self.status in (
            "disabled",
            "hydrated",
            "up_to_date",
            "dry_run",
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


@dataclass(slots=True)
class _PlannedDownload:
    object_name: str
    size: int
    target: Path


def _safe_target_path(dataset_dir: Path, object_name: str) -> Path:
    """Map a GCS object name to a local path, rejecting traversal outside `dataset_dir`."""
    cleaned = object_name.lstrip("/")
    if not cleaned or cleaned.endswith("/"):
        raise ValueError(f"GCS object {object_name!r} is a directory placeholder")
    target = (dataset_dir / cleaned).resolve()
    if not target.is_relative_to(dataset_dir):
        raise ValueError(f"GCS object {object_name!r} escapes dataset dir {dataset_dir}")
    return target


def _scope_prefixes(scope: str, tickers: frozenset[str]) -> list[str]:
    _kind, base = HYDRATION_SCOPES[scope]
    if scope in _TICKER_SCOPED_PREFIXES and tickers:
        return [f"{base}{ticker}/" for ticker in sorted(tickers)]
    return [base]


def _build_storage_client(project_id: str) -> Any:
    from google.cloud import storage  # Imported lazily so `off` mode never touches GCP libs.

    return storage.Client(project=project_id)


def _finalize(result: CorpusHydrationResult, started: float) -> CorpusHydrationResult:
    result.duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
    result.completed_at = datetime.now(UTC).isoformat()
    return result


def hydrate_local_corpus(
    config: CorpusHydrationConfig,
    *,
    storage_client: Any | None = None,
) -> CorpusHydrationResult:
    """Download missing corpus artifacts for the configured scopes from GCS.

    Existing local files are skipped unless `config.overwrite` is set; downloads are
    written to a temporary sibling file and atomically renamed so partially written
    artifacts are never observed by the tool loaders.
    """
    started = time.perf_counter()
    result = CorpusHydrationResult(
        mode=config.mode,
        status="up_to_date",
        bucket=config.bucket_name,
        dataset_dir=str(config.dataset_dir),
        scopes=list(config.scopes),
        tickers=sorted(config.tickers),
    )

    try:
        client = storage_client or _build_storage_client(config.project_id)
        bucket = client.bucket(config.bucket_name)
    except Exception as exc:  # DefaultCredentialsError, invalid project, etc.
        result.status = "failed"
        result.errors.append(
            f"Unable to initialize GCS client for gs://{config.bucket_name}: {exc}"
        )
        return _finalize(result, started)

    planned: dict[str, _PlannedDownload] = {}

    def _plan(object_name: str, size: int | None) -> None:
        if object_name in planned:
            return
        try:
            target = _safe_target_path(config.dataset_dir, object_name)
        except ValueError as exc:
            if not object_name.endswith("/"):
                result.errors.append(str(exc))
            return
        result.objects_considered += 1
        if target.exists() and not config.overwrite:
            result.skipped_existing += 1
            return
        planned[object_name] = _PlannedDownload(object_name, int(size or 0), target)

    for scope in config.scopes:
        kind, base = HYDRATION_SCOPES[scope]
        try:
            if kind == "object":
                local_target = _safe_target_path(config.dataset_dir, base)
                if local_target.exists() and not config.overwrite:
                    result.objects_considered += 1
                    result.skipped_existing += 1
                    continue
                blob = bucket.get_blob(base)
                if blob is None:
                    result.errors.append(f"Missing GCS object gs://{config.bucket_name}/{base}")
                    continue
                _plan(blob.name, getattr(blob, "size", None))
            else:
                for prefix in _scope_prefixes(scope, config.tickers):
                    for blob in client.list_blobs(config.bucket_name, prefix=prefix):
                        _plan(blob.name, getattr(blob, "size", None))
        except Exception as exc:
            result.errors.append(
                f"Failed listing scope '{scope}' in gs://{config.bucket_name}/{base}: {exc}"
            )

    if config.dry_run:
        result.planned_objects = sorted(planned)[:_DRY_RUN_SAMPLE_LIMIT]
        result.status = "failed" if result.errors and not planned else "dry_run"
        logger.info(
            "Corpus hydration dry run: %d object(s) (%d bytes) would be downloaded from gs://%s",
            len(planned),
            sum(p.size for p in planned.values()),
            config.bucket_name,
        )
        return _finalize(result, started)

    lock = threading.Lock()

    def _download(item: _PlannedDownload) -> int:
        item.target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = item.target.with_name(f".{item.target.name}.{uuid.uuid4().hex}.part")
        try:
            bucket.blob(item.object_name).download_to_filename(
                str(tmp_path), timeout=config.timeout_seconds
            )
            os.replace(tmp_path, item.target)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return item.target.stat().st_size

    if planned:
        logger.info(
            "Hydrating %d corpus object(s) from gs://%s into %s (scopes=%s)",
            len(planned),
            config.bucket_name,
            config.dataset_dir,
            ",".join(config.scopes),
        )
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = {pool.submit(_download, item): item for item in planned.values()}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    size = future.result()
                except Exception as exc:
                    with lock:
                        result.errors.append(
                            f"Failed downloading gs://{config.bucket_name}/{item.object_name}: {exc}"
                        )
                    continue
                with lock:
                    result.downloaded_objects += 1
                    result.bytes_downloaded += size

    if result.errors:
        result.status = "partial" if result.downloaded_objects else "failed"
    elif result.downloaded_objects:
        result.status = "hydrated"
    else:
        result.status = "up_to_date"

    return _finalize(result, started)


# ---------------------------------------------------------------------------
# Startup integration & status reporting
# ---------------------------------------------------------------------------

_LAST_HYDRATION_RESULT: CorpusHydrationResult | None = None


def run_startup_corpus_hydration(
    config: CorpusHydrationConfig | None = None,
    *,
    storage_client: Any | None = None,
) -> CorpusHydrationResult:
    """Run hydration according to `config.mode` and remember the result for `/health`.

    * `off`: no GCS calls; returns a `disabled` result.
    * `auto`: best effort; failures are logged and the app starts with whatever is local.
    * `required`: raises `CorpusHydrationError` when any requested artifact fails to hydrate.
    """
    global _LAST_HYDRATION_RESULT
    cfg = config or CorpusHydrationConfig.from_env()

    if cfg.mode == "off":
        result = CorpusHydrationResult(
            mode="off",
            status="disabled",
            bucket=cfg.bucket_name,
            dataset_dir=str(cfg.dataset_dir),
            scopes=list(cfg.scopes),
            tickers=sorted(cfg.tickers),
            completed_at=datetime.now(UTC).isoformat(),
        )
        _LAST_HYDRATION_RESULT = result
        return result

    try:
        result = hydrate_local_corpus(cfg, storage_client=storage_client)
    except Exception as exc:  # Defensive: never let `auto` mode crash startup.
        result = CorpusHydrationResult(
            mode=cfg.mode,
            status="failed",
            bucket=cfg.bucket_name,
            dataset_dir=str(cfg.dataset_dir),
            scopes=list(cfg.scopes),
            tickers=sorted(cfg.tickers),
            errors=[f"Unexpected corpus hydration error: {exc}"],
            completed_at=datetime.now(UTC).isoformat(),
        )

    _LAST_HYDRATION_RESULT = result

    if result.ok:
        logger.info(
            "Corpus hydration %s: downloaded=%d skipped_existing=%d bytes=%d in %.1f ms",
            result.status,
            result.downloaded_objects,
            result.skipped_existing,
            result.bytes_downloaded,
            result.duration_ms,
        )
    elif cfg.mode == "required":
        raise CorpusHydrationError(
            f"Required corpus hydration from gs://{cfg.bucket_name} {result.status}: "
            + "; ".join(result.errors[:5])
        )
    else:
        logger.warning(
            "Best-effort corpus hydration %s (continuing with local/synthesized fallbacks): %s",
            result.status,
            "; ".join(result.errors[:5]),
        )
    return result


def get_last_corpus_hydration_result() -> CorpusHydrationResult | None:
    return _LAST_HYDRATION_RESULT


def reset_corpus_hydration_state() -> None:
    """Clear the remembered hydration result (used by tests)."""
    global _LAST_HYDRATION_RESULT
    _LAST_HYDRATION_RESULT = None


def describe_local_corpus(dataset_dir: Path | None = None) -> dict[str, Any]:
    """Cheaply report which local fallback artifacts are present and how tools will source data."""
    root = Path(dataset_dir or resolve_curated_dataset_dir()).resolve()
    financials_manifest = (root / FINANCIALS_MANIFEST_REL).is_file()
    structured_dir = root / "structured"
    structured_present = (
        structured_dir.is_dir()
        and next(structured_dir.rglob("*_financials.json"), None) is not None
    )
    chunks_manifest = (root / CHUNKS_MANIFEST_REL).is_file()
    unstructured_dir = root / "unstructured"
    markdown_present = (
        unstructured_dir.is_dir() and next(unstructured_dir.rglob("*.md"), None) is not None
    )

    if financials_manifest:
        profile_source = "financials_manifest"
    elif structured_present:
        profile_source = "structured_json"
    else:
        profile_source = "missing"

    return {
        "dataset_dir": str(root),
        "financials_manifest_present": financials_manifest,
        "structured_profiles_present": structured_present,
        "chunks_manifest_present": chunks_manifest,
        "unstructured_markdown_present": markdown_present,
        "profile_source": profile_source,
        "chunk_source": "chunks_manifest" if chunks_manifest else "synthesized_from_profiles",
        "section_source": "markdown_files" if markdown_present else "synthesized_from_profiles",
    }


def get_corpus_status_metadata() -> dict[str, Any]:
    """Return local corpus layout + last hydration outcome for the `/health` endpoint."""
    try:
        cfg_meta: dict[str, Any] = CorpusHydrationConfig.from_env().to_public_dict()
    except CorpusHydrationError as exc:
        cfg_meta = {"mode": "invalid", "error": str(exc)}
    last = _LAST_HYDRATION_RESULT
    return {
        **describe_local_corpus(),
        "hydration": {
            **cfg_meta,
            "last_result": last.to_dict() if last is not None else None,
        },
    }
