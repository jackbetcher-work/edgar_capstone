"""Google Cloud Storage bucket provisioning and parallel dataset uploader."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from google.auth import exceptions as auth_exceptions

from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.gcp.defaults import (
    DEFAULT_BUCKET_NAME,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
)

try:
    from google.api_core import exceptions as gcp_exceptions
    from google.cloud import storage
    from google.cloud.storage.retry import DEFAULT_RETRY
except ImportError:  # pragma: no cover - optional `[ingest]` extra
    gcp_exceptions = None  # type: ignore[assignment]
    storage = None  # type: ignore[assignment]
    DEFAULT_RETRY = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class GCSAccessError(RuntimeError):
    """Raised when GCS credentials, IAM permissions, or optional dependencies prevent access."""


def _require_storage_sdk() -> Any:
    """Return the `google.cloud.storage` module or raise a clear `[ingest]` install error."""
    if storage is None:
        raise GCSAccessError(
            "google-cloud-storage is not installed. Install the optional ingestion dependencies "
            "with: pip install -e '.[ingest]'"
        )
    return storage


def create_storage_client(*, project_id: str = DEFAULT_PROJECT_ID) -> Any:
    """Create a `google.cloud.storage.Client` with friendly credential/dependency errors."""
    sdk = _require_storage_sdk()
    try:
        return sdk.Client(project=project_id)
    except auth_exceptions.DefaultCredentialsError as exc:
        raise GCSAccessError(
            f"Google Cloud Application Default Credentials not found for project '{project_id}': "
            f"{exc}\n"
            "Run `gcloud auth application-default login` with your Argolis sandbox identity, "
            "then re-run with `--upload-only`."
        ) from exc


def normalize_bucket_name(raw_bucket: str) -> str:
    """Normalize `gs://bucket-name/` or `bucket-name` to bare bucket name."""
    cleaned = raw_bucket.strip()
    if cleaned.startswith("gs://"):
        cleaned = cleaned[len("gs://") :]
    cleaned = cleaned.strip("/")
    if not cleaned or "/" in cleaned:
        raise ValueError(
            f"Invalid GCS bucket name {raw_bucket!r}. Expected 'bucket-name' or 'gs://bucket-name'."
        )
    return cleaned


def content_type_for_path(path: Path) -> str:
    """Determine appropriate HTTP Content-Type for GCS objects."""
    suffix = path.suffix.lower()
    if suffix == ".md":
        # Use text/plain so Vertex AI Search unstructured ingestion parses .md cleanly
        return "text/plain; charset=utf-8"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    if suffix == ".jsonl":
        return "application/x-ndjson; charset=utf-8"
    return "application/octet-stream"


def profile_blob_metadata(profile: FinancialDataProfile) -> dict[str, str]:
    """Build standard GCS custom object metadata for a structured/unstructured filing blob."""
    return {
        "profile_id": profile.profile_id,
        "ticker": profile.ticker,
        "cik": profile.cik,
        "company_name": profile.company_name,
        "gics_sector": profile.gics_sector,
        "fiscal_year": str(profile.fiscal_year),
        "form_type": profile.form_type,
        "accession_number": profile.accession_number,
    }


def _upload_blob_from_filename(
    blob: Any,
    local_path: Path,
    *,
    content_type: str,
    timeout: float,
    retry: Any,
) -> None:
    """Invoke `blob.upload_from_filename`, gracefully supporting minimal test doubles."""
    kwargs: dict[str, Any] = {"content_type": content_type}
    try:
        sig = inspect.signature(blob.upload_from_filename)
    except (TypeError, ValueError):
        if retry is not None:
            kwargs["retry"] = retry
        kwargs["timeout"] = timeout
    else:
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if retry is not None and (has_var_kw or "retry" in sig.parameters):
            kwargs["retry"] = retry
        if has_var_kw or "timeout" in sig.parameters:
            kwargs["timeout"] = timeout
    blob.upload_from_filename(str(local_path), **kwargs)


UploadItem = tuple[Path, str] | tuple[Path, str, dict[str, str] | None]


def upload_files_to_bucket(
    bucket: Any,
    items: Sequence[UploadItem],
    *,
    max_workers: int = 8,
    timeout: float = 120.0,
    retry: Any = DEFAULT_RETRY,
    content_type_fn: Callable[[Path, str], str] | None = None,
) -> int:
    """Upload `(local_path, blob_path[, metadata])` tuples concurrently to a GCS bucket.

    Preserves custom `blob.metadata`, resolves Content-Type via `content_type_fn` (defaulting to
    `content_type_for_path`), applies `DEFAULT_RETRY` and `timeout`, and wraps errors in
    `GCSAccessError`.
    """
    if not items:
        return 0

    bucket_label = getattr(bucket, "name", None) or "gcs-bucket"

    def _unpack(item: UploadItem) -> tuple[Path, str, dict[str, str] | None]:
        if len(item) == 2:
            return item[0], item[1], None
        return item[0], item[1], item[2]

    def _upload_one(local_path: Path, blob_path: str, metadata: dict[str, str] | None) -> str:
        blob = bucket.blob(blob_path)
        if metadata is not None:
            blob.metadata = dict(metadata)
        ct = (
            content_type_fn(local_path, blob_path)
            if content_type_fn is not None
            else content_type_for_path(local_path)
        )
        _upload_blob_from_filename(
            blob,
            local_path,
            content_type=ct,
            timeout=timeout,
            retry=retry,
        )
        return blob_path

    if len(items) == 1 or max_workers <= 1:
        uploaded = 0
        for item in items:
            local_path, blob_path, metadata = _unpack(item)
            try:
                _upload_one(local_path, blob_path, metadata)
                uploaded += 1
            except Exception as exc:
                raise GCSAccessError(
                    f"Failed uploading '{blob_path}' to gs://{bucket_label}: {exc}"
                ) from exc
        return uploaded

    uploaded_count = 0
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {}
        for item in items:
            local_path, blob_path, metadata = _unpack(item)
            futures[pool.submit(_upload_one, local_path, blob_path, metadata)] = blob_path
        for future in as_completed(futures):
            blob_path = futures[future]
            try:
                future.result()
                uploaded_count += 1
            except Exception as exc:
                raise GCSAccessError(
                    f"Failed uploading '{blob_path}' to gs://{bucket_label}: {exc}"
                ) from exc
    return uploaded_count


def ensure_gcs_bucket(
    client: Any,
    *,
    bucket_name: str,
    project_id: str = DEFAULT_PROJECT_ID,
    location: str = DEFAULT_LOCATION,
    create_if_missing: bool = True,
) -> Any:
    """Look up the target GCS bucket, creating it in `location` with uniform access if missing."""
    normalized = normalize_bucket_name(bucket_name)
    auth_errors: tuple[type[BaseException], ...] = (auth_exceptions.DefaultCredentialsError,)
    conflict_errors: tuple[type[BaseException], ...] = ()
    if gcp_exceptions is not None:
        auth_errors = (
            auth_exceptions.DefaultCredentialsError,
            gcp_exceptions.Forbidden,
            gcp_exceptions.Unauthorized,
        )
        conflict_errors = (gcp_exceptions.Conflict,)

    try:
        existing = client.lookup_bucket(normalized)
        if existing is not None:
            logger.info(
                "Verified existing GCS bucket gs://%s (location=%s)",
                normalized,
                existing.location,
            )
            return existing

        if not create_if_missing:
            raise GCSAccessError(
                f"GCS bucket gs://{normalized} does not exist and bucket creation is disabled."
            )

        logger.info(
            "Creating GCS bucket gs://%s in project '%s' (location=%s)...",
            normalized,
            project_id,
            location,
        )
        new_bucket = client.bucket(normalized)
        new_bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        new_bucket.iam_configuration.public_access_prevention = "enforced"
        created = client.create_bucket(
            new_bucket,
            project=project_id,
            location=location,
        )
        logger.info("Successfully created GCS bucket gs://%s", normalized)
        return created

    except auth_errors as exc:
        raise GCSAccessError(
            f"GCS authentication/permission error for project '{project_id}' and bucket "
            f"'gs://{normalized}': {exc}\n"
            "Ensure Application Default Credentials are set for an account with Storage Admin / "
            "Storage Object Admin in the Argolis project:\n"
            "  gcloud auth application-default login\n"
            f"  gcloud config set project {project_id}\n"
            "Once authenticated, re-run with `--upload-only` to sync the locally staged dataset "
            "without re-downloading from SEC EDGAR."
        ) from exc
    except conflict_errors as exc:
        raise GCSAccessError(
            f"GCS bucket name 'gs://{normalized}' is already taken globally by another project: "
            f"{exc}. Specify a unique bucket name with `--bucket <unique-bucket-name>`."
        ) from exc


def upload_staged_dataset(
    *,
    output_dir: Path,
    project_id: str = DEFAULT_PROJECT_ID,
    bucket_name: str = DEFAULT_BUCKET_NAME,
    location: str = DEFAULT_LOCATION,
    create_bucket_if_missing: bool = True,
    profiles_by_blob: dict[str, FinancialDataProfile] | None = None,
    max_workers: int = 8,
    storage_client: Any | None = None,
) -> int:
    """Upload all staged unstructured, structured, and manifest files to GCS concurrently."""
    normalized_bucket = normalize_bucket_name(bucket_name)
    client = storage_client or create_storage_client(project_id=project_id)

    bucket = ensure_gcs_bucket(
        client,
        bucket_name=normalized_bucket,
        project_id=project_id,
        location=location,
        create_if_missing=create_bucket_if_missing,
    )

    items: list[tuple[Path, str, dict[str, str] | None]] = []
    for subdir in ("unstructured", "structured", "manifests"):
        folder = output_dir / subdir
        if not folder.exists():
            continue
        for file_path in sorted(folder.rglob("*")):
            if file_path.is_file():
                rel_blob = file_path.relative_to(output_dir).as_posix()
                metadata = (
                    profile_blob_metadata(profiles_by_blob[rel_blob])
                    if profiles_by_blob and rel_blob in profiles_by_blob
                    else None
                )
                items.append((file_path, rel_blob, metadata))

    if not items:
        logger.warning("No staged files found in %s to upload.", output_dir)
        return 0

    logger.info(
        "Uploading %d objects from %s to gs://%s...",
        len(items),
        output_dir,
        normalized_bucket,
    )
    uploaded_count = upload_files_to_bucket(
        bucket,
        items,
        max_workers=max_workers,
        timeout=120.0,
        retry=DEFAULT_RETRY,
    )
    logger.info(
        "Successfully uploaded %d objects to gs://%s",
        uploaded_count,
        normalized_bucket,
    )
    return uploaded_count
