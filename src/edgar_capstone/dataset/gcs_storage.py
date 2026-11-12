"""Google Cloud Storage bucket provisioning and parallel dataset uploader."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path

from google.api_core import exceptions as gcp_exceptions
from google.auth import exceptions as auth_exceptions
from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY

from edgar_capstone.dataset.models import FinancialDataProfile

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_ID = "iron-radio-514217-q7"
DEFAULT_BUCKET_NAME = "fde-sec-edgar-sandbox-dev-sec-corpus"
DEFAULT_LOCATION = "us-central1"


class GCSAccessError(RuntimeError):
    """Raised when GCS credentials or IAM permissions prevent bucket access/upload."""


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


def ensure_gcs_bucket(
    client: storage.Client,
    *,
    bucket_name: str,
    project_id: str = DEFAULT_PROJECT_ID,
    location: str = DEFAULT_LOCATION,
    create_if_missing: bool = True,
) -> storage.Bucket:
    """Look up the target GCS bucket, creating it in `location` with uniform access if missing."""
    normalized = normalize_bucket_name(bucket_name)
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
        created = client.create_bucket(
            new_bucket,
            project=project_id,
            location=location,
        )
        logger.info("Successfully created GCS bucket gs://%s", normalized)
        return created

    except (
        auth_exceptions.DefaultCredentialsError,
        gcp_exceptions.Forbidden,
        gcp_exceptions.Unauthorized,
    ) as exc:
        raise GCSAccessError(
            f"GCS authentication/permission error for project '{project_id}' and bucket "
            f"'gs://{normalized}': {exc}\n"
            "Ensure Application Default Credentials are set for an account with Storage Admin / "
            "Storage Object Admin in the Argolis project:\n"
            f"  gcloud auth application-default login\n"
            f"  gcloud config set project {project_id}\n"
            "Once authenticated, re-run with `--upload-only` to sync the locally staged dataset "
            "without re-downloading from SEC EDGAR."
        ) from exc
    except gcp_exceptions.Conflict as exc:
        raise GCSAccessError(
            f"GCS bucket name 'gs://{normalized}' is already taken globally by another project: {exc}. "
            "Specify a unique bucket name with `--bucket <unique-bucket-name>`."
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
    storage_client: storage.Client | None = None,
) -> int:
    """Upload all staged unstructured, structured, and manifest files to GCS concurrently."""
    normalized_bucket = normalize_bucket_name(bucket_name)
    try:
        client = storage_client or storage.Client(project=project_id)
    except auth_exceptions.DefaultCredentialsError as exc:
        raise GCSAccessError(
            f"Google Cloud Application Default Credentials not found for project '{project_id}': {exc}\n"
            "Run `gcloud auth application-default login` with your Argolis sandbox identity, "
            "then re-run with `--upload-only`."
        ) from exc

    bucket = ensure_gcs_bucket(
        client,
        bucket_name=normalized_bucket,
        project_id=project_id,
        location=location,
        create_if_missing=create_bucket_if_missing,
    )

    files_to_upload: list[tuple[Path, str]] = []
    for subdir in ("unstructured", "structured", "manifests"):
        folder = output_dir / subdir
        if not folder.exists():
            continue
        for file_path in sorted(folder.rglob("*")):
            if file_path.is_file():
                rel_blob = file_path.relative_to(output_dir).as_posix()
                files_to_upload.append((file_path, rel_blob))

    if not files_to_upload:
        logger.warning("No staged files found in %s to upload.", output_dir)
        return 0

    def _upload_one(local_path: Path, blob_path: str) -> str:
        blob = bucket.blob(blob_path)
        if profiles_by_blob and blob_path in profiles_by_blob:
            prof = profiles_by_blob[blob_path]
            blob.metadata = {
                "profile_id": prof.profile_id,
                "ticker": prof.ticker,
                "cik": prof.cik,
                "company_name": prof.company_name,
                "gics_sector": prof.gics_sector,
                "fiscal_year": str(prof.fiscal_year),
                "form_type": prof.form_type,
                "accession_number": prof.accession_number,
            }
        blob.upload_from_filename(
            str(local_path),
            content_type=content_type_for_path(local_path),
            retry=DEFAULT_RETRY,
            timeout=120,
        )
        return blob_path

    uploaded_count = 0
    logger.info(
        "Uploading %d objects from %s to gs://%s...",
        len(files_to_upload),
        output_dir,
        normalized_bucket,
    )
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {
            pool.submit(_upload_one, local_path, blob_path): blob_path
            for local_path, blob_path in files_to_upload
        }
        for future in as_completed(futures):
            blob_path = futures[future]
            try:
                future.result()
                uploaded_count += 1
            except Exception as exc:
                raise GCSAccessError(
                    f"Failed uploading '{blob_path}' to gs://{normalized_bucket}: {exc}"
                ) from exc

    logger.info(
        "Successfully uploaded %d objects to gs://%s",
        uploaded_count,
        normalized_bucket,
    )
    return uploaded_count
