"""Single source of truth for GCP project/location/bucket defaults and env resolution."""

from __future__ import annotations

import os

DEFAULT_PROJECT_ID = "iron-radio-514217-q7"
DEFAULT_PROJECT_NAME = "fde-sec-edgar-sandbox-dev"
DEFAULT_BUCKET_NAME = "fde-sec-edgar-sandbox-dev-sec-corpus"
DEFAULT_LOCATION = "us-central1"
DEFAULT_BIGQUERY_DATASET_ID = "sec_edgar"
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Pinned by SEC fair-access policy & `test_rbac_and_agent_identities.py`
DEFAULT_SEC_USER_AGENT = (
    "SEC-EDGAR-Analyst-Capstone sec-analyst-sa@fde-sec-edgar-sandbox-dev.iam.gserviceaccount.com"
)


def _first_non_empty_env(*names: str) -> str | None:
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return None


def resolve_gcp_project_id(
    explicit: str | None = None, *, default: str = DEFAULT_PROJECT_ID
) -> str:
    """Resolve the active GCP project ID with uniform precedence across the codebase.

    Precedence: explicit arg > ``GCP_PROJECT_ID`` > ``GOOGLE_CLOUD_PROJECT`` >
    ``GCLOUD_PROJECT`` > legacy ``EDGAR_GCP_PROJECT_ID`` > ``default``.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    return (
        _first_non_empty_env(
            "GCP_PROJECT_ID",
            "GOOGLE_CLOUD_PROJECT",
            "GCLOUD_PROJECT",
            "EDGAR_GCP_PROJECT_ID",
        )
        or default
    )


def resolve_gcp_location(
    explicit: str | None = None,
    *,
    default: str = DEFAULT_LOCATION,
    allow_global: bool = True,
) -> str:
    """Resolve the active GCP regional location with uniform precedence."""
    raw = (
        (explicit or "").strip()
        or _first_non_empty_env(
            "GCP_REGION",
            "GOOGLE_CLOUD_LOCATION",
            "BIGQUERY_LOCATION",
            "EDGAR_GCP_REGION",
        )
        or default
    )
    if not allow_global and raw.lower() == "global":
        return default
    return raw


def resolve_corpus_bucket(
    explicit: str | None = None,
    *,
    default: str = DEFAULT_BUCKET_NAME,
) -> str:
    """Resolve the SEC corpus GCS bucket name (accepting canonical and legacy env vars)."""
    raw = (
        (explicit or "").strip()
        or _first_non_empty_env("SEC_CORPUS_BUCKET", "EDGAR_GCS_BUCKET")
        or default
    )
    if raw.startswith("gs://"):
        raw = raw[len("gs://") :]
    return raw.strip("/")


def resolve_bq_dataset_id(
    explicit: str | None = None,
    *,
    default: str = DEFAULT_BIGQUERY_DATASET_ID,
) -> str:
    """Resolve the BigQuery dataset ID (accepting canonical and legacy env vars)."""
    if explicit and explicit.strip():
        return explicit.strip()
    return _first_non_empty_env("BIGQUERY_DATASET_ID", "EDGAR_BQ_DATASET") or default


def resolve_sec_user_agent(
    explicit: str | None = None,
    *,
    default: str = DEFAULT_SEC_USER_AGENT,
) -> str:
    """Resolve the SEC EDGAR User-Agent header (accepting both ``SEC_EDGAR_USER_AGENT`` and ``SEC_USER_AGENT``)."""
    if explicit and explicit.strip():
        return explicit.strip()
    return _first_non_empty_env("SEC_EDGAR_USER_AGENT", "SEC_USER_AGENT") or default
