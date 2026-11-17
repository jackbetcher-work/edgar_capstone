"""Shared pytest configuration: every test that touches data runs against live GCP.

There is no local corpus or offline mode. Tests read BigQuery `sec_edgar`, Vertex AI Search,
Vertex AI embeddings, and the GCS SEC corpus bucket, and chat turns call live Gemini on Vertex AI
unless a test injects a scripted ADK `BaseLlm`. Telemetry and feedback stay in memory so test
runs never write to the production telemetry tables.

Override the target project with `EDGAR_TEST_PROJECT_ID`. Developer shells often export an
unrelated `GCP_PROJECT_ID`, so it is pinned here rather than inherited.
"""

from __future__ import annotations

import os

import pytest

_TEST_PROJECT_ID = os.environ.get("EDGAR_TEST_PROJECT_ID", "iron-radio-514217-q7")

os.environ["GCP_PROJECT_ID"] = _TEST_PROJECT_ID
os.environ["GOOGLE_CLOUD_PROJECT"] = _TEST_PROJECT_ID
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("EDGAR_GEMINI_LOCATION", "global")
os.environ["EDGAR_TELEMETRY_BACKEND"] = "local"
os.environ["EDGAR_FEEDBACK_BACKEND"] = "local"


@pytest.fixture(scope="session", autouse=True)
def _require_application_default_credentials() -> None:
    """Fail fast with an actionable message when ADC is missing or expired."""
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError, RefreshError
    from google.auth.transport.requests import Request

    try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())
    except (DefaultCredentialsError, RefreshError) as exc:
        pytest.exit(
            "The test suite reads live GCP data and calls Vertex AI, so it needs valid "
            "Application Default Credentials. Run `gcloud auth application-default login` "
            f"(or run in CI with a service account). Details: {exc}",
            returncode=4,
        )
