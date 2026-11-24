"""Shared pytest configuration and reusable fixtures.

Default runs exercise both fast, deterministic unit/integration tests (using scripted ADK
`BaseLlm`s, `httpx.MockTransport` GCP adapters, and the local Model Armor emulator) AND a
small `@pytest.mark.live` smoke suite that hits real GCP services (BigQuery, Vertex AI Search,
Vertex AI embeddings, GCS, Model Armor, and Gemini on Vertex AI).

Run only offline tests (zero network / no ADC required):
    pytest -m "not live"

Run only the live GCP smoke tests:
    pytest -m live

Override the target project with `EDGAR_TEST_PROJECT_ID`. Developer shells often export an
unrelated `GCP_PROJECT_ID`, so it is pinned here rather than inherited. Telemetry and feedback
stay in memory so test runs never write to the production telemetry tables.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from edgar_capstone.agents.adapters.registry import GcpToolAdapters
from edgar_capstone.security.model_armor import reset_model_armor_guardrail
from tests.fake_gcp import use_fake_gcp_adapters

_TEST_PROJECT_ID = os.environ.get("EDGAR_TEST_PROJECT_ID", "iron-radio-514217-q7")

os.environ["GCP_PROJECT_ID"] = _TEST_PROJECT_ID
os.environ["GOOGLE_CLOUD_PROJECT"] = _TEST_PROJECT_ID
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("EDGAR_GEMINI_LOCATION", "global")
os.environ["EDGAR_TELEMETRY_BACKEND"] = "local"
os.environ["EDGAR_FEEDBACK_BACKEND"] = "local"
# Default to the wire-compatible local Model Armor emulator so unit/scripted tests do not
# silently fire live Model Armor / Cloud DLP REST calls. Tests that explicitly exercise live
# Model Armor request the `live_model_armor` fixture (or set mode="gcp"/"auto").
os.environ.setdefault("EDGAR_MODEL_ARMOR_MODE", "emulator")
# Default semantic scope/continuity classification to deterministic fallback in offline tests
# (tests that exercise Flash-Lite pass an explicit `scope_llm` or opt in with `live`).
os.environ.setdefault("EDGAR_SCOPE_LLM_MODE", "deterministic")


@pytest.fixture(scope="session", autouse=True)
def _require_application_default_credentials(request: pytest.FixtureRequest) -> None:
    """Fail fast with an actionable message when live tests are selected and ADC is missing."""
    has_live = any(item.get_closest_marker("live") is not None for item in request.session.items)
    if not has_live:
        return

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
            "Selected `@pytest.mark.live` tests read live GCP data and call Vertex AI, so they "
            "need valid Application Default Credentials. Run `gcloud auth application-default "
            f"login` (or run offline tests with `pytest -m 'not live'`). Details: {exc}",
            returncode=4,
        )


@pytest.fixture(scope="session")
def session_idp_keys() -> dict[str, Any]:
    """Generate ephemeral ES256 and RSA-2048 IdP keypairs ONCE per pytest session."""
    return {
        "iap_ec_key": ec.generate_private_key(ec.SECP256R1()),
        "google_rsa_key": rsa.generate_private_key(public_exponent=65537, key_size=2048),
        "okta_rsa_key": rsa.generate_private_key(public_exponent=65537, key_size=2048),
    }


@pytest.fixture()
def fake_gcp_adapters() -> Iterator[tuple[GcpToolAdapters, list[tuple[str, str]]]]:
    """Temporarily install deterministic `httpx.MockTransport` GCP adapters without wiping live caches."""
    with use_fake_gcp_adapters() as bundle:
        yield bundle


@pytest.fixture()
def live_model_armor(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Enable live Google Cloud Model Armor (`mode='auto'`) for a single live smoke test."""
    monkeypatch.setenv("EDGAR_MODEL_ARMOR_MODE", "auto")
    reset_model_armor_guardrail()
    try:
        yield
    finally:
        monkeypatch.setenv("EDGAR_MODEL_ARMOR_MODE", "emulator")
        reset_model_armor_guardrail()
