"""Security, Secret Manager retrieval, and secret hygiene scanning package."""

from __future__ import annotations

from edgar_capstone.security.secrets import (
    DEFAULT_PROJECT_ID,
    MANAGED_SECRETS,
    RedactedSecret,
    SecretFinding,
    SecretManagerClient,
    SecretSpec,
    scan_repository_for_exposed_secrets,
    scan_text_for_exposed_secrets,
)

__all__ = [
    "DEFAULT_PROJECT_ID",
    "MANAGED_SECRETS",
    "RedactedSecret",
    "SecretFinding",
    "SecretManagerClient",
    "SecretSpec",
    "scan_repository_for_exposed_secrets",
    "scan_text_for_exposed_secrets",
]
