"""Google Cloud Secret Manager integration and pipeline secret-leak scanning utilities."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import time
from typing import Any

import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
import httpx

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_ID = "iron-radio-514217-q7"


@dataclass(frozen=True)
class SecretSpec:
    """Metadata specification for a Secret Manager secret used by the SEC Analyst system."""

    env_var: str
    secret_id: str
    description: str
    required: bool = True
    version: str = "latest"

    def resource_name(self, project_id: str = DEFAULT_PROJECT_ID) -> str:
        """Return the fully-qualified Secret Manager version resource name."""
        return f"projects/{project_id}/secrets/{self.secret_id}/versions/{self.version}"

    def cloud_run_mount_spec(self) -> str:
        """Return the Cloud Run `--set-secrets` ENV=SECRET:VERSION binding string."""
        return f"{self.env_var}={self.secret_id}:{self.version}"


MANAGED_SECRETS: tuple[SecretSpec, ...] = (
    SecretSpec(
        env_var="SEC_EDGAR_USER_AGENT",
        secret_id="sec-edgar-user-agent",
        description="SEC EDGAR API compliant User-Agent identity header",
        required=True,
    ),
    SecretSpec(
        env_var="GEMINI_API_KEY",
        secret_id="sec-edgar-gemini-api-key",
        description="Gemini / Vertex AI API credential for fallback evaluation and synthesis",
        required=True,
    ),
    SecretSpec(
        env_var="INTERNAL_IAP_AUDIENCE",
        secret_id="sec-edgar-iap-audience",
        description="Identity-Aware Proxy (IAP) OAuth2 client audience for JWT verification",
        required=False,
    ),
)


class RedactedSecret:
    """Container for a sensitive secret value that prevents accidental logging or repr leaks."""

    __slots__ = ("_name", "_value")

    def __init__(self, value: str, *, name: str = "secret") -> None:
        self._value = value
        self._name = name

    def get_secret_value(self) -> str:
        """Explicitly unwrap the underlying secret value when needed for an API call."""
        return self._value

    def __repr__(self) -> str:
        return f"RedactedSecret(name={self._name!r}, value='***REDACTED***')"

    def __str__(self) -> str:
        return "***REDACTED***"

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RedactedSecret):
            return self._value == other._value
        return False


class SecretManagerClient:
    """Retrieves secrets from Google Cloud Secret Manager without exposing credentials in code."""

    def __init__(
        self,
        project_id: str | None = None,
        *,
        cache_ttl_seconds: float = 300.0,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        self.project_id = (
            project_id
            or os.environ.get("GCP_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or DEFAULT_PROJECT_ID
        )
        self.cache_ttl_seconds = cache_ttl_seconds
        self._http_client = http_client
        self._credentials = credentials
        self._cache: dict[str, tuple[float, RedactedSecret]] = {}

    def _get_auth_token(self) -> str:
        if self._credentials is None:
            self._credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not self._credentials.valid or not getattr(self._credentials, "token", None):
            self._credentials.refresh(GoogleAuthRequest())
        return str(self._credentials.token)

    def get_secret(
        self,
        secret_id: str,
        *,
        version: str = "latest",
        env_fallback: str | None = None,
    ) -> RedactedSecret:
        """Retrieve a secret from Secret Manager (or Cloud Run Secret Manager env mount)."""
        cache_key = f"{self.project_id}/{secret_id}/{version}"
        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if cached is not None and (now - cached[0]) < self.cache_ttl_seconds:
            return cached[1]

        # If Cloud Run already mounted the Secret Manager secret into the environment, use it first
        if env_fallback and os.environ.get(env_fallback):
            secret_obj = RedactedSecret(os.environ[env_fallback], name=secret_id)
            self._cache[cache_key] = (now, secret_obj)
            return secret_obj

        url = (
            f"https://secretmanager.googleapis.com/v1/projects/{self.project_id}"
            f"/secrets/{secret_id}/versions/{version}:access"
        )
        token = self._get_auth_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        if self._http_client is not None:
            resp = self._http_client.get(url, headers=headers)
        else:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, headers=headers)

        resp.raise_for_status()
        body = resp.json()
        encoded_data = body.get("payload", {}).get("data", "")
        if not encoded_data:
            raise ValueError(
                f"Secret Manager returned empty payload for {secret_id} (version={version})"
            )

        decoded = base64.b64decode(encoded_data).decode("utf-8")
        secret_obj = RedactedSecret(decoded, name=secret_id)
        self._cache[cache_key] = (now, secret_obj)
        return secret_obj

    def load_managed_secrets(
        self,
        specs: tuple[SecretSpec, ...] = MANAGED_SECRETS,
    ) -> dict[str, RedactedSecret]:
        """Load all managed secrets from Secret Manager, returning redacted wrappers keyed by env_var."""
        loaded: dict[str, RedactedSecret] = {}
        for spec in specs:
            try:
                loaded[spec.env_var] = self.get_secret(
                    spec.secret_id,
                    version=spec.version,
                    env_fallback=spec.env_var,
                )
            except Exception as exc:
                if spec.required:
                    raise RuntimeError(
                        f"Failed to retrieve required secret '{spec.secret_id}' "
                        f"from Secret Manager: {type(exc).__name__}"
                    ) from exc
                logger.debug(
                    "Optional secret %s not available: %s",
                    spec.secret_id,
                    type(exc).__name__,
                )
        return loaded


# ---------------------------------------------------------------------------
# Secret-Exposure & Hardcoded Credential Scanner for CI/CD Pipelines
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key_block",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "gcp_service_account_json_key",
        re.compile(r'"type"\s*:\s*"service_account"[\s\S]{0,400}"private_key_id"\s*:'),
    ),
    (
        "google_api_key",
        re.compile(r"\bAIza[0-9A-Za-z\-_]{30,45}\b"),
    ),
    (
        "google_oauth_access_token",
        re.compile(r"\bya29\.[0-9A-Za-z\-_]{20,}\b"),
    ),
    (
        "github_pat_token",
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{22,}\b"),
    ),
    (
        "aws_access_key_id",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ),
    (
        "slack_token",
        re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,}\b"),
    ),
    (
        "hardcoded_secret_assignment",
        re.compile(
            r"(?i)\b(?:[a-z0-9_]*_)?(?:api[_-]?key|client[_-]?secret|private[_-]?key|access[_-]?token|auth[_-]?token|password|passwd)"
            r"\s*[:=]\s*['\"](?!\$\{|\$[A-Z_]|projects/|sec-edgar-|\*\*\*REDACTED\*\*\*|YOUR_|REPLACE_|example|dummy|test|fake|none|null)[A-Za-z0-9_\-\./+=]{16,}['\"]"
        ),
    ),
)


@dataclass(frozen=True)
class SecretFinding:
    """Represents a potential exposed secret or credential found during scanning."""

    source: str
    rule_id: str
    line_number: int
    snippet_redacted: str


def scan_text_for_exposed_secrets(
    content: str,
    *,
    source_name: str = "<memory>",
) -> list[SecretFinding]:
    """Scan text content for hardcoded keys, tokens, or credentials."""
    findings: list[SecretFinding] = []

    # Check multi-line patterns (e.g. JSON service account keys or PEM blocks)
    for rule_id in ("private_key_block", "gcp_service_account_json_key"):
        pattern = dict(_SECRET_PATTERNS)[rule_id]
        for match in pattern.finditer(content):
            line_no = content[: match.start()].count("\n") + 1
            findings.append(
                SecretFinding(
                    source=source_name,
                    rule_id=rule_id,
                    line_number=line_no,
                    snippet_redacted=f"[{rule_id} detected on line {line_no}]",
                )
            )

    # Check line-by-line patterns
    line_rules = [
        (r_id, pat)
        for r_id, pat in _SECRET_PATTERNS
        if r_id not in ("private_key_block", "gcp_service_account_json_key")
    ]
    for idx, line in enumerate(content.splitlines(), start=1):
        for rule_id, pattern in line_rules:
            if pattern.search(line):
                findings.append(
                    SecretFinding(
                        source=source_name,
                        rule_id=rule_id,
                        line_number=idx,
                        snippet_redacted=f"[{rule_id} detected on line {idx}]",
                    )
                )

    return findings


def scan_repository_for_exposed_secrets(repo_root: Path) -> list[SecretFinding]:
    """Scan pipeline, infrastructure, and source files in the repository for exposed secrets."""
    scan_targets: list[Path] = []
    include_globs = (
        ".github/workflows/*.yml",
        ".github/workflows/*.yaml",
        "cloudbuild.yaml",
        "Dockerfile",
        "terraform/*.tf",
        "terraform/*.tfvars.example",
        "src/**/*.py",
        "scripts/**/*.py",
    )
    for pattern in include_globs:
        scan_targets.extend(sorted(repo_root.glob(pattern)))

    all_findings: list[SecretFinding] = []
    for path in scan_targets:
        if not path.is_file():
            continue
        rel = str(path.relative_to(repo_root))
        text = path.read_text(encoding="utf-8", errors="replace")
        all_findings.extend(scan_text_for_exposed_secrets(text, source_name=rel))

    return all_findings
