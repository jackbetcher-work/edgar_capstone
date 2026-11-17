"""Identity-Aware Proxy (IAP) & Google Workspace Okta/SSO JWT Bearer Authentication.

Protects internal FastAPI endpoints using cryptographic JWT bearer verification
integrated with Google Cloud Identity-Aware Proxy (IAP), Google Workspace OIDC,
and Okta SSO federation, alongside Cloud Identity group RBAC and 10-K access
audit logging.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

import httpx
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import Depends, HTTPException, Request, status

from edgar_capstone.common.domain import sha256_text
from edgar_capstone.common.env import env_bool, split_csv_env
from edgar_capstone.gcp.defaults import DEFAULT_PROJECT_ID, DEFAULT_PROJECT_NAME
from edgar_capstone.observability.tracing import traced
from edgar_capstone.security.agent_identity import (
    S2S_AGENT_ASSERTION_HEADER,
    AgentAuthorizationError,
    get_agent_s2s_broker,
    resolve_agent_by_service_account_id,
    resolve_agent_identity,
)
from edgar_capstone.security.jwt_codec import (
    PublicKey,
    b64url_decode,
    b64url_encode,
    ec_public_key_to_jwk,
    jwk_to_public_key,
    parse_jwt_header_and_claims,
    rsa_public_key_to_jwk,
    sign_jwt_es256,
    sign_jwt_rs256,
    verify_jws_signature,
)
from edgar_capstone.security.rbac import (
    ANALYST_PERMISSIONS,
    CUSTOM_RUNTIME_SA_ACCOUNT_ID,
    DEFAULT_CLOUD_IDENTITY_DOMAIN,
    CloudIdentityRBACPolicy,
    Permission,
    RuntimeServiceAccountConfig,
    is_default_gcp_service_account,
    validate_custom_service_account_email,
)
from edgar_capstone.telemetry import get_telemetry_sink

logger = logging.getLogger(__name__)

GOOGLE_IAP_ISSUER = "https://cloud.google.com/iap"
GOOGLE_IAP_JWKS_URI = "https://www.gstatic.com/iap/verify/public_key-jwk"

GOOGLE_WORKSPACE_ISSUERS: frozenset[str] = frozenset(
    {
        "https://accounts.google.com",
        "accounts.google.com",
    }
)
GOOGLE_WORKSPACE_JWKS_URI = "https://www.googleapis.com/oauth2/v3/certs"

LOCAL_DEV_SSO_ISSUER = "https://iap.local.edgar-analyst.internal"
DEFAULT_LOCAL_IAP_AUDIENCE = (
    f"/projects/{DEFAULT_PROJECT_ID}/locations/us-central1/services/sec-edgar-analyst"
)

ALLOWED_JWS_ALGORITHMS: frozenset[str] = frozenset({"RS256", "ES256"})
MAX_JWT_LENGTH_BYTES = 16_384

IAP_ASSERTION_HEADER = "X-Goog-IAP-JWT-Assertion"

# Roles that IAP_ASSERTION_DEFAULT_ROLE may grant to IAP-verified users who carry no
# Cloud Identity group claims. Admin is deliberately excluded: it must come from a group.
IAP_DEFAULT_ROLE_CHOICES: frozenset[str] = frozenset({"analyst", "viewer"})

_AUTH_ENV_KEYS: tuple[str, ...] = (
    "K_SERVICE",
    "IAP_STRICT_MODE",
    "EDGAR_ENABLE_LOCAL_SSO_DEV_BROKER",
    "INTERNAL_IAP_AUDIENCE",
    "ADDITIONAL_IAP_AUDIENCES",
    "OKTA_CLIENT_ID",
    "GOOGLE_OAUTH_CLIENT_ID",
    "OKTA_ISSUER_URI",
    "OKTA_JWKS_URI",
    "ALLOWED_WORKSPACE_DOMAINS",
    "REQUIRED_ANALYST_GROUPS",
    "ADMIN_CLOUD_IDENTITY_GROUPS",
    "ADMIN_WORKSPACE_GROUPS",
    "ANALYST_CLOUD_IDENTITY_GROUPS",
    "ANALYST_WORKSPACE_GROUPS",
    "VIEWER_CLOUD_IDENTITY_GROUPS",
    "VIEWER_WORKSPACE_GROUPS",
    "CLOUD_IDENTITY_DOMAIN",
    "ALLOWED_CUSTOM_SERVICE_ACCOUNTS",
    "ALLOW_GCP_SERVICE_ACCOUNTS",
    "IAP_ASSERTION_DEFAULT_ROLE",
    "GCP_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "EDGAR_GCP_PROJECT_ID",
    "GCP_PROJECT_NAME",
    "RUNTIME_SERVICE_ACCOUNT_ID",
    "RUNTIME_SERVICE_ACCOUNT",
    "ALLOWED_SERVICE_ACCOUNT_PROJECTS",
)


class AuthenticationError(Exception):
    """Raised when a request lacks a valid JWT bearer credential (HTTP 401)."""

    def __init__(self, message: str, *, error_code: str = "invalid_token") -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class AuthorizationError(Exception):
    """Raised when an authenticated principal lacks domain or group permissions (HTTP 403)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _sanitize_www_auth_value(raw: str, *, max_len: int = 200) -> str:
    """Sanitize a string for safe embedding inside an RFC 9110 quoted-string header value."""
    printable = "".join(ch if 0x20 <= ord(ch) <= 0x7E else " " for ch in str(raw))
    collapsed = " ".join(printable.split())[:max_len]
    return collapsed.replace("\\", "\\\\").replace('"', '\\"')


def _derive_okta_endpoints(okta_issuer_uri: str) -> tuple[str, str]:
    """Derive Okta JWKS URI and OAuth2 authorization endpoint from an Okta issuer URL."""
    issuer = okta_issuer_uri.strip().rstrip("/")
    if "/oauth2/" in issuer:
        return f"{issuer}/v1/keys", f"{issuer}/v1/authorize"
    return f"{issuer}/oauth2/v1/keys", f"{issuer}/oauth2/v1/authorize"


def is_production_iap_mode() -> bool:
    """Return True when running in Cloud Run or when strict IAP mode is enabled."""
    if os.environ.get("K_SERVICE"):
        return True
    if env_bool("IAP_STRICT_MODE", False):
        return True
    if not env_bool("EDGAR_ENABLE_LOCAL_SSO_DEV_BROKER", True):
        return True
    return False


def _auth_env_snapshot() -> tuple[str | None, ...]:
    return tuple(os.environ.get(key) for key in _AUTH_ENV_KEYS)


@dataclass(frozen=True)
class AuthSettings:
    """Runtime authentication and authorization settings loaded from environment variables."""

    audiences: tuple[str, ...]
    okta_issuer_uri: str | None
    okta_jwks_uri: str | None
    okta_client_id: str | None
    google_oauth_client_id: str | None
    allowed_workspace_domains: tuple[str, ...]
    required_analyst_groups: tuple[str, ...]
    admin_workspace_groups: tuple[str, ...]
    analyst_workspace_groups: tuple[str, ...]
    allow_gcp_service_accounts: bool
    strict_production_mode: bool
    clock_skew_seconds: int = 30
    viewer_workspace_groups: tuple[str, ...] = (
        "sec-edgar-viewers",
        "sec-edgar-auditors",
        "compliance-auditors",
    )
    cloud_identity_domain: str = DEFAULT_CLOUD_IDENTITY_DOMAIN
    runtime_service_account: str = (
        f"{CUSTOM_RUNTIME_SA_ACCOUNT_ID}@{DEFAULT_PROJECT_ID}.iam.gserviceaccount.com"
    )
    allowed_service_account_projects: tuple[str, ...] = (
        DEFAULT_PROJECT_ID,
        DEFAULT_PROJECT_NAME,
    )
    allowed_custom_service_accounts: tuple[str, ...] = ()
    # Role granted to users authenticated only by a Cloud IAP assertion that carries no
    # matching group claims. IAP has already checked roles/iap.httpsResourceAccessor, so
    # this only decides what an admitted user can do in the app. None = deny.
    iap_default_role: str | None = None

    @classmethod
    def from_env(cls) -> AuthSettings:
        strict_mode = is_production_iap_mode()
        raw_iap_aud = os.environ.get("INTERNAL_IAP_AUDIENCE", "").strip()
        okta_client_id = os.environ.get("OKTA_CLIENT_ID", "").strip() or None
        google_client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "").strip() or None
        extra_audiences = split_csv_env(os.environ.get("ADDITIONAL_IAP_AUDIENCES"))

        aud_list: list[str] = []
        for candidate in (*split_csv_env(raw_iap_aud), *extra_audiences):
            if candidate and candidate not in aud_list:
                aud_list.append(candidate)
        for cid in (okta_client_id, google_client_id):
            if cid and cid not in aud_list:
                aud_list.append(cid)

        if not aud_list and not strict_mode:
            aud_list.append(DEFAULT_LOCAL_IAP_AUDIENCE)

        okta_issuer = os.environ.get("OKTA_ISSUER_URI", "").strip().rstrip("/") or None
        if okta_issuer and not okta_issuer.startswith("https://"):
            raise ValueError("OKTA_ISSUER_URI must use https://")

        okta_jwks = os.environ.get("OKTA_JWKS_URI", "").strip() or None
        if okta_issuer and not okta_jwks:
            okta_jwks, _ = _derive_okta_endpoints(okta_issuer)

        allowed_domains = tuple(
            d.lower() for d in split_csv_env(os.environ.get("ALLOWED_WORKSPACE_DOMAINS"))
        )
        required_groups = tuple(
            g.lower() for g in split_csv_env(os.environ.get("REQUIRED_ANALYST_GROUPS"))
        )
        admin_groups = tuple(
            g.lower()
            for g in (
                split_csv_env(os.environ.get("ADMIN_CLOUD_IDENTITY_GROUPS"))
                or split_csv_env(os.environ.get("ADMIN_WORKSPACE_GROUPS"))
                or ("sec-edgar-admins", "gcp-security-admins", "okta-sec-admins")
            )
        )
        analyst_groups = tuple(
            g.lower()
            for g in (
                split_csv_env(os.environ.get("ANALYST_CLOUD_IDENTITY_GROUPS"))
                or split_csv_env(os.environ.get("ANALYST_WORKSPACE_GROUPS"))
                or ("sec-edgar-analysts", "financial-analysts")
            )
        )
        viewer_groups = tuple(
            g.lower()
            for g in (
                split_csv_env(os.environ.get("VIEWER_CLOUD_IDENTITY_GROUPS"))
                or split_csv_env(os.environ.get("VIEWER_WORKSPACE_GROUPS"))
                or ("sec-edgar-viewers", "sec-edgar-auditors", "compliance-auditors")
            )
        )
        cloud_identity_domain = os.environ.get("CLOUD_IDENTITY_DOMAIN", "").strip().lower() or (
            allowed_domains[0] if allowed_domains else DEFAULT_CLOUD_IDENTITY_DOMAIN
        )
        runtime_sa_cfg = RuntimeServiceAccountConfig.from_env()
        allowed_custom_sas = tuple(
            sa.lower() for sa in split_csv_env(os.environ.get("ALLOWED_CUSTOM_SERVICE_ACCOUNTS"))
        )
        allow_sa = env_bool("ALLOW_GCP_SERVICE_ACCOUNTS", True)
        raw_iap_default_role = os.environ.get("IAP_ASSERTION_DEFAULT_ROLE", "").strip().lower()
        if raw_iap_default_role and raw_iap_default_role not in IAP_DEFAULT_ROLE_CHOICES:
            raise ValueError(
                "IAP_ASSERTION_DEFAULT_ROLE must be one of "
                f"{sorted(IAP_DEFAULT_ROLE_CHOICES)} or empty, got {raw_iap_default_role!r}"
            )

        return cls(
            audiences=tuple(aud_list),
            okta_issuer_uri=okta_issuer,
            okta_jwks_uri=okta_jwks,
            okta_client_id=okta_client_id,
            google_oauth_client_id=google_client_id,
            allowed_workspace_domains=allowed_domains,
            required_analyst_groups=required_groups,
            admin_workspace_groups=admin_groups,
            analyst_workspace_groups=analyst_groups,
            allow_gcp_service_accounts=allow_sa,
            strict_production_mode=strict_mode,
            viewer_workspace_groups=viewer_groups,
            cloud_identity_domain=cloud_identity_domain,
            runtime_service_account=runtime_sa_cfg.email,
            allowed_service_account_projects=runtime_sa_cfg.allowed_projects(),
            allowed_custom_service_accounts=allowed_custom_sas,
            iap_default_role=raw_iap_default_role or None,
        )

    def rbac_policy(self) -> CloudIdentityRBACPolicy:
        """Build the active Cloud Identity Group RBAC policy engine."""
        return CloudIdentityRBACPolicy.from_groups(
            admin_groups=self.admin_workspace_groups,
            analyst_groups=self.analyst_workspace_groups,
            viewer_groups=self.viewer_workspace_groups,
            required_analyst_groups=self.required_analyst_groups,
            cloud_identity_domain=self.cloud_identity_domain,
        )

    def authorization_endpoint(self) -> str | None:
        """Return the external OIDC authorization URL when Okta or Google OAuth is configured."""
        if self.okta_issuer_uri and self.okta_client_id:
            _, auth_url = _derive_okta_endpoints(self.okta_issuer_uri)
            return auth_url
        if self.google_oauth_client_id:
            return "https://accounts.google.com/o/oauth2/v2/auth"
        return None


@dataclass(frozen=True)
class AuthenticatedUser:
    """Verified user, custom runtime SA, or per-agent principal extracted from a JWT."""

    sub: str
    email: str
    name: str
    hosted_domain: str | None
    identity_provider: str
    sso_federation: str
    issuer: str
    audience: str
    groups: tuple[str, ...]
    roles: tuple[str, ...]
    issued_at: int
    expires_at: int
    token_fingerprint: str
    iap_assertion_verified: bool = False
    permissions: tuple[str, ...] = ()
    cloud_identity_groups: tuple[str, ...] = ()
    agent_identity: str | None = None
    spiffe_id: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "sub": self.sub,
            "email": self.email,
            "name": self.name,
            "hosted_domain": self.hosted_domain,
            "identity_provider": self.identity_provider,
            "sso_federation": self.sso_federation,
            "issuer": self.issuer,
            "audience": self.audience,
            "groups": list(self.groups),
            "cloud_identity_groups": list(self.cloud_identity_groups),
            "roles": list(self.roles),
            "permissions": list(self.permissions),
            "agent_identity": self.agent_identity,
            "spiffe_id": self.spiffe_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "iap_assertion_verified": self.iap_assertion_verified,
        }


# ---------------------------------------------------------------------------
# Ephemeral Local Development SSO Broker Keypair (Lazily Initialized, Zero Secrets)
# ---------------------------------------------------------------------------


class _EphemeralDevSsoSigner:
    """Generates an in-memory P-256 keypair per process for local dev SSO testing."""

    def __init__(self) -> None:
        self._private_key = ec.generate_private_key(ec.SECP256R1())
        self.kid = f"local-dev-es256-{uuid.uuid4().hex[:12]}"
        self.public_jwk = ec_public_key_to_jwk(self._private_key.public_key(), kid=self.kid)

    def mint_token(
        self,
        *,
        email: str,
        name: str,
        hosted_domain: str,
        groups: list[str],
        audience: str,
        ttl_seconds: int = 3600,
    ) -> tuple[str, int]:
        now = int(time.time())
        exp = now + ttl_seconds
        claims: dict[str, Any] = {
            "iss": LOCAL_DEV_SSO_ISSUER,
            "sub": f"okta_workspace:{email.lower()}",
            "aud": audience,
            "email": email.lower(),
            "email_verified": True,
            "name": name,
            "hd": hosted_domain.lower(),
            "idp": "okta_sso",
            "amr": ["sso", "okta", "mfa"],
            "groups": groups,
            "jti": uuid.uuid4().hex,
            "iat": now,
            "exp": exp,
        }
        return sign_jwt_es256(claims, self._private_key, kid=self.kid), exp


_LOCAL_SIGNER_LOCK = threading.Lock()
_LOCAL_SIGNER_HOLDER: dict[str, _EphemeralDevSsoSigner | None] = {"instance": None}


def _get_local_dev_signer() -> _EphemeralDevSsoSigner:
    signer = _LOCAL_SIGNER_HOLDER["instance"]
    if signer is None:
        with _LOCAL_SIGNER_LOCK:
            signer = _LOCAL_SIGNER_HOLDER["instance"]
            if signer is None:
                signer = _EphemeralDevSsoSigner()
                _LOCAL_SIGNER_HOLDER["instance"] = signer
    return signer


# ---------------------------------------------------------------------------
# Token Revocation & Security Audit Trail (Single Source of Truth in telemetry.py)
# ---------------------------------------------------------------------------

_REVOKED_TOKENS_LOCK = threading.Lock()
_REVOKED_TOKEN_FINGERPRINTS: dict[str, int] = {}


def compute_token_fingerprint(raw_jwt: str) -> str:
    """Compute a non-reversible SHA-256 digest of a JWT for revocation and audit tracking."""
    return sha256_text(raw_jwt)


def revoke_bearer_token(raw_jwt: str, *, expires_at: int) -> None:
    """Record a JWT digest as revoked until its expiration epoch."""
    now = int(time.time())
    fp = compute_token_fingerprint(raw_jwt)
    with _REVOKED_TOKENS_LOCK:
        expired_keys = [k for k, exp in _REVOKED_TOKEN_FINGERPRINTS.items() if exp <= now]
        for k in expired_keys:
            _REVOKED_TOKEN_FINGERPRINTS.pop(k, None)
        _REVOKED_TOKEN_FINGERPRINTS[fp] = max(expires_at, now + 60)


def is_bearer_token_revoked(raw_jwt: str) -> bool:
    """Check whether a JWT digest has been explicitly signed out / revoked."""
    now = int(time.time())
    fp = compute_token_fingerprint(raw_jwt)
    with _REVOKED_TOKENS_LOCK:
        exp = _REVOKED_TOKEN_FINGERPRINTS.get(fp)
        if exp is None:
            return False
        if exp <= now:
            _REVOKED_TOKEN_FINGERPRINTS.pop(fp, None)
            return False
        return True


def record_security_audit_event(
    *,
    event_type: str,
    user: AuthenticatedUser | None,
    resource: str,
    status_outcome: str = "allowed",
    tickers: list[str] | str | None = None,
    detail: str = "",
) -> dict[str, Any]:
    """Append a structured authentication or 10-K data access event to the audit trail."""
    if isinstance(tickers, str):
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    elif tickers is not None:
        ticker_list = [str(t).strip().upper() for t in tickers if str(t).strip()]
    else:
        ticker_list = []

    now_epoch = round(time.time(), 3)
    now_iso = datetime.fromtimestamp(now_epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry: dict[str, Any] = {
        "event_id": f"aud-{uuid.uuid4().hex[:12]}",
        "recorded_at": now_iso,
        "recorded_at_epoch": now_epoch,
        "event_type": event_type,
        "status": status_outcome,
        "resource": resource,
        "tickers": ticker_list,
        "user_email": user.email if user else "unauthenticated",
        "user_sub": user.sub if user else None,
        "hosted_domain": user.hosted_domain if user else None,
        "identity_provider": user.identity_provider if user else None,
        "roles": list(user.roles) if user else [],
        "cloud_identity_groups": list(user.cloud_identity_groups) if user else [],
        "agent_identity": user.agent_identity if user else None,
        "spiffe_id": user.spiffe_id if user else None,
        "iap_assertion_verified": user.iap_assertion_verified if user else False,
        "detail": detail,
    }
    # BigQueryTelemetrySink is the single source of truth: it buffers locally, emits the
    # structured Cloud Logging sink record, and streams to BigQuery when enabled.
    result = get_telemetry_sink().record_security_audit_log(entry)
    row = result.get("row") if isinstance(result, dict) else None
    return row if isinstance(row, dict) else entry


def get_security_audit_events(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent structured security and 10-K access audit events."""
    bounded = max(1, min(limit, 500))
    sink = get_telemetry_sink()
    events = sink.query_security_audit_logs(limit=bounded)
    if events or not sink.is_gcp_enabled():
        return events
    return []


def clear_security_audit_events() -> None:
    """Clear in-memory security audit events (intended for isolated test suites)."""
    get_telemetry_sink().clear_local_audit_events()


# ---------------------------------------------------------------------------
# JWKS Public Key Resolver & Cryptographic IAP / Okta / Workspace Verifier
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ParsedJwksCacheEntry:
    fetched_at: float
    raw_by_kid: dict[str, dict[str, Any]]
    parsed_keys: dict[str, PublicKey]
    parse_errors: dict[str, str]


class JwksKeyStore:
    """Fetches and caches parsed JWKS public keys for Google IAP, Google Workspace, and Okta."""

    def __init__(
        self,
        *,
        cache_ttl_seconds: float = 300.0,
        min_refresh_interval_seconds: float = 15.0,
        http_client: httpx.Client | None = None,
        static_jwks_by_uri: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.cache_ttl_seconds = cache_ttl_seconds
        self.min_refresh_interval_seconds = min_refresh_interval_seconds
        self._http_client = http_client
        self._pooled_http_client: httpx.Client | None = None
        self._static_jwks: dict[str, dict[str, Any]] = dict(static_jwks_by_uri or {})
        self._cache: dict[str, dict[str, dict[str, Any]]] = {}
        self._parsed_keys: dict[str, _ParsedJwksCacheEntry] = {}
        self._uri_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def _get_http_client(self) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        with self._lock:
            if self._pooled_http_client is None or self._pooled_http_client.is_closed:
                self._pooled_http_client = httpx.Client(
                    timeout=5.0,
                    limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
                )
            return self._pooled_http_client

    def close(self) -> None:
        """Close the internally managed pooled ``httpx.Client`` (leaves injected clients open)."""
        with self._lock:
            if self._pooled_http_client is not None and not self._pooled_http_client.is_closed:
                self._pooled_http_client.close()
            self._pooled_http_client = None

    def __enter__(self) -> JwksKeyStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _get_uri_lock(self, jwks_uri: str) -> threading.Lock:
        with self._lock:
            uri_lock = self._uri_locks.get(jwks_uri)
            if uri_lock is None:
                uri_lock = threading.Lock()
                self._uri_locks[jwks_uri] = uri_lock
            return uri_lock

    def register_static_jwks(self, jwks_uri: str, jwks_payload: dict[str, Any]) -> None:
        with self._lock:
            if self._static_jwks.get(jwks_uri) == jwks_payload:
                return
            self._static_jwks[jwks_uri] = jwks_payload
            self._cache.pop(jwks_uri, None)
            self._parsed_keys.pop(jwks_uri, None)

    @staticmethod
    def _parse_jwks_payload(payload: Any) -> _ParsedJwksCacheEntry:
        if not isinstance(payload, dict):
            raise AuthenticationError("Identity provider returned an empty or invalid JWKS keyset.")
        keys_list = payload.get("keys")
        if not isinstance(keys_list, list) or not keys_list:
            raise AuthenticationError("Identity provider returned an empty or invalid JWKS keyset.")

        raw_by_kid: dict[str, dict[str, Any]] = {}
        parsed_keys: dict[str, PublicKey] = {}
        parse_errors: dict[str, str] = {}

        for item in keys_list:
            if not isinstance(item, dict):
                continue
            raw_kid = item.get("kid")
            if not raw_kid:
                continue
            kid_str = str(raw_kid)
            raw_by_kid[kid_str] = item
            try:
                parsed_keys[kid_str] = jwk_to_public_key(item)
            except (ValueError, KeyError, TypeError) as exc:
                parse_errors[kid_str] = str(exc)

        return _ParsedJwksCacheEntry(
            fetched_at=time.monotonic(),
            raw_by_kid=raw_by_kid,
            parsed_keys=parsed_keys,
            parse_errors=parse_errors,
        )

    def _fetch_jwks_entry(self, jwks_uri: str) -> _ParsedJwksCacheEntry:
        with self._lock:
            static_payload = self._static_jwks.get(jwks_uri)

        if static_payload is not None:
            payload: Any = static_payload
        else:
            try:
                resp = self._get_http_client().get(jwks_uri)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:
                raise AuthenticationError(
                    f"Unable to retrieve JWKS signing keys from identity provider ({type(exc).__name__})."
                ) from exc

        return self._parse_jwks_payload(payload)

    @staticmethod
    def _resolve_from_entry(entry: _ParsedJwksCacheEntry, kid: str) -> PublicKey | None:
        if kid in entry.parsed_keys:
            return entry.parsed_keys[kid]
        if kid in entry.parse_errors:
            raise AuthenticationError(
                f"Invalid JWK signing key for kid={kid!r}: {entry.parse_errors[kid]}"
            )
        return None

    def get_public_key(
        self,
        *,
        jwks_uri: str,
        kid: str,
    ) -> PublicKey:
        now = time.monotonic()
        with self._lock:
            cached = self._parsed_keys.get(jwks_uri)
            is_static = jwks_uri in self._static_jwks

        if cached is not None and (now - cached.fetched_at) < self.cache_ttl_seconds:
            resolved = self._resolve_from_entry(cached, kid)
            if resolved is not None:
                return resolved
            if (now - cached.fetched_at) < self.min_refresh_interval_seconds and not is_static:
                raise AuthenticationError(f"Signing key kid={kid!r} not found in cached JWKS.")

        # Deduplicate concurrent refreshes for the same JWKS URI across threads.
        uri_lock = self._get_uri_lock(jwks_uri)
        with uri_lock:
            now = time.monotonic()
            with self._lock:
                cached = self._parsed_keys.get(jwks_uri)
                is_static = jwks_uri in self._static_jwks
            if cached is not None and (now - cached.fetched_at) < self.cache_ttl_seconds:
                resolved = self._resolve_from_entry(cached, kid)
                if resolved is not None:
                    return resolved
                if (now - cached.fetched_at) < self.min_refresh_interval_seconds and not is_static:
                    raise AuthenticationError(f"Signing key kid={kid!r} not found in cached JWKS.")

            fresh_entry = self._fetch_jwks_entry(jwks_uri)
            with self._lock:
                self._parsed_keys[jwks_uri] = fresh_entry
                self._cache[jwks_uri] = fresh_entry.raw_by_kid

        resolved = self._resolve_from_entry(fresh_entry, kid)
        if resolved is None:
            raise AuthenticationError(
                f"Signing key kid={kid!r} not found in identity provider JWKS."
            )
        return resolved


class IAPJwtVerifier:
    """Verifies JWT bearer assertions from IAP, Google Workspace OIDC, and Okta SSO."""

    def __init__(
        self,
        *,
        settings: AuthSettings | None = None,
        jwks_store: JwksKeyStore | None = None,
    ) -> None:
        self.settings = settings or AuthSettings.from_env()
        self.jwks_store = jwks_store or JwksKeyStore()
        if not self.settings.strict_production_mode:
            signer = _get_local_dev_signer()
            self.jwks_store.register_static_jwks(
                f"{LOCAL_DEV_SSO_ISSUER}/.well-known/jwks.json",
                {"keys": [signer.public_jwk]},
            )

    def _resolve_issuer_target(self, issuer: str) -> tuple[str, str]:
        """Return `(jwks_uri, identity_provider_id)` for allowlisted issuers only."""
        if issuer == GOOGLE_IAP_ISSUER:
            return GOOGLE_IAP_JWKS_URI, "google_iap"

        if issuer in GOOGLE_WORKSPACE_ISSUERS:
            return GOOGLE_WORKSPACE_JWKS_URI, "google_workspace_oidc"

        if self.settings.okta_issuer_uri and issuer.rstrip("/") == self.settings.okta_issuer_uri:
            if not self.settings.okta_jwks_uri:
                raise AuthenticationError("Okta JWKS URI is not configured on the server.")
            return self.settings.okta_jwks_uri, "okta_sso"

        if not self.settings.strict_production_mode and issuer == LOCAL_DEV_SSO_ISSUER:
            return f"{LOCAL_DEV_SSO_ISSUER}/.well-known/jwks.json", "local_workspace_okta_dev"

        raise AuthenticationError(f"Untrusted JWT issuer: {issuer!r}")

    @staticmethod
    def _normalize_email(claims: dict[str, Any]) -> str:
        raw_email = claims.get("email") or claims.get("preferred_username") or ""
        if not isinstance(raw_email, str) or not raw_email.strip():
            raise AuthenticationError("JWT is missing a user or service principal email claim.")
        cleaned = raw_email.strip()
        if cleaned.startswith("accounts.google.com:"):
            cleaned = cleaned.split(":", 1)[1].strip()
        if "@" not in cleaned:
            raise AuthenticationError("JWT email claim is invalid.")
        return cleaned.lower()

    @staticmethod
    def _extract_groups(claims: dict[str, Any]) -> tuple[str, ...]:
        found: list[str] = []
        raw_groups = claims.get("groups")
        if isinstance(raw_groups, list):
            for g in raw_groups:
                if isinstance(g, str) and g.strip():
                    norm = g.strip().lower()
                    if norm not in found:
                        found.append(norm)

        google_claim = claims.get("google")
        if isinstance(google_claim, dict):
            for field_name in ("groups", "access_levels"):
                items = google_claim.get(field_name)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str) and item.strip():
                            norm = item.strip().lower()
                            if norm not in found:
                                found.append(norm)
        return tuple(found)

    def _validate_temporal_claims(self, claims: dict[str, Any]) -> tuple[int, int]:
        now = int(time.time())
        skew = self.settings.clock_skew_seconds

        exp_raw = claims.get("exp")
        if not isinstance(exp_raw, (int, float)) or isinstance(exp_raw, bool):
            raise AuthenticationError("JWT is missing numeric 'exp' claim.")
        exp = int(exp_raw)
        if now - skew >= exp:
            raise AuthenticationError("JWT bearer token has expired.")

        iat_raw = claims.get("iat", now)
        if not isinstance(iat_raw, (int, float)) or isinstance(iat_raw, bool):
            raise AuthenticationError("JWT has invalid 'iat' claim.")
        iat = int(iat_raw)
        if iat > now + skew:
            raise AuthenticationError("JWT 'iat' timestamp is in the future.")

        nbf_raw = claims.get("nbf")
        if nbf_raw is not None:
            if not isinstance(nbf_raw, (int, float)) or isinstance(nbf_raw, bool):
                raise AuthenticationError("JWT has invalid 'nbf' claim.")
            if int(nbf_raw) > now + skew:
                raise AuthenticationError("JWT bearer token is not yet valid ('nbf').")

        return iat, exp

    def _validate_audience(self, claims: dict[str, Any]) -> str:
        if not self.settings.audiences:
            raise AuthenticationError("IAP OAuth2 / JWT audience is not configured on the server.")

        aud_claim = claims.get("aud")
        if isinstance(aud_claim, str):
            token_audiences = [aud_claim]
        elif isinstance(aud_claim, list) and all(isinstance(x, str) for x in aud_claim):
            token_audiences = list(aud_claim)
        else:
            raise AuthenticationError("JWT is missing a valid 'aud' claim.")

        for candidate in token_audiences:
            if candidate in self.settings.audiences:
                return candidate

        raise AuthenticationError("JWT audience ('aud') does not match configured IAP audience.")

    def _enforce_service_account_and_agent_policy(
        self,
        email: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...], str | None, str | None]:
        """Enforce custom runtime SA (`sec-analyst-sa`) and per-agent least-privilege S2S rules."""
        if not self.settings.allow_gcp_service_accounts:
            raise AuthorizationError(
                "Service account principals are disabled by current access policy."
            )

        if is_default_gcp_service_account(email):
            raise AuthorizationError(
                f"Default GCP service account {email!r} is prohibited; only custom runtime "
                f"({CUSTOM_RUNTIME_SA_ACCOUNT_ID}@<project>.iam.gserviceaccount.com) and "
                "granular per-agent service accounts are permitted."
            )

        try:
            account_id, project_name = validate_custom_service_account_email(
                email,
                expected_account_id=None,
            )
        except ValueError as exc:
            raise AuthorizationError(str(exc)) from exc

        is_explicitly_allowed = email in self.settings.allowed_custom_service_accounts
        if (
            not is_explicitly_allowed
            and self.settings.allowed_service_account_projects
            and project_name not in self.settings.allowed_service_account_projects
        ):
            raise AuthorizationError(
                f"Service account project {project_name!r} is not authorized for SEC EDGAR S2S access."
            )

        runtime_account_id = self.settings.runtime_service_account.split("@", 1)[0].lower()
        if account_id in {CUSTOM_RUNTIME_SA_ACCOUNT_ID, runtime_account_id}:
            return (
                ("authenticated", "service_account", "analyst"),
                tuple(sorted(ANALYST_PERMISSIONS)),
                None,
                f"spiffe://{project_name}.svc.id.goog/ns/sec-edgar/sa/{account_id}",
            )

        agent_spec = resolve_agent_by_service_account_id(account_id)
        if agent_spec is not None:
            return (
                (
                    "authenticated",
                    "service_account",
                    "agent",
                    f"agent:{agent_spec.agent_name}",
                ),
                tuple(sorted(agent_spec.api_permissions)),
                agent_spec.agent_name,
                agent_spec.spiffe_id(project_name),
            )

        if account_id == "sec-edgar-cicd-sa":
            return (
                ("authenticated", "service_account", "cicd_probe"),
                (Permission.RBAC_INSPECT.value,),
                None,
                f"spiffe://{project_name}.svc.id.goog/ns/sec-edgar/sa/{account_id}",
            )

        if is_explicitly_allowed:
            return (
                ("authenticated", "service_account"),
                (Permission.FILINGS_READ.value, Permission.RBAC_INSPECT.value),
                None,
                f"spiffe://{project_name}.svc.id.goog/ns/sec-edgar/sa/{account_id}",
            )

        raise AuthorizationError(
            f"Unrecognized service account {email!r}; only custom runtime SA "
            f"'{CUSTOM_RUNTIME_SA_ACCOUNT_ID}@<project>.iam.gserviceaccount.com' and registered "
            "per-agent service accounts are authorized."
        )

    def _enforce_workspace_and_rbac(
        self,
        *,
        email: str,
        claims: dict[str, Any],
        idp: str,
        groups: tuple[str, ...],
    ) -> tuple[
        str | None,
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        str | None,
        str | None,
    ]:
        is_gcp_sa = email.endswith(".gserviceaccount.com")

        if idp == "google_workspace_oidc" and claims.get("email_verified") is False:
            raise AuthenticationError("Google Workspace email address is not verified.")

        hd_claim = claims.get("hd")
        if isinstance(hd_claim, str) and hd_claim.strip():
            hosted_domain: str | None = hd_claim.strip().lower()
        elif not is_gcp_sa and "@" in email:
            email_domain = email.rsplit("@", 1)[1].lower()
            hosted_domain = (
                None if email_domain in {"gmail.com", "googlemail.com"} else email_domain
            )
        else:
            hosted_domain = None

        if is_gcp_sa:
            sa_roles, sa_perms, agent_id, spiffe_id = (
                self._enforce_service_account_and_agent_policy(email)
            )
            return None, sa_roles, sa_perms, (), agent_id, spiffe_id

        if self.settings.allowed_workspace_domains:
            if not hosted_domain or hosted_domain not in self.settings.allowed_workspace_domains:
                raise AuthorizationError(
                    f"Google Workspace domain {hosted_domain or 'consumer-account'!r} is not "
                    "authorized for SEC EDGAR internal APIs."
                )

        policy = self.settings.rbac_policy()
        roles, permissions, matched_ci_groups = policy.resolve_human_access(
            user_groups=groups,
            hosted_domain=hosted_domain,
        )
        if not roles and idp == "google_iap" and self.settings.iap_default_role:
            # IAP assertions carry no group claims. IAP only signs one after the caller
            # passed roles/iap.httpsResourceAccessor on this service, so grant the
            # configured default role instead of rejecting every IAP-only user.
            for role_def in policy.role_definitions:
                if role_def.role == self.settings.iap_default_role:
                    roles = ("authenticated", role_def.role)
                    permissions = tuple(sorted(role_def.permissions))
                    break
        if not roles:
            raise AuthorizationError(
                "Authenticated Workspace/Okta user is not a member of any required Cloud Identity RBAC group."
            )

        return hosted_domain, roles, permissions, matched_ci_groups, None, None

    @traced("auth.verify_jwt")
    def verify_jwt(
        self,
        raw_jwt: str,
        *,
        require_iap_issuer: bool = False,
    ) -> AuthenticatedUser:
        """Cryptographically verify a compact JWT and return the authenticated principal."""
        token = raw_jwt.strip()
        if not token or len(token) > MAX_JWT_LENGTH_BYTES:
            raise AuthenticationError("Missing or oversized JWT bearer token.")

        if is_bearer_token_revoked(token):
            raise AuthenticationError("JWT bearer token has been signed out or revoked.")

        parts = token.split(".")
        if len(parts) != 3 or not all(parts):
            raise AuthenticationError("JWT must contain header, payload, and signature segments.")

        header_b64, payload_b64, sig_b64 = parts
        try:
            header, claims = parse_jwt_header_and_claims(header_b64, payload_b64)
        except ValueError as exc:
            raise AuthenticationError(str(exc)) from exc

        alg = header.get("alg")
        if not isinstance(alg, str) or alg not in ALLOWED_JWS_ALGORITHMS:
            raise AuthenticationError(f"Disallowed or missing JWT algorithm: {alg!r}")

        kid = header.get("kid")
        if not isinstance(kid, str) or not kid.strip():
            raise AuthenticationError("JWT header is missing required 'kid' key identifier.")

        issuer = claims.get("iss")
        if not isinstance(issuer, str) or not issuer.strip():
            raise AuthenticationError("JWT payload is missing 'iss' claim.")
        if require_iap_issuer and issuer != GOOGLE_IAP_ISSUER:
            raise AuthenticationError(
                "IAP assertion header must be issued by https://cloud.google.com/iap."
            )

        jwks_uri, idp = self._resolve_issuer_target(issuer)
        if idp == "google_iap" and alg != "ES256":
            raise AuthenticationError("Google Cloud IAP assertions must use ES256.")

        public_key = self.jwks_store.get_public_key(jwks_uri=jwks_uri, kid=kid.strip())
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        try:
            signature_bytes = b64url_decode(sig_b64)
            verify_jws_signature(
                alg=alg,
                signing_input=signing_input,
                signature_bytes=signature_bytes,
                public_key=public_key,
            )
        except ValueError as exc:
            raise AuthenticationError(str(exc)) from exc

        sub = claims.get("sub")
        if not isinstance(sub, str) or not sub.strip():
            raise AuthenticationError("JWT payload is missing 'sub' subject claim.")

        iat, exp = self._validate_temporal_claims(claims)
        matched_aud = self._validate_audience(claims)
        email = self._normalize_email(claims)
        groups = self._extract_groups(claims)
        (
            hosted_domain,
            roles,
            permissions,
            cloud_identity_groups,
            agent_identity,
            spiffe_id,
        ) = self._enforce_workspace_and_rbac(
            email=email,
            claims=claims,
            idp=idp,
            groups=groups,
        )

        raw_name = claims.get("name")
        display_name = (
            raw_name.strip()
            if isinstance(raw_name, str) and raw_name.strip()
            else email.split("@", 1)[0].replace(".", " ").title()
        )

        sso_federation = "google_workspace_okta_sso"
        if email.endswith(".gserviceaccount.com"):
            sso_federation = (
                "gcp_agent_workload_identity" if agent_identity else "gcp_iam_service_account"
            )

        return AuthenticatedUser(
            sub=sub.strip(),
            email=email,
            name=display_name,
            hosted_domain=hosted_domain,
            identity_provider=idp,
            sso_federation=sso_federation,
            issuer=issuer,
            audience=matched_aud,
            groups=groups,
            roles=roles,
            issued_at=iat,
            expires_at=exp,
            token_fingerprint=compute_token_fingerprint(token),
            iap_assertion_verified=(idp == "google_iap"),
            permissions=permissions,
            cloud_identity_groups=cloud_identity_groups,
            agent_identity=agent_identity,
            spiffe_id=spiffe_id,
        )


# ---------------------------------------------------------------------------
# FastAPI Dependencies & Verifier Lifecycle
# ---------------------------------------------------------------------------

_DEFAULT_JWKS_STORE = JwksKeyStore()
_VERIFIER_CACHE_LOCK = threading.Lock()
_VERIFIER_CACHE: dict[str, tuple[tuple[str | None, ...], IAPJwtVerifier] | None] = {"entry": None}


def get_auth_verifier() -> IAPJwtVerifier:
    """Return a cached `IAPJwtVerifier` keyed by the active auth environment snapshot."""
    snapshot = _auth_env_snapshot()
    with _VERIFIER_CACHE_LOCK:
        cached = _VERIFIER_CACHE["entry"]
        if cached is not None and cached[0] == snapshot:
            return cached[1]
        verifier = IAPJwtVerifier(
            settings=AuthSettings.from_env(),
            jwks_store=_DEFAULT_JWKS_STORE,
        )
        _VERIFIER_CACHE["entry"] = (snapshot, verifier)
        return verifier


def extract_bearer_token_from_request(request: Request) -> str:
    """Extract the raw JWT from `Authorization: Bearer <token>` or raise `AuthenticationError`."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        raise AuthenticationError(
            "Missing Authorization header. Internal APIs are protected by Identity-Aware Proxy (IAP) "
            "and require JWT-based Bearer authentication.",
            error_code="missing_token",
        )
    scheme, _, credentials = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not credentials.strip():
        raise AuthenticationError(
            "Invalid Authorization header scheme. Expected 'Authorization: Bearer <jwt>'.",
            error_code="invalid_request",
        )
    return credentials.strip()


def require_authenticated_user(
    request: Request,
    verifier: IAPJwtVerifier = Depends(get_auth_verifier),
) -> AuthenticatedUser:
    """FastAPI dependency enforcing IAP + Google Workspace Okta/SSO JWT Bearer authentication.

    Browser sessions behind Cloud IAP send no `Authorization` header; IAP injects a signed
    `X-Goog-IAP-JWT-Assertion` instead. That assertion alone is accepted when it verifies
    as ES256, issued by https://cloud.google.com/iap, for a configured audience.
    """
    try:
        iap_assertion = request.headers.get(IAP_ASSERTION_HEADER, "").strip()
        iap_session_only = bool(iap_assertion) and not request.headers.get("Authorization")

        if iap_session_only:
            raw_bearer = ""
            user = verifier.verify_jwt(iap_assertion, require_iap_issuer=True)
        else:
            raw_bearer = extract_bearer_token_from_request(request)
            user = verifier.verify_jwt(raw_bearer)

        # Defense-in-depth: if routed through Google Cloud IAP proxy with X-Goog-IAP-JWT-Assertion,
        # cryptographically validate the proxy assertion as well and ensure identity consistency.
        # SECURITY: Keep Bearer JWT's roles, permissions, groups, and cloud_identity_groups
        # strictly authoritative. Never union iap_user's roles/permissions (which may come from
        # `iap_default_role` fallback or a different principal when calling with a service account).
        if iap_assertion and not iap_session_only:
            iap_user = verifier.verify_jwt(iap_assertion, require_iap_issuer=True)
            is_sa_principal = user.email.endswith(".gserviceaccount.com")
            if not is_sa_principal and iap_user.email != user.email:
                raise AuthenticationError(
                    "Principal mismatch between Bearer JWT and X-Goog-IAP-JWT-Assertion header."
                )
            user = AuthenticatedUser(
                sub=user.sub,
                email=user.email,
                name=user.name,
                hosted_domain=(
                    user.hosted_domain or (None if is_sa_principal else iap_user.hosted_domain)
                ),
                identity_provider=user.identity_provider,
                sso_federation=user.sso_federation,
                issuer=user.issuer,
                audience=user.audience,
                groups=user.groups,
                roles=user.roles,
                issued_at=user.issued_at,
                expires_at=min(user.expires_at, iap_user.expires_at),
                token_fingerprint=user.token_fingerprint,
                iap_assertion_verified=True,
                permissions=user.permissions,
                cloud_identity_groups=user.cloud_identity_groups,
                agent_identity=user.agent_identity,
                spiffe_id=user.spiffe_id,
            )

        # Per-agent S2S Least Privilege attenuation: when a service-to-service request
        # carries an `X-Agent-Identity-Assertion` JWT, verify it cryptographically and
        # attenuate effective permissions strictly to that agent's least-privilege scope.
        s2s_assertion = request.headers.get(S2S_AGENT_ASSERTION_HEADER, "").strip()
        if s2s_assertion:
            if "service_account" not in user.roles:
                raise AuthorizationError(
                    f"{S2S_AGENT_ASSERTION_HEADER} header is restricted to service-to-service "
                    "service account principals."
                )
            try:
                s2s_claims = get_agent_s2s_broker().verify_s2s_assertion(s2s_assertion)
            except AgentAuthorizationError as exc:
                raise AuthorizationError(exc.message) from exc
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise AuthorizationError(
                    f"Invalid {S2S_AGENT_ASSERTION_HEADER} assertion: {exc}"
                ) from exc

            # If Bearer token is itself a per-agent SA, ensure it matches the S2S assertion agent
            if user.agent_identity and user.agent_identity != s2s_claims.agent_name:
                raise AuthenticationError(
                    f"Agent identity mismatch between Bearer SA ({user.agent_identity}) "
                    f"and {S2S_AGENT_ASSERTION_HEADER} ({s2s_claims.agent_name})."
                )

            attenuated_permissions = tuple(
                sorted(set(user.permissions).intersection(s2s_claims.permissions))
            )
            agent_roles = tuple(
                dict.fromkeys(
                    (
                        "authenticated",
                        "service_account",
                        "agent",
                        f"agent:{s2s_claims.agent_name}",
                    )
                )
            )
            user = AuthenticatedUser(
                sub=user.sub,
                email=user.email,
                name=user.name,
                hosted_domain=user.hosted_domain,
                identity_provider=user.identity_provider,
                sso_federation="gcp_agent_workload_identity",
                issuer=user.issuer,
                audience=user.audience,
                groups=user.groups,
                roles=agent_roles,
                issued_at=user.issued_at,
                expires_at=min(user.expires_at, s2s_claims.expires_at),
                token_fingerprint=user.token_fingerprint,
                iap_assertion_verified=user.iap_assertion_verified,
                permissions=attenuated_permissions,
                cloud_identity_groups=user.cloud_identity_groups,
                agent_identity=s2s_claims.agent_name,
                spiffe_id=s2s_claims.spiffe_id,
            )

        request.state.authenticated_user = user
        request.state.raw_bearer_token = raw_bearer
        return user

    except AuthenticationError as exc:
        record_security_audit_event(
            event_type="api_authentication_failed",
            user=None,
            resource=request.url.path,
            status_outcome="denied_401",
            detail=exc.message,
        )
        safe_code = _sanitize_www_auth_value(exc.error_code, max_len=64)
        safe_desc = _sanitize_www_auth_value(exc.message, max_len=200)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "unauthorized",
                "error": exc.error_code,
                "message": exc.message,
                "identity_provider": "Google Workspace (Okta SSO)",
                "access_pattern": "IAP JWT Bearer",
            },
            headers={
                "WWW-Authenticate": (
                    f'Bearer realm="sec-edgar-iap", error="{safe_code}", '
                    f'error_description="{safe_desc}"'
                )
            },
        ) from exc

    except AuthorizationError as exc:
        record_security_audit_event(
            event_type="api_authorization_denied",
            user=None,
            resource=request.url.path,
            status_outcome="denied_403",
            detail=exc.message,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "status": "forbidden",
                "error": "insufficient_permissions",
                "message": exc.message,
            },
        ) from exc


def require_permissions(
    *required_permissions: str,
    agent_tool: str | None = None,
):
    """Build a FastAPI dependency enforcing Cloud Identity RBAC or per-agent S2S permissions."""

    def _dependency(
        request: Request,
        user: AuthenticatedUser = Depends(require_authenticated_user),
    ) -> AuthenticatedUser:
        missing = [p for p in required_permissions if p not in user.permissions]
        if missing:
            record_security_audit_event(
                event_type="api_authorization_denied",
                user=user,
                resource=request.url.path,
                status_outcome="denied_403",
                detail=(
                    f"Missing required permission(s) {missing}; "
                    f"principal={user.email} roles={list(user.roles)} "
                    f"agent_identity={user.agent_identity}."
                ),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "status": "forbidden",
                    "error": "insufficient_permissions",
                    "message": (
                        f"Principal '{user.email}' lacks required RBAC / agent permission(s): "
                        f"{missing}."
                    ),
                    "required_permissions": list(required_permissions),
                    "granted_permissions": list(user.permissions),
                    "agent_identity": user.agent_identity,
                },
            )

        if agent_tool is not None and user.agent_identity is not None:
            agent_spec = resolve_agent_identity(user.agent_identity)
            if agent_tool not in agent_spec.allowed_tools:
                record_security_audit_event(
                    event_type="agent_s2s_tool_boundary_denied",
                    user=user,
                    resource=request.url.path,
                    status_outcome="denied_403",
                    detail=(
                        f"Agent '{agent_spec.agent_name}' ({user.email}) is forbidden from "
                        f"calling endpoint bound to tool '{agent_tool}'."
                    ),
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "status": "forbidden",
                        "error": "agent_least_privilege_violation",
                        "message": (
                            f"Agent identity '{agent_spec.agent_name}' ({user.email}) is not "
                            f"authorized for tool boundary '{agent_tool}' under Least Privilege."
                        ),
                        "agent_identity": agent_spec.agent_name,
                        "required_tool": agent_tool,
                        "allowed_tools": sorted(agent_spec.allowed_tools),
                    },
                )
        return user

    return _dependency


def issue_local_dev_workspace_sso_token(
    *,
    email: str = "analyst@fde-workspace.corp",
    name: str = "Senior Financial Analyst",
    groups: list[str] | None = None,
    verifier: IAPJwtVerifier | None = None,
) -> tuple[str, AuthenticatedUser]:
    """Mint and verify an ephemeral ES256 JWT for local Workspace/Okta SSO development.

    Strictly disabled in Cloud Run (`K_SERVICE`) and when `IAP_STRICT_MODE=true`.
    """
    active_verifier = verifier or get_auth_verifier()
    if active_verifier.settings.strict_production_mode:
        raise AuthorizationError(
            "Local Google Workspace / Okta SSO token broker is disabled in Cloud Run / IAP strict mode."
        )

    cleaned_email = email.strip().lower()
    if "@" not in cleaned_email:
        raise AuthenticationError("Valid corporate Google Workspace email is required.")
    hosted_domain = cleaned_email.rsplit("@", 1)[1]

    audience = (
        active_verifier.settings.audiences[0]
        if active_verifier.settings.audiences
        else DEFAULT_LOCAL_IAP_AUDIENCE
    )
    resolved_groups = (
        [g.strip().lower() for g in groups if g.strip()]
        if groups is not None
        else ["sec-edgar-analysts"]
    )

    raw_jwt, _ = _get_local_dev_signer().mint_token(
        email=cleaned_email,
        name=name.strip() or "SEC Analyst",
        hosted_domain=hosted_domain,
        groups=resolved_groups,
        audience=audience,
    )
    verified_user = active_verifier.verify_jwt(raw_jwt)
    return raw_jwt, verified_user


__all__ = [
    "ALLOWED_JWS_ALGORITHMS",
    "DEFAULT_LOCAL_IAP_AUDIENCE",
    "GOOGLE_IAP_ISSUER",
    "GOOGLE_IAP_JWKS_URI",
    "GOOGLE_WORKSPACE_ISSUERS",
    "GOOGLE_WORKSPACE_JWKS_URI",
    "IAP_ASSERTION_HEADER",
    "IAP_DEFAULT_ROLE_CHOICES",
    "LOCAL_DEV_SSO_ISSUER",
    "MAX_JWT_LENGTH_BYTES",
    "AuthSettings",
    "AuthenticatedUser",
    "AuthenticationError",
    "AuthorizationError",
    "IAPJwtVerifier",
    "JwksKeyStore",
    "b64url_decode",
    "b64url_encode",
    "clear_security_audit_events",
    "compute_token_fingerprint",
    "ec_public_key_to_jwk",
    "extract_bearer_token_from_request",
    "get_auth_verifier",
    "get_security_audit_events",
    "is_bearer_token_revoked",
    "is_production_iap_mode",
    "issue_local_dev_workspace_sso_token",
    "jwk_to_public_key",
    "record_security_audit_event",
    "require_authenticated_user",
    "require_permissions",
    "revoke_bearer_token",
    "rsa_public_key_to_jwk",
    "sign_jwt_es256",
    "sign_jwt_rs256",
]
