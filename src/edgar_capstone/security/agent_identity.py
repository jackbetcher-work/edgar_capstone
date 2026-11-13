"""Granular Per-Agent Identities & Least-Privilege Service-to-Service (S2S) Security.

Enforces per-agent cryptographic and IAM identities across:
- `RootOrchestrator` (`root-orchestrator-agent-sa@<project>.iam.gserviceaccount.com`)
- `FinancialAnalystAgent` (`financial-analyst-agent-sa@<project>.iam.gserviceaccount.com`)
- `ContextCacheManager` / `ContextCacheAgent` (`context-cache-agent-sa@<project>.iam.gserviceaccount.com`)
- `ModelArmorGuardrail` (`model-armor-guardrail-sa@<project>.iam.gserviceaccount.com`)

Each agent receives a distinct SPIFFE workload identity, dedicated GCP service
account, explicit allowed-caller / allowed-target S2S policy graph, allowed tool
boundary, and short-lived audience-bound S2S token attestation.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
import time
from typing import Any
import uuid

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
import httpx

from edgar_capstone.security.rbac import (
    CUSTOM_RUNTIME_SA_ACCOUNT_ID,
    Permission,
    RuntimeServiceAccountConfig,
)
from edgar_capstone.security.secrets import DEFAULT_PROJECT_ID

S2S_AGENT_ASSERTION_HEADER = "X-Agent-Identity-Assertion"
AGENT_IDENTITY_ISSUER = "https://agent-identity.sec-edgar.internal"
MAX_AGENT_S2S_TTL_SECONDS = 300


class AgentAuthorizationError(PermissionError):
    """Raised when an agent attempts a tool or S2S call outside its least-privilege scope."""

    def __init__(
        self,
        message: str,
        *,
        agent_name: str,
        target: str,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.agent_name = agent_name
        self.target = target


@dataclass(frozen=True)
class AgentIdentitySpec:
    """Granular least-privilege identity and S2S communication policy for a single agent."""

    agent_name: str
    service_account_id: str
    description: str
    allowed_callers: frozenset[str]
    allowed_target_services: frozenset[str]
    allowed_tools: frozenset[str]
    api_permissions: frozenset[str]
    gcp_iam_roles: tuple[str, ...]
    oauth_scopes: tuple[str, ...]

    def service_account_email(self, project_name: str | None = None) -> str:
        resolved_project = (
            (project_name or "").strip().lower()
            or os.environ.get("GCP_PROJECT_ID", "").strip().lower()
            or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip().lower()
            or DEFAULT_PROJECT_ID
        )
        return f"{self.service_account_id}@{resolved_project}.iam.gserviceaccount.com"

    def spiffe_id(self, project_name: str | None = None) -> str:
        resolved_project = (
            (project_name or "").strip().lower()
            or os.environ.get("GCP_PROJECT_ID", "").strip().lower()
            or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip().lower()
            or DEFAULT_PROJECT_ID
        )
        return (
            f"spiffe://{resolved_project}.svc.id.goog"
            f"/ns/sec-edgar/sa/{self.service_account_id}/agent/{self.agent_name}"
        )

    def to_public_dict(self, project_name: str | None = None) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "service_account_id": self.service_account_id,
            "service_account_email": self.service_account_email(project_name),
            "spiffe_id": self.spiffe_id(project_name),
            "description": self.description,
            "allowed_callers": sorted(self.allowed_callers),
            "allowed_target_services": sorted(self.allowed_target_services),
            "allowed_tools": sorted(self.allowed_tools),
            "api_permissions": sorted(self.api_permissions),
            "gcp_iam_roles": list(self.gcp_iam_roles),
            "oauth_scopes": list(self.oauth_scopes),
        }


# Logical backing service boundaries for ADK tools & internal S2S hops
TOOL_TARGET_SERVICE_MAP: dict[str, str] = {
    "classify_query_intent": "intent-router-service",
    "fetch_financial_metrics": "bigquery-metrics-service",
    "query_looker_financial_explore": "looker-semantic-service",
    "calculate_period_variances": "variance-calculation-service",
    "search_10k_chunks_rag": "hybrid-rag-search-service",
    "load_cached_10k_section": "context-cache-service",
    "before_model_safety_and_latency_callback": "model-armor-service",
    "after_model_fact_check_callback": "fact-check-verification-service",
}

AGENT_TARGET_SERVICE_IDS: dict[str, str] = {
    "RootOrchestrator": "root-orchestrator-agent",
    "FinancialAnalystAgent": "financial-analyst-agent",
    "ContextCacheManager": "context-cache-agent",
    "ContextCacheAgent": "context-cache-agent",
    "ModelArmorGuardrail": "model-armor-guardrail",
}

PER_AGENT_IDENTITIES: dict[str, AgentIdentitySpec] = {
    "RootOrchestrator": AgentIdentitySpec(
        agent_name="RootOrchestrator",
        service_account_id="root-orchestrator-agent-sa",
        description=(
            "Central ADK supervisor for intent classification, structured metric routing, "
            "and delegating deep 10-K analysis and context caching to specialist sub-agents."
        ),
        allowed_callers=frozenset({CUSTOM_RUNTIME_SA_ACCOUNT_ID}),
        allowed_target_services=frozenset(
            {
                "intent-router-service",
                "bigquery-metrics-service",
                "looker-semantic-service",
                "variance-calculation-service",
                "financial-analyst-agent",
                "context-cache-agent",
                "model-armor-guardrail",
                "fact-check-verification-service",
            }
        ),
        allowed_tools=frozenset(
            {
                "classify_query_intent",
                "fetch_financial_metrics",
                "query_looker_financial_explore",
                "calculate_period_variances",
            }
        ),
        api_permissions=frozenset(
            {
                Permission.FILINGS_READ.value,
                Permission.VARIANCES_CALCULATE.value,
                Permission.CHAT_QUERY.value,
                Permission.CHUNK_HEALTH_READ.value,
                Permission.RBAC_INSPECT.value,
            }
        ),
        gcp_iam_roles=(
            "roles/aiplatform.user",
            "roles/run.invoker",
            "roles/logging.logWriter",
            "roles/monitoring.metricWriter",
        ),
        oauth_scopes=(
            "https://www.googleapis.com/auth/cloud-platform.read-only",
            "https://www.googleapis.com/auth/bigquery.readonly",
        ),
    ),
    "FinancialAnalystAgent": AgentIdentitySpec(
        agent_name="FinancialAnalystAgent",
        service_account_id="financial-analyst-agent-sa",
        description=(
            "Specialist SEC 10-K reasoning agent scoped to deterministic BigQuery financial "
            "variances, Discovery Engine / Vector Search Hybrid RAG, and section cache reads."
        ),
        allowed_callers=frozenset({"RootOrchestrator", CUSTOM_RUNTIME_SA_ACCOUNT_ID}),
        allowed_target_services=frozenset(
            {
                "variance-calculation-service",
                "bigquery-metrics-service",
                "looker-semantic-service",
                "hybrid-rag-search-service",
                "context-cache-service",
                "context-cache-agent",
                "model-armor-guardrail",
                "fact-check-verification-service",
            }
        ),
        allowed_tools=frozenset(
            {
                "calculate_period_variances",
                "fetch_financial_metrics",
                "query_looker_financial_explore",
                "search_10k_chunks_rag",
                "load_cached_10k_section",
            }
        ),
        api_permissions=frozenset(
            {
                Permission.FILINGS_READ.value,
                Permission.VARIANCES_CALCULATE.value,
                Permission.CHUNK_HEALTH_READ.value,
            }
        ),
        gcp_iam_roles=(
            "roles/bigquery.jobUser",
            "roles/bigquery.connectionUser",
            "roles/bigquery.dataViewer",
            "roles/discoveryengine.viewer",
            "roles/aiplatform.user",
            "roles/logging.logWriter",
            "roles/monitoring.metricWriter",
        ),
        oauth_scopes=(
            "https://www.googleapis.com/auth/bigquery.readonly",
            "https://www.googleapis.com/auth/cloud-platform.read-only",
        ),
    ),
    "ContextCacheManager": AgentIdentitySpec(
        agent_name="ContextCacheManager",
        service_account_id="context-cache-agent-sa",
        description=(
            "Least-privilege 10-K section loader and Vertex AI Context Cache specialist "
            "scoped strictly to reading curated SEC 10-K Markdown objects from GCS."
        ),
        allowed_callers=frozenset(
            {"RootOrchestrator", "FinancialAnalystAgent", CUSTOM_RUNTIME_SA_ACCOUNT_ID}
        ),
        allowed_target_services=frozenset(
            {
                "context-cache-service",
                "gcs-sec-corpus-bucket",
            }
        ),
        allowed_tools=frozenset({"load_cached_10k_section"}),
        api_permissions=frozenset({Permission.FILINGS_READ.value}),
        gcp_iam_roles=(
            "roles/storage.objectViewer",
            "roles/aiplatform.user",
            "roles/logging.logWriter",
        ),
        oauth_scopes=(
            "https://www.googleapis.com/auth/devstorage.read_only",
        ),
    ),
    "ModelArmorGuardrail": AgentIdentitySpec(
        agent_name="ModelArmorGuardrail",
        service_account_id="model-armor-guardrail-sa",
        description=(
            "Isolated prompt-safety and grounding-lock inspection identity with zero "
            "access to BigQuery financial tables, GCS corpus buckets, or secrets."
        ),
        allowed_callers=frozenset(
            {"RootOrchestrator", "FinancialAnalystAgent", CUSTOM_RUNTIME_SA_ACCOUNT_ID}
        ),
        allowed_target_services=frozenset({"model-armor-service"}),
        allowed_tools=frozenset(),
        api_permissions=frozenset(),
        gcp_iam_roles=(
            "roles/aiplatform.user",
            "roles/logging.logWriter",
        ),
        oauth_scopes=(
            "https://www.googleapis.com/auth/cloud-platform.read-only",
        ),
    ),
}

# Alias ADK `ContextCacheAgent` to the shared `context-cache-agent-sa` identity spec
AGENT_NAME_ALIASES: dict[str, str] = {
    "ContextCacheAgent": "ContextCacheManager",
}

SERVICE_ACCOUNT_ID_TO_AGENT: dict[str, AgentIdentitySpec] = {
    spec.service_account_id: spec for spec in PER_AGENT_IDENTITIES.values()
}

TARGET_SERVICE_TO_AGENT_NAME: dict[str, str] = {
    "root-orchestrator-agent": "RootOrchestrator",
    "financial-analyst-agent": "FinancialAnalystAgent",
    "context-cache-agent": "ContextCacheManager",
    "model-armor-guardrail": "ModelArmorGuardrail",
}


def resolve_agent_identity(agent_name: str) -> AgentIdentitySpec:
    """Return the `AgentIdentitySpec` for `agent_name` (or raise `AgentAuthorizationError`)."""
    canonical = AGENT_NAME_ALIASES.get(agent_name, agent_name)
    spec = PER_AGENT_IDENTITIES.get(canonical)
    if spec is None:
        raise AgentAuthorizationError(
            f"Unregistered agent identity {agent_name!r}; every agent must declare a least-privilege identity.",
            agent_name=agent_name,
            target="identity-registry",
        )
    return spec


def resolve_agent_by_service_account_id(account_id: str) -> AgentIdentitySpec | None:
    """Look up a per-agent identity specification by its GCP service account ID."""
    return SERVICE_ACCOUNT_ID_TO_AGENT.get(account_id.strip().lower())


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    padded = segment + "=" * ((4 - len(segment) % 4) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


@dataclass(frozen=True)
class VerifiedAgentS2SClaims:
    """Verified cryptographic claims from an audience-bound per-agent S2S assertion."""

    agent_name: str
    service_account_email: str
    spiffe_id: str
    caller: str
    target_service: str
    audience: str
    allowed_tools: tuple[str, ...]
    permissions: tuple[str, ...]
    issued_at: int
    expires_at: int
    jti: str


class AgentS2STokenBroker:
    """Mints and verifies short-lived ES256 audience-bound S2S assertions per agent."""

    def __init__(self) -> None:
        self._ephemeral_ec_key = ec.generate_private_key(ec.SECP256R1())
        self._public_ec_key = self._ephemeral_ec_key.public_key()
        self.kid = f"agent-s2s-es256-{uuid.uuid4().hex[:12]}"

    def mint_s2s_assertion(
        self,
        *,
        agent_name: str,
        target_service: str,
        caller: str = CUSTOM_RUNTIME_SA_ACCOUNT_ID,
        project_name: str | None = None,
        ttl_seconds: int = 120,
    ) -> str:
        """Mint a short-lived ES256 S2S JWT bound strictly to `(agent_name -> target_service)`."""
        spec = resolve_agent_identity(agent_name)
        if ttl_seconds <= 0 or ttl_seconds > MAX_AGENT_S2S_TTL_SECONDS:
            raise ValueError(
                f"Agent S2S assertion TTL must be between 1 and {MAX_AGENT_S2S_TTL_SECONDS} seconds."
            )

        # 1. Verify caller is authorized to invoke/act through this agent
        if caller not in spec.allowed_callers and caller != spec.agent_name:
            raise AgentAuthorizationError(
                f"Caller {caller!r} is not permitted to invoke agent {spec.agent_name!r} "
                f"(allowed_callers={sorted(spec.allowed_callers)}).",
                agent_name=spec.agent_name,
                target=target_service,
            )

        # 2. Verify outbound target_service is within this agent's least-privilege allowlist
        if target_service not in spec.allowed_target_services:
            raise AgentAuthorizationError(
                f"Agent {spec.agent_name!r} ({spec.service_account_id}) is forbidden from "
                f"communicating with service {target_service!r} under Least Privilege policy.",
                agent_name=spec.agent_name,
                target=target_service,
            )

        # 3. If target_service is another agent, enforce mutual inbound caller policy on the callee
        callee_agent_name = TARGET_SERVICE_TO_AGENT_NAME.get(target_service)
        if callee_agent_name is not None:
            callee_spec = resolve_agent_identity(callee_agent_name)
            if spec.agent_name not in callee_spec.allowed_callers:
                raise AgentAuthorizationError(
                    f"Target agent {callee_spec.agent_name!r} rejects inbound S2S calls from "
                    f"{spec.agent_name!r} (allowed_callers={sorted(callee_spec.allowed_callers)}).",
                    agent_name=spec.agent_name,
                    target=target_service,
                )

        now = int(time.time())
        exp = now + ttl_seconds
        aud = (
            target_service
            if target_service.startswith(("https://", "http://", "service://", "/projects/"))
            else f"service://{target_service}"
        )
        header = {"alg": "ES256", "typ": "at+jwt", "kid": self.kid}
        claims: dict[str, Any] = {
            "iss": AGENT_IDENTITY_ISSUER,
            "sub": spec.spiffe_id(project_name),
            "aud": aud,
            "agent_name": spec.agent_name,
            "email": spec.service_account_email(project_name),
            "caller": caller,
            "target_service": target_service,
            "allowed_tools": sorted(spec.allowed_tools),
            "permissions": sorted(spec.api_permissions),
            "iat": now,
            "exp": exp,
            "jti": uuid.uuid4().hex,
        }
        h_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        p_b64 = _b64url_encode(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{h_b64}.{p_b64}".encode("ascii")
        der_sig = self._ephemeral_ec_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der_sig)
        sig_b64 = _b64url_encode(r.to_bytes(32, "big") + s.to_bytes(32, "big"))
        return f"{h_b64}.{p_b64}.{sig_b64}"

    def verify_s2s_assertion(
        self,
        raw_assertion: str,
        *,
        expected_target_service: str | None = None,
        required_tool: str | None = None,
        required_permission: str | None = None,
        clock_skew_seconds: int = 15,
    ) -> VerifiedAgentS2SClaims:
        """Cryptographically verify a per-agent S2S assertion and enforce audience/tool/permission bounds."""
        parts = raw_assertion.strip().split(".")
        if len(parts) != 3 or not all(parts):
            raise ValueError("Malformed S2S agent assertion JWT.")

        h_b64, p_b64, sig_b64 = parts
        header = json.loads(_b64url_decode(h_b64).decode("utf-8"))
        claims = json.loads(_b64url_decode(p_b64).decode("utf-8"))

        if header.get("alg") != "ES256" or header.get("kid") != self.kid:
            raise ValueError("Untrusted S2S agent assertion key or algorithm.")

        sig_bytes = _b64url_decode(sig_b64)
        if len(sig_bytes) != 64:
            raise ValueError("Invalid ES256 signature length in S2S agent assertion.")
        r = int.from_bytes(sig_bytes[:32], "big")
        s = int.from_bytes(sig_bytes[32:], "big")
        try:
            self._public_ec_key.verify(
                encode_dss_signature(r, s),
                f"{h_b64}.{p_b64}".encode("ascii"),
                ec.ECDSA(hashes.SHA256()),
            )
        except InvalidSignature as exc:
            raise ValueError("S2S agent assertion signature verification failed.") from exc

        if claims.get("iss") != AGENT_IDENTITY_ISSUER:
            raise ValueError(f"Unexpected S2S assertion issuer: {claims.get('iss')!r}")

        now = int(time.time())
        iat = int(claims["iat"])
        exp = int(claims["exp"])
        if now - clock_skew_seconds >= exp:
            raise ValueError("S2S agent assertion has expired.")
        if iat > now + clock_skew_seconds:
            raise ValueError("S2S agent assertion 'iat' is in the future.")

        agent_name = str(claims["agent_name"])
        spec = resolve_agent_identity(agent_name)
        target_service = str(claims["target_service"])
        aud = str(claims["aud"])

        if expected_target_service is not None:
            expected_aud = (
                expected_target_service
                if expected_target_service.startswith(
                    ("https://", "http://", "service://", "/projects/")
                )
                else f"service://{expected_target_service}"
            )
            if aud != expected_aud or target_service != expected_target_service:
                raise AgentAuthorizationError(
                    f"Confused-deputy S2S replay blocked: token bound to {aud!r}, "
                    f"cannot authenticate to {expected_aud!r}.",
                    agent_name=spec.agent_name,
                    target=expected_target_service,
                )

        if required_tool is not None and required_tool not in spec.allowed_tools:
            raise AgentAuthorizationError(
                f"Agent {spec.agent_name!r} is not permitted to invoke tool {required_tool!r}.",
                agent_name=spec.agent_name,
                target=required_tool,
            )

        if required_permission is not None and required_permission not in spec.api_permissions:
            raise AgentAuthorizationError(
                f"Agent {spec.agent_name!r} lacks required S2S permission {required_permission!r}.",
                agent_name=spec.agent_name,
                target=required_permission,
            )

        return VerifiedAgentS2SClaims(
            agent_name=spec.agent_name,
            service_account_email=str(claims["email"]),
            spiffe_id=str(claims["sub"]),
            caller=str(claims["caller"]),
            target_service=target_service,
            audience=aud,
            allowed_tools=tuple(claims.get("allowed_tools") or ()),
            permissions=tuple(claims.get("permissions") or ()),
            issued_at=iat,
            expires_at=exp,
            jti=str(claims["jti"]),
        )


_DEFAULT_S2S_BROKER = AgentS2STokenBroker()


def get_agent_s2s_broker() -> AgentS2STokenBroker:
    """Return the process-wide per-agent S2S cryptographic assertion broker."""
    return _DEFAULT_S2S_BROKER


def authorize_agent_tool_invocation(
    *,
    agent_name: str,
    tool_name: str,
    caller: str = CUSTOM_RUNTIME_SA_ACCOUNT_ID,
    project_name: str | None = None,
    broker: AgentS2STokenBroker | None = None,
) -> VerifiedAgentS2SClaims:
    """Enforce per-agent tool & backing-service least privilege and return verified S2S claims."""
    spec = resolve_agent_identity(agent_name)
    if tool_name not in spec.allowed_tools:
        raise AgentAuthorizationError(
            f"Agent {spec.agent_name!r} ({spec.service_account_id}) is not authorized "
            f"to invoke tool {tool_name!r} (allowed_tools={sorted(spec.allowed_tools)}).",
            agent_name=spec.agent_name,
            target=tool_name,
        )

    target_service = TOOL_TARGET_SERVICE_MAP.get(tool_name, f"tool:{tool_name}")
    active_broker = broker or _DEFAULT_S2S_BROKER
    raw_assertion = active_broker.mint_s2s_assertion(
        agent_name=spec.agent_name,
        target_service=target_service,
        caller=caller,
        project_name=project_name,
    )
    return active_broker.verify_s2s_assertion(
        raw_assertion,
        expected_target_service=target_service,
        required_tool=tool_name,
    )


def authorize_agent_s2s_hop(
    *,
    caller_agent: str,
    callee_agent: str,
    project_name: str | None = None,
    broker: AgentS2STokenBroker | None = None,
) -> VerifiedAgentS2SClaims:
    """Verify and attest a service-to-service delegation hop from `caller_agent` to `callee_agent`."""
    target_service = AGENT_TARGET_SERVICE_IDS.get(callee_agent)
    if not target_service:
        raise AgentAuthorizationError(
            f"Unknown target agent service boundary for {callee_agent!r}.",
            agent_name=caller_agent,
            target=callee_agent,
        )

    active_broker = broker or _DEFAULT_S2S_BROKER
    if caller_agent == CUSTOM_RUNTIME_SA_ACCOUNT_ID:
        # Host runtime SA (`sec-analyst-sa`) initiating entry into a registered agent
        callee_spec = resolve_agent_identity(callee_agent)
        if CUSTOM_RUNTIME_SA_ACCOUNT_ID not in callee_spec.allowed_callers:
            raise AgentAuthorizationError(
                f"Host SA {CUSTOM_RUNTIME_SA_ACCOUNT_ID!r} is not permitted to invoke {callee_spec.agent_name!r}.",
                agent_name=callee_spec.agent_name,
                target=target_service,
            )
        first_target = next(iter(sorted(callee_spec.allowed_target_services)))
        raw_assertion = active_broker.mint_s2s_assertion(
            agent_name=callee_spec.agent_name,
            target_service=first_target,
            caller=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
            project_name=project_name,
        )
        return active_broker.verify_s2s_assertion(
            raw_assertion,
            expected_target_service=first_target,
        )

    raw_assertion = active_broker.mint_s2s_assertion(
        agent_name=caller_agent,
        target_service=target_service,
        caller=caller_agent,
        project_name=project_name,
    )
    return active_broker.verify_s2s_assertion(
        raw_assertion,
        expected_target_service=target_service,
    )


class GcpAgentIamCredentialsClient:
    """Impersonates granular per-agent GCP service accounts via IAM Credentials API.

    Allows the Cloud Run host identity (`sec-analyst-sa@<project>.iam.gserviceaccount.com`)
    to mint short-lived, audience-bound OIDC ID tokens and downscoped OAuth2 tokens
    for a specific agent service account without any static keys.
    """

    def __init__(
        self,
        *,
        project_id: str | None = None,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        runtime_cfg = RuntimeServiceAccountConfig.from_env()
        self.project_id = (project_id or runtime_cfg.project_id).strip().lower()
        self.host_service_account = runtime_cfg.email
        self._http_client = http_client
        self._credentials = credentials

    def _get_host_bearer_token(self) -> str:
        if self._credentials is None:
            self._credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not self._credentials.valid or not getattr(self._credentials, "token", None):
            self._credentials.refresh(GoogleAuthRequest())
        return str(self._credentials.token)

    def generate_agent_id_token(
        self,
        *,
        agent_name: str,
        target_service: str,
        audience: str,
    ) -> str:
        """Mint a Google-signed OIDC ID token for `agent_name`'s dedicated service account."""
        spec = resolve_agent_identity(agent_name)
        if target_service not in spec.allowed_target_services:
            raise AgentAuthorizationError(
                f"Agent {spec.agent_name!r} is forbidden from requesting OIDC token for {target_service!r}.",
                agent_name=spec.agent_name,
                target=target_service,
            )

        agent_sa = spec.service_account_email(self.project_id)
        url = (
            f"https://iamcredentials.googleapis.com/v1/projects/-"
            f"/serviceAccounts/{agent_sa}:generateIdToken"
        )
        headers = {
            "Authorization": f"Bearer {self._get_host_bearer_token()}",
            "Content-Type": "application/json",
        }
        payload = {
            "audience": audience,
            "includeEmail": True,
        }
        if self._http_client is not None:
            resp = self._http_client.post(url, headers=headers, json=payload)
        else:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        signed_id_jwt = str(resp.json().get("token") or "").strip()
        if not signed_id_jwt:
            raise RuntimeError(f"IAM Credentials API returned empty ID token for {agent_sa}.")
        return signed_id_jwt

    def generate_downscoped_agent_token(
        self,
        *,
        agent_name: str,
        target_service: str,
        lifetime_seconds: int = 300,
    ) -> dict[str, Any]:
        """Mint a short-lived downscoped GCP OAuth2 token scoped strictly to `spec.oauth_scopes`."""
        spec = resolve_agent_identity(agent_name)
        if target_service not in spec.allowed_target_services:
            raise AgentAuthorizationError(
                f"Agent {spec.agent_name!r} is forbidden from requesting credentials for {target_service!r}.",
                agent_name=spec.agent_name,
                target=target_service,
            )
        bounded_lifetime = max(60, min(lifetime_seconds, MAX_AGENT_S2S_TTL_SECONDS))
        agent_sa = spec.service_account_email(self.project_id)
        url = (
            f"https://iamcredentials.googleapis.com/v1/projects/-"
            f"/serviceAccounts/{agent_sa}:generateAccessToken"
        )
        headers = {
            "Authorization": f"Bearer {self._get_host_bearer_token()}",
            "Content-Type": "application/json",
        }
        payload = {
            "scope": list(spec.oauth_scopes),
            "lifetime": f"{bounded_lifetime}s",
        }
        if self._http_client is not None:
            resp = self._http_client.post(url, headers=headers, json=payload)
        else:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return {
            "agent_name": spec.agent_name,
            "service_account_email": agent_sa,
            "scopes": list(spec.oauth_scopes),
            "expire_time": data.get("expireTime"),
            "credential_present": bool(data.get("accessToken")),
        }
