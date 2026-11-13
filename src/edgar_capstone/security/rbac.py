"""Cloud Identity Group-Based RBAC and Custom Runtime Service Account Governance.

Implements:
1. Role-Based Access Control (RBAC): Group-based access definitions managed via
   Google Cloud Identity Groups (`group:<name>@<domain>`).
2. Custom Service Account Enforcement: Ensures the application runs under and
   validates the dedicated custom service account
   `sec-analyst-sa@<project_name>.iam.gserviceaccount.com`, rejecting default
   Compute Engine / App Engine service accounts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import os
import re
from typing import Any

from edgar_capstone.security.secrets import DEFAULT_PROJECT_ID

DEFAULT_CLOUD_IDENTITY_DOMAIN = "fde-workspace.corp"
DEFAULT_PROJECT_NAME = "fde-sec-edgar-sandbox-dev"
CUSTOM_RUNTIME_SA_ACCOUNT_ID = "sec-analyst-sa"

_CUSTOM_SA_EMAIL_RE = re.compile(
    r"^(?P<account_id>[a-z][a-z0-9\-]{4,28}[a-z0-9])@(?P<project>[a-z][a-z0-9\-]{3,61}[a-z0-9])\.iam\.gserviceaccount\.com$"
)
_DISALLOWED_DEFAULT_SA_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\d+-compute@developer\.gserviceaccount\.com$", re.IGNORECASE),
    re.compile(r"^.+@appspot\.gserviceaccount\.com$", re.IGNORECASE),
    re.compile(r"^.+@cloudservices\.gserviceaccount\.com$", re.IGNORECASE),
    re.compile(r"^.+@cloudbuild\.gserviceaccount\.com$", re.IGNORECASE),
)


class Permission(StrEnum):
    """Granular permission actions governed by Cloud Identity Group RBAC and Agent Identities."""

    FILINGS_READ = "filings:read"
    VARIANCES_CALCULATE = "variances:calculate"
    CHAT_QUERY = "chat:query"
    FEEDBACK_SUBMIT = "feedback:submit"
    CHUNK_HEALTH_READ = "chunk_health:read"
    AUDIT_READ = "audit:read"
    SECRETS_INSPECT = "secrets:inspect"
    RBAC_INSPECT = "rbac:inspect"
    RBAC_ADMIN = "rbac:admin"


ADMIN_PERMISSIONS: frozenset[str] = frozenset(p.value for p in Permission)

ANALYST_PERMISSIONS: frozenset[str] = frozenset(
    {
        Permission.FILINGS_READ.value,
        Permission.VARIANCES_CALCULATE.value,
        Permission.CHAT_QUERY.value,
        Permission.FEEDBACK_SUBMIT.value,
        Permission.CHUNK_HEALTH_READ.value,
        Permission.AUDIT_READ.value,
        Permission.SECRETS_INSPECT.value,
        Permission.RBAC_INSPECT.value,
    }
)

VIEWER_PERMISSIONS: frozenset[str] = frozenset(
    {
        Permission.FILINGS_READ.value,
        Permission.CHUNK_HEALTH_READ.value,
        Permission.AUDIT_READ.value,
        Permission.RBAC_INSPECT.value,
    }
)


def _split_csv(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def parse_cloud_identity_group(raw_group: str) -> tuple[str, str | None]:
    """Parse a Cloud Identity group string into `(group_local_id, domain_or_none)`.

    Supports IAM `group:sec-edgar-analysts@corp.com`, OIDC `sec-edgar-analysts@corp.com`,
    and federated short CN `sec-edgar-analysts`.
    """
    cleaned = raw_group.strip().lower()
    if cleaned.startswith("group:"):
        cleaned = cleaned[6:].strip()
    if "@" in cleaned:
        local_part, domain_part = cleaned.split("@", 1)
        return local_part.strip(), domain_part.strip() or None
    return cleaned, None


def format_cloud_identity_group_principal(
    raw_group: str,
    *,
    default_domain: str = DEFAULT_CLOUD_IDENTITY_DOMAIN,
) -> str:
    """Format a group identifier into canonical GCP IAM `group:<id>@<domain>` notation."""
    local_part, domain_part = parse_cloud_identity_group(raw_group)
    resolved_domain = domain_part or default_domain.strip().lower()
    return f"group:{local_part}@{resolved_domain}"


def _matches_configured_groups(
    user_groups: tuple[str, ...],
    configured_groups: tuple[str, ...],
) -> list[str]:
    """Return canonical matched Cloud Identity groups when user group claims match policy."""
    if not user_groups or not configured_groups:
        return []

    parsed_configured = [parse_cloud_identity_group(cfg) for cfg in configured_groups]
    matched: list[str] = []

    for raw_user_group in user_groups:
        u_local, u_domain = parse_cloud_identity_group(raw_user_group)
        if not u_local:
            continue
        for c_local, c_domain in parsed_configured:
            if u_local != c_local:
                continue
            # If policy pins a specific Cloud Identity domain and user claim carries a domain,
            # require exact domain match.
            if c_domain and u_domain and c_domain != u_domain:
                continue
            canonical = (
                f"group:{u_local}@{u_domain}"
                if u_domain
                else (f"group:{c_local}@{c_domain}" if c_domain else f"group:{u_local}")
            )
            if canonical not in matched:
                matched.append(canonical)
    return matched


@dataclass(frozen=True)
class CloudIdentityRoleDefinition:
    """Declarative RBAC role definition backed by Google Cloud Identity Groups."""

    role: str
    display_name: str
    description: str
    cloud_identity_groups: tuple[str, ...]
    gcp_iam_roles: tuple[str, ...]
    permissions: frozenset[str]

    def to_public_dict(self, *, default_domain: str = DEFAULT_CLOUD_IDENTITY_DOMAIN) -> dict[str, Any]:
        return {
            "role": self.role,
            "display_name": self.display_name,
            "description": self.description,
            "cloud_identity_groups": [
                format_cloud_identity_group_principal(g, default_domain=default_domain)
                for g in self.cloud_identity_groups
            ],
            "gcp_iam_roles": list(self.gcp_iam_roles),
            "permissions": sorted(self.permissions),
        }


@dataclass(frozen=True)
class CloudIdentityRBACPolicy:
    """Group-based RBAC policy engine managed via Google Cloud Identity Groups."""

    cloud_identity_domain: str
    role_definitions: tuple[CloudIdentityRoleDefinition, ...]
    required_analyst_groups: tuple[str, ...] = ()

    @classmethod
    def from_groups(
        cls,
        *,
        admin_groups: tuple[str, ...],
        analyst_groups: tuple[str, ...],
        viewer_groups: tuple[str, ...] = (
            "sec-edgar-viewers",
            "sec-edgar-auditors",
            "compliance-auditors",
        ),
        required_analyst_groups: tuple[str, ...] = (),
        cloud_identity_domain: str = DEFAULT_CLOUD_IDENTITY_DOMAIN,
    ) -> CloudIdentityRBACPolicy:
        effective_analyst_groups = (
            required_analyst_groups if required_analyst_groups else analyst_groups
        )
        definitions = (
            CloudIdentityRoleDefinition(
                role="admin",
                display_name="SEC Platform Administrator",
                description=(
                    "Full administrative governance, IAP access, audit log inspection, "
                    "and RBAC policy management via Cloud Identity Admin Groups."
                ),
                cloud_identity_groups=admin_groups,
                gcp_iam_roles=(
                    "roles/iap.httpsResourceAccessor",
                    "roles/run.viewer",
                    "roles/logging.privateLogViewer",
                    "roles/monitoring.viewer",
                    "roles/bigquery.metadataViewer",
                ),
                permissions=ADMIN_PERMISSIONS,
            ),
            CloudIdentityRoleDefinition(
                role="analyst",
                display_name="SEC Financial Analyst",
                description=(
                    "Conversational Hybrid RAG 10-K synthesis, deterministic variance "
                    "calculations, source section inspection, and analyst feedback via "
                    "Cloud Identity Analyst Groups."
                ),
                cloud_identity_groups=effective_analyst_groups,
                gcp_iam_roles=(
                    "roles/iap.httpsResourceAccessor",
                    "roles/bigquery.dataViewer",
                    "roles/discoveryengine.viewer",
                ),
                permissions=ANALYST_PERMISSIONS,
            ),
            CloudIdentityRoleDefinition(
                role="viewer",
                display_name="SEC Compliance Viewer / Auditor",
                description=(
                    "Read-only access to curated 10-K filings, structured metrics, chunk "
                    "health diagnostics, and security audit trails via Cloud Identity Viewer Groups."
                ),
                cloud_identity_groups=viewer_groups,
                gcp_iam_roles=(
                    "roles/iap.httpsResourceAccessor",
                    "roles/bigquery.metadataViewer",
                ),
                permissions=VIEWER_PERMISSIONS,
            ),
        )
        return cls(
            cloud_identity_domain=cloud_identity_domain,
            role_definitions=definitions,
            required_analyst_groups=required_analyst_groups,
        )

    def resolve_human_access(
        self,
        *,
        user_groups: tuple[str, ...],
        hosted_domain: str | None = None,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Resolve `(roles, permissions, matched_cloud_identity_groups)` from Cloud Identity groups.

        Returns empty `roles` if the user does not belong to any authorized Cloud Identity group.
        """
        effective_domain = hosted_domain or self.cloud_identity_domain
        roles: list[str] = []
        permissions: set[str] = set()
        matched_groups: list[str] = []

        for role_def in self.role_definitions:
            hits = _matches_configured_groups(user_groups, role_def.cloud_identity_groups)
            if hits:
                if role_def.role not in roles:
                    roles.append(role_def.role)
                permissions.update(role_def.permissions)
                for h in hits:
                    formatted = format_cloud_identity_group_principal(
                        h, default_domain=effective_domain
                    )
                    if formatted not in matched_groups:
                        matched_groups.append(formatted)

        if not roles:
            return (), (), ()

        ordered_roles: list[str] = ["authenticated"]
        if "admin" in roles:
            # Admins inherit analyst operational capabilities alongside administrative controls
            if "analyst" not in roles:
                ordered_roles.append("analyst")
            ordered_roles.append("admin")
        else:
            for r in roles:
                if r not in ordered_roles:
                    ordered_roles.append(r)

        return tuple(ordered_roles), tuple(sorted(permissions)), tuple(matched_groups)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "management_plane": "Google Cloud Identity Groups",
            "principal_syntax": "group:<group-id>@<workspace-domain>",
            "cloud_identity_domain": self.cloud_identity_domain,
            "individual_user_bindings_allowed": False,
            "roles": [
                rd.to_public_dict(default_domain=self.cloud_identity_domain)
                for rd in self.role_definitions
            ],
        }


# ---------------------------------------------------------------------------
# Custom Application Runtime Service Account (`sec-analyst-sa@<project>.iam.gserviceaccount.com`)
# ---------------------------------------------------------------------------


def is_default_gcp_service_account(email: str) -> bool:
    """Return True if `email` is a GCP-managed default Compute, App Engine, or Cloud Build SA."""
    cleaned = email.strip().lower()
    return any(pat.match(cleaned) for pat in _DISALLOWED_DEFAULT_SA_PATTERNS)


def build_custom_service_account_email(
    project_name: str,
    *,
    account_id: str = CUSTOM_RUNTIME_SA_ACCOUNT_ID,
) -> str:
    """Construct `sec-analyst-sa@<project_name>.iam.gserviceaccount.com` and validate format."""
    cleaned_project = project_name.strip().lower()
    cleaned_account = account_id.strip().lower()
    candidate = f"{cleaned_account}@{cleaned_project}.iam.gserviceaccount.com"
    validate_custom_service_account_email(candidate, expected_account_id=cleaned_account)
    return candidate


def validate_custom_service_account_email(
    email: str,
    *,
    expected_account_id: str | None = CUSTOM_RUNTIME_SA_ACCOUNT_ID,
    allowed_projects: tuple[str, ...] | None = None,
) -> tuple[str, str]:
    """Validate that `email` is a custom IAM service account (never a default compute SA).

    Returns `(account_id, project_name)` or raises `ValueError`.
    """
    cleaned = email.strip().lower()
    if is_default_gcp_service_account(cleaned):
        raise ValueError(
            f"Default GCP service account {cleaned!r} is prohibited; application must run "
            f"under custom service account '{CUSTOM_RUNTIME_SA_ACCOUNT_ID}@<project>.iam.gserviceaccount.com'."
        )

    match = _CUSTOM_SA_EMAIL_RE.match(cleaned)
    if not match:
        raise ValueError(
            f"Service account {cleaned!r} does not match custom IAM format "
            "'<account-id>@<project-name>.iam.gserviceaccount.com'."
        )

    account_id = match.group("account_id")
    project_name = match.group("project")

    if expected_account_id and account_id != expected_account_id.strip().lower():
        raise ValueError(
            f"Service account ID {account_id!r} does not match required custom runtime "
            f"service account {expected_account_id!r}."
        )

    if allowed_projects:
        normalized_projects = {p.strip().lower() for p in allowed_projects if p.strip()}
        if normalized_projects and project_name not in normalized_projects:
            raise ValueError(
                f"Service account project {project_name!r} is not in allowed projects "
                f"{sorted(normalized_projects)!r}."
            )

    return account_id, project_name


@dataclass(frozen=True)
class RuntimeServiceAccountConfig:
    """Verified configuration for the custom application runtime service account."""

    account_id: str
    project_id: str
    project_name: str
    email: str
    project_name_alias_email: str

    @classmethod
    def from_env(cls) -> RuntimeServiceAccountConfig:
        project_id = (
            os.environ.get("GCP_PROJECT_ID", "").strip()
            or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
            or DEFAULT_PROJECT_ID
        )
        project_name = (
            os.environ.get("GCP_PROJECT_NAME", "").strip()
            or DEFAULT_PROJECT_NAME
        )
        account_id = (
            os.environ.get("RUNTIME_SERVICE_ACCOUNT_ID", "").strip()
            or CUSTOM_RUNTIME_SA_ACCOUNT_ID
        )
        explicit_sa = os.environ.get("RUNTIME_SERVICE_ACCOUNT", "").strip()
        if explicit_sa:
            parsed_acc, parsed_proj = validate_custom_service_account_email(
                explicit_sa,
                expected_account_id=account_id,
            )
            email = f"{parsed_acc}@{parsed_proj}.iam.gserviceaccount.com"
        else:
            email = build_custom_service_account_email(project_id, account_id=account_id)

        alias_email = build_custom_service_account_email(project_name, account_id=account_id)
        return cls(
            account_id=account_id,
            project_id=project_id,
            project_name=project_name,
            email=email,
            project_name_alias_email=alias_email,
        )

    def allowed_projects(self) -> tuple[str, ...]:
        extra = _split_csv(os.environ.get("ALLOWED_SERVICE_ACCOUNT_PROJECTS"))
        ordered: list[str] = []
        for candidate in (self.project_id, self.project_name, DEFAULT_PROJECT_ID, DEFAULT_PROJECT_NAME, *extra):
            norm = candidate.strip().lower()
            if norm and norm not in ordered:
                ordered.append(norm)
        return tuple(ordered)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "email": self.email,
            "project_name_email": self.project_name_alias_email,
            "pattern": f"{self.account_id}@<project-name>.iam.gserviceaccount.com",
            "project_id": self.project_id,
            "project_name": self.project_name,
            "default_compute_sa_prohibited": True,
        }
