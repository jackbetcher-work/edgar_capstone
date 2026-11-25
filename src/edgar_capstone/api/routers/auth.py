"""Authentication, IAP/Okta SSO token brokerage, and RBAC governance routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from edgar_capstone.api.dependencies import (
    LocalSsoBrokerForbiddenError,
    audited_permissions,
    format_token_exchange_response,
    get_runtime_sa_config,
)
from edgar_capstone.api.schemas import (
    DevWorkspaceSsoLoginRequest,
    SsoTokenExchangeRequest,
)
from edgar_capstone.security.auth import (
    GOOGLE_IAP_ISSUER,
    AuthenticatedUser,
    AuthenticationError,
    AuthorizationError,
    IAPJwtVerifier,
    get_auth_verifier,
    get_security_audit_events,
    issue_local_dev_workspace_sso_token,
    record_security_audit_event,
    require_authenticated_user,
    require_permissions,
    revoke_bearer_token,
)
from edgar_capstone.security.rbac import Permission, RuntimeServiceAccountConfig

router = APIRouter(tags=["Authentication & RBAC"])


@router.get("/api/v1/auth/config")
def auth_discovery_config(
    verifier: IAPJwtVerifier = Depends(get_auth_verifier),
    runtime_sa: RuntimeServiceAccountConfig = Depends(get_runtime_sa_config),
) -> dict[str, Any]:
    """Return public SSO, IAP, Cloud Identity RBAC, and Agent Identity metadata (no secrets)."""
    settings = verifier.settings
    return {
        "status": "ok",
        "auth_required": True,
        "access_pattern": "iap_jwt_bearer",
        "identity_provider": {
            "name": "Google Workspace / Cloud IAP",
            "protocol": "Google Workspace OIDC + Direct Cloud Run IAP",
            "google_issuer": "https://accounts.google.com",
            "iap_issuer": GOOGLE_IAP_ISSUER,
            "google_client_id": settings.google_oauth_client_id,
            "authorization_endpoint": settings.authorization_endpoint(),
            "allowed_workspace_domains": list(settings.allowed_workspace_domains),
            "required_analyst_groups": list(settings.required_analyst_groups),
            "local_sso_dev_broker_enabled": not settings.strict_production_mode,
        },
        "iap": {
            "enabled": True,
            "strict_production_mode": settings.strict_production_mode,
            "bearer_header": "Authorization",
            "assertion_header": "X-Goog-IAP-JWT-Assertion",
            "audience_configured": bool(settings.audiences),
            # In Cloud Run the browser can sign in through the IAP session alone; the
            # frontend probes /api/v1/auth/me without a Bearer token when this is true.
            "session_auth_enabled": settings.strict_production_mode,
            "default_role": settings.iap_default_role,
        },
        "rbac": settings.rbac_policy().to_public_dict(),
        "service_account": runtime_sa.to_public_dict(),
    }


@router.post("/api/v1/auth/sso/exchange")
def exchange_sso_id_token(
    req: SsoTokenExchangeRequest,
    verifier: IAPJwtVerifier = Depends(get_auth_verifier),
) -> dict[str, Any]:
    """Validate a Google Workspace / Okta SSO / IAP OIDC ID token for API Bearer access."""
    cleaned_jwt = req.id_token.strip()
    if cleaned_jwt.lower().startswith("bearer "):
        cleaned_jwt = cleaned_jwt[7:].strip()

    try:
        user = verifier.verify_jwt(cleaned_jwt)
    except AuthenticationError as exc:
        record_security_audit_event(
            event_type="sso_token_exchange_failed",
            user=None,
            resource="/api/v1/auth/sso/exchange",
            status_outcome="denied_401",
            detail=exc.message,
        )
        raise
    except AuthorizationError as exc:
        record_security_audit_event(
            event_type="sso_token_exchange_forbidden",
            user=None,
            resource="/api/v1/auth/sso/exchange",
            status_outcome="denied_403",
            detail=exc.message,
        )
        raise

    record_security_audit_event(
        event_type="sso_token_exchange",
        user=user,
        resource="/api/v1/auth/sso/exchange",
        status_outcome="allowed",
    )
    return format_token_exchange_response(cleaned_jwt, user)


@router.post("/api/v1/auth/sso/dev-token")
def mint_local_dev_sso_token(
    req: DevWorkspaceSsoLoginRequest,
    verifier: IAPJwtVerifier = Depends(get_auth_verifier),
) -> dict[str, Any]:
    """Issue a signed ES256 Workspace/Okta JWT for local development (disabled in Cloud Run)."""
    try:
        signed_jwt, user = issue_local_dev_workspace_sso_token(
            email=req.email,
            name=req.name,
            groups=req.groups,
            verifier=verifier,
        )
    except AuthenticationError as exc:
        record_security_audit_event(
            event_type="sso_dev_token_failed",
            user=None,
            resource="/api/v1/auth/sso/dev-token",
            status_outcome="denied_401",
            detail=exc.message,
        )
        raise
    except AuthorizationError as exc:
        record_security_audit_event(
            event_type="sso_dev_token_forbidden",
            user=None,
            resource="/api/v1/auth/sso/dev-token",
            status_outcome="denied_403",
            detail=exc.message,
        )
        raise LocalSsoBrokerForbiddenError(exc.message) from exc

    record_security_audit_event(
        event_type="sso_workspace_okta_login",
        user=user,
        resource="/api/v1/auth/sso/dev-token",
        status_outcome="allowed",
    )
    return format_token_exchange_response(signed_jwt, user)


@router.get("/api/v1/auth/me")
def get_authenticated_session(
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Return the verified Google Workspace / Okta SSO & IAP principal for the active Bearer JWT."""
    return {
        "status": "authenticated",
        "user": user.to_public_dict(),
    }


@router.post("/api/v1/auth/logout")
def logout_authenticated_session(
    request: Request,
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Revoke the active Bearer JWT and record a session logout audit event."""
    raw_token = getattr(request.state, "raw_bearer_token", "")
    if raw_token:
        revoke_bearer_token(raw_token, expires_at=user.expires_at)
    record_security_audit_event(
        event_type="sso_logout",
        user=user,
        resource="/api/v1/auth/logout",
        status_outcome="revoked",
    )
    return {
        "status": "logged_out",
        "revoked_sub": user.sub,
        "user_email": user.email,
    }


@router.get("/api/v1/auth/audit")
def list_security_audit_trail(
    limit: int = Query(50, ge=1, le=500),
    user: AuthenticatedUser = Depends(require_permissions(Permission.AUDIT_READ.value)),
) -> dict[str, Any]:
    """Retrieve structured authentication and 10-K report access audit logs."""
    events = get_security_audit_events(limit=limit)
    return {
        "status": "ok",
        "requested_by": user.email,
        "event_count": len(events),
        "events": events,
    }


@router.get("/api/v1/admin/rbac/policies")
def admin_rbac_governance_policies(
    user: AuthenticatedUser = Depends(
        audited_permissions(
            Permission.RBAC_ADMIN.value,
            event_type="admin_rbac_governance_inspected",
        )
    ),
    verifier: IAPJwtVerifier = Depends(get_auth_verifier),
    runtime_sa: RuntimeServiceAccountConfig = Depends(get_runtime_sa_config),
) -> dict[str, Any]:
    """Administrative endpoint restricted strictly to Cloud Identity `admin` group principals."""
    return {
        "status": "ok",
        "administered_by": user.email,
        "cloud_identity_groups": list(user.cloud_identity_groups),
        "rbac_policy": verifier.settings.rbac_policy().to_public_dict(),
        "runtime_service_account": runtime_sa.to_public_dict(),
    }
