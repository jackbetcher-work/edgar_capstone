"""Security, Cloud Identity RBAC, Custom Service Account, Agent Identities, and Secret Hygiene.

Uses PEP 562 lazy module attribute loading so ``import edgar_capstone.security`` and
submodule imports (e.g. ``from edgar_capstone.security.secrets import ...``) do not
eagerly pull in ``auth``, ``model_armor``, ADK/FastAPI dependencies, or generate
ephemeral ECDSA keys.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edgar_capstone.security.agent_identity import (
        AGENT_IDENTITY_ISSUER,
        AGENT_NAME_ALIASES,
        AGENT_TARGET_SERVICE_IDS,
        MAX_AGENT_S2S_TTL_SECONDS,
        PER_AGENT_IDENTITIES,
        S2S_AGENT_ASSERTION_HEADER,
        SERVICE_ACCOUNT_ID_TO_AGENT,
        TOOL_TARGET_SERVICE_MAP,
        AgentAuthorizationError,
        AgentIdentitySpec,
        AgentS2STokenBroker,
        GcpAgentIamCredentialsClient,
        VerifiedAgentS2SClaims,
        authorize_agent_s2s_hop,
        authorize_agent_tool_invocation,
        get_agent_s2s_broker,
        resolve_agent_by_service_account_id,
        resolve_agent_identity,
    )
    from edgar_capstone.security.auth import (
        DEFAULT_LOCAL_IAP_AUDIENCE,
        GOOGLE_IAP_ISSUER,
        GOOGLE_IAP_JWKS_URI,
        GOOGLE_WORKSPACE_ISSUERS,
        GOOGLE_WORKSPACE_JWKS_URI,
        LOCAL_DEV_SSO_ISSUER,
        AuthenticatedUser,
        AuthenticationError,
        AuthorizationError,
        AuthSettings,
        IAPJwtVerifier,
        JwksKeyStore,
        clear_security_audit_events,
        compute_token_fingerprint,
        extract_bearer_token_from_request,
        get_auth_verifier,
        get_security_audit_events,
        is_bearer_token_revoked,
        is_production_iap_mode,
        issue_local_dev_workspace_sso_token,
        record_security_audit_event,
        require_authenticated_user,
        require_permissions,
        revoke_bearer_token,
    )
    from edgar_capstone.security.jwt_codec import (
        b64url_decode,
        b64url_encode,
        ec_public_key_to_jwk,
        jwk_to_public_key,
        rsa_public_key_to_jwk,
        sign_jwt_es256,
        sign_jwt_rs256,
    )
    from edgar_capstone.security.model_armor import (
        DEFAULT_MODEL_ARMOR_LOCATION,
        DEFAULT_MODEL_ARMOR_TEMPLATE_ID,
        FinancialProjectionFilter,
        GuardrailDecision,
        LocalModelArmorEmulatorTransport,
        ModelArmorClient,
        ModelArmorConfig,
        ModelArmorGuardrailService,
        ModelArmorSafetyPlugin,
        ModelArmorSanitizationSummary,
        ProjectionFilterResult,
        configure_model_armor_guardrail,
        get_model_armor_guardrail,
        get_model_armor_status_metadata,
        parse_model_armor_response,
        reset_model_armor_guardrail,
        schema_safe_block_text,
    )
    from edgar_capstone.security.rbac import (
        ADMIN_PERMISSIONS,
        ANALYST_PERMISSIONS,
        CUSTOM_RUNTIME_SA_ACCOUNT_ID,
        DEFAULT_CLOUD_IDENTITY_DOMAIN,
        DEFAULT_PROJECT_NAME,
        VIEWER_PERMISSIONS,
        CloudIdentityRBACPolicy,
        CloudIdentityRoleDefinition,
        Permission,
        RuntimeServiceAccountConfig,
        build_custom_service_account_email,
        format_cloud_identity_group_principal,
        is_default_gcp_service_account,
        parse_cloud_identity_group,
        validate_custom_service_account_email,
    )
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

_EXPORTS: dict[str, tuple[str, str]] = {
    # agent_identity
    "AGENT_IDENTITY_ISSUER": ("edgar_capstone.security.agent_identity", "AGENT_IDENTITY_ISSUER"),
    "AGENT_NAME_ALIASES": ("edgar_capstone.security.agent_identity", "AGENT_NAME_ALIASES"),
    "AGENT_TARGET_SERVICE_IDS": (
        "edgar_capstone.security.agent_identity",
        "AGENT_TARGET_SERVICE_IDS",
    ),
    "MAX_AGENT_S2S_TTL_SECONDS": (
        "edgar_capstone.security.agent_identity",
        "MAX_AGENT_S2S_TTL_SECONDS",
    ),
    "PER_AGENT_IDENTITIES": ("edgar_capstone.security.agent_identity", "PER_AGENT_IDENTITIES"),
    "S2S_AGENT_ASSERTION_HEADER": (
        "edgar_capstone.security.agent_identity",
        "S2S_AGENT_ASSERTION_HEADER",
    ),
    "SERVICE_ACCOUNT_ID_TO_AGENT": (
        "edgar_capstone.security.agent_identity",
        "SERVICE_ACCOUNT_ID_TO_AGENT",
    ),
    "TOOL_TARGET_SERVICE_MAP": (
        "edgar_capstone.security.agent_identity",
        "TOOL_TARGET_SERVICE_MAP",
    ),
    "AgentAuthorizationError": (
        "edgar_capstone.security.agent_identity",
        "AgentAuthorizationError",
    ),
    "AgentIdentitySpec": ("edgar_capstone.security.agent_identity", "AgentIdentitySpec"),
    "AgentS2STokenBroker": ("edgar_capstone.security.agent_identity", "AgentS2STokenBroker"),
    "GcpAgentIamCredentialsClient": (
        "edgar_capstone.security.agent_identity",
        "GcpAgentIamCredentialsClient",
    ),
    "VerifiedAgentS2SClaims": ("edgar_capstone.security.agent_identity", "VerifiedAgentS2SClaims"),
    "authorize_agent_s2s_hop": (
        "edgar_capstone.security.agent_identity",
        "authorize_agent_s2s_hop",
    ),
    "authorize_agent_tool_invocation": (
        "edgar_capstone.security.agent_identity",
        "authorize_agent_tool_invocation",
    ),
    "get_agent_s2s_broker": ("edgar_capstone.security.agent_identity", "get_agent_s2s_broker"),
    "resolve_agent_by_service_account_id": (
        "edgar_capstone.security.agent_identity",
        "resolve_agent_by_service_account_id",
    ),
    "resolve_agent_identity": ("edgar_capstone.security.agent_identity", "resolve_agent_identity"),
    # auth
    "DEFAULT_LOCAL_IAP_AUDIENCE": ("edgar_capstone.security.auth", "DEFAULT_LOCAL_IAP_AUDIENCE"),
    "GOOGLE_IAP_ISSUER": ("edgar_capstone.security.auth", "GOOGLE_IAP_ISSUER"),
    "GOOGLE_IAP_JWKS_URI": ("edgar_capstone.security.auth", "GOOGLE_IAP_JWKS_URI"),
    "GOOGLE_WORKSPACE_ISSUERS": ("edgar_capstone.security.auth", "GOOGLE_WORKSPACE_ISSUERS"),
    "GOOGLE_WORKSPACE_JWKS_URI": ("edgar_capstone.security.auth", "GOOGLE_WORKSPACE_JWKS_URI"),
    "LOCAL_DEV_SSO_ISSUER": ("edgar_capstone.security.auth", "LOCAL_DEV_SSO_ISSUER"),
    "AuthSettings": ("edgar_capstone.security.auth", "AuthSettings"),
    "AuthenticatedUser": ("edgar_capstone.security.auth", "AuthenticatedUser"),
    "AuthenticationError": ("edgar_capstone.security.auth", "AuthenticationError"),
    "AuthorizationError": ("edgar_capstone.security.auth", "AuthorizationError"),
    "IAPJwtVerifier": ("edgar_capstone.security.auth", "IAPJwtVerifier"),
    "JwksKeyStore": ("edgar_capstone.security.auth", "JwksKeyStore"),
    "clear_security_audit_events": ("edgar_capstone.security.auth", "clear_security_audit_events"),
    "compute_token_fingerprint": ("edgar_capstone.security.auth", "compute_token_fingerprint"),
    "extract_bearer_token_from_request": (
        "edgar_capstone.security.auth",
        "extract_bearer_token_from_request",
    ),
    "get_auth_verifier": ("edgar_capstone.security.auth", "get_auth_verifier"),
    "get_security_audit_events": ("edgar_capstone.security.auth", "get_security_audit_events"),
    "is_bearer_token_revoked": ("edgar_capstone.security.auth", "is_bearer_token_revoked"),
    "is_production_iap_mode": ("edgar_capstone.security.auth", "is_production_iap_mode"),
    "issue_local_dev_workspace_sso_token": (
        "edgar_capstone.security.auth",
        "issue_local_dev_workspace_sso_token",
    ),
    "record_security_audit_event": ("edgar_capstone.security.auth", "record_security_audit_event"),
    "require_authenticated_user": ("edgar_capstone.security.auth", "require_authenticated_user"),
    "require_permissions": ("edgar_capstone.security.auth", "require_permissions"),
    "revoke_bearer_token": ("edgar_capstone.security.auth", "revoke_bearer_token"),
    # jwt_codec
    "b64url_decode": ("edgar_capstone.security.jwt_codec", "b64url_decode"),
    "b64url_encode": ("edgar_capstone.security.jwt_codec", "b64url_encode"),
    "ec_public_key_to_jwk": ("edgar_capstone.security.jwt_codec", "ec_public_key_to_jwk"),
    "jwk_to_public_key": ("edgar_capstone.security.jwt_codec", "jwk_to_public_key"),
    "rsa_public_key_to_jwk": ("edgar_capstone.security.jwt_codec", "rsa_public_key_to_jwk"),
    "sign_jwt_es256": ("edgar_capstone.security.jwt_codec", "sign_jwt_es256"),
    "sign_jwt_rs256": ("edgar_capstone.security.jwt_codec", "sign_jwt_rs256"),
    # model_armor
    "DEFAULT_MODEL_ARMOR_LOCATION": (
        "edgar_capstone.security.model_armor",
        "DEFAULT_MODEL_ARMOR_LOCATION",
    ),
    "DEFAULT_MODEL_ARMOR_TEMPLATE_ID": (
        "edgar_capstone.security.model_armor",
        "DEFAULT_MODEL_ARMOR_TEMPLATE_ID",
    ),
    "FinancialProjectionFilter": (
        "edgar_capstone.security.model_armor",
        "FinancialProjectionFilter",
    ),
    "GuardrailDecision": ("edgar_capstone.security.model_armor", "GuardrailDecision"),
    "LocalModelArmorEmulatorTransport": (
        "edgar_capstone.security.model_armor",
        "LocalModelArmorEmulatorTransport",
    ),
    "ModelArmorClient": ("edgar_capstone.security.model_armor", "ModelArmorClient"),
    "ModelArmorConfig": ("edgar_capstone.security.model_armor", "ModelArmorConfig"),
    "ModelArmorGuardrailService": (
        "edgar_capstone.security.model_armor",
        "ModelArmorGuardrailService",
    ),
    "ModelArmorSafetyPlugin": ("edgar_capstone.security.model_armor", "ModelArmorSafetyPlugin"),
    "ModelArmorSanitizationSummary": (
        "edgar_capstone.security.model_armor",
        "ModelArmorSanitizationSummary",
    ),
    "ProjectionFilterResult": ("edgar_capstone.security.model_armor", "ProjectionFilterResult"),
    "configure_model_armor_guardrail": (
        "edgar_capstone.security.model_armor",
        "configure_model_armor_guardrail",
    ),
    "get_model_armor_guardrail": (
        "edgar_capstone.security.model_armor",
        "get_model_armor_guardrail",
    ),
    "get_model_armor_status_metadata": (
        "edgar_capstone.security.model_armor",
        "get_model_armor_status_metadata",
    ),
    "parse_model_armor_response": (
        "edgar_capstone.security.model_armor",
        "parse_model_armor_response",
    ),
    "reset_model_armor_guardrail": (
        "edgar_capstone.security.model_armor",
        "reset_model_armor_guardrail",
    ),
    "schema_safe_block_text": ("edgar_capstone.security.model_armor", "schema_safe_block_text"),
    # rbac
    "ADMIN_PERMISSIONS": ("edgar_capstone.security.rbac", "ADMIN_PERMISSIONS"),
    "ANALYST_PERMISSIONS": ("edgar_capstone.security.rbac", "ANALYST_PERMISSIONS"),
    "CUSTOM_RUNTIME_SA_ACCOUNT_ID": (
        "edgar_capstone.security.rbac",
        "CUSTOM_RUNTIME_SA_ACCOUNT_ID",
    ),
    "DEFAULT_CLOUD_IDENTITY_DOMAIN": (
        "edgar_capstone.security.rbac",
        "DEFAULT_CLOUD_IDENTITY_DOMAIN",
    ),
    "DEFAULT_PROJECT_NAME": ("edgar_capstone.security.rbac", "DEFAULT_PROJECT_NAME"),
    "VIEWER_PERMISSIONS": ("edgar_capstone.security.rbac", "VIEWER_PERMISSIONS"),
    "CloudIdentityRBACPolicy": ("edgar_capstone.security.rbac", "CloudIdentityRBACPolicy"),
    "CloudIdentityRoleDefinition": ("edgar_capstone.security.rbac", "CloudIdentityRoleDefinition"),
    "Permission": ("edgar_capstone.security.rbac", "Permission"),
    "RuntimeServiceAccountConfig": ("edgar_capstone.security.rbac", "RuntimeServiceAccountConfig"),
    "build_custom_service_account_email": (
        "edgar_capstone.security.rbac",
        "build_custom_service_account_email",
    ),
    "format_cloud_identity_group_principal": (
        "edgar_capstone.security.rbac",
        "format_cloud_identity_group_principal",
    ),
    "is_default_gcp_service_account": (
        "edgar_capstone.security.rbac",
        "is_default_gcp_service_account",
    ),
    "parse_cloud_identity_group": ("edgar_capstone.security.rbac", "parse_cloud_identity_group"),
    "validate_custom_service_account_email": (
        "edgar_capstone.security.rbac",
        "validate_custom_service_account_email",
    ),
    # secrets
    "DEFAULT_PROJECT_ID": ("edgar_capstone.security.secrets", "DEFAULT_PROJECT_ID"),
    "MANAGED_SECRETS": ("edgar_capstone.security.secrets", "MANAGED_SECRETS"),
    "RedactedSecret": ("edgar_capstone.security.secrets", "RedactedSecret"),
    "SecretFinding": ("edgar_capstone.security.secrets", "SecretFinding"),
    "SecretManagerClient": ("edgar_capstone.security.secrets", "SecretManagerClient"),
    "SecretSpec": ("edgar_capstone.security.secrets", "SecretSpec"),
    "scan_repository_for_exposed_secrets": (
        "edgar_capstone.security.secrets",
        "scan_repository_for_exposed_secrets",
    ),
    "scan_text_for_exposed_secrets": (
        "edgar_capstone.security.secrets",
        "scan_text_for_exposed_secrets",
    ),
}

__all__ = [
    "ADMIN_PERMISSIONS",
    "AGENT_IDENTITY_ISSUER",
    "AGENT_NAME_ALIASES",
    "AGENT_TARGET_SERVICE_IDS",
    "ANALYST_PERMISSIONS",
    "CUSTOM_RUNTIME_SA_ACCOUNT_ID",
    "DEFAULT_CLOUD_IDENTITY_DOMAIN",
    "DEFAULT_LOCAL_IAP_AUDIENCE",
    "DEFAULT_MODEL_ARMOR_LOCATION",
    "DEFAULT_MODEL_ARMOR_TEMPLATE_ID",
    "DEFAULT_PROJECT_ID",
    "DEFAULT_PROJECT_NAME",
    "GOOGLE_IAP_ISSUER",
    "GOOGLE_IAP_JWKS_URI",
    "GOOGLE_WORKSPACE_ISSUERS",
    "GOOGLE_WORKSPACE_JWKS_URI",
    "LOCAL_DEV_SSO_ISSUER",
    "MANAGED_SECRETS",
    "MAX_AGENT_S2S_TTL_SECONDS",
    "PER_AGENT_IDENTITIES",
    "S2S_AGENT_ASSERTION_HEADER",
    "SERVICE_ACCOUNT_ID_TO_AGENT",
    "TOOL_TARGET_SERVICE_MAP",
    "VIEWER_PERMISSIONS",
    "AgentAuthorizationError",
    "AgentIdentitySpec",
    "AgentS2STokenBroker",
    "AuthSettings",
    "AuthenticatedUser",
    "AuthenticationError",
    "AuthorizationError",
    "CloudIdentityRBACPolicy",
    "CloudIdentityRoleDefinition",
    "FinancialProjectionFilter",
    "GcpAgentIamCredentialsClient",
    "GuardrailDecision",
    "IAPJwtVerifier",
    "JwksKeyStore",
    "LocalModelArmorEmulatorTransport",
    "ModelArmorClient",
    "ModelArmorConfig",
    "ModelArmorGuardrailService",
    "ModelArmorSafetyPlugin",
    "ModelArmorSanitizationSummary",
    "Permission",
    "ProjectionFilterResult",
    "RedactedSecret",
    "RuntimeServiceAccountConfig",
    "SecretFinding",
    "SecretManagerClient",
    "SecretSpec",
    "VerifiedAgentS2SClaims",
    "authorize_agent_s2s_hop",
    "authorize_agent_tool_invocation",
    "b64url_decode",
    "b64url_encode",
    "build_custom_service_account_email",
    "clear_security_audit_events",
    "compute_token_fingerprint",
    "configure_model_armor_guardrail",
    "ec_public_key_to_jwk",
    "extract_bearer_token_from_request",
    "format_cloud_identity_group_principal",
    "get_agent_s2s_broker",
    "get_auth_verifier",
    "get_model_armor_guardrail",
    "get_model_armor_status_metadata",
    "get_security_audit_events",
    "is_bearer_token_revoked",
    "is_default_gcp_service_account",
    "is_production_iap_mode",
    "issue_local_dev_workspace_sso_token",
    "jwk_to_public_key",
    "parse_cloud_identity_group",
    "parse_model_armor_response",
    "record_security_audit_event",
    "require_authenticated_user",
    "require_permissions",
    "reset_model_armor_guardrail",
    "resolve_agent_by_service_account_id",
    "resolve_agent_identity",
    "revoke_bearer_token",
    "rsa_public_key_to_jwk",
    "scan_repository_for_exposed_secrets",
    "scan_text_for_exposed_secrets",
    "schema_safe_block_text",
    "sign_jwt_es256",
    "sign_jwt_rs256",
    "validate_custom_service_account_email",
]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
