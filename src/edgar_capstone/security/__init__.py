"""Security package: IAP/Workspace auth, RBAC, Model Armor guardrails, and secret scanning."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
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
    "decode_verified_jwt": ("edgar_capstone.security.jwt_codec", "decode_verified_jwt"),
    "ec_public_key_to_jwk": ("edgar_capstone.security.jwt_codec", "ec_public_key_to_jwk"),
    "jwk_to_public_key": ("edgar_capstone.security.jwt_codec", "jwk_to_public_key"),
    "parse_unverified_jwt": ("edgar_capstone.security.jwt_codec", "parse_unverified_jwt"),
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
    "LocalModelArmorEmulator": (
        "edgar_capstone.security.model_armor",
        "LocalModelArmorEmulator",
    ),
    "ModelArmorClient": ("edgar_capstone.security.model_armor", "ModelArmorClient"),
    "ModelArmorConfig": ("edgar_capstone.security.model_armor", "ModelArmorConfig"),
    "ModelArmorGuardrailService": (
        "edgar_capstone.security.model_armor",
        "ModelArmorGuardrailService",
    ),
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
    "parse_model_armor_proto": (
        "edgar_capstone.security.model_armor",
        "parse_model_armor_proto",
    ),
    "reset_model_armor_guardrail": (
        "edgar_capstone.security.model_armor",
        "reset_model_armor_guardrail",
    ),
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
    "SecretFinding": ("edgar_capstone.security.secrets", "SecretFinding"),
    "scan_repository_for_exposed_secrets": (
        "edgar_capstone.security.secrets",
        "scan_repository_for_exposed_secrets",
    ),
    "scan_text_for_exposed_secrets": (
        "edgar_capstone.security.secrets",
        "scan_text_for_exposed_secrets",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value
