"""Comprehensive tests for Google Workspace Okta/SSO & Identity-Aware Proxy (IAP) JWT authentication."""

from __future__ import annotations

import base64
import dataclasses
import json
import time
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.testclient import TestClient

from edgar_capstone import security as security_pkg
from edgar_capstone.security import (
    GOOGLE_IAP_ISSUER,
    GOOGLE_IAP_JWKS_URI,
    GOOGLE_WORKSPACE_JWKS_URI,
    AuthenticationError,
    AuthSettings,
    IAPJwtVerifier,
    JwksKeyStore,
    clear_security_audit_events,
    ec_public_key_to_jwk,
    get_auth_verifier,
    get_security_audit_events,
    rsa_public_key_to_jwk,
    sign_jwt_es256,
    sign_jwt_rs256,
)
from edgar_capstone.security import auth as security_auth_mod
from edgar_capstone.server import app

TEST_IAP_AUDIENCE = "/projects/991122334455/global/backendServices/887766554433"
TEST_OKTA_ISSUER = "https://acme-workspace.okta.com/oauth2/default"
TEST_OKTA_JWKS_URI = "https://acme-workspace.okta.com/oauth2/default/v1/keys"
TEST_OKTA_CLIENT_ID = "0oa9sec_edgar_okta_client"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


@pytest.fixture()
def idp_keyring():
    """Generate ephemeral in-memory ES256 (IAP) and RS256 (Google Workspace & Okta) keypairs."""
    iap_ec_key = ec.generate_private_key(ec.SECP256R1())
    google_rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    okta_rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    jwks_by_uri: dict[str, dict[str, Any]] = {
        GOOGLE_IAP_JWKS_URI: {
            "keys": [ec_public_key_to_jwk(iap_ec_key.public_key(), kid="iap-es256-k1")]
        },
        GOOGLE_WORKSPACE_JWKS_URI: {
            "keys": [rsa_public_key_to_jwk(google_rsa_key.public_key(), kid="gws-rs256-k1")]
        },
        TEST_OKTA_JWKS_URI: {
            "keys": [rsa_public_key_to_jwk(okta_rsa_key.public_key(), kid="okta-rs256-k1")]
        },
    }

    def _mock_transport(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in jwks_by_uri:
            return httpx.Response(200, json=jwks_by_uri[url])
        return httpx.Response(404, json={"error": "not_found"})

    jwks_store = JwksKeyStore(
        http_client=httpx.Client(transport=httpx.MockTransport(_mock_transport))
    )
    settings = AuthSettings(
        audiences=(TEST_IAP_AUDIENCE, TEST_OKTA_CLIENT_ID),
        okta_issuer_uri=TEST_OKTA_ISSUER,
        okta_jwks_uri=TEST_OKTA_JWKS_URI,
        okta_client_id=TEST_OKTA_CLIENT_ID,
        google_oauth_client_id=None,
        allowed_workspace_domains=("acme-corp.com",),
        required_analyst_groups=("sec-edgar-analysts",),
        admin_workspace_groups=("sec-edgar-admins",),
        analyst_workspace_groups=("sec-edgar-analysts",),
        allow_gcp_service_accounts=True,
        strict_production_mode=True,
    )
    verifier = IAPJwtVerifier(settings=settings, jwks_store=jwks_store)
    return {
        "iap_ec_key": iap_ec_key,
        "google_rsa_key": google_rsa_key,
        "okta_rsa_key": okta_rsa_key,
        "jwks_by_uri": jwks_by_uri,
        "jwks_store": jwks_store,
        "settings": settings,
        "verifier": verifier,
    }


def test_all_internal_apis_require_jwt_bearer_authentication(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every retained internal /api/v1/* endpoint must reject unauthenticated calls with HTTP 401."""
    # Mount a real frontend/dist bundle to verify `/healthz` still returns 404 (never shadowed by SPA)
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<!doctype html><html></html>", encoding="utf-8")
    monkeypatch.setenv("FRONTEND_DIST_DIR", str(dist_dir))

    client = TestClient(app)

    # Public probe and OIDC discovery endpoints remain reachable
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/healthz").status_code == 404
    cfg_resp = client.get("/api/v1/auth/config")
    assert cfg_resp.status_code == 200
    assert cfg_resp.json()["auth_required"] is True
    assert cfg_resp.json()["access_pattern"] == "iap_jwt_bearer"
    assert cfg_resp.json()["identity_provider"]["name"] == "Google Workspace (Okta SSO)"

    # Verify all 12 pruned routes are removed:
    # GET paths return 404 via spa.py's explicit `/api/*` & `/healthz` guard;
    # POST paths return 404 or 405 (Starlette's catch-all `GET /{full_path:path}` returns 405 on POST).
    pruned_routes = [
        ("GET", "/healthz", None, {404}),
        (
            "POST",
            "/api/v1/section",
            {"ticker": "NVDA", "fiscal_year": 2026, "section_id": "item_7"},
            {405},
        ),
        ("GET", "/api/v1/auth/rbac", None, {404}),
        ("GET", "/api/v1/explore/yoy_variances", None, {404}),
        ("GET", "/api/v1/looker/catalog", None, {404}),
        ("GET", "/api/v1/telemetry/context-latency", None, {404}),
        ("POST", "/api/v1/intent", {"query": "Compare NVDA and AAPL"}, {404, 405}),
        ("POST", "/api/v1/evaluate", {"query": "test"}, {404, 405}),
        ("POST", "/api/v1/coding-assist", {"prompt": "test"}, {404, 405}),
        (
            "POST",
            "/api/v1/variances",
            {"tickers": "NVDA", "start_year": 2024, "end_year": 2026},
            {404, 405},
        ),
        ("POST", "/api/v1/search", {"query": "Blackwell architecture"}, {404, 405}),
        ("GET", "/api/v1/metrics", None, {404}),
    ]
    for method, path, payload, allowed_statuses in pruned_routes:
        resp = (
            client.request(method, path, json=payload)
            if payload is not None
            else client.request(method, path)
        )
        assert resp.status_code in allowed_statuses, (
            f"Expected one of {sorted(allowed_statuses)} on pruned {method} {path}, "
            f"got {resp.status_code}"
        )

    # Confirm none of the pruned routes remain registered in the OpenAPI schema
    schema_paths = app.openapi()["paths"]
    assert "post" not in schema_paths.get("/api/v1/section", {})
    for removed_path in (
        "/healthz",
        "/api/v1/auth/rbac",
        "/api/v1/explore/{explore_name}",
        "/api/v1/looker/catalog",
        "/api/v1/telemetry/context-latency",
        "/api/v1/intent",
        "/api/v1/evaluate",
        "/api/v1/coding-assist",
        "/api/v1/variances",
        "/api/v1/search",
        "/api/v1/metrics",
    ):
        assert removed_path not in schema_paths

    # ALL 14 retained authenticated routes must reject unauthenticated calls with HTTP 401
    protected_calls = [
        ("GET", "/api/v1/auth/me", None),
        ("POST", "/api/v1/auth/logout", None),
        ("GET", "/api/v1/auth/audit", None),
        ("GET", "/api/v1/admin/rbac/policies", None),
        ("GET", "/api/v1/secrets/status", None),
        ("GET", "/api/v1/companies", None),
        ("GET", "/api/v1/section?ticker=NVDA&fiscal_year=2026&section_id=item_7", None),
        ("POST", "/api/v1/chat", {"query": "NVDA FY2026 revenue drivers"}),
        ("POST", "/api/v1/chat/stream", {"query": "NVDA FY2026 revenue drivers"}),
        ("GET", "/api/v1/sessions", None),
        ("DELETE", "/api/v1/sessions/nonexistent", None),
        ("GET", "/api/v1/telemetry/sessions", None),
        ("POST", "/api/v1/feedback", {"message_id": "m1", "rating": "up"}),
        ("GET", "/api/v1/feedback", None),
    ]
    assert len(protected_calls) == 14

    for method, path, payload in protected_calls:
        resp = (
            client.request(method, path, json=payload)
            if payload is not None
            else client.request(method, path)
        )
        assert resp.status_code == 401, f"Expected 401 on {method} {path}, got {resp.status_code}"
        assert "Bearer" in resp.headers.get("WWW-Authenticate", "")

    # Non-Bearer Authorization schemes must also be rejected with 401
    basic_resp = client.get(
        "/api/v1/companies",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    assert basic_resp.status_code == 401


def test_iap_es256_google_workspace_rs256_and_okta_sso_verification(idp_keyring) -> None:
    """Verify real ES256 (Cloud IAP) and RS256 (Google Workspace & Okta SSO) tokens end-to-end."""
    verifier: IAPJwtVerifier = idp_keyring["verifier"]
    app.dependency_overrides[get_auth_verifier] = lambda: verifier
    clear_security_audit_events()

    try:
        client = TestClient(app)
        now = int(time.time())

        # 1. Google Cloud IAP ES256 JWT (with accounts.google.com: prefix normalization)
        iap_jwt = sign_jwt_es256(
            {
                "iss": GOOGLE_IAP_ISSUER,
                "sub": "accounts.google.com:1122334455",
                "aud": TEST_IAP_AUDIENCE,
                "email": "accounts.google.com:jane.doe@acme-corp.com",
                "hd": "acme-corp.com",
                "google": {"groups": ["group:sec-edgar-analysts@acme-corp.com"]},
                "iat": now,
                "exp": now + 1800,
            },
            idp_keyring["iap_ec_key"],
            kid="iap-es256-k1",
        )
        me_iap = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {iap_jwt}"},
        )
        assert me_iap.status_code == 200
        iap_user = me_iap.json()["user"]
        assert iap_user["email"] == "jane.doe@acme-corp.com"
        assert iap_user["hosted_domain"] == "acme-corp.com"
        assert iap_user["identity_provider"] == "google_iap"
        assert iap_user["iap_assertion_verified"] is True

        # 2. Google Workspace OIDC RS256 JWT federated with Okta SSO
        gws_jwt = sign_jwt_rs256(
            {
                "iss": "https://accounts.google.com",
                "sub": "10987654321",
                "aud": TEST_IAP_AUDIENCE,
                "email": "alex.chen@acme-corp.com",
                "email_verified": True,
                "name": "Alex Chen",
                "hd": "acme-corp.com",
                "groups": ["sec-edgar-analysts"],
                "iat": now,
                "exp": now + 1800,
            },
            idp_keyring["google_rsa_key"],
            kid="gws-rs256-k1",
        )
        me_gws = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {gws_jwt}"},
        )
        assert me_gws.status_code == 200
        assert me_gws.json()["user"]["identity_provider"] == "google_workspace_oidc"
        assert me_gws.json()["user"]["iap_assertion_verified"] is False

        # 3. Okta SSO OIDC RS256 JWT + accompanying X-Goog-IAP-JWT-Assertion (dual-header defense)
        okta_jwt = sign_jwt_rs256(
            {
                "iss": TEST_OKTA_ISSUER,
                "sub": "00u8okta9911",
                "aud": TEST_OKTA_CLIENT_ID,
                "preferred_username": "jane.doe@acme-corp.com",
                "name": "Jane Doe",
                "groups": ["sec-edgar-admins"],
                "iat": now,
                "exp": now + 1800,
            },
            idp_keyring["okta_rsa_key"],
            kid="okta-rs256-k1",
        )
        dual_resp = client.get(
            "/api/v1/auth/me",
            headers={
                "Authorization": f"Bearer {okta_jwt}",
                "X-Goog-IAP-JWT-Assertion": iap_jwt,
            },
        )
        assert dual_resp.status_code == 200
        dual_user = dual_resp.json()["user"]
        assert dual_user["identity_provider"] == "okta_sso"
        assert dual_user["iap_assertion_verified"] is True
        assert dual_user["roles"] == ["authenticated", "analyst", "admin"]
        # Bearer groups stay strictly authoritative (no union from IAP JWT's analyst group)
        assert dual_user["groups"] == ["sec-edgar-admins"]

        # Mismatched identity between Bearer JWT and X-Goog-IAP-JWT-Assertion must fail (401)
        mismatch_resp = client.get(
            "/api/v1/auth/me",
            headers={
                "Authorization": f"Bearer {gws_jwt}",  # alex.chen@acme-corp.com
                "X-Goog-IAP-JWT-Assertion": iap_jwt,  # jane.doe@acme-corp.com
            },
        )
        assert mismatch_resp.status_code == 401
    finally:
        app.dependency_overrides.pop(get_auth_verifier, None)


def test_cryptographic_attacks_temporal_bounds_and_key_rotation(idp_keyring) -> None:
    """Reject tampered payloads, alg=none/HS256, expired/future tokens, bad aud/iss, and test JWKS rotation."""
    verifier: IAPJwtVerifier = idp_keyring["verifier"]
    now = int(time.time())

    valid_jwt = sign_jwt_es256(
        {
            "iss": GOOGLE_IAP_ISSUER,
            "sub": "accounts.google.com:777",
            "aud": TEST_IAP_AUDIENCE,
            "email": "analyst@acme-corp.com",
            "hd": "acme-corp.com",
            "groups": ["sec-edgar-analysts"],
            "iat": now,
            "exp": now + 600,
        },
        idp_keyring["iap_ec_key"],
        kid="iap-es256-k1",
    )

    # 1. Tampered payload with original signature -> fails verification
    h_b64, _, s_b64 = valid_jwt.split(".")
    tampered_payload = _b64url(
        json.dumps(
            {
                "iss": GOOGLE_IAP_ISSUER,
                "sub": "accounts.google.com:777",
                "aud": TEST_IAP_AUDIENCE,
                "email": "ceo@acme-corp.com",
                "hd": "acme-corp.com",
                "groups": ["sec-edgar-admins"],
                "iat": now,
                "exp": now + 600,
            }
        ).encode("utf-8")
    )
    with pytest.raises(AuthenticationError):
        verifier.verify_jwt(f"{h_b64}.{tampered_payload}.{s_b64}")

    # 2. Algorithm confusion: alg="none" and alg="HS256"
    for bad_alg in ("none", "HS256"):
        bad_hdr = _b64url(json.dumps({"alg": bad_alg, "kid": "iap-es256-k1"}).encode("utf-8"))
        with pytest.raises(Exception, match="Disallowed or missing JWT algorithm"):
            verifier.verify_jwt(f"{bad_hdr}.{tampered_payload}.sig")

    # 3. Expired token, future iat, premature nbf, wrong audience, untrusted issuer
    expired_jwt = sign_jwt_es256(
        {
            "iss": GOOGLE_IAP_ISSUER,
            "sub": "u1",
            "aud": TEST_IAP_AUDIENCE,
            "email": "analyst@acme-corp.com",
            "groups": ["sec-edgar-analysts"],
            "iat": now - 7200,
            "exp": now - 120,
        },
        idp_keyring["iap_ec_key"],
        kid="iap-es256-k1",
    )
    with pytest.raises(Exception, match="expired"):
        verifier.verify_jwt(expired_jwt)

    premature_jwt = sign_jwt_es256(
        {
            "iss": GOOGLE_IAP_ISSUER,
            "sub": "u1",
            "aud": TEST_IAP_AUDIENCE,
            "email": "analyst@acme-corp.com",
            "groups": ["sec-edgar-analysts"],
            "iat": now,
            "nbf": now + 600,
            "exp": now + 1200,
        },
        idp_keyring["iap_ec_key"],
        kid="iap-es256-k1",
    )
    with pytest.raises(Exception, match="nbf"):
        verifier.verify_jwt(premature_jwt)

    wrong_aud_jwt = sign_jwt_es256(
        {
            "iss": GOOGLE_IAP_ISSUER,
            "sub": "u1",
            "aud": "/projects/000/global/backendServices/wrong",
            "email": "analyst@acme-corp.com",
            "groups": ["sec-edgar-analysts"],
            "iat": now,
            "exp": now + 600,
        },
        idp_keyring["iap_ec_key"],
        kid="iap-es256-k1",
    )
    with pytest.raises(Exception, match="audience"):
        verifier.verify_jwt(wrong_aud_jwt)

    untrusted_iss_jwt = sign_jwt_es256(
        {
            "iss": "https://evil-idp.example.org",
            "sub": "u1",
            "aud": TEST_IAP_AUDIENCE,
            "email": "analyst@acme-corp.com",
            "groups": ["sec-edgar-analysts"],
            "iat": now,
            "exp": now + 600,
        },
        idp_keyring["iap_ec_key"],
        kid="iap-es256-k1",
    )
    with pytest.raises(Exception, match="Untrusted JWT issuer"):
        verifier.verify_jwt(untrusted_iss_jwt)

    # 4. Seamless JWKS key rotation when a newly rotated kid appears
    rotated_ec_key = ec.generate_private_key(ec.SECP256R1())
    idp_keyring["jwks_store"].min_refresh_interval_seconds = 0.0
    idp_keyring["jwks_by_uri"][GOOGLE_IAP_JWKS_URI]["keys"].append(
        ec_public_key_to_jwk(rotated_ec_key.public_key(), kid="iap-es256-k2-rotated")
    )
    rotated_jwt = sign_jwt_es256(
        {
            "iss": GOOGLE_IAP_ISSUER,
            "sub": "accounts.google.com:888",
            "aud": TEST_IAP_AUDIENCE,
            "email": "rotated@acme-corp.com",
            "hd": "acme-corp.com",
            "groups": ["sec-edgar-analysts"],
            "iat": now,
            "exp": now + 600,
        },
        rotated_ec_key,
        kid="iap-es256-k2-rotated",
    )
    rotated_user = verifier.verify_jwt(rotated_jwt)
    assert rotated_user.email == "rotated@acme-corp.com"


def test_workspace_domain_rbac_production_lockdown_logout_and_audit_trail(idp_keyring) -> None:
    """Enforce Workspace hd domain + Okta group RBAC (403), Cloud Run dev-broker lockdown, logout revocation, and 10-K audit logs."""
    verifier: IAPJwtVerifier = idp_keyring["verifier"]
    app.dependency_overrides[get_auth_verifier] = lambda: verifier
    clear_security_audit_events()

    try:
        client = TestClient(app)
        now = int(time.time())

        # Strict production mode disables local dev SSO broker with 403 Forbidden
        dev_broker_resp = client.post(
            "/api/v1/auth/sso/dev-token",
            json={"email": "analyst@acme-corp.com"},
        )
        assert dev_broker_resp.status_code == 403

        # Unauthorized Google Workspace domain -> 403 Forbidden
        wrong_domain_jwt = sign_jwt_rs256(
            {
                "iss": "https://accounts.google.com",
                "sub": "u-ext",
                "aud": TEST_IAP_AUDIENCE,
                "email": "contractor@external-vendor.io",
                "email_verified": True,
                "hd": "external-vendor.io",
                "groups": ["sec-edgar-analysts"],
                "iat": now,
                "exp": now + 600,
            },
            idp_keyring["google_rsa_key"],
            kid="gws-rs256-k1",
        )
        assert (
            client.get(
                "/api/v1/companies",
                headers={"Authorization": f"Bearer {wrong_domain_jwt}"},
            ).status_code
            == 403
        )

        # Valid domain but missing required Cloud Identity / Okta analyst group -> 403 Forbidden
        missing_group_jwt = sign_jwt_rs256(
            {
                "iss": TEST_OKTA_ISSUER,
                "sub": "00u-nogroup",
                "aud": TEST_OKTA_CLIENT_ID,
                "email": "hr.user@acme-corp.com",
                "groups": ["general-employees"],
                "iat": now,
                "exp": now + 600,
            },
            idp_keyring["okta_rsa_key"],
            kid="okta-rs256-k1",
        )
        assert (
            client.get(
                "/api/v1/companies",
                headers={"Authorization": f"Bearer {missing_group_jwt}"},
            ).status_code
            == 403
        )

        # GCP Runtime Service Account is permitted for service-to-service calls
        sa_jwt = sign_jwt_rs256(
            {
                "iss": "https://accounts.google.com",
                "sub": "114455667788",
                "aud": TEST_IAP_AUDIENCE,
                "email": "sec-analyst-sa@iron-radio-514217-q7.iam.gserviceaccount.com",
                "email_verified": True,
                "iat": now,
                "exp": now + 600,
            },
            idp_keyring["google_rsa_key"],
            kid="gws-rs256-k1",
        )
        assert (
            client.get(
                "/api/v1/companies",
                headers={"Authorization": f"Bearer {sa_jwt}"},
            ).status_code
            == 200
        )

        # Authorized analyst accesses 10-K section -> tracked in audit trail, then logs out -> token revoked
        analyst_jwt = sign_jwt_rs256(
            {
                "iss": TEST_OKTA_ISSUER,
                "sub": "00u-analyst-1",
                "aud": TEST_OKTA_CLIENT_ID,
                "email": "lead.analyst@acme-corp.com",
                "name": "Lead Analyst",
                "groups": ["sec-edgar-analysts"],
                "jti": "session-okta-001",
                "iat": now,
                "exp": now + 1800,
            },
            idp_keyring["okta_rsa_key"],
            kid="okta-rs256-k1",
        )
        headers = {"Authorization": f"Bearer {analyst_jwt}"}

        sec_resp = client.get(
            "/api/v1/section",
            headers=headers,
            params={"ticker": "AAPL", "fiscal_year": 2026, "section_id": "item_1a"},
        )
        assert sec_resp.status_code == 200

        audit_resp = client.get("/api/v1/auth/audit", headers=headers)
        assert audit_resp.status_code == 200
        events = audit_resp.json()["events"]
        section_events = [e for e in events if e["event_type"] == "filing_10k_section_read"]
        assert len(section_events) >= 1
        assert section_events[0]["user_email"] == "lead.analyst@acme-corp.com"
        assert section_events[0]["tickers"] == ["AAPL"]
        assert section_events[0]["identity_provider"] == "okta_sso"

        # Logout revokes the Bearer JWT immediately
        logout_resp = client.post("/api/v1/auth/logout", headers=headers)
        assert logout_resp.status_code == 200
        assert logout_resp.json()["status"] == "logged_out"

        replay_resp = client.get("/api/v1/companies", headers=headers)
        assert replay_resp.status_code == 401
    finally:
        app.dependency_overrides.pop(get_auth_verifier, None)


TEST_CLOUD_RUN_IAP_AUDIENCE = (
    "/projects/991122334455/locations/us-central1/services/sec-edgar-analyst"
)


def test_cloud_run_iap_assertion_only_browser_session(idp_keyring) -> None:
    """Behind IAP on Cloud Run the browser sends only X-Goog-IAP-JWT-Assertion (no Bearer)."""
    settings = dataclasses.replace(
        idp_keyring["settings"],
        audiences=(TEST_CLOUD_RUN_IAP_AUDIENCE,),
        allowed_workspace_domains=(),
        iap_default_role="analyst",
    )
    verifier = IAPJwtVerifier(settings=settings, jwks_store=idp_keyring["jwks_store"])
    app.dependency_overrides[get_auth_verifier] = lambda: verifier
    now = int(time.time())

    def _iap_assertion(*, aud: str = TEST_CLOUD_RUN_IAP_AUDIENCE, key: Any = None) -> str:
        return sign_jwt_es256(
            {
                "iss": GOOGLE_IAP_ISSUER,
                "sub": "accounts.google.com:5566778899",
                "aud": aud,
                "email": "accounts.google.com:colleague@jackbetcher.altostrat.com",
                "hd": "jackbetcher.altostrat.com",
                "iat": now,
                "exp": now + 600,
            },
            key or idp_keyring["iap_ec_key"],
            kid="iap-es256-k1",
        )

    try:
        client = TestClient(app)

        cfg = client.get("/api/v1/auth/config").json()
        assert cfg["iap"]["session_auth_enabled"] is True
        assert cfg["iap"]["default_role"] == "analyst"

        # IAP assertion alone, no group claims -> configured default role (analyst)
        headers = {"X-Goog-IAP-JWT-Assertion": _iap_assertion()}
        me = client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200, me.text
        user = me.json()["user"]
        assert user["email"] == "colleague@jackbetcher.altostrat.com"
        assert user["identity_provider"] == "google_iap"
        assert user["iap_assertion_verified"] is True
        assert user["roles"] == ["authenticated", "analyst"]
        assert client.get("/api/v1/companies", headers=headers).status_code == 200

        # Admin-only endpoints stay closed: admin must come from a Cloud Identity group
        assert client.get("/api/v1/admin/rbac/policies", headers=headers).status_code == 403

        # Assertion minted for a different IAP resource (audience mismatch) -> 401
        wrong_aud = {
            "X-Goog-IAP-JWT-Assertion": _iap_assertion(
                aud="/projects/991122334455/global/backendServices/111"
            )
        }
        assert client.get("/api/v1/auth/me", headers=wrong_aud).status_code == 401

        # Forged assertion (not signed by the IAP key) -> 401
        forged = {
            "X-Goog-IAP-JWT-Assertion": _iap_assertion(key=ec.generate_private_key(ec.SECP256R1()))
        }
        assert client.get("/api/v1/auth/me", headers=forged).status_code == 401

        # A non-IAP issuer in the IAP header is rejected even if otherwise valid -> 401
        gws_jwt = sign_jwt_rs256(
            {
                "iss": "https://accounts.google.com",
                "sub": "10987654321",
                "aud": TEST_CLOUD_RUN_IAP_AUDIENCE,
                "email": "colleague@jackbetcher.altostrat.com",
                "email_verified": True,
                "groups": ["sec-edgar-analysts"],
                "iat": now,
                "exp": now + 600,
            },
            idp_keyring["google_rsa_key"],
            kid="gws-rs256-k1",
        )
        assert (
            client.get("/api/v1/auth/me", headers={"X-Goog-IAP-JWT-Assertion": gws_jwt}).status_code
            == 401
        )

        # Without a default role, an IAP-only user with no group is authenticated but forbidden
        no_default = dataclasses.replace(settings, iap_default_role=None)
        app.dependency_overrides[get_auth_verifier] = lambda: IAPJwtVerifier(
            settings=no_default, jwks_store=idp_keyring["jwks_store"]
        )
        assert client.get("/api/v1/auth/me", headers=headers).status_code == 403
    finally:
        app.dependency_overrides.pop(get_auth_verifier, None)


def test_iap_default_role_env_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAP_ASSERTION_DEFAULT_ROLE", "Viewer")
    assert AuthSettings.from_env().iap_default_role == "viewer"

    monkeypatch.setenv("IAP_ASSERTION_DEFAULT_ROLE", "")
    assert AuthSettings.from_env().iap_default_role is None

    monkeypatch.setenv("IAP_ASSERTION_DEFAULT_ROLE", "admin")
    with pytest.raises(ValueError, match="IAP_ASSERTION_DEFAULT_ROLE"):
        AuthSettings.from_env()


def test_sso_token_exchange_endpoint_and_audit_trail(idp_keyring) -> None:
    """Verify POST /api/v1/auth/sso/exchange handles valid JWTs, Bearer prefix stripping, 401, and 403."""
    verifier: IAPJwtVerifier = idp_keyring["verifier"]
    app.dependency_overrides[get_auth_verifier] = lambda: verifier
    clear_security_audit_events()

    try:
        client = TestClient(app)
        now = int(time.time())

        valid_okta_jwt = sign_jwt_rs256(
            {
                "iss": TEST_OKTA_ISSUER,
                "sub": "00u-exchange-1",
                "aud": TEST_OKTA_CLIENT_ID,
                "email": "exchange.analyst@acme-corp.com",
                "name": "Exchange Analyst",
                "groups": ["sec-edgar-analysts"],
                "iat": now,
                "exp": now + 1800,
            },
            idp_keyring["okta_rsa_key"],
            kid="okta-rs256-k1",
        )

        # 1. Valid raw JWT -> 200 + sso_token_exchange audit event
        ok_resp = client.post("/api/v1/auth/sso/exchange", json={"id_token": valid_okta_jwt})
        assert ok_resp.status_code == 200, ok_resp.text
        ok_body = ok_resp.json()
        assert ok_body["status"] == "authenticated"
        assert ok_body["token_type"] == "Bearer"
        assert ok_body["access_token"] == valid_okta_jwt
        assert ok_body["expires_at"] == now + 1800
        assert ok_body["user"]["email"] == "exchange.analyst@acme-corp.com"
        assert ok_body["user"]["identity_provider"] == "okta_sso"
        assert ok_body["user"]["roles"] == ["authenticated", "analyst"]

        # 2. `Bearer ` prefix (including mixed-case and whitespace) is stripped cleanly
        prefixed_resp = client.post(
            "/api/v1/auth/sso/exchange",
            json={"id_token": f"  Bearer {valid_okta_jwt}  "},
        )
        assert prefixed_resp.status_code == 200, prefixed_resp.text
        assert prefixed_resp.json()["access_token"] == valid_okta_jwt

        lower_prefixed_resp = client.post(
            "/api/v1/auth/sso/exchange",
            json={"id_token": f"bearer {valid_okta_jwt}"},
        )
        assert lower_prefixed_resp.status_code == 200, lower_prefixed_resp.text
        assert lower_prefixed_resp.json()["access_token"] == valid_okta_jwt

        # 3. Expired JWT -> 401 + WWW-Authenticate header + sso_token_exchange_failed audit event
        expired_jwt = sign_jwt_rs256(
            {
                "iss": TEST_OKTA_ISSUER,
                "sub": "00u-expired",
                "aud": TEST_OKTA_CLIENT_ID,
                "email": "exchange.analyst@acme-corp.com",
                "groups": ["sec-edgar-analysts"],
                "iat": now - 7200,
                "exp": now - 120,
            },
            idp_keyring["okta_rsa_key"],
            kid="okta-rs256-k1",
        )
        exp_resp = client.post("/api/v1/auth/sso/exchange", json={"id_token": expired_jwt})
        assert exp_resp.status_code == 401
        assert "Bearer" in exp_resp.headers.get("WWW-Authenticate", "")
        assert exp_resp.json()["detail"]["status"] == "unauthorized"

        # 4. Malformed / invalid JWT -> 401 + sso_token_exchange_failed audit event
        bad_resp = client.post(
            "/api/v1/auth/sso/exchange",
            json={"id_token": "not.a.valid_jwt_token_payload"},
        )
        assert bad_resp.status_code == 401
        assert "Bearer" in bad_resp.headers.get("WWW-Authenticate", "")

        # 5. Forbidden group -> 403 + sso_token_exchange_forbidden audit event
        no_group_jwt = sign_jwt_rs256(
            {
                "iss": TEST_OKTA_ISSUER,
                "sub": "00u-nogroup",
                "aud": TEST_OKTA_CLIENT_ID,
                "email": "hr.user@acme-corp.com",
                "groups": ["general-employees"],
                "iat": now,
                "exp": now + 600,
            },
            idp_keyring["okta_rsa_key"],
            kid="okta-rs256-k1",
        )
        forb_group_resp = client.post("/api/v1/auth/sso/exchange", json={"id_token": no_group_jwt})
        assert forb_group_resp.status_code == 403
        assert forb_group_resp.json()["detail"]["status"] == "forbidden"

        # 6. Forbidden Workspace domain -> 403 + sso_token_exchange_forbidden audit event
        bad_domain_jwt = sign_jwt_rs256(
            {
                "iss": "https://accounts.google.com",
                "sub": "u-ext-domain",
                "aud": TEST_IAP_AUDIENCE,
                "email": "contractor@external-vendor.io",
                "email_verified": True,
                "hd": "external-vendor.io",
                "groups": ["sec-edgar-analysts"],
                "iat": now,
                "exp": now + 600,
            },
            idp_keyring["google_rsa_key"],
            kid="gws-rs256-k1",
        )
        forb_domain_resp = client.post(
            "/api/v1/auth/sso/exchange", json={"id_token": bad_domain_jwt}
        )
        assert forb_domain_resp.status_code == 403
        assert forb_domain_resp.json()["detail"]["status"] == "forbidden"

        # Verify all expected audit events were recorded
        events = get_security_audit_events(limit=50)
        allowed_exchanges = [
            e
            for e in events
            if e["event_type"] == "sso_token_exchange" and e["status"] == "allowed"
        ]
        failed_exchanges = [
            e
            for e in events
            if e["event_type"] == "sso_token_exchange_failed" and e["status"] == "denied_401"
        ]
        forbidden_exchanges = [
            e
            for e in events
            if e["event_type"] == "sso_token_exchange_forbidden" and e["status"] == "denied_403"
        ]
        assert len(allowed_exchanges) == 3
        assert all(e["resource"] == "/api/v1/auth/sso/exchange" for e in allowed_exchanges)
        assert len(failed_exchanges) == 2
        assert len(forbidden_exchanges) == 2
    finally:
        app.dependency_overrides.pop(get_auth_verifier, None)


def test_dual_iap_and_bearer_never_escalates_privileges_and_require_roles_removed(
    idp_keyring,
) -> None:
    """Dual IAP + Bearer must keep Bearer RBAC strictly authoritative (no escalation from IAP)."""
    # Verify legacy dead-code `require_roles` dependency has been deleted from security exports
    assert not hasattr(security_auth_mod, "require_roles")
    assert not hasattr(security_pkg, "require_roles")

    settings = dataclasses.replace(
        idp_keyring["settings"],
        required_analyst_groups=(),
        viewer_workspace_groups=("sec-edgar-viewers",),
        iap_default_role="analyst",
    )
    verifier = IAPJwtVerifier(settings=settings, jwks_store=idp_keyring["jwks_store"])
    app.dependency_overrides[get_auth_verifier] = lambda: verifier
    clear_security_audit_events()

    try:
        client = TestClient(app)
        now = int(time.time())

        viewer_okta_jwt = sign_jwt_rs256(
            {
                "iss": TEST_OKTA_ISSUER,
                "sub": "00u-viewer-1",
                "aud": TEST_OKTA_CLIENT_ID,
                "email": "auditor@acme-corp.com",
                "groups": ["sec-edgar-viewers"],
                "iat": now,
                "exp": now + 1200,
            },
            idp_keyring["okta_rsa_key"],
            kid="okta-rs256-k1",
        )

        # Case 1: IAP assertion has no groups (alone would receive iap_default_role="analyst")
        iap_default_analyst_jwt = sign_jwt_es256(
            {
                "iss": GOOGLE_IAP_ISSUER,
                "sub": "accounts.google.com:viewer-1",
                "aud": TEST_IAP_AUDIENCE,
                "email": "accounts.google.com:auditor@acme-corp.com",
                "hd": "acme-corp.com",
                "iat": now,
                "exp": now + 600,
            },
            idp_keyring["iap_ec_key"],
            kid="iap-es256-k1",
        )
        # Sanity check: IAP assertion alone yields `analyst` via iap_default_role
        iap_only = client.get(
            "/api/v1/auth/me",
            headers={"X-Goog-IAP-JWT-Assertion": iap_default_analyst_jwt},
        )
        assert iap_only.status_code == 200
        assert iap_only.json()["user"]["roles"] == ["authenticated", "analyst"]

        # Combined with Viewer Bearer JWT: must stay strictly `viewer`, NEVER escalate to `analyst`!
        dual_viewer_headers = {
            "Authorization": f"Bearer {viewer_okta_jwt}",
            "X-Goog-IAP-JWT-Assertion": iap_default_analyst_jwt,
        }
        me_dual = client.get("/api/v1/auth/me", headers=dual_viewer_headers)
        assert me_dual.status_code == 200
        user_dual = me_dual.json()["user"]
        assert user_dual["iap_assertion_verified"] is True
        assert user_dual["roles"] == ["authenticated", "viewer"]
        assert "analyst" not in user_dual["roles"]
        assert "chat:query" not in user_dual["permissions"]
        assert client.get("/api/v1/secrets/status", headers=dual_viewer_headers).status_code == 403
        assert (
            client.post(
                "/api/v1/feedback",
                headers=dual_viewer_headers,
                json={"message_id": "m1", "rating": "up"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/v1/chat",
                headers=dual_viewer_headers,
                json={"query": "Summarize NVDA revenue"},
            ).status_code
            == 403
        )

        # Case 2: IAP assertion carries admin group while Bearer is viewer -> must NOT escalate to admin
        iap_admin_jwt = sign_jwt_es256(
            {
                "iss": GOOGLE_IAP_ISSUER,
                "sub": "accounts.google.com:viewer-1",
                "aud": TEST_IAP_AUDIENCE,
                "email": "accounts.google.com:auditor@acme-corp.com",
                "hd": "acme-corp.com",
                "groups": ["sec-edgar-admins"],
                "iat": now,
                "exp": now + 600,
            },
            idp_keyring["iap_ec_key"],
            kid="iap-es256-k1",
        )
        dual_viewer_vs_admin_iap = {
            "Authorization": f"Bearer {viewer_okta_jwt}",
            "X-Goog-IAP-JWT-Assertion": iap_admin_jwt,
        }
        me_v_admin = client.get("/api/v1/auth/me", headers=dual_viewer_vs_admin_iap)
        assert me_v_admin.status_code == 200
        assert me_v_admin.json()["user"]["roles"] == ["authenticated", "viewer"]
        assert (
            client.get("/api/v1/admin/rbac/policies", headers=dual_viewer_vs_admin_iap).status_code
            == 403
        )

        # Case 3: Service account Bearer + human admin IAP assertion -> SA MUST NOT escalate!
        sa_guardrail_jwt = sign_jwt_rs256(
            {
                "iss": "https://accounts.google.com",
                "sub": "114455667799",
                "aud": TEST_IAP_AUDIENCE,
                "email": "model-armor-guardrail-sa@iron-radio-514217-q7.iam.gserviceaccount.com",
                "email_verified": True,
                "iat": now,
                "exp": now + 600,
            },
            idp_keyring["google_rsa_key"],
            kid="gws-rs256-k1",
        )
        dual_sa_headers = {
            "Authorization": f"Bearer {sa_guardrail_jwt}",
            "X-Goog-IAP-JWT-Assertion": iap_admin_jwt,
        }
        me_sa = client.get("/api/v1/auth/me", headers=dual_sa_headers)
        assert me_sa.status_code == 200
        sa_user = me_sa.json()["user"]
        assert sa_user["iap_assertion_verified"] is True
        assert sa_user["agent_identity"] == "ModelArmorGuardrail"
        assert "analyst" not in sa_user["roles"]
        assert "admin" not in sa_user["roles"]
        assert sa_user["permissions"] == []
        assert client.get("/api/v1/companies", headers=dual_sa_headers).status_code == 403
        assert client.get("/api/v1/admin/rbac/policies", headers=dual_sa_headers).status_code == 403
    finally:
        app.dependency_overrides.pop(get_auth_verifier, None)
