"""Comprehensive tests for Google Workspace Okta/SSO & Identity-Aware Proxy (IAP) JWT authentication."""

from __future__ import annotations

import base64
import json
import time
from typing import Any

from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.testclient import TestClient
import httpx
import pytest

from edgar_capstone.security import (
    GOOGLE_IAP_ISSUER,
    GOOGLE_IAP_JWKS_URI,
    GOOGLE_WORKSPACE_JWKS_URI,
    AuthSettings,
    IAPJwtVerifier,
    JwksKeyStore,
    clear_security_audit_events,
    ec_public_key_to_jwk,
    get_auth_verifier,
    rsa_public_key_to_jwk,
    sign_jwt_es256,
    sign_jwt_rs256,
)
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


def test_all_internal_apis_require_jwt_bearer_authentication() -> None:
    """Every internal /api/v1/* data endpoint must reject unauthenticated calls with HTTP 401."""
    client = TestClient(app)

    # Public probe and OIDC discovery endpoints remain reachable
    assert client.get("/health").status_code == 200
    assert client.get("/healthz").status_code == 200
    cfg_resp = client.get("/api/v1/auth/config")
    assert cfg_resp.status_code == 200
    assert cfg_resp.json()["auth_required"] is True
    assert cfg_resp.json()["access_pattern"] == "iap_jwt_bearer"
    assert cfg_resp.json()["identity_provider"]["name"] == "Google Workspace (Okta SSO)"

    protected_calls = [
        ("GET", "/api/v1/auth/me", None),
        ("POST", "/api/v1/auth/logout", None),
        ("GET", "/api/v1/auth/audit", None),
        ("GET", "/api/v1/secrets/status", None),
        ("POST", "/api/v1/variances", {"tickers": "NVDA", "start_year": 2024, "end_year": 2026}),
        ("POST", "/api/v1/search", {"query": "Blackwell architecture"}),
        ("POST", "/api/v1/intent", {"query": "Compare NVDA and AAPL"}),
        ("GET", "/api/v1/metrics", None),
        ("GET", "/api/v1/explore/yoy_variances", None),
        ("GET", "/api/v1/companies", None),
        ("POST", "/api/v1/chat", {"query": "NVDA FY2026 revenue drivers"}),
        ("GET", "/api/v1/section?ticker=NVDA&fiscal_year=2026&section_id=item_7", None),
        ("POST", "/api/v1/section", {"ticker": "NVDA", "fiscal_year": 2026, "section_id": "item_7"}),
        ("POST", "/api/v1/feedback", {"message_id": "m1", "rating": "up"}),
    ]

    for method, path, payload in protected_calls:
        if method == "GET":
            resp = client.get(path)
        else:
            resp = client.post(path, json=payload)
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
        assert "admin" in dual_user["roles"]

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
    with pytest.raises( Exception ):
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
