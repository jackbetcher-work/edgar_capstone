"""RFC 7515 / RFC 7517 JWT & JWK helpers backed by ``PyJWT[crypto]``."""

from __future__ import annotations

import json
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

PublicKey = ec.EllipticCurvePublicKey | rsa.RSAPublicKey


def jwk_to_public_key(jwk: dict[str, Any]) -> PublicKey:
    """Construct a cryptographic public key from an EC (P-256) or RSA JWK dict via PyJWT."""
    if not isinstance(jwk, dict):
        raise ValueError("JWK entry must be a JSON object.")

    kty = jwk.get("kty")
    if kty == "EC":
        crv = jwk.get("crv")
        if crv != "P-256":
            raise ValueError(f"Unsupported EC curve in JWK: {crv!r}")
    elif kty != "RSA":
        raise ValueError(f"Unsupported JWK key type: {kty!r}")

    try:
        key = jwt.PyJWK.from_dict(jwk).key
    except Exception as exc:
        raise ValueError(f"Invalid {kty} JWK public key: {exc}") from exc

    if not isinstance(key, (ec.EllipticCurvePublicKey, rsa.RSAPublicKey)):
        raise ValueError(f"Unsupported public key type produced from JWK: {type(key).__name__}")
    return key


def ec_public_key_to_jwk(
    public_key: ec.EllipticCurvePublicKey,
    *,
    kid: str,
) -> dict[str, Any]:
    """Serialize a P-256 public key into standard RFC 7517 JWK representation."""
    raw = json.loads(ECAlgorithm.to_jwk(public_key))
    raw.update({"alg": "ES256", "use": "sig", "kid": kid})
    return raw


def rsa_public_key_to_jwk(
    public_key: rsa.RSAPublicKey,
    *,
    kid: str,
) -> dict[str, Any]:
    """Serialize an RSA public key into standard RFC 7517 JWK representation."""
    raw = json.loads(RSAAlgorithm.to_jwk(public_key))
    raw.update({"alg": "RS256", "use": "sig", "kid": kid})
    return raw


def sign_jwt_es256(
    claims: dict[str, Any],
    private_key: ec.EllipticCurvePrivateKey,
    *,
    kid: str,
    typ: str = "JWT",
) -> str:
    """Create a compact JWS signed with ECDSA P-256 + SHA-256 (ES256) via PyJWT."""
    return str(
        jwt.encode(
            claims,
            private_key,
            algorithm="ES256",
            headers={"kid": kid, "typ": typ},
        )
    )


def sign_jwt_rs256(
    claims: dict[str, Any],
    private_key: rsa.RSAPrivateKey,
    *,
    kid: str,
    typ: str = "JWT",
) -> str:
    """Create a compact JWS signed with RSA PKCS#1 v1.5 + SHA-256 (RS256) via PyJWT."""
    return str(
        jwt.encode(
            claims,
            private_key,
            algorithm="RS256",
            headers={"kid": kid, "typ": typ},
        )
    )


def parse_unverified_jwt(token: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Inspect unverified JWT header & payload via PyJWT for issuer/kid routing only."""
    try:
        header = jwt.get_unverified_header(token)
        claims = jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_nbf": False,
                "verify_iat": False,
                "verify_aud": False,
            },
        )
    except jwt.PyJWTError as exc:
        raise ValueError(f"Malformed JWT structure: {exc}") from exc

    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise ValueError("JWT header and payload must be JSON objects.")
    return header, claims


def decode_verified_jwt(
    token: str,
    public_key: PublicKey,
    *,
    alg: str,
    leeway_seconds: int = 0,
) -> dict[str, Any]:
    """Cryptographically verify a JWS signature and standard temporal claims via PyJWT."""
    if alg == "ES256" and not isinstance(public_key, ec.EllipticCurvePublicKey):
        raise ValueError("JWK key type does not match ES256 header.")
    if alg == "RS256" and not isinstance(public_key, rsa.RSAPublicKey):
        raise ValueError("JWK key type does not match RS256 header.")
    if alg not in ("ES256", "RS256"):
        raise ValueError(f"Unsupported JWT signing algorithm: {alg!r}")

    try:
        claims = jwt.decode(
            token,
            public_key,
            algorithms=[alg],
            leeway=max(0, int(leeway_seconds)),
            options={
                "verify_signature": True,
                "verify_exp": False,
                "verify_nbf": False,
                "verify_iat": False,
                "verify_aud": False,
            },
        )
    except jwt.PyJWTError as exc:
        raise ValueError(f"{alg} JWT signature verification failed: {exc}") from exc
    return dict(claims)


__all__ = [
    "PublicKey",
    "decode_verified_jwt",
    "ec_public_key_to_jwk",
    "jwk_to_public_key",
    "parse_unverified_jwt",
    "rsa_public_key_to_jwk",
    "sign_jwt_es256",
    "sign_jwt_rs256",
]
