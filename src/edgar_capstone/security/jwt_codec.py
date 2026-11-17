"""Shared RFC 7515 / RFC 7517 base64url, JWK, and ES256/RS256 JWT codec helpers."""

from __future__ import annotations

import base64
import binascii
import json
import string
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)

_B64URL_CHARS = frozenset(string.ascii_letters + string.digits + "-_=")

PublicKey = ec.EllipticCurvePublicKey | rsa.RSAPublicKey


def b64url_encode(raw: bytes) -> str:
    """Encode bytes into unpadded RFC 4648 base64url ASCII text."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def b64url_decode(segment: str) -> bytes:
    """Decode a base64url segment strictly, raising ``ValueError`` on malformed input."""
    if not isinstance(segment, str) or not segment:
        raise ValueError("Empty or non-string base64url segment in JWT.")
    stripped = segment.rstrip("=")
    if not stripped or len(stripped) % 4 == 1 or any(ch not in _B64URL_CHARS for ch in segment):
        raise ValueError("Malformed base64url segment in JWT.")
    if "=" in stripped:
        raise ValueError("Malformed base64url padding in JWT.")
    padded = stripped + "=" * ((4 - len(stripped) % 4) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, binascii.Error, UnicodeEncodeError) as exc:
        raise ValueError("Malformed base64url segment in JWT.") from exc


def int_to_b64url(value: int, byte_length: int | None = None) -> str:
    """Encode a non-negative integer as big-endian base64url bytes."""
    if byte_length is None:
        byte_length = max(1, (value.bit_length() + 7) // 8)
    return b64url_encode(value.to_bytes(byte_length, "big"))


def b64url_to_int(segment: str) -> int:
    """Decode a big-endian base64url unsigned integer, raising ``ValueError`` if empty/invalid."""
    raw = b64url_decode(segment)
    if not raw:
        raise ValueError("Empty numeric parameter in JWK.")
    return int.from_bytes(raw, "big")


def parse_jwt_header_and_claims(
    header_b64: str,
    payload_b64: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode and parse compact JWS header & payload JSON objects, raising ``ValueError`` if invalid."""
    try:
        header = json.loads(b64url_decode(header_b64).decode("utf-8"))
        claims = json.loads(b64url_decode(payload_b64).decode("utf-8"))
    except ValueError:
        raise
    except (UnicodeDecodeError, TypeError) as exc:
        raise ValueError("JWT header or payload is not valid UTF-8 JSON.") from exc

    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise ValueError("JWT header and payload must be JSON objects.")
    return header, claims


def jwk_to_public_key(jwk: dict[str, Any]) -> PublicKey:
    """Construct a cryptographic public key from an EC (P-256) or RSA JWK dictionary.

    Raises ``ValueError`` for missing required fields, unsupported key types/curves,
    or invalid key parameters (never uncaught ``KeyError``).
    """
    if not isinstance(jwk, dict):
        raise ValueError("JWK entry must be a JSON object.")

    kty = jwk.get("kty")
    if kty == "EC":
        crv = jwk.get("crv")
        if crv != "P-256":
            raise ValueError(f"Unsupported EC curve in JWK: {crv!r}")
        x_raw = jwk.get("x")
        y_raw = jwk.get("y")
        if not isinstance(x_raw, str) or not x_raw or not isinstance(y_raw, str) or not y_raw:
            raise ValueError("EC JWK is missing required 'x' or 'y' coordinate.")
        x = b64url_to_int(x_raw)
        y = b64url_to_int(y_raw)
        try:
            return ec.EllipticCurvePublicNumbers(x, y, ec.SECP256R1()).public_key()
        except ValueError as exc:
            raise ValueError(f"Invalid EC P-256 public key coordinates in JWK: {exc}") from exc

    if kty == "RSA":
        n_raw = jwk.get("n")
        e_raw = jwk.get("e")
        if not isinstance(n_raw, str) or not n_raw or not isinstance(e_raw, str) or not e_raw:
            raise ValueError("RSA JWK is missing required 'n' or 'e' parameter.")
        n = b64url_to_int(n_raw)
        e = b64url_to_int(e_raw)
        try:
            return rsa.RSAPublicNumbers(e, n).public_key()
        except ValueError as exc:
            raise ValueError(f"Invalid RSA public key parameters in JWK: {exc}") from exc

    raise ValueError(f"Unsupported JWK key type: {kty!r}")


def ec_public_key_to_jwk(
    public_key: ec.EllipticCurvePublicKey,
    *,
    kid: str,
) -> dict[str, Any]:
    """Serialize a P-256 public key into standard RFC 7517 JWK representation."""
    numbers = public_key.public_numbers()
    return {
        "kty": "EC",
        "crv": "P-256",
        "alg": "ES256",
        "use": "sig",
        "kid": kid,
        "x": int_to_b64url(numbers.x, 32),
        "y": int_to_b64url(numbers.y, 32),
    }


def rsa_public_key_to_jwk(
    public_key: rsa.RSAPublicKey,
    *,
    kid: str,
) -> dict[str, Any]:
    """Serialize an RSA public key into standard RFC 7517 JWK representation."""
    numbers = public_key.public_numbers()
    return {
        "kty": "RSA",
        "alg": "RS256",
        "use": "sig",
        "kid": kid,
        "n": int_to_b64url(numbers.n),
        "e": int_to_b64url(numbers.e),
    }


def sign_jwt_es256(
    claims: dict[str, Any],
    private_key: ec.EllipticCurvePrivateKey,
    *,
    kid: str,
    typ: str = "JWT",
) -> str:
    """Create a compact JWS signed with ECDSA P-256 + SHA-256 (ES256)."""
    header = {"alg": "ES256", "typ": typ, "kid": kid}
    header_b64 = b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = b64url_encode(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    der_sig = private_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der_sig)
    raw_sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return f"{header_b64}.{payload_b64}.{b64url_encode(raw_sig)}"


def sign_jwt_rs256(
    claims: dict[str, Any],
    private_key: rsa.RSAPrivateKey,
    *,
    kid: str,
    typ: str = "JWT",
) -> str:
    """Create a compact JWS signed with RSA PKCS#1 v1.5 + SHA-256 (RS256)."""
    header = {"alg": "RS256", "typ": typ, "kid": kid}
    header_b64 = b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = b64url_encode(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{header_b64}.{payload_b64}.{b64url_encode(sig)}"


def verify_es256_signature(
    signing_input: bytes,
    signature_bytes: bytes,
    public_key: ec.EllipticCurvePublicKey,
) -> None:
    """Verify a raw 64-byte IEEE P1363 ES256 signature, raising ``ValueError`` on failure."""
    if len(signature_bytes) != 64:
        raise ValueError("Invalid ES256 signature length.")
    r = int.from_bytes(signature_bytes[:32], "big")
    s = int.from_bytes(signature_bytes[32:], "big")
    if r <= 0 or s <= 0:
        raise ValueError("Invalid ES256 signature scalar.")
    der_sig = encode_dss_signature(r, s)
    try:
        public_key.verify(der_sig, signing_input, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as exc:
        raise ValueError("ES256 JWT signature verification failed.") from exc


def verify_jws_signature(
    *,
    alg: str,
    signing_input: bytes,
    signature_bytes: bytes,
    public_key: PublicKey,
) -> None:
    """Verify an ES256 or RS256 JWS signature against ``public_key``, raising ``ValueError`` on failure."""
    if alg == "ES256":
        if not isinstance(public_key, ec.EllipticCurvePublicKey):
            raise ValueError("JWK key type does not match ES256 header.")
        verify_es256_signature(signing_input, signature_bytes, public_key)
        return

    if alg == "RS256":
        if not isinstance(public_key, rsa.RSAPublicKey):
            raise ValueError("JWK key type does not match RS256 header.")
        try:
            public_key.verify(
                signature_bytes,
                signing_input,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
        except InvalidSignature as exc:
            raise ValueError("RS256 JWT signature verification failed.") from exc
        return

    raise ValueError(f"Unsupported JWT signing algorithm: {alg!r}")


__all__ = [
    "PublicKey",
    "b64url_decode",
    "b64url_encode",
    "b64url_to_int",
    "ec_public_key_to_jwk",
    "int_to_b64url",
    "jwk_to_public_key",
    "parse_jwt_header_and_claims",
    "rsa_public_key_to_jwk",
    "sign_jwt_es256",
    "sign_jwt_rs256",
    "verify_es256_signature",
    "verify_jws_signature",
]
