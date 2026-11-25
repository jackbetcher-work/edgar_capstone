"""Shared GCP SDK exception hierarchy, error translation, and ClientOptions helpers."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Mapping
from typing import Any


class GcpClientError(RuntimeError):
    """Base exception for GCP SDK / API failures."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GcpHttpError(GcpClientError):
    """Raised when a GCP API returns a non-retryable 4xx/5xx error."""

    def __init__(self, status_code: int, url: str, response_text: str) -> None:
        super().__init__(
            f"GCP API request failed (HTTP {status_code}) for {url}: {response_text}",
            status_code=status_code,
        )
        self.url = url
        self.response_text = response_text


class GcpNotFoundError(GcpClientError):
    """Raised when a GCP resource is not found (HTTP 404 / gRPC NOT_FOUND)."""

    def __init__(self, message: str, *, status_code: int | None = 404) -> None:
        super().__init__(message, status_code=status_code)


class GcpRateLimitError(GcpClientError):
    """Raised when a GCP API returns HTTP 429 / gRPC RESOURCE_EXHAUSTED."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after_seconds = retry_after_seconds


class GcpTimeoutError(GcpClientError, TimeoutError):
    """Raised when a GCP API request times out or hits a transient upstream failure."""


def parse_retry_after(resp_or_headers: Any) -> float | None:
    """Parse a numeric ``Retry-After`` header in seconds from a response or header mapping."""
    if resp_or_headers is None:
        return None
    headers = getattr(resp_or_headers, "headers", resp_or_headers)
    if not isinstance(headers, Mapping):
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def build_client_options(
    *,
    api_endpoint: str | None = None,
    quota_project_id: str | None = None,
) -> Any | None:
    """Build a ``google.api_core.client_options.ClientOptions`` if any option is set."""
    endpoint = (api_endpoint or "").strip() or None
    qp = (quota_project_id or "").strip() or None
    if not endpoint and not qp:
        return None
    from google.api_core.client_options import ClientOptions

    kwargs: dict[str, Any] = {}
    if endpoint:
        kwargs["api_endpoint"] = endpoint
    if qp:
        kwargs["quota_project_id"] = qp
    return ClientOptions(**kwargs)


_GRPC_STATUS_TO_HTTP: dict[str, int] = {
    "INVALID_ARGUMENT": 400,
    "FAILED_PRECONDITION": 400,
    "OUT_OF_RANGE": 400,
    "UNAUTHENTICATED": 401,
    "PERMISSION_DENIED": 403,
    "NOT_FOUND": 404,
    "ABORTED": 409,
    "ALREADY_EXISTS": 409,
    "RESOURCE_EXHAUSTED": 429,
    "CANCELLED": 499,
    "UNKNOWN": 500,
    "INTERNAL": 500,
    "DATA_LOSS": 500,
    "UNIMPLEMENTED": 501,
    "UNAVAILABLE": 503,
    "DEADLINE_EXCEEDED": 504,
}

_GRPC_INT_TO_NAME: dict[int, str] = {
    1: "CANCELLED",
    2: "UNKNOWN",
    3: "INVALID_ARGUMENT",
    4: "DEADLINE_EXCEEDED",
    5: "NOT_FOUND",
    6: "ALREADY_EXISTS",
    7: "PERMISSION_DENIED",
    8: "RESOURCE_EXHAUSTED",
    9: "FAILED_PRECONDITION",
    10: "ABORTED",
    11: "OUT_OF_RANGE",
    12: "UNIMPLEMENTED",
    13: "INTERNAL",
    14: "UNAVAILABLE",
    15: "DATA_LOSS",
    16: "UNAUTHENTICATED",
}


def _extract_status_info(exc: BaseException) -> tuple[int | None, str | None]:
    """Return ``(http_status_code, grpc_status_name)`` extracted from a GCP/gRPC exception."""
    code = getattr(exc, "code", None)
    if callable(code):
        try:
            code = code()
        except Exception:
            code = None
    if isinstance(code, int) and not isinstance(code, bool):
        if code >= 100:
            return code, None
        grpc_name = _GRPC_INT_TO_NAME.get(code)
        if grpc_name is not None:
            return _GRPC_STATUS_TO_HTTP[grpc_name], grpc_name
    else:
        code_name = getattr(code, "name", None)
        if isinstance(code_name, str) and code_name.upper() in _GRPC_STATUS_TO_HTTP:
            norm_name = code_name.upper()
            return _GRPC_STATUS_TO_HTTP[norm_name], norm_name
        val = getattr(code, "value", None)
        if isinstance(val, tuple) and val and isinstance(val[0], int):
            grpc_name = _GRPC_INT_TO_NAME.get(val[0])
            if grpc_name is not None:
                return _GRPC_STATUS_TO_HTTP[grpc_name], grpc_name

    resp = getattr(exc, "response", None)
    resp_code = getattr(resp, "status_code", None)
    return (int(resp_code), None) if isinstance(resp_code, int) else (None, None)


def translate_gcp_sdk_error(
    exc: BaseException,
    *,
    context: str = "GCP API",
) -> GcpClientError:
    """Translate ``google.api_core`` / ``google.genai`` / timeout errors into ``GcpClientError``."""
    if isinstance(exc, GcpClientError):
        return exc

    if isinstance(exc, (TimeoutError, concurrent.futures.TimeoutError)):
        return GcpTimeoutError(f"Timed out calling {context}: {exc}")

    # Avoid eager top-level imports of google.api_core / requests / google.genai
    exc_type_name = type(exc).__name__
    exc_module = type(exc).__module__ or ""

    if exc_module.startswith("requests.exceptions") and "Timeout" in exc_type_name:
        return GcpTimeoutError(f"Timed out calling {context}: {exc}")
    if exc_module.startswith("httpx") and "Timeout" in exc_type_name:
        return GcpTimeoutError(f"Timed out calling {context}: {exc}")

    status_code, grpc_name = _extract_status_info(exc)
    retry_after = parse_retry_after(getattr(exc, "response", None))

    # google.api_core.exceptions, google.genai.errors.APIError, & raw grpc.RpcError
    if exc_type_name in ("TooManyRequests", "ResourceExhausted") or status_code == 429:
        return GcpRateLimitError(
            f"{context} rate limited (HTTP 429): {exc}",
            retry_after_seconds=retry_after,
        )
    if exc_type_name == "NotFound" or status_code == 404:
        return GcpNotFoundError(f"{context} resource not found (HTTP 404): {exc}", status_code=404)
    if (
        exc_type_name
        in (
            "DeadlineExceeded",
            "ServiceUnavailable",
            "InternalServerError",
            "GatewayTimeout",
            "BadGateway",
            "Aborted",
            "RetryError",
        )
        or grpc_name == "ABORTED"
        or status_code in (408, 500, 502, 503, 504)
    ):
        return GcpTimeoutError(
            f"{context} transient failure ({exc_type_name}): {exc}",
            status_code=status_code,
        )
    if status_code is not None and status_code >= 400:
        return GcpHttpError(status_code=status_code, url=context, response_text=str(exc)[:500])

    return GcpClientError(f"{context} failed ({exc_type_name}): {exc}", status_code=status_code)


__all__ = [
    "GcpClientError",
    "GcpHttpError",
    "GcpNotFoundError",
    "GcpRateLimitError",
    "GcpTimeoutError",
    "build_client_options",
    "parse_retry_after",
    "translate_gcp_sdk_error",
]
