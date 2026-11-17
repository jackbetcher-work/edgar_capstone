"""Thread-safe ADC OAuth2 bearer auth and pooled ``httpx.Client`` base transport for GCP REST APIs."""

from __future__ import annotations

import threading
from types import TracebackType
from typing import Any

import google.auth
import httpx
from google.auth.exceptions import DefaultCredentialsError
from google.auth.transport.requests import Request as GoogleAuthRequest

from edgar_capstone.gcp.defaults import CLOUD_PLATFORM_SCOPE


class GcpRestError(RuntimeError):
    """Base exception for GCP REST API failures."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GcpHttpError(GcpRestError):
    """Raised when a GCP REST API returns a non-retryable HTTP 4xx/5xx error."""

    def __init__(self, status_code: int, url: str, response_text: str) -> None:
        super().__init__(
            f"GCP REST request failed (HTTP {status_code}) for {url}: {response_text}",
            status_code=status_code,
        )
        self.url = url
        self.response_text = response_text


class GcpNotFoundError(GcpRestError):
    """Raised when a GCP REST resource returns HTTP 404 Not Found."""


class GcpRateLimitError(GcpRestError):
    """Raised when a GCP REST API returns HTTP 429 Too Many Requests."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after_seconds = retry_after_seconds


class GcpTimeoutError(GcpRestError, TimeoutError):
    """Raised when a GCP REST API request times out."""


def parse_retry_after(resp: httpx.Response) -> float | None:
    """Parse the numeric HTTP ``Retry-After`` header in seconds (if present and valid)."""
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


class GcpRestClient:
    """Reusable thread-safe ADC bearer authentication and pooled ``httpx.Client`` base class.

    Unifies credential refresh locking, empty-token guards, connection pooling, optional
    ``x-goog-user-project`` quota header injection, resource cleanup (``close()`` / context
    manager), and HTTP status translation across BigQuery, Firestore, Vertex AI Search,
    Vertex AI embeddings, GCS REST, Model Armor, DLP, Secret Manager, and IAM Credentials.
    """

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
        default_timeout: float = 30.0,
        scopes: tuple[str, ...] = (CLOUD_PLATFORM_SCOPE,),
        quota_project_id: str | None = None,
        max_connections: int = 48,
        max_keepalive_connections: int = 24,
    ) -> None:
        self._http_client = http_client
        self._pooled_http_client: httpx.Client | None = None
        self._credentials = credentials
        self._default_timeout = float(default_timeout)
        self._scopes = list(scopes)
        self._quota_project_id = (quota_project_id or "").strip() or None
        self._max_connections = max_connections
        self._max_keepalive_connections = max_keepalive_connections
        self._auth_lock = threading.Lock()
        self._client_lock = threading.Lock()

    def _get_http_client(self) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        with self._client_lock:
            if self._pooled_http_client is None or self._pooled_http_client.is_closed:
                self._pooled_http_client = httpx.Client(
                    timeout=self._default_timeout,
                    limits=httpx.Limits(
                        max_connections=self._max_connections,
                        max_keepalive_connections=self._max_keepalive_connections,
                    ),
                )
            return self._pooled_http_client

    def close(self) -> None:
        """Close the internally managed pooled ``httpx.Client`` (leaves injected clients open)."""
        with self._client_lock:
            if self._pooled_http_client is not None and not self._pooled_http_client.is_closed:
                self._pooled_http_client.close()
            self._pooled_http_client = None

    def __enter__(self) -> GcpRestClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _get_bearer_token(self) -> str:
        """Obtain a valid OAuth2 bearer token from ADC with thread-safe refresh."""
        with self._auth_lock:
            if self._credentials is None:
                self._credentials, _ = google.auth.default(scopes=self._scopes)
            if not getattr(self._credentials, "valid", False) or not getattr(
                self._credentials, "token", None
            ):
                self._credentials.refresh(GoogleAuthRequest())
            token = getattr(self._credentials, "token", None)
        if not token:
            raise DefaultCredentialsError(
                "Application Default Credentials did not return an OAuth2 bearer token."
            )
        return str(token)

    def _build_auth_headers(
        self,
        *,
        extra_headers: dict[str, str] | None = None,
        has_json_body: bool = False,
        quota_project_id: str | None = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {"Authorization": f"Bearer {self._get_bearer_token()}"}
        qp = (quota_project_id if quota_project_id is not None else self._quota_project_id) or ""
        if qp:
            headers["x-goog-user-project"] = qp
        if has_json_body:
            headers["Content-Type"] = "application/json"
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _raw_request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        content: bytes | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        quota_project_id: str | None = None,
    ) -> httpx.Response:
        req_headers = self._build_auth_headers(
            extra_headers=headers,
            has_json_body=json_body is not None,
            quota_project_id=quota_project_id,
        )
        client = self._get_http_client()
        kwargs: dict[str, Any] = {
            "json": json_body,
            "content": content,
            "params": params,
            "headers": req_headers,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        return client.request(method, url, **kwargs)

    def _send_request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        content: bytes | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        quota_project_id: str | None = None,
        raise_5xx_as_httpx: bool = False,
        not_found_as_dedicated_error: bool = False,
    ) -> httpx.Response:
        """Execute an authenticated HTTP request and translate timeouts / 429 / 404 / 4xx / 5xx."""
        try:
            resp = self._raw_request(
                method,
                url,
                json_body=json_body,
                content=content,
                params=params,
                headers=headers,
                timeout=timeout,
                quota_project_id=quota_project_id,
            )
        except httpx.TimeoutException as exc:
            raise GcpTimeoutError(f"Timed out calling GCP endpoint {url}: {exc}") from exc

        if resp.status_code == 429:
            raise GcpRateLimitError(
                f"GCP API HTTP 429 Too Many Requests for {url}: {resp.text[:300]}",
                retry_after_seconds=parse_retry_after(resp),
            )
        if not_found_as_dedicated_error and resp.status_code == 404:
            raise GcpNotFoundError(
                f"GCP REST resource not found (HTTP 404) for {url}: {resp.text[:300]}",
                status_code=404,
            )
        if raise_5xx_as_httpx and resp.status_code in (500, 502, 503, 504):
            resp.raise_for_status()
        if resp.status_code >= 400:
            raise GcpHttpError(
                status_code=resp.status_code,
                url=url,
                response_text=resp.text[:500],
            )
        return resp

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        content: bytes | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        quota_project_id: str | None = None,
    ) -> Any:
        resp = self._send_request(
            method,
            url,
            json_body=json_body,
            content=content,
            params=params,
            headers=headers,
            timeout=timeout,
            quota_project_id=quota_project_id,
        )
        if not resp.content:
            return {}
        return resp.json()

    def _request_text(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> str:
        resp = self._send_request(
            method,
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        return resp.text
