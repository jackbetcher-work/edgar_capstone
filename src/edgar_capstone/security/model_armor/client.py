"""Regional Google Cloud Model Armor v1 REST client."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal, TypeVar

import httpx

from edgar_capstone.common.env import env_bool
from edgar_capstone.gcp.rest_client import (
    GcpHttpError,
    GcpRateLimitError,
    GcpRestClient,
    GcpTimeoutError,
    parse_retry_after,
)
from edgar_capstone.security.model_armor.config import (
    ModelArmorConfig,
    ModelArmorSanitizationSummary,
    parse_model_armor_response,
)
from edgar_capstone.security.model_armor.emulator import LocalModelArmorEmulatorTransport

logger = logging.getLogger(__name__)

_R = TypeVar("_R")


def _call_with_retries(
    operation_name: str,
    fn: Callable[[], _R],
    *,
    max_retries: int = 3,
    base_delay_seconds: float = 0.05,
) -> _R:
    """Invoke ``fn`` wrapped with ``retry_transient_errors`` (lazy import avoids package cycle)."""
    from edgar_capstone.agents.core.resilience import retry_transient_errors

    wrapped = retry_transient_errors(
        operation_name=operation_name,
        max_retries=max_retries,
        base_delay_seconds=base_delay_seconds,
    )(fn)
    return wrapped()


def _raise_for_security_http_status(resp: httpx.Response, service_label: str) -> None:
    if resp.status_code < 400:
        return
    snippet = resp.text[:300]
    if resp.status_code == 429:
        raise GcpRateLimitError(
            f"{service_label} HTTP 429 rate limited: {snippet}",
            retry_after_seconds=parse_retry_after(resp),
        )
    if resp.status_code in (408, 500, 502, 503, 504):
        raise GcpTimeoutError(f"{service_label} transient HTTP {resp.status_code}: {snippet}")
    raise GcpHttpError(
        status_code=resp.status_code,
        url=str(resp.request.url) if resp.request is not None else service_label,
        response_text=snippet,
    )


def _is_retryable_dlp_error(exc: BaseException) -> bool:
    from edgar_capstone.agents.core.resilience import RateLimitExceededError, UpstreamTimeoutError

    if isinstance(
        exc,
        (
            GcpRateLimitError,
            GcpTimeoutError,
            RateLimitExceededError,
            UpstreamTimeoutError,
            TimeoutError,
            httpx.TimeoutException,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


class ModelArmorClient(GcpRestClient):
    """Regional Google Cloud Model Armor v1 & Cloud SDP/DLP REST client with S2S auth & retries."""

    def __init__(
        self,
        config: ModelArmorConfig | None = None,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        self.config = config or ModelArmorConfig.from_env()
        super().__init__(
            http_client=http_client,
            credentials=credentials,
            default_timeout=self.config.timeout_seconds,
            quota_project_id=self.config.project_id,
            max_connections=32,
            max_keepalive_connections=16,
        )
        self._emulator_client: httpx.Client | None = None

    def _get_emulator_client(self) -> httpx.Client:
        """Lazily initialize the offline wire-emulator client only when emulator mode is needed."""
        with self._client_lock:
            if self._emulator_client is None or self._emulator_client.is_closed:
                self._emulator_client = httpx.Client(transport=LocalModelArmorEmulatorTransport())
            return self._emulator_client

    def close(self) -> None:
        """Close both the pooled live GCP client and any lazily created emulator client."""
        super().close()
        with self._client_lock:
            if self._emulator_client is not None and not self._emulator_client.is_closed:
                self._emulator_client.close()
            self._emulator_client = None

    def _should_use_live_gcp(self) -> bool:
        if self.config.mode in ("disabled", "emulator"):
            return False
        if self.config.mode in ("gcp", "live"):
            return True
        if self._http_client is not None or self._credentials is not None:
            return True
        return env_bool("GOOGLE_GENAI_USE_VERTEXAI", False)

    @property
    def is_live_gcp_active(self) -> bool:
        """Return True when live Google Cloud Model Armor & DLP REST calls are enabled."""
        return self._should_use_live_gcp()

    def _get_auth_token(self, *, target_service: str = "model-armor-service") -> str:
        _ = target_service
        return self._get_bearer_token()

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        service_label: str,
        target_service: str,
    ) -> tuple[dict[str, Any], str]:
        if not self._should_use_live_gcp():
            resp = self._get_emulator_client().post(url, json=payload)
            _raise_for_security_http_status(resp, f"{service_label} (Emulator)")
            return resp.json(), "emulator_model_armor_v1"

        from edgar_capstone.agents.core.resilience import ToolExecutionError

        try:
            token = self._get_auth_token(target_service=target_service)
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                # User ADC (developer shells, local CI runs) has no quota project, and the
                # DLP API rejects such calls; bill them to the configured project explicitly.
                "x-goog-user-project": self.config.project_id,
            }
            client = self._get_http_client()
            try:
                resp = client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException as exc:
                raise GcpTimeoutError(f"{service_label} timed out calling {url}: {exc}") from exc
            _raise_for_security_http_status(resp, service_label)
            return resp.json(), "gcp_model_armor_v1"
        except Exception as exc:
            if _is_retryable_dlp_error(exc):
                raise
            if (
                self.config.mode == "auto"
                and self._http_client is None
                and self._credentials is None
            ):
                logger.warning(
                    "%s live GCP call unavailable in auto mode (%s); routing via Model Armor wire emulator.",
                    service_label,
                    type(exc).__name__,
                )
                resp = self._get_emulator_client().post(url, json=payload)
                _raise_for_security_http_status(resp, f"{service_label} (Emulator)")
                return resp.json(), "emulator_model_armor_v1"
            raise ToolExecutionError(f"{service_label} request failed: {exc}") from exc

    def _sanitize(
        self,
        stage: Literal["prompt", "response"],
        text: str,
    ) -> ModelArmorSanitizationSummary:
        if stage == "prompt":
            rpc_suffix = "sanitizeUserPrompt"
            payload = {"userPromptData": {"text": text}}
            op_name = "model_armor_sanitize_user_prompt"
            label = "Model Armor sanitizeUserPrompt"
        else:
            rpc_suffix = "sanitizeModelResponse"
            payload = {"modelResponseData": {"text": text}}
            op_name = "model_armor_sanitize_model_response"
            label = "Model Armor sanitizeModelResponse"

        url = (
            f"{self.config.regional_api_endpoint}/v1/"
            f"{self.config.template_resource_name}:{rpc_suffix}"
        )

        def _invoke() -> ModelArmorSanitizationSummary:
            raw, backend = self._post_json(
                url,
                payload,
                service_label=label,
                target_service="model-armor-service",
            )
            return parse_model_armor_response(raw, backend_used=backend)

        return _call_with_retries(op_name, _invoke)

    def sanitize_user_prompt(self, text: str) -> ModelArmorSanitizationSummary:
        """Invoke Google Cloud Model Armor `templates.sanitizeUserPrompt`."""
        return self._sanitize("prompt", text)

    def sanitize_model_response(self, text: str) -> ModelArmorSanitizationSummary:
        """Invoke Google Cloud Model Armor `templates.sanitizeModelResponse`."""
        return self._sanitize("response", text)
