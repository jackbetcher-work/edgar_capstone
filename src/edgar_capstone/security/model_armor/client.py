"""Regional Google Cloud Model Armor v1 SDK client."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Any, Literal, TypeVar

from edgar_capstone.common.env import env_bool
from edgar_capstone.gcp.errors import (
    GcpRateLimitError,
    GcpTimeoutError,
    build_client_options,
    translate_gcp_sdk_error,
)
from edgar_capstone.security.model_armor.config import (
    ModelArmorConfig,
    ModelArmorSanitizationSummary,
    parse_model_armor_proto,
)
from edgar_capstone.security.model_armor.emulator import LocalModelArmorEmulator

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


def _is_retryable_model_armor_error(exc: BaseException) -> bool:
    from edgar_capstone.agents.core.resilience import RateLimitExceededError, UpstreamTimeoutError

    return isinstance(
        exc,
        (
            GcpRateLimitError,
            GcpTimeoutError,
            RateLimitExceededError,
            UpstreamTimeoutError,
            TimeoutError,
        ),
    )


class ModelArmorClient:
    """Regional Google Cloud Model Armor v1 client backed by ``google-cloud-modelarmor``."""

    def __init__(
        self,
        config: ModelArmorConfig | None = None,
        *,
        sdk_client: Any | None = None,
        credentials: Any | None = None,
    ) -> None:
        self.config = config or ModelArmorConfig.from_env()
        self._injected_client = sdk_client
        self._owned_client: Any | None = None
        self._credentials = credentials
        self._client_lock = threading.Lock()
        self._emulator = LocalModelArmorEmulator()

    def _get_sdk_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        with self._client_lock:
            if self._owned_client is None:
                from google.cloud import modelarmor_v1

                opts = build_client_options(
                    api_endpoint=self.config.regional_api_host,
                    quota_project_id=self.config.project_id,
                )
                kwargs: dict[str, Any] = {"client_options": opts}
                if self._credentials is not None:
                    kwargs["credentials"] = self._credentials
                self._owned_client = modelarmor_v1.ModelArmorClient(**kwargs)
            return self._owned_client

    def close(self) -> None:
        """Close the internally managed SDK transport (leaves injected clients open)."""
        with self._client_lock:
            if self._owned_client is not None:
                transport = getattr(self._owned_client, "transport", None)
                close_fn = getattr(transport, "close", None)
                if callable(close_fn):
                    close_fn()
                self._owned_client = None

    def __enter__(self) -> ModelArmorClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _should_use_live_gcp(self) -> bool:
        if self.config.mode in ("disabled", "emulator"):
            return False
        if self.config.mode == "gcp":
            return True
        if self._injected_client is not None or self._credentials is not None:
            return True
        return env_bool("GOOGLE_GENAI_USE_VERTEXAI", False)

    @property
    def is_live_gcp_active(self) -> bool:
        """Return True when live Google Cloud Model Armor SDK calls are enabled."""
        return self._should_use_live_gcp()

    def _sanitize(
        self,
        stage: Literal["prompt", "response"],
        text: str,
    ) -> ModelArmorSanitizationSummary:
        from google.cloud import modelarmor_v1

        if stage == "prompt":
            request: Any = modelarmor_v1.SanitizeUserPromptRequest(
                name=self.config.template_resource_name,
                user_prompt_data=modelarmor_v1.DataItem(text=text),
            )
            op_name = "model_armor_sanitize_user_prompt"
            label = "Model Armor sanitizeUserPrompt"
            method_name = "sanitize_user_prompt"
        else:
            request = modelarmor_v1.SanitizeModelResponseRequest(
                name=self.config.template_resource_name,
                model_response_data=modelarmor_v1.DataItem(text=text),
            )
            op_name = "model_armor_sanitize_model_response"
            label = "Model Armor sanitizeModelResponse"
            method_name = "sanitize_model_response"

        def _invoke() -> ModelArmorSanitizationSummary:
            if not self._should_use_live_gcp():
                resp = getattr(self._emulator, method_name)(request)
                return parse_model_armor_proto(resp, backend_used="emulator_model_armor_v1")

            from edgar_capstone.agents.core.resilience import ToolExecutionError

            try:
                client = self._get_sdk_client()
                try:
                    resp = getattr(client, method_name)(
                        request=request,
                        timeout=self.config.timeout_seconds,
                    )
                except Exception as exc:
                    raise translate_gcp_sdk_error(exc, context=label) from exc
                return parse_model_armor_proto(resp, backend_used="gcp_model_armor_v1")
            except Exception as exc:
                if _is_retryable_model_armor_error(exc):
                    raise
                if (
                    self.config.mode == "auto"
                    and self._injected_client is None
                    and self._credentials is None
                ):
                    logger.warning(
                        "%s live GCP call unavailable in auto mode (%s); routing via Model Armor emulator.",
                        label,
                        type(exc).__name__,
                    )
                    resp = getattr(self._emulator, method_name)(request)
                    return parse_model_armor_proto(resp, backend_used="emulator_model_armor_v1")
                raise ToolExecutionError(f"{label} request failed: {exc}") from exc

        return _call_with_retries(op_name, _invoke)

    def sanitize_user_prompt(self, text: str) -> ModelArmorSanitizationSummary:
        """Invoke Google Cloud Model Armor ``templates.sanitizeUserPrompt``."""
        return self._sanitize("prompt", text)

    def sanitize_model_response(self, text: str) -> ModelArmorSanitizationSummary:
        """Invoke Google Cloud Model Armor ``templates.sanitizeModelResponse``."""
        return self._sanitize("response", text)
