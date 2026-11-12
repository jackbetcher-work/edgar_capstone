"""Standardized exception wrappers, rate-limit (HTTP 429) retry logic, and tool telemetry."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import functools
import inspect
import logging
import time
from typing import Any, TypeVar

import httpx

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class ToolExecutionError(RuntimeError):
    """Base exception for ADK tool execution failures."""


class RateLimitExceededError(ToolExecutionError):
    """Raised when an upstream API returns HTTP 429 Too Many Requests."""

    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class UpstreamTimeoutError(ToolExecutionError):
    """Raised when an external API or query times out."""


def _is_retryable_exception(exc: BaseException) -> tuple[bool, float | None]:
    """Determine if an exception is a transient rate limit (HTTP 429) or timeout."""
    if isinstance(exc, RateLimitExceededError):
        return True, exc.retry_after_seconds
    if isinstance(exc, (UpstreamTimeoutError, TimeoutError, httpx.TimeoutException)):
        return True, None
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (429, 500, 502, 503, 504):
            retry_after = exc.response.headers.get("Retry-After")
            delay: float | None = None
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = None
            return True, delay
    return False, None


def _record_tool_telemetry(
    tool_context: Any | None,
    *,
    tool_name: str,
    status: str,
    duration_ms: float,
    attempts: int,
    error_message: str | None = None,
) -> None:
    """Append tool execution telemetry to `tool_context.state['tool_telemetry']`."""
    if tool_context is None or not hasattr(tool_context, "state"):
        return
    try:
        state = tool_context.state
        telemetry_log = list(state.get("tool_telemetry", []))
        entry: dict[str, Any] = {
            "tool_name": tool_name,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "attempts": attempts,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if error_message:
            entry["error"] = error_message
        telemetry_log.append(entry)
        state["tool_telemetry"] = telemetry_log
    except Exception as exc:
        logger.debug("Skipping tool telemetry state update: %s", exc)


def with_tool_resilience(
    *,
    tool_name: str | None = None,
    max_retries: int = 3,
    base_delay_seconds: float = 0.25,
    max_delay_seconds: float = 8.0,
) -> Callable[[F], F]:
    """Decorator wrapping an ADK tool function with HTTP 429/timeout retries and telemetry.

    Preserves the original function signature (`__signature__`, `__name__`, `__doc__`)
    so Google ADK's `FunctionTool` schema introspection works without alteration.
    """

    def decorator(func: F) -> F:
        resolved_name = tool_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            start_ts = time.perf_counter()
            tool_ctx = kwargs.get("tool_context")
            if tool_ctx is None:
                for arg in args:
                    if hasattr(arg, "state") and hasattr(arg, "actions"):
                        tool_ctx = arg
                        break

            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                    _record_tool_telemetry(
                        tool_ctx,
                        tool_name=resolved_name,
                        status="success",
                        duration_ms=elapsed_ms,
                        attempts=attempt,
                    )
                    return result
                except Exception as exc:
                    retryable, explicit_delay = _is_retryable_exception(exc)
                    if retryable and attempt < max_retries:
                        backoff = (
                            explicit_delay
                            if explicit_delay is not None
                            else min(max_delay_seconds, base_delay_seconds * (2 ** (attempt - 1)))
                        )
                        logger.warning(
                            "Tool '%s' encountered transient error on attempt %d/%d (%s); "
                            "retrying in %.2fs...",
                            resolved_name,
                            attempt,
                            max_retries,
                            exc,
                            backoff,
                        )
                        time.sleep(backoff)
                        continue

                    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                    err_type = type(exc).__name__
                    err_msg = str(exc)
                    logger.error(
                        "Tool '%s' failed after %d attempt(s) (%.1f ms): [%s] %s",
                        resolved_name,
                        attempt,
                        elapsed_ms,
                        err_type,
                        err_msg,
                    )
                    _record_tool_telemetry(
                        tool_ctx,
                        tool_name=resolved_name,
                        status="error",
                        duration_ms=elapsed_ms,
                        attempts=attempt,
                        error_message=f"{err_type}: {err_msg}",
                    )
                    return {
                        "status": "error",
                        "tool_name": resolved_name,
                        "error_type": err_type,
                        "message": err_msg,
                        "attempts": attempt,
                    }

            return {
                "status": "error",
                "tool_name": resolved_name,
                "error_type": "MaxRetriesExceeded",
                "message": f"Exhausted {max_retries} retries for {resolved_name}",
            }

        wrapper.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
