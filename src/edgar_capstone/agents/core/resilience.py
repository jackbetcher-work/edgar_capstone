"""Standardized exception wrappers, rate-limit (HTTP 429) retry logic, and tool telemetry."""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, ParamSpec, TypeVar

import httpx

from edgar_capstone.gcp.rest_client import GcpRateLimitError, GcpTimeoutError, parse_retry_after
from edgar_capstone.observability.tracing import (
    add_span_event,
    mark_span_error,
    set_span_attributes,
    start_span,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")
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
    """Determine if an exception is a transient rate limit (HTTP 429), 5xx, or timeout."""
    if isinstance(exc, (RateLimitExceededError, GcpRateLimitError)):
        return True, exc.retry_after_seconds
    if hasattr(exc, "retry_after_seconds"):
        return True, getattr(exc, "retry_after_seconds", None)
    if isinstance(
        exc, (UpstreamTimeoutError, GcpTimeoutError, TimeoutError, httpx.TimeoutException)
    ):
        return True, None
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (429, 500, 502, 503, 504):
            return True, parse_retry_after(exc.response)
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
    """Append tool execution telemetry to ``tool_context.state['tool_telemetry']``."""
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
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if error_message:
            entry["error"] = error_message
        telemetry_log.append(entry)
        state["tool_telemetry"] = telemetry_log
    except Exception as exc:
        logger.debug("Skipping tool telemetry state update: %s", exc)


def retry_transient_errors(
    *,
    operation_name: str | None = None,
    max_retries: int = 3,
    base_delay_seconds: float = 0.25,
    max_delay_seconds: float = 8.0,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry transient HTTP 429/5xx/timeout errors with exponential backoff and re-raise on failure.

    Unlike ``with_tool_resilience`` (which converts final failures into ADK tool error dicts),
    this decorator preserves the wrapped function's return type and propagates exceptions to
    callers (e.g., internal RPC clients such as Model Armor, DLP, GCS/Vertex prompt stores).
    """
    eff_retries = max(1, int(max_retries))

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        resolved_name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with start_span(
                f"rpc.{resolved_name}",
                attributes={"edgar.rpc.name": resolved_name, "code.function": func.__qualname__},
            ) as span:
                for attempt in range(1, eff_retries + 1):
                    try:
                        result = func(*args, **kwargs)
                        set_span_attributes(
                            span,
                            **{"edgar.rpc.attempts": attempt, "edgar.rpc.status": "success"},
                        )
                        return result
                    except Exception as exc:
                        retryable, explicit_delay = _is_retryable_exception(exc)
                        if retryable and attempt < eff_retries:
                            backoff = (
                                explicit_delay
                                if explicit_delay is not None
                                else min(
                                    max_delay_seconds,
                                    base_delay_seconds * (2 ** (attempt - 1)),
                                )
                            )
                            logger.warning(
                                "RPC '%s' encountered transient error on attempt %d/%d (%s); "
                                "retrying in %.2fs...",
                                resolved_name,
                                attempt,
                                eff_retries,
                                exc,
                                backoff,
                            )
                            add_span_event(
                                "rpc.retry",
                                attempt=attempt,
                                backoff_seconds=backoff,
                                error_type=type(exc).__name__,
                            )
                            time.sleep(backoff)
                            continue
                        set_span_attributes(
                            span,
                            **{"edgar.rpc.attempts": attempt, "edgar.rpc.status": "error"},
                        )
                        mark_span_error(
                            str(exc),
                            span,
                            **{"edgar.rpc.error_type": type(exc).__name__},
                        )
                        raise
                raise AssertionError("unreachable")  # pragma: no cover

        return wrapper

    return decorator


def with_tool_resilience(
    *,
    tool_name: str | None = None,
    max_retries: int = 3,
    base_delay_seconds: float = 0.25,
    max_delay_seconds: float = 8.0,
) -> Callable[[F], F]:
    """Decorator wrapping an ADK tool function with HTTP 429/timeout retries and telemetry.

    Each invocation runs inside an OpenTelemetry ``tool.<name>`` span (child of the ADK
    ``execute_tool`` span or of the chat-turn span) that records attempts, retry events,
    and the final status for Cloud Trace.

    Preserves the original function signature (``__signature__``, ``__name__``, ``__doc__``)
    so Google ADK's ``FunctionTool`` schema introspection works without alteration.
    """
    eff_retries = max(1, int(max_retries))

    def decorator(func: F) -> F:
        resolved_name = tool_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            with start_span(
                f"tool.{resolved_name}",
                attributes={"edgar.tool.name": resolved_name, "code.function": func.__qualname__},
            ) as span:
                result = _invoke_with_retries(*args, **kwargs)
                status = (
                    str(result.get("status", "success")) if isinstance(result, dict) else "success"
                )
                set_span_attributes(span, **{"edgar.tool.status": status})
                if status == "error" and isinstance(result, dict):
                    mark_span_error(
                        str(result.get("message") or "tool error"),
                        span,
                        **{"edgar.tool.error_type": result.get("error_type")},
                    )
                return result

        def _invoke_with_retries(*args: Any, **kwargs: Any) -> dict[str, Any]:
            start_ts = time.perf_counter()
            tool_ctx = kwargs.get("tool_context")
            if tool_ctx is None:
                for arg in args:
                    if hasattr(arg, "state") and hasattr(arg, "actions"):
                        tool_ctx = arg
                        break

            for attempt in range(1, eff_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                    set_span_attributes(**{"edgar.tool.attempts": attempt})
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
                    if retryable and attempt < eff_retries:
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
                            eff_retries,
                            exc,
                            backoff,
                        )
                        add_span_event(
                            "tool.retry",
                            attempt=attempt,
                            backoff_seconds=backoff,
                            error_type=type(exc).__name__,
                        )
                        time.sleep(backoff)
                        continue

                    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                    err_type = type(exc).__name__
                    err_msg = str(exc)
                    set_span_attributes(**{"edgar.tool.attempts": attempt})
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
            raise AssertionError("unreachable")  # pragma: no cover

        wrapper.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
