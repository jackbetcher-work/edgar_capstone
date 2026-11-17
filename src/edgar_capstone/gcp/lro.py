"""Generic Long-Running Operation (LRO) & job poller with per-poll OAuth2 token refresh."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_POLL_STATUSES = frozenset({429, 500, 502, 503, 504})


class OperationError(RuntimeError):
    """Raised when a polled GCP operation/job fails with an API or terminal error."""


class OperationTimeoutError(OperationError, TimeoutError):
    """Raised when a polled GCP operation/job does not finish before its deadline."""


def poll_operation(
    poll_fn: Callable[[], httpx.Response],
    is_done_fn: Callable[[dict[str, Any]], bool],
    *,
    label: str = "GCP operation",
    timeout_seconds: float = 600.0,
    initial_interval_seconds: float = 2.0,
    max_interval_seconds: float = 15.0,
    backoff_multiplier: float = 1.5,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll a GCP operation/job endpoint until ``is_done_fn(payload)`` returns True.

    - Invokes ``poll_fn()`` on every attempt so callers can acquire a fresh OAuth2 bearer
      token across long-running jobs (> 30–60 minutes).
    - Retries **only** transient HTTP 429/5xx or network timeout responses.
    - Fails fast on fatal HTTP 4xx errors (`401`, `403`, `404`, `400`) instead of spinning
      silently until timeout.
    """
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    interval = max(0.0, float(initial_interval_seconds))
    max_interval = max(interval, float(max_interval_seconds))

    while True:
        try:
            resp = poll_fn()
        except httpx.TimeoutException as exc:
            if time.monotonic() >= deadline:
                raise OperationTimeoutError(
                    f"Timed out polling {label} after {timeout_seconds:.0f}s: {exc}"
                ) from exc
            logger.warning("Transient timeout polling %s (%s); retrying...", label, exc)
            sleep_fn(min(interval, max(0.0, deadline - time.monotonic())))
            interval = min(max_interval, interval * backoff_multiplier)
            continue

        if resp.status_code in RETRYABLE_POLL_STATUSES:
            if time.monotonic() >= deadline:
                raise OperationTimeoutError(
                    f"Timed out polling {label} (last HTTP {resp.status_code}: {resp.text[:300]})"
                )
            logger.warning(
                "Transient HTTP %d polling %s; retrying in %.1fs...",
                resp.status_code,
                label,
                interval,
            )
            sleep_fn(min(interval, max(0.0, deadline - time.monotonic())))
            interval = min(max_interval, interval * backoff_multiplier)
            continue

        if resp.status_code >= 400:
            raise OperationError(
                f"{label} poll failed with fatal HTTP {resp.status_code}: {resp.text[:500]}"
            )

        payload = resp.json() if resp.content else {}
        if isinstance(payload, dict) and is_done_fn(payload):
            return payload

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OperationTimeoutError(
                f"Timed out waiting for {label} after {timeout_seconds:.0f}s."
            )
        sleep_fn(min(interval, remaining))
        interval = min(max_interval, interval * backoff_multiplier)
