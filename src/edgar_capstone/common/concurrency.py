"""Concurrency utilities with automatic ``contextvars`` propagation."""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")


def run_parallel_with_context(
    items: Sequence[_T],
    fn: Callable[[_T], _R],
    *,
    max_workers: int = 6,
) -> list[_R]:
    """Execute ``fn(item)`` in parallel while preserving the caller's ``contextvars`` (OTel spans).

    Preserves input order. For 0 or 1 items (or ``max_workers <= 1``), executes inline without
    spawning a thread pool.
    """
    if not items:
        return []
    if len(items) == 1 or max_workers <= 1:
        return [fn(item) for item in items]

    worker_count = min(max_workers, len(items))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(contextvars.copy_context().run, fn, item) for item in items]
        return [fut.result() for fut in futures]
