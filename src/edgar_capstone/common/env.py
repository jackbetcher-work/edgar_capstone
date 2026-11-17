"""Consistent, fail-safe environment variable parsing helpers."""

from __future__ import annotations

import os

TRUTHY_ENV_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})
FALSY_ENV_VALUES: frozenset[str] = frozenset({"0", "false", "no", "off"})


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable safely.

    Unlike naive ``raw.lower() in TRUTHY`` checks, empty strings and unrecognized values
    fall back to ``default`` rather than silently failing open when ``default=True``.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    norm = raw.strip().lower()
    if not norm:
        return default
    if norm in TRUTHY_ENV_VALUES:
        return True
    if norm in FALSY_ENV_VALUES:
        return False
    return default


def split_csv_env(raw: str | None) -> tuple[str, ...]:
    """Split a comma-separated environment string into trimmed, deduplicated entries."""
    if not raw:
        return ()
    seen: list[str] = []
    for item in raw.split(","):
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return tuple(seen)


def env_positive_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a positive integer env var, falling back to ``default`` if missing or invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed >= minimum else default


def env_positive_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    """Read a numeric float env var >= ``minimum``, falling back to ``default``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    return parsed if parsed >= minimum else default
