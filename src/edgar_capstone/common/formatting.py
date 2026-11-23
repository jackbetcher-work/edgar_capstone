"""Shared numeric formatting, percentage/ratio math, statistics, and text helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

_M = TypeVar("_M", bound=BaseModel)

_QUOTE_TRANSLATE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)
_WHITESPACE_RE = re.compile(r"\s+")
_MD_ESCAPE_RE = re.compile(r"\\([.\-(),&$%])")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL | re.IGNORECASE)


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 ``Z`` format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_usd(value: float | int | None) -> str:
    """Format a USD figure concisely using T / B / M / K suffixes with thousand separators."""
    if value is None:
        return "N/A"
    num = float(value)
    sign = "-" if num < 0 else ""
    mag = abs(num)
    if mag >= 1_000_000_000_000:
        return f"{sign}${mag / 1_000_000_000_000:,.2f}T"
    if mag >= 1_000_000_000:
        return f"{sign}${mag / 1_000_000_000:,.2f}B"
    if mag >= 1_000_000:
        return f"{sign}${mag / 1_000_000:,.2f}M"
    if mag >= 1_000:
        return f"{sign}${mag / 1_000:,.2f}K"
    return f"{sign}${mag:,.0f}"


def fmt_pct(value: float | int | None, *, signed: bool = True) -> str:
    """Format a percentage change or value (e.g. ``+12.34%`` or ``12.34%``)."""
    if value is None:
        return "N/A"
    num = float(value)
    if signed:
        if num > 0:
            return f"+{num:.2f}%"
        return f"{num:.2f}%"
    return f"{num:.2f}%"


def fmt_bps(value: float | int | None, *, signed: bool = True) -> str:
    """Format a basis-points delta (e.g. ``+210 bps`` or ``-45 bps``)."""
    if value is None:
        return "N/A"
    num = float(value)
    if signed and num > 0:
        return f"+{num:.0f} bps"
    return f"{num:.0f} bps"


def pct_change(start: float | None, end: float | None, *, ndigits: int = 2) -> float | None:
    """Compute ``(end - start) / |start| * 100`` safely (returns ``None`` when undefined)."""
    if start is None or end is None or start == 0:
        return None
    return round(((end - start) / abs(start)) * 100.0, ndigits)


def ratio_pct(
    numerator: float | None, denominator: float | None, *, ndigits: int = 2
) -> float | None:
    """Compute ``numerator / denominator * 100`` safely (returns ``None`` when denominator <= 0)."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round((numerator / denominator) * 100.0, ndigits)


def margin_bps_delta(start_pct: float | None, end_pct: float | None) -> float | None:
    """Compute basis-points change ``(end_pct - start_pct) * 100`` rounded to 1 decimal."""
    if start_pct is None or end_pct is None:
        return None
    return round((end_pct - start_pct) * 100.0, 1)


def percentile(values: Sequence[float], pct: float) -> float:
    """Compute linear-interpolated percentile ``pct`` in ``[0, 100]`` over a sequence of floats."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] + frac * (ordered[hi] - ordered[lo]))


def normalize_quote_text(text: str) -> str:
    """Normalize smart punctuation, dashes, markdown escapes, and whitespace for quote matching."""
    cleaned = (text or "").translate(_QUOTE_TRANSLATE)
    cleaned = _MD_ESCAPE_RE.sub(r"\1", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip().lower()
    return cleaned


def _extract_json_dict(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if not text:
        return None
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    elif text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
            text = "\n".join(body).strip()
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        try:
            loaded = json.loads(text[first : last + 1])
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            return None
    return None


def parse_model_output(
    raw: Any,
    model_cls: type[_M],
    *,
    fallback_text: str | None = None,
) -> _M | None:
    """Validate a structured LLM response (instance, dict, or JSON string) with text fallback.

    Unlike naive ``if ... elif`` chains, when ``raw`` fails validation this still attempts
    ``fallback_text`` before returning ``None``.
    """
    if isinstance(raw, model_cls):
        return raw
    if isinstance(raw, dict):
        try:
            return model_cls.model_validate(raw)
        except ValidationError:
            pass
    elif isinstance(raw, str) and raw.strip():
        candidate = _extract_json_dict(raw)
        if candidate is not None:
            try:
                return model_cls.model_validate(candidate)
            except ValidationError:
                pass
    if fallback_text and fallback_text.strip():
        candidate = _extract_json_dict(fallback_text)
        if candidate is not None:
            try:
                return model_cls.model_validate(candidate)
            except ValidationError:
                return None
    return None
