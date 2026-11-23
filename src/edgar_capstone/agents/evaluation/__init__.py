"""Claim verification, deterministic report rendering, and LLM-as-a-Judge evaluation."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edgar_capstone.agents.evaluation.judge import (
        evaluate_variance_explanation,
        evaluate_variance_explanation_with_latency,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "evaluate_variance_explanation": (
        "edgar_capstone.agents.evaluation.judge",
        "evaluate_variance_explanation",
    ),
    "evaluate_variance_explanation_with_latency": (
        "edgar_capstone.agents.evaluation.judge",
        "evaluate_variance_explanation_with_latency",
    ),
}

__all__ = [
    "evaluate_variance_explanation",
    "evaluate_variance_explanation_with_latency",
]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
