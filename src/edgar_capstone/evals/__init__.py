"""Golden eval dataset + regression gate (`make eval-all`).

Import-light on purpose: uses PEP 562 lazy loading and nothing here imports
`edgar_capstone.agents` at module load.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edgar_capstone.evals.golden import (
        DEFAULT_GOLDEN_DATASET_PATH,
        ExpectedDelta,
        ExpectedFlag,
        ExpectedSection,
        ExpectedVariance,
        GoldenDatasetError,
        GoldenEvalItem,
        RequestFilters,
        dataset_fingerprint,
        load_golden_dataset,
        validate_golden_dataset,
    )
    from edgar_capstone.evals.runner import (
        EvalRunReport,
        EvalThresholds,
        format_report,
        run_golden_evals,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_GOLDEN_DATASET_PATH": (
        "edgar_capstone.evals.golden",
        "DEFAULT_GOLDEN_DATASET_PATH",
    ),
    "EvalRunReport": ("edgar_capstone.evals.runner", "EvalRunReport"),
    "EvalThresholds": ("edgar_capstone.evals.runner", "EvalThresholds"),
    "ExpectedDelta": ("edgar_capstone.evals.golden", "ExpectedDelta"),
    "ExpectedFlag": ("edgar_capstone.evals.golden", "ExpectedFlag"),
    "ExpectedSection": ("edgar_capstone.evals.golden", "ExpectedSection"),
    "ExpectedVariance": ("edgar_capstone.evals.golden", "ExpectedVariance"),
    "GoldenDatasetError": ("edgar_capstone.evals.golden", "GoldenDatasetError"),
    "GoldenEvalItem": ("edgar_capstone.evals.golden", "GoldenEvalItem"),
    "RequestFilters": ("edgar_capstone.evals.golden", "RequestFilters"),
    "dataset_fingerprint": ("edgar_capstone.evals.golden", "dataset_fingerprint"),
    "format_report": ("edgar_capstone.evals.runner", "format_report"),
    "load_golden_dataset": ("edgar_capstone.evals.golden", "load_golden_dataset"),
    "run_golden_evals": ("edgar_capstone.evals.runner", "run_golden_evals"),
    "validate_golden_dataset": ("edgar_capstone.evals.golden", "validate_golden_dataset"),
}

__all__ = [
    "DEFAULT_GOLDEN_DATASET_PATH",
    "EvalRunReport",
    "EvalThresholds",
    "ExpectedDelta",
    "ExpectedFlag",
    "ExpectedSection",
    "ExpectedVariance",
    "GoldenDatasetError",
    "GoldenEvalItem",
    "RequestFilters",
    "dataset_fingerprint",
    "format_report",
    "load_golden_dataset",
    "run_golden_evals",
    "validate_golden_dataset",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
