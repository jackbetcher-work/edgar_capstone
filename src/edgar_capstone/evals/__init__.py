"""Golden eval dataset + regression gate (`make eval-all`).

Import-light on purpose: nothing here imports `edgar_capstone.agents` at module load.
"""

from edgar_capstone.evals.golden import (
    DEFAULT_GOLDEN_DATASET_PATH,
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

__all__ = [
    "DEFAULT_GOLDEN_DATASET_PATH",
    "EvalRunReport",
    "EvalThresholds",
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
