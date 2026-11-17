"""Golden eval dataset + regression gate (`make eval-all`).

Import-light on purpose: nothing here imports `edgar_capstone.agents` at module load, so
`configure_eval_environment()` can pin the execution env before the agents read it.
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
    configure_eval_environment,
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
    "configure_eval_environment",
    "dataset_fingerprint",
    "format_report",
    "load_golden_dataset",
    "run_golden_evals",
    "validate_golden_dataset",
]
