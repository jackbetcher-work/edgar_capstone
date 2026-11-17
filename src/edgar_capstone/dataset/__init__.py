"""Static SEC 10-K dataset curation and GCS ingestion pipeline.

Uses PEP 562 lazy attribute loading so importing lightweight submodules such as
``edgar_capstone.dataset.models`` never eagerly loads ``edgartools``, ``pandas``,
or ``google-cloud-storage``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edgar_capstone.dataset.constituents import (
        CURATED_STARTER_TICKERS,
        load_constituents,
        normalize_edgar_ticker,
    )
    from edgar_capstone.dataset.models import (
        CompanyConstituent,
        DatasetBuildSummary,
        ExtractedFilingBundle,
        FinancialDataProfile,
        MetricProvenance,
        VertexAISearchContent,
        VertexAISearchManifestRecord,
    )
    from edgar_capstone.dataset.pipeline import DatasetBuildConfig, run_dataset_pipeline

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CURATED_STARTER_TICKERS": ("edgar_capstone.dataset.constituents", "CURATED_STARTER_TICKERS"),
    "CompanyConstituent": ("edgar_capstone.dataset.models", "CompanyConstituent"),
    "DatasetBuildConfig": ("edgar_capstone.dataset.pipeline", "DatasetBuildConfig"),
    "DatasetBuildSummary": ("edgar_capstone.dataset.models", "DatasetBuildSummary"),
    "ExtractedFilingBundle": ("edgar_capstone.dataset.models", "ExtractedFilingBundle"),
    "FinancialDataProfile": ("edgar_capstone.dataset.models", "FinancialDataProfile"),
    "MetricProvenance": ("edgar_capstone.dataset.models", "MetricProvenance"),
    "VertexAISearchContent": ("edgar_capstone.dataset.models", "VertexAISearchContent"),
    "VertexAISearchManifestRecord": (
        "edgar_capstone.dataset.models",
        "VertexAISearchManifestRecord",
    ),
    "load_constituents": ("edgar_capstone.dataset.constituents", "load_constituents"),
    "normalize_edgar_ticker": ("edgar_capstone.dataset.constituents", "normalize_edgar_ticker"),
    "run_dataset_pipeline": ("edgar_capstone.dataset.pipeline", "run_dataset_pipeline"),
}

__all__ = [
    "CURATED_STARTER_TICKERS",
    "CompanyConstituent",
    "DatasetBuildConfig",
    "DatasetBuildSummary",
    "ExtractedFilingBundle",
    "FinancialDataProfile",
    "MetricProvenance",
    "VertexAISearchContent",
    "VertexAISearchManifestRecord",
    "load_constituents",
    "normalize_edgar_ticker",
    "run_dataset_pipeline",
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
