"""Filesystem layout of the local ingestion staging area (`data/`).

Used dynamically by `dataset.pipeline`, `dataset.constituents`, `vector_db.pipeline`,
and the ingestion scripts (`build-sec-dataset`, `build-vector-db`, `backfill-rd-metric`)
to stage 10-K profiles, Markdown, and manifests under `data/curated_dataset/` before
publishing them to GCS, BigQuery, and Vertex AI Search. The serving runtime never reads
this directory; it queries those GCP services directly.

The data root is resolved from (in order) the `EDGAR_DATA_ROOT` override, the source
checkout (`<repo>/data`), or `<cwd>/data`, so the scripts work from an editable install,
a non-editable install, or an arbitrary working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

CURATED_DATASET_DIRNAME = "curated_dataset"
SP500_CONSTITUENTS_FILENAME = "sp500_constituents.csv"


def _source_tree_data_root() -> Path:
    """Return `<repo>/data` for source checkouts and editable installs."""
    return Path(__file__).resolve().parents[3] / "data"


def resolve_data_root() -> Path:
    """Locate the directory that holds `curated_dataset/` and `sp500_constituents.csv`.

    Resolution order:
      1. `EDGAR_DATA_ROOT` environment override.
      2. Source checkout (`<repo>/data`) when it exists.
      3. `<cwd>/data` when it exists.
      4. Fall back to the source-tree location (created on demand by build scripts).
    """
    env_root = os.environ.get("EDGAR_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    source_root = _source_tree_data_root()
    if source_root.is_dir():
        return source_root

    cwd_data = Path.cwd() / "data"
    if cwd_data.is_dir():
        return cwd_data.resolve()

    return source_root


def resolve_curated_dataset_dir() -> Path:
    """Return the active curated dataset directory (`EDGAR_DATASET_DIR` wins when set)."""
    env_dir = os.environ.get("EDGAR_DATASET_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return resolve_data_root() / CURATED_DATASET_DIRNAME


def resolve_sp500_constituents_csv() -> Path:
    """Return the default S&P 500 constituents CSV location."""
    return resolve_data_root() / SP500_CONSTITUENTS_FILENAME
