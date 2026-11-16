"""Filesystem layout resolution for the curated SEC 10-K corpus.

The package can run from three layouts:

1. A source checkout / editable install (`<repo>/src/edgar_capstone/...`), where
   `<repo>/data/` sits three directories above this module.
2. A non-editable install inside the production container
   (`/opt/venv/lib/python3.11/site-packages/edgar_capstone/...`), where walking up
   from `__file__` lands inside the virtualenv instead of `/app/data/`.
3. An arbitrary working directory containing a `data/` folder.

Resolving the data root by probing these candidates (with explicit environment
overrides taking precedence) keeps the baked container manifests and any
GCS-hydrated fallback artifacts discoverable regardless of how the package was
installed.
"""

from __future__ import annotations

import os
from pathlib import Path

CONTAINER_DATA_ROOT = Path("/app/data")
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
      3. Container layout (`/app/data`) when it exists.
      4. `<cwd>/data` when it exists.
      5. Fall back to the source-tree location (created on demand by build scripts).
    """
    env_root = os.environ.get("EDGAR_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    source_root = _source_tree_data_root()
    if source_root.is_dir():
        return source_root

    for candidate in (CONTAINER_DATA_ROOT, Path.cwd() / "data"):
        if candidate.is_dir():
            return candidate.resolve()

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
