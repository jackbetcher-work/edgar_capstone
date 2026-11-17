"""Shared CLI helpers for repository scripts (`scripts/*.py`).

Importing this module automatically calls `ensure_src_on_path()` so top-level
`from edgar_capstone...` imports work cleanly both from editable/wheel installs
and from raw source checkouts without needing per-import `# noqa: E402` comments.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def ensure_src_on_path() -> Path:
    """Ensure `<repo>/src` is importable only when `edgar_capstone` is not already installed.

    Preserves the installed package inside containers (e.g. `/opt/venv/lib/.../site-packages`)
    so in-process checks exercise the exact packaged code.
    """
    if importlib.util.find_spec("edgar_capstone") is None and SRC_DIR.is_dir():
        src_str = str(SRC_DIR)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
    if importlib.util.find_spec("scripts") is None:
        root_str = str(PROJECT_ROOT)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
    return PROJECT_ROOT


# Execute at import time so scripts can import `edgar_capstone.*` at the top of the file.
ensure_src_on_path()

from edgar_capstone.common.domain import normalize_edgar_ticker  # noqa: E402
from edgar_capstone.gcp.defaults import (  # noqa: E402
    DEFAULT_BIGQUERY_DATASET_ID,
    DEFAULT_BUCKET_NAME,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
)


def parse_ticker_args(raw_tickers: Iterable[str] | None) -> list[str] | None:
    """Normalize space- or comma-separated ticker arguments (e.g., `BRK.B` -> `BRK-B`)."""
    if not raw_tickers:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_tickers:
        for part in str(item).split(","):
            ticker = normalize_edgar_ticker(part)
            if ticker and ticker not in seen:
                seen.add(ticker)
                normalized.append(ticker)
    return normalized or None


def add_gcp_args(
    parser: argparse.ArgumentParser,
    *,
    include_project: bool = True,
    include_bucket: bool = False,
    include_location: bool = False,
    include_dataset: bool = False,
    project_default: str = DEFAULT_PROJECT_ID,
    bucket_default: str = DEFAULT_BUCKET_NAME,
    location_default: str = DEFAULT_LOCATION,
    dataset_default: str = DEFAULT_BIGQUERY_DATASET_ID,
    location_flag: str = "--location",
    dataset_flag: str = "--dataset-id",
    project_help: str = "Google Cloud project ID.",
    bucket_help: str = "Target GCS corpus bucket name.",
    location_help: str = "Google Cloud region/location.",
    dataset_help: str = "Target BigQuery dataset ID.",
) -> None:
    """Register standard GCP CLI flags on an `argparse.ArgumentParser`."""
    if include_project:
        parser.add_argument("--project-id", type=str, default=project_default, help=project_help)
    if include_bucket:
        parser.add_argument("--bucket", type=str, default=bucket_default, help=bucket_help)
    if include_dataset:
        parser.add_argument(dataset_flag, type=str, default=dataset_default, help=dataset_help)
    if include_location:
        parser.add_argument(location_flag, type=str, default=location_default, help=location_help)


def add_logging_args(
    parser: argparse.ArgumentParser,
    *,
    default_level: str = "INFO",
    use_verbose: bool = False,
    restricted_choices: bool = False,
) -> None:
    """Register standard logging verbosity flags (`--log-level` or `--verbose`)."""
    if use_verbose:
        parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
        return
    kwargs: dict[str, object] = {"default": default_level, "help": "Logging verbosity."}
    if restricted_choices:
        kwargs["type"] = str
        kwargs["choices"] = ["DEBUG", "INFO", "WARNING", "ERROR"]
    parser.add_argument("--log-level", **kwargs)  # type: ignore[arg-type]


def configure_script_logging(
    *,
    level: str | None = None,
    verbose: bool = False,
    fmt: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt: str | None = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure root logging consistently for CLI scripts."""
    if verbose:
        resolved_level = logging.DEBUG
    elif level:
        resolved_level = getattr(logging, str(level).upper(), logging.INFO)
    else:
        resolved_level = logging.INFO
    kwargs: dict[str, object] = {"level": resolved_level, "format": fmt}
    if datefmt is not None:
        kwargs["datefmt"] = datefmt
    logging.basicConfig(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "PROJECT_ROOT",
    "SRC_DIR",
    "add_gcp_args",
    "add_logging_args",
    "configure_script_logging",
    "ensure_src_on_path",
    "parse_ticker_args",
]
