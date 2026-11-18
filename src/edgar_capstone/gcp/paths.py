"""Wheel-safe repository root and asset directory discovery helpers."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_repo_root() -> Path:
    """Locate the repository root safely across source checkouts, editable installs, and Docker images."""
    source_root = Path(__file__).resolve().parents[3]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file() or (cwd / "prompts").is_dir():
        return cwd
    app_dir = Path("/app")
    if app_dir.is_dir():
        return app_dir
    return cwd


def resolve_asset_dir(name: str, *, env_var: str | None = None) -> Path:
    """Resolve a runtime asset directory (``prompts``, ``terraform/schemas``, etc.)."""
    if env_var:
        override = os.environ.get(env_var, "").strip()
        if override:
            return Path(override).resolve()
    candidates = [
        Path.cwd() / name,
        Path(__file__).resolve().parents[3] / name,
        Path("/app") / name,
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()
    return candidates[1].resolve()
