"""Cached LookML semantic layer catalog introspection over ``looker/``."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

_LOOKML_EXPLORE_RE = re.compile(r"^\s*explore:\s*([A-Za-z0-9_]+)\s*\{", re.MULTILINE)
_LOOKML_VIEW_RE = re.compile(r"^\s*view:\s*([A-Za-z0-9_]+)\s*\{", re.MULTILINE)
_LOOKML_DIMENSION_RE = re.compile(
    r"^\s*(?:dimension|dimension_group):\s*([A-Za-z0-9_]+)\s*\{", re.MULTILINE
)
_LOOKML_MEASURE_RE = re.compile(r"^\s*measure:\s*([A-Za-z0-9_]+)\s*\{", re.MULTILINE)
_LOOKML_SQL_TABLE_RE = re.compile(r"sql_table_name:\s*([^;]+);;", re.MULTILINE)
_LOOKML_CONNECTION_RE = re.compile(r'connection:\s*"([^"]+)"')
_LOOKML_DASHBOARD_RE = re.compile(r"^\s*-\s*dashboard:\s*([A-Za-z0-9_]+)", re.MULTILINE)
_LOOKML_TITLE_RE = re.compile(r"^\s*title:\s*(.+)$", re.MULTILINE)
_LOOKML_ELEMENT_NAME_RE = re.compile(r"^\s*-\s*name:\s*([A-Za-z0-9_]+)", re.MULTILINE)

_CACHE_LOCK = threading.Lock()
_CATALOG_CACHE: dict[Path, tuple[tuple[tuple[str, int], ...], dict[str, Any]]] = {}


def resolve_looker_project_dir() -> Path:
    """Locate the repository ``looker/`` semantic layer directory."""
    env_dir = os.environ.get("EDGAR_LOOKML_DIR", "").strip()
    if env_dir:
        return Path(env_dir).resolve()
    candidates = [
        Path.cwd() / "looker",
        Path(__file__).resolve().parents[2] / "looker",
        Path("/app/looker"),
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()
    return candidates[1].resolve()


def _dir_signature(root: Path) -> tuple[tuple[str, int], ...]:
    """Compute a lightweight mtime signature across ``looker/`` files for cache invalidation."""
    entries: list[tuple[str, int]] = []
    manifest = root / "manifest.lkml"
    if manifest.is_file():
        entries.append(("manifest.lkml", manifest.stat().st_mtime_ns))
    for sub, pattern in (
        ("models", "*.model.lkml"),
        ("views", "*.view.lkml"),
        ("dashboards", "*.dashboard.lookml"),
    ):
        sub_dir = root / sub
        if sub_dir.is_dir():
            for path in sorted(sub_dir.glob(pattern)):
                entries.append((f"{sub}/{path.name}", path.stat().st_mtime_ns))
    return tuple(entries)


def clear_lookml_catalog_cache() -> None:
    """Reset the mtime-keyed LookML catalog cache (used by tests)."""
    with _CACHE_LOCK:
        _CATALOG_CACHE.clear()


def load_lookml_project_catalog(looker_dir: Path | None = None) -> dict[str, Any]:
    """Parse LookML models, views, explores, and dashboards in ``looker/`` (cached by file mtimes)."""
    root = (looker_dir or resolve_looker_project_dir()).resolve()
    if not root.is_dir():
        return {
            "status": "missing",
            "looker_dir": str(root),
            "models": [],
            "views": [],
            "dashboards": [],
        }

    sig = _dir_signature(root)
    with _CACHE_LOCK:
        cached = _CATALOG_CACHE.get(root)
        if cached is not None and cached[0] == sig:
            return cached[1]

    manifest_path = root / "manifest.lkml"
    models: list[dict[str, Any]] = []
    for model_file in (
        sorted((root / "models").glob("*.model.lkml")) if (root / "models").is_dir() else []
    ):
        content = model_file.read_text(encoding="utf-8")
        conn_match = _LOOKML_CONNECTION_RE.search(content)
        explores = _LOOKML_EXPLORE_RE.findall(content)
        models.append(
            {
                "model_name": model_file.name.replace(".model.lkml", ""),
                "file": f"looker/models/{model_file.name}",
                "connection": conn_match.group(1) if conn_match else None,
                "explores": explores,
                "explore_count": len(explores),
            }
        )

    views: list[dict[str, Any]] = []
    for view_file in (
        sorted((root / "views").glob("*.view.lkml")) if (root / "views").is_dir() else []
    ):
        content = view_file.read_text(encoding="utf-8")
        view_names = _LOOKML_VIEW_RE.findall(content)
        sql_table_match = _LOOKML_SQL_TABLE_RE.search(content)
        dims = _LOOKML_DIMENSION_RE.findall(content)
        measures = _LOOKML_MEASURE_RE.findall(content)
        views.append(
            {
                "view_name": view_names[0] if view_names else view_file.stem.replace(".view", ""),
                "file": f"looker/views/{view_file.name}",
                "sql_table_name": (sql_table_match.group(1).strip() if sql_table_match else None),
                "dimensions": dims,
                "measures": measures,
                "dimension_count": len(dims),
                "measure_count": len(measures),
            }
        )

    dashboards: list[dict[str, Any]] = []
    for dash_file in (
        sorted((root / "dashboards").glob("*.dashboard.lookml"))
        if (root / "dashboards").is_dir()
        else []
    ):
        content = dash_file.read_text(encoding="utf-8")
        dash_ids = _LOOKML_DASHBOARD_RE.findall(content)
        titles = _LOOKML_TITLE_RE.findall(content)
        tiles = _LOOKML_ELEMENT_NAME_RE.findall(content)
        dashboards.append(
            {
                "dashboard_id": (
                    dash_ids[0] if dash_ids else dash_file.name.replace(".dashboard.lookml", "")
                ),
                "title": titles[0].strip().strip("\"'") if titles else "",
                "file": f"looker/dashboards/{dash_file.name}",
                "elements": tiles,
                "element_count": len(tiles),
            }
        )

    catalog = {
        "status": "ok",
        "looker_dir": str(root),
        "manifest_present": manifest_path.is_file(),
        "model_count": len(models),
        "view_count": len(views),
        "dashboard_count": len(dashboards),
        "models": models,
        "views": views,
        "dashboards": dashboards,
    }
    with _CACHE_LOCK:
        _CATALOG_CACHE[root] = (sig, catalog)
    return catalog
