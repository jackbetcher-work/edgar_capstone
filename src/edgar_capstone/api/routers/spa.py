"""Single-origin Terminal UI static bundle router (must be registered last)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

router = APIRouter(include_in_schema=False)


def _resolve_frontend_dist_dir() -> Path | None:
    """Locate the compiled frontend bundle (`frontend/dist/index.html`)."""
    candidates: list[Path] = []
    env_override = os.environ.get("FRONTEND_DIST_DIR", "").strip()
    if env_override:
        candidates.append(Path(env_override))
    candidates.append(Path.cwd() / "frontend" / "dist")
    resolved_file = Path(__file__).resolve()
    if len(resolved_file.parents) > 4:
        candidates.append(resolved_file.parents[4] / "frontend" / "dist")
    for directory in candidates:
        if (directory / "index.html").is_file():
            return directory.resolve()
    return None


@router.get("/{full_path:path}", include_in_schema=False)
@router.get("/", include_in_schema=False)
def serve_terminal_ui(full_path: str = "") -> FileResponse:
    """Serve the compiled SEC EDGAR Terminal UI identically in local preview and Cloud Run."""
    normalized = full_path.strip("/")
    if normalized == "api" or normalized.startswith("api/") or normalized == "healthz":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    dist_dir = _resolve_frontend_dist_dir()
    if dist_dir is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend bundle not found. Run 'make build' to compile frontend/dist.",
        )

    if normalized:
        asset_path = (dist_dir / normalized).resolve()
        if asset_path.is_relative_to(dist_dir) and asset_path.is_file():
            return FileResponse(asset_path)

    return FileResponse(dist_dir / "index.html")
