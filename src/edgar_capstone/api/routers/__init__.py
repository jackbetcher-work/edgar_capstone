"""Modular FastAPI APIRouter registry for the SEC EDGAR Analyst service."""

from __future__ import annotations

from fastapi import APIRouter

from edgar_capstone.api.routers.auth import router as auth_router
from edgar_capstone.api.routers.chat import router as chat_router
from edgar_capstone.api.routers.feedback import router as feedback_router
from edgar_capstone.api.routers.filings import router as filings_router
from edgar_capstone.api.routers.health import router as health_router
from edgar_capstone.api.routers.sessions import router as sessions_router
from edgar_capstone.api.routers.spa import router as spa_router
from edgar_capstone.api.routers.telemetry import router as telemetry_router

# Registration order matters: `spa_router` holds the catch-all `/{full_path:path}`
# route and MUST remain last so it never shadows explicit API or OpenAPI routes.
ALL_ROUTERS: tuple[APIRouter, ...] = (
    health_router,
    auth_router,
    filings_router,
    chat_router,
    sessions_router,
    telemetry_router,
    feedback_router,
    spa_router,
)

__all__ = [
    "ALL_ROUTERS",
    "auth_router",
    "chat_router",
    "feedback_router",
    "filings_router",
    "health_router",
    "sessions_router",
    "spa_router",
    "telemetry_router",
]
