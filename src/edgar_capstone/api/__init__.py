"""Modular FastAPI API package for the SEC EDGAR Natural Language Analyst service."""

from __future__ import annotations

from edgar_capstone.api.dependencies import (
    WWW_AUTHENTICATE_BEARER,
    LiveRuntimeUnavailableError,
    LocalSsoBrokerForbiddenError,
    audited_permissions,
    format_token_exchange_response,
    get_runtime_sa_config,
    register_exception_handlers,
)
from edgar_capstone.api.routers import (
    ALL_ROUTERS,
    auth_router,
    chat_router,
    feedback_router,
    filings_router,
    health_router,
    sessions_router,
    spa_router,
    telemetry_router,
)
from edgar_capstone.api.schemas import (
    ChatQueryRequest,
    DevWorkspaceSsoLoginRequest,
    FeedbackRequest,
    FilingScopeMixin,
    SsoTokenExchangeRequest,
)

__all__ = [
    "ALL_ROUTERS",
    "ChatQueryRequest",
    "DevWorkspaceSsoLoginRequest",
    "FeedbackRequest",
    "FilingScopeMixin",
    "LiveRuntimeUnavailableError",
    "LocalSsoBrokerForbiddenError",
    "SsoTokenExchangeRequest",
    "WWW_AUTHENTICATE_BEARER",
    "audited_permissions",
    "auth_router",
    "chat_router",
    "feedback_router",
    "filings_router",
    "format_token_exchange_response",
    "get_runtime_sa_config",
    "health_router",
    "register_exception_handlers",
    "sessions_router",
    "spa_router",
    "telemetry_router",
]
