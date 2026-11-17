"""FastAPI Cloud Run service entrypoint for the SEC EDGAR Natural Language Analyst."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import metadata

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from edgar_capstone.agents.chat_engine import run_conversational_rag_turn
from edgar_capstone.agents.react_loop import flush_pending_evaluations
from edgar_capstone.api.dependencies import (
    LiveRuntimeUnavailableError,
    register_exception_handlers,
)
from edgar_capstone.api.routers import ALL_ROUTERS
from edgar_capstone.api.routers.spa import _resolve_frontend_dist_dir
from edgar_capstone.api.schemas import (
    ChatQueryRequest,
    DevWorkspaceSsoLoginRequest,
    FeedbackRequest,
    FilingScopeMixin,
    SsoTokenExchangeRequest,
)
from edgar_capstone.observability import (
    REQUEST_ID_HEADER,
    TRACE_ID_HEADER,
    RequestTelemetryMiddleware,
    configure_logging,
    configure_tracing,
    flush_tracing,
    instrument_fastapi_app,
    logging_status_metadata,
    tracing_status_metadata,
)
from edgar_capstone.security.agent_identity import S2S_AGENT_ASSERTION_HEADER
from edgar_capstone.telemetry import get_telemetry_sink, reset_telemetry_stores

# Structured JSON logs (Cloud Logging) + OpenTelemetry tracing (Cloud Trace) must be configured
# before the app object is built so the FastAPI instrumentation and every module logger see them.
configure_logging()
configure_tracing()

logger = logging.getLogger(__name__)

_CORS_ALLOW_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def _resolve_package_version() -> str:
    """Resolve the installed `edgar-capstone` distribution version dynamically."""
    try:
        return metadata.version("edgar-capstone")
    except metadata.PackageNotFoundError:
        return "0.1.0"


@asynccontextmanager
async def _app_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Log startup metadata and flush/close background workers, telemetry stores, and trace spans."""
    logger.info(
        "SEC EDGAR analyst API ready",
        extra={
            "json_fields": {
                "event": "service_started",
                "tracing": tracing_status_metadata(),
                "logging": logging_status_metadata(),
            }
        },
    )
    try:
        yield
    finally:
        # Drain async background LLM-as-a-Judge evaluations, flush/close BigQuery telemetry writes,
        # and flush OpenTelemetry trace spans even if an earlier cleanup step raises.
        try:
            await asyncio.to_thread(flush_pending_evaluations, 10.0)
        finally:
            try:
                await asyncio.to_thread(get_telemetry_sink().flush_async_writes, 10.0)
            finally:
                try:
                    await asyncio.to_thread(reset_telemetry_stores)
                finally:
                    await asyncio.to_thread(flush_tracing)


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application with middleware, handlers, and routers."""
    application = FastAPI(
        lifespan=_app_lifespan,
        title="SEC EDGAR Natural Language Analyst API",
        version=_resolve_package_version(),
        description=(
            "Cloud Run FastAPI backend for deterministic SEC 10-K financial variance "
            "calculations, metadata-filtered Hybrid RAG search, and ADK agent orchestration. "
            "Internal APIs are protected via Identity-Aware Proxy (IAP) and Google Workspace "
            "Okta/SSO JWT Bearer authentication."
        ),
    )

    register_exception_handlers(application)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Goog-IAP-JWT-Assertion",
            S2S_AGENT_ASSERTION_HEADER,
            "traceparent",
            "tracestate",
            "X-Cloud-Trace-Context",
            REQUEST_ID_HEADER,
        ],
        expose_headers=[
            "WWW-Authenticate",
            TRACE_ID_HEADER,
            REQUEST_ID_HEADER,
            "Server-Timing",
        ],
    )
    # Added after CORS so it wraps it: preflight responses also carry trace/latency headers.
    application.add_middleware(RequestTelemetryMiddleware)
    # Outermost: OpenTelemetry server span, continuing upstream `traceparent` / X-Cloud-Trace-Context.
    instrument_fastapi_app(application)

    for router in ALL_ROUTERS:
        application.include_router(router)

    return application


app = create_app()

__all__ = [
    "ChatQueryRequest",
    "DevWorkspaceSsoLoginRequest",
    "FeedbackRequest",
    "FilingScopeMixin",
    "LiveRuntimeUnavailableError",
    "SsoTokenExchangeRequest",
    "_resolve_frontend_dist_dir",
    "app",
    "create_app",
    "run_conversational_rag_turn",
]
