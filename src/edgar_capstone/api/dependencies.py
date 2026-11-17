"""Shared FastAPI dependencies, custom exceptions, exception handlers, and response helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse

from edgar_capstone.security.auth import (
    AuthenticatedUser,
    AuthenticationError,
    AuthorizationError,
    record_security_audit_event,
    require_permissions,
)
from edgar_capstone.security.rbac import RuntimeServiceAccountConfig

WWW_AUTHENTICATE_BEARER = 'Bearer realm="sec-edgar-iap", error="invalid_token"'


class LiveRuntimeUnavailableError(RuntimeError):
    """Raised when the live ADK / Vertex AI execution runtime cannot serve a turn (HTTP 503)."""


class LocalSsoBrokerForbiddenError(AuthorizationError):
    """Raised when the local dev SSO broker is called while strict production mode is active."""

    error_code: str = "local_sso_broker_forbidden"


def register_exception_handlers(app: FastAPI) -> None:
    """Attach centralized HTTP error handlers for domain auth and live-runtime exceptions."""

    @app.exception_handler(LiveRuntimeUnavailableError)
    async def _handle_live_runtime_unavailable(
        _request: Request,
        exc: LiveRuntimeUnavailableError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": {
                    "status": "error",
                    "error": "live_adk_unavailable",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(AuthenticationError)
    async def _handle_authentication_error(
        _request: Request,
        exc: AuthenticationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "detail": {
                    "status": "unauthorized",
                    "error": getattr(exc, "error_code", None) or "invalid_token",
                    "message": exc.message,
                }
            },
            headers={"WWW-Authenticate": WWW_AUTHENTICATE_BEARER},
        )

    @app.exception_handler(AuthorizationError)
    async def _handle_authorization_error(
        _request: Request,
        exc: AuthorizationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "detail": {
                    "status": "forbidden",
                    "error": getattr(exc, "error_code", None) or "insufficient_permissions",
                    "message": exc.message,
                }
            },
        )


def get_runtime_sa_config() -> RuntimeServiceAccountConfig:
    """Resolve the active custom runtime service account configuration from environment."""
    return RuntimeServiceAccountConfig.from_env()


def audited_permissions(
    permission: str,
    *extra_permissions: str,
    event_type: str,
    agent_tool: str | None = None,
) -> Callable[..., AuthenticatedUser]:
    """Enforce RBAC / S2S permissions and automatically record a static security audit event."""
    permission_dep = require_permissions(permission, *extra_permissions, agent_tool=agent_tool)

    def _dependency(
        request: Request,
        user: AuthenticatedUser = Depends(permission_dep),
    ) -> AuthenticatedUser:
        record_security_audit_event(
            event_type=event_type,
            user=user,
            resource=request.url.path,
        )
        return user

    return _dependency


def format_token_exchange_response(
    access_token: str,
    user: AuthenticatedUser,
) -> dict[str, Any]:
    """Format a standard Bearer token response shared by SSO exchange and dev-token endpoints."""
    return {
        "status": "authenticated",
        "token_type": "Bearer",
        "access_token": access_token,
        "expires_at": user.expires_at,
        "user": user.to_public_dict(),
    }


__all__ = [
    "LiveRuntimeUnavailableError",
    "LocalSsoBrokerForbiddenError",
    "WWW_AUTHENTICATE_BEARER",
    "audited_permissions",
    "format_token_exchange_response",
    "get_runtime_sa_config",
    "register_exception_handlers",
]
