"""Session and tool-usage telemetry routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from edgar_capstone.security.auth import (
    AuthenticatedUser,
    record_security_audit_event,
    require_permissions,
)
from edgar_capstone.security.rbac import Permission
from edgar_capstone.telemetry import get_telemetry_sink

router = APIRouter(tags=["Telemetry & Audit"])


@router.get("/api/v1/telemetry/sessions")
def get_session_and_tool_telemetry(
    limit: int = Query(50, ge=1, le=500),
    session_id: str | None = Query(default=None),
    user: AuthenticatedUser = Depends(require_permissions(Permission.AUDIT_READ.value)),
) -> dict[str, Any]:
    """Retrieve conversational session length, companies analyzed, report counts, tool usage, and BigQuery sink metrics."""
    record_security_audit_event(
        event_type="session_telemetry_read",
        user=user,
        resource="/api/v1/telemetry/sessions",
        detail=f"limit={limit} session_id={session_id or 'ALL'}",
    )
    sink = get_telemetry_sink()
    events = sink.query_session_telemetry(limit=limit, session_id=session_id)
    return {
        "status": "ok",
        "requested_by": user.email,
        "telemetry_sink": sink.status_metadata(),
        "summary": sink.summarize_session_telemetry(),
        "event_count": len(events),
        "events": events,
    }
