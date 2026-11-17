"""Persistent user-scoped analyst chat session routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from edgar_capstone.agents.context_memory import get_conversation_memory_manager
from edgar_capstone.security.auth import (
    AuthenticatedUser,
    record_security_audit_event,
    require_permissions,
)
from edgar_capstone.security.rbac import Permission

router = APIRouter(tags=["Sessions"])


@router.get("/api/v1/sessions")
def list_user_chat_sessions(
    limit: int = Query(25, ge=1, le=100),
    user: AuthenticatedUser = Depends(require_permissions(Permission.CHAT_QUERY.value)),
) -> dict[str, Any]:
    """List persistent analyst chat sessions belonging solely to the authenticated user."""
    memory_mgr = get_conversation_memory_manager()
    sessions = memory_mgr.list_user_sessions(user.email, limit=limit)
    record_security_audit_event(
        event_type="chat_sessions_listed",
        user=user,
        resource="/api/v1/sessions",
        detail=f"count={len(sessions)}",
    )
    return {
        "status": "ok",
        "user_email": user.email,
        "persistence": memory_mgr.persistence_status(),
        "session_count": len(sessions),
        "sessions": sessions,
    }


@router.delete("/api/v1/sessions/{session_id}")
def delete_user_chat_session(
    session_id: str,
    user: AuthenticatedUser = Depends(require_permissions(Permission.CHAT_QUERY.value)),
) -> dict[str, Any]:
    """Delete a persistent analyst chat session scoped strictly to the authenticated user."""
    memory_mgr = get_conversation_memory_manager()
    deleted = memory_mgr.delete_session(session_id, user_id=user.email)
    record_security_audit_event(
        event_type="chat_session_deleted",
        user=user,
        resource=f"/api/v1/sessions/{session_id}",
        status_outcome="deleted" if deleted else "not_found",
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "status": "not_found",
                "message": f"Session '{session_id}' was not found for {user.email}.",
            },
        )
    return {
        "status": "deleted",
        "session_id": session_id,
        "user_email": user.email,
    }
