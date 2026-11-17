"""Analyst feedback persistence and audit listing routes."""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query

from edgar_capstone.api.schemas import FeedbackRequest
from edgar_capstone.security.auth import (
    AuthenticatedUser,
    record_security_audit_event,
    require_permissions,
)
from edgar_capstone.security.rbac import Permission
from edgar_capstone.telemetry import get_feedback_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Feedback"])


@router.post("/api/v1/feedback")
def submit_analyst_feedback(
    req: FeedbackRequest,
    user: AuthenticatedUser = Depends(require_permissions(Permission.FEEDBACK_SUBMIT.value)),
) -> dict[str, Any]:
    """Persist analyst thumbs-up/down audit feedback and optional comment to Firestore."""
    persisted = get_feedback_store().record_feedback(
        message_id=req.message_id,
        rating=req.rating,
        comment=req.comment,
        query=req.query,
        user_email=user.email,
        identity_provider=user.identity_provider,
        session_id=req.session_id,
        tickers=req.tickers,
        intent=req.intent,
    )
    entry = persisted["entry"]
    record_security_audit_event(
        event_type="analyst_feedback_submitted",
        user=user,
        resource="/api/v1/feedback",
        tickers=req.tickers,
        detail=f"message_id={req.message_id} rating={req.rating} backend={persisted['backend_used']}",
    )
    logger.info(
        "Recorded analyst feedback message_id=%s rating=%s user=%s backend=%s",
        req.message_id,
        req.rating,
        user.email,
        persisted["backend_used"],
    )
    return {
        "status": "success",
        "recorded": entry,
        "persistence": {
            "backend": persisted["backend_used"],
            "database_id": persisted["database_id"],
            "collection": persisted["collection"],
            "document_name": persisted["document_name"],
        },
        "total_feedback_count": persisted["total_feedback_count"],
    }


@router.get("/api/v1/feedback")
def list_analyst_feedback(
    limit: int = Query(50, ge=1, le=500),
    rating: Literal["up", "down"] | None = Query(default=None),
    user: AuthenticatedUser = Depends(require_permissions(Permission.AUDIT_READ.value)),
) -> dict[str, Any]:
    """List persisted analyst feedback ratings and audit comments from Firestore."""
    record_security_audit_event(
        event_type="analyst_feedback_listed",
        user=user,
        resource="/api/v1/feedback",
        detail=f"limit={limit} rating={rating or 'ALL'}",
    )
    result = get_feedback_store().list_feedback(limit=limit, rating=rating)
    return {
        "status": "ok",
        "requested_by": user.email,
        **result,
    }
