"""Conversational Hybrid RAG analyst routes (`POST /api/v1/chat` and `/api/v1/chat/stream`)."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from edgar_capstone.agents.orchestration.chat_engine import run_conversational_rag_turn
from edgar_capstone.api.dependencies import LiveRuntimeUnavailableError
from edgar_capstone.api.schemas import ChatQueryRequest
from edgar_capstone.observability import (
    current_trace_metadata,
    set_span_attributes,
)
from edgar_capstone.security.auth import (
    AuthenticatedUser,
    bind_audit_principal,
    record_security_audit_event,
    require_permissions,
    reset_audit_principal,
)
from edgar_capstone.security.rbac import Permission

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Conversational Analyst"])

_GENERIC_STREAM_ERROR_MESSAGE = (
    "An internal error occurred while processing the request. "
    "Please check server logs with the attached trace ID."
)

ProgressCallback = Callable[[str, str, dict[str, Any] | None], None]


def _format_sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


def _execute_and_enrich_chat_turn(
    req: ChatQueryRequest,
    user: AuthenticatedUser,
    resource: str,
    *,
    streaming: bool = False,
    on_progress: ProgressCallback | None = None,
    trace_meta_fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one grounded RAG turn and attach security audit, principal, OTel span, and trace metadata."""
    turn_kwargs: dict[str, Any] = {
        "query": req.query,
        "tickers": req.tickers,
        "gics_sector": req.gics_sector,
        "section_id": req.section_id,
        "start_year": req.start_year,
        "end_year": req.end_year,
        "top_k": req.top_k,
        "session_id": req.session_id,
        "user_id": user.email,
    }
    if on_progress is not None:
        turn_kwargs["on_progress"] = on_progress

    # Attribute guardrail block audit events (Model Armor / SDP / projection filter), which are
    # raised deep inside the ADK turn without an `AuthenticatedUser`, to this request principal.
    principal_token = bind_audit_principal(user)
    try:
        turn_result = run_conversational_rag_turn(**turn_kwargs)
    except LiveRuntimeUnavailableError:
        raise
    except RuntimeError as exc:
        raise LiveRuntimeUnavailableError(str(exc)) from exc
    finally:
        reset_audit_principal(principal_token)

    resolved_tickers = turn_result.get("resolved_scope", {}).get("tickers") or req.tickers
    record_security_audit_event(
        event_type="conversational_rag_chat",
        user=user,
        resource=resource,
        tickers=resolved_tickers,
        status_outcome=str(turn_result.get("status", "success")),
        detail=(
            f"intent={turn_result.get('intent', 'unknown')} "
            f"verified={turn_result.get('answer_verified')}"
        ),
    )
    turn_result["authenticated_principal"] = {
        "email": user.email,
        "identity_provider": user.identity_provider,
        "iap_assertion_verified": user.iap_assertion_verified,
        "roles": list(user.roles),
        "cloud_identity_groups": list(user.cloud_identity_groups),
        "agent_identity": user.agent_identity,
    }
    span_attrs: dict[str, Any] = {
        "edgar.chat.intent": turn_result.get("intent"),
        "edgar.chat.status": turn_result.get("status"),
        "edgar.chat.execution_mode": turn_result.get("execution_mode"),
        "edgar.chat.citations_count": len(turn_result.get("citations") or []),
        "edgar.chat.total_latency_ms": turn_result.get("total_latency_ms"),
        "edgar.chat.session_id": req.session_id,
    }
    if streaming:
        span_attrs["edgar.chat.streaming"] = True
    set_span_attributes(**span_attrs)
    turn_result["observability"] = current_trace_metadata() or trace_meta_fallback
    return turn_result


@router.post("/api/v1/chat")
def conversational_rag_chat(
    req: ChatQueryRequest,
    user: AuthenticatedUser = Depends(require_permissions(Permission.CHAT_QUERY.value)),
) -> dict[str, Any]:
    """Execute a source-grounded conversational Hybrid RAG query with ADK Runner / ReAct trace & fact-check."""
    return _execute_and_enrich_chat_turn(
        req,
        user,
        resource="/api/v1/chat",
        streaming=False,
    )


@router.post("/api/v1/chat/stream")
async def conversational_rag_chat_stream(
    req: ChatQueryRequest,
    user: AuthenticatedUser = Depends(require_permissions(Permission.CHAT_QUERY.value)),
) -> StreamingResponse:
    """Stream live Server-Sent Events (SSE) progress updates followed by the final grounded chat response."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
    t_start = time.perf_counter()
    trace_meta_snapshot = current_trace_metadata()
    worker_ctx = contextvars.copy_context()

    def _on_progress(stage: str, message: str, meta: dict[str, Any] | None = None) -> None:
        elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
        payload = {
            "stage": stage,
            "message": message,
            "meta": meta,
            "elapsed_ms": elapsed_ms,
        }
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", payload))

    def _worker() -> None:
        try:
            turn_result = _execute_and_enrich_chat_turn(
                req,
                user,
                resource="/api/v1/chat/stream",
                streaming=True,
                on_progress=_on_progress,
                trace_meta_fallback=trace_meta_snapshot,
            )
            loop.call_soon_threadsafe(queue.put_nowait, ("result", turn_result))
        except LiveRuntimeUnavailableError as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                (
                    "error",
                    {
                        "status": "error",
                        "error": "live_adk_unavailable",
                        "message": str(exc),
                        "observability": trace_meta_snapshot,
                    },
                ),
            )
        except Exception:
            logger.exception("Unhandled error during /api/v1/chat/stream")
            loop.call_soon_threadsafe(
                queue.put_nowait,
                (
                    "error",
                    {
                        "status": "error",
                        "error": "internal_server_error",
                        "message": _GENERIC_STREAM_ERROR_MESSAGE,
                        "observability": trace_meta_snapshot,
                    },
                ),
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def _event_generator() -> AsyncIterator[str]:
        yield _format_sse_event(
            "progress",
            {
                "stage": "connected",
                "message": "Connected to SEC EDGAR analyst stream...",
                "meta": {"trace_id": (trace_meta_snapshot or {}).get("trace_id")},
                "elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
            },
        )
        worker_task = asyncio.ensure_future(loop.run_in_executor(None, worker_ctx.run, _worker))
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if item is None:
                    break
                event_name, payload = item
                yield _format_sse_event(event_name, payload)
        finally:
            if not worker_task.done():
                await asyncio.shield(worker_task)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
