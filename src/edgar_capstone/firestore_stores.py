"""Google Cloud Firestore Native REST v1 stores for analyst feedback and session memory."""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any, Literal

import httpx

from edgar_capstone.common.formatting import utc_now_iso
from edgar_capstone.gcp.firestore import (
    DEFAULT_FIRESTORE_DATABASE_ID,
    FirestoreCollectionStore,
    encode_firestore_fields,
)
from edgar_capstone.gcp.rest_client import GcpRestError

if TYPE_CHECKING:
    from edgar_capstone.telemetry import TelemetryConfig

logger = logging.getLogger(__name__)

DEFAULT_FIRESTORE_FEEDBACK_COLLECTION = "analyst_feedback"
DEFAULT_FIRESTORE_SESSION_COLLECTION = "analyst_sessions"
DEFAULT_MAX_LOCAL_RECORDS = 500


class _TelemetryFirestoreStore(FirestoreCollectionStore):
    """`FirestoreCollectionStore` subclass translating `GcpRestError` into `Telemetry*Error`."""

    def _firestore_request_json(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        from edgar_capstone.telemetry import _translate_gcp_error

        try:
            return super()._firestore_request_json(
                method,
                url,
                json_body=json_body,
                params=params,
            )
        except GcpRestError as exc:
            raise _translate_gcp_error(exc) from exc


class FirestoreFeedbackStore(_TelemetryFirestoreStore):
    """Persists analyst thumbs-up/down ratings and audit comments to Google Cloud Firestore."""

    def __init__(
        self,
        config: TelemetryConfig | None = None,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
        max_local_records: int = DEFAULT_MAX_LOCAL_RECORDS,
    ) -> None:
        from edgar_capstone.telemetry import TelemetryConfig as _TelemetryConfig

        self.config = config or _TelemetryConfig.from_env()
        super().__init__(
            project_id=self.config.project_id,
            database_id=self.config.firestore_database_id,
            collection_id=self.config.firestore_feedback_collection,
            http_client=http_client,
            credentials=credentials,
            default_timeout=15.0,
        )
        self._local_feedback: deque[dict[str, Any]] = deque(maxlen=max_local_records)
        self._persisted_firestore_count = 0

    def _should_attempt_gcp(self) -> tuple[bool, bool]:
        mode = self.config.feedback_backend
        if mode == "local":
            return False, False
        if mode == "gcp":
            return True, True
        return self.config.gcp_auto_triggered, False

    def record_feedback(
        self,
        *,
        message_id: str,
        rating: Literal["up", "down"],
        comment: str = "",
        query: str = "",
        user_email: str,
        identity_provider: str | None = None,
        session_id: str | None = None,
        tickers: list[str] | None = None,
        intent: str = "",
    ) -> dict[str, Any]:
        """Persist an analyst feedback entry to Firestore (or local memory in local mode)."""
        from edgar_capstone.telemetry import (
            _should_fallback_to_local,
            emit_cloud_logging_sink_record,
        )

        now_epoch = round(time.time(), 3)
        now_iso = utc_now_iso()
        feedback_id = f"fbk-{uuid.uuid4().hex[:12]}"
        norm_tickers = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]

        entry: dict[str, Any] = {
            "feedback_id": feedback_id,
            "message_id": message_id.strip(),
            "session_id": session_id.strip() if session_id and session_id.strip() else None,
            "rating": rating,
            "comment": comment.strip(),
            "query": query.strip(),
            "tickers": norm_tickers,
            "intent": intent.strip(),
            "user_email": user_email,
            "identity_provider": identity_provider or "unknown",
            "recorded_at": now_iso,
            "recorded_at_epoch": now_epoch,
        }

        emit_cloud_logging_sink_record(self.collection_id, entry)

        document_name = (
            f"projects/{self.project_id}/databases/{self.database_id}"
            f"/documents/{self.collection_id}/{feedback_id}"
        )
        backend_used = "local_memory"
        attempt_gcp, strict_gcp = self._should_attempt_gcp()

        if attempt_gcp:
            doc_body = {"fields": encode_firestore_fields(entry)}
            try:
                created_doc = self._firestore_request_json(
                    "POST",
                    self.collection_url(),
                    json_body=doc_body,
                    params={"documentId": feedback_id},
                )
                if isinstance(created_doc, dict) and created_doc.get("name"):
                    document_name = str(created_doc["name"])
                backend_used = "firestore"
                with self._lock:
                    self._persisted_firestore_count += 1
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                if not _should_fallback_to_local(exc, strict_gcp=strict_gcp):
                    raise
                logger.warning(
                    "Firestore feedback write failed in auto mode (%s); persisting in local buffer.",
                    exc,
                )
                backend_used = "local_memory_fallback"

        with self._lock:
            self._local_feedback.append(entry)
            total_count = len(self._local_feedback)

        return {
            "entry": entry,
            "backend_used": backend_used,
            "collection": self.collection_id,
            "database_id": self.database_id,
            "document_name": document_name,
            "total_feedback_count": total_count,
        }

    def list_feedback(
        self,
        *,
        limit: int = 50,
        rating: Literal["up", "down"] | None = None,
    ) -> dict[str, Any]:
        """List persisted analyst feedback entries from Firestore `:runQuery` or local buffer."""
        from edgar_capstone.telemetry import _should_fallback_to_local

        bounded = max(1, min(int(limit), 500))
        attempt_gcp, strict_gcp = self._should_attempt_gcp()
        items: list[dict[str, Any]] | None = None
        source = "local_memory"

        if attempt_gcp:
            structured_query: dict[str, Any] = {
                "from": [{"collectionId": self.collection_id}],
                "orderBy": [
                    {
                        "field": {"fieldPath": "recorded_at_epoch"},
                        "direction": "DESCENDING",
                    }
                ],
                "limit": bounded,
            }
            if rating in ("up", "down"):
                structured_query["where"] = {
                    "fieldFilter": {
                        "field": {"fieldPath": "rating"},
                        "op": "EQUAL",
                        "value": {"stringValue": rating},
                    }
                }
            try:
                items = self.run_structured_query(structured_query)
                source = "firestore"
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                if not _should_fallback_to_local(exc, strict_gcp=strict_gcp):
                    raise
                logger.warning(
                    "Firestore feedback query failed in auto mode (%s); reading local buffer.",
                    exc,
                )
                source = "local_memory_fallback"

        if items is None:
            with self._lock:
                local_copy = list(reversed(self._local_feedback))
            if rating in ("up", "down"):
                local_copy = [x for x in local_copy if x.get("rating") == rating]
            items = local_copy[:bounded]

        up_count = sum(1 for x in items if x.get("rating") == "up")
        down_count = sum(1 for x in items if x.get("rating") == "down")
        total = len(items)
        return {
            "source": source,
            "database_id": self.database_id,
            "collection": self.collection_id,
            "feedback_count": total,
            "thumbs_up_count": up_count,
            "thumbs_down_count": down_count,
            "positive_rating_ratio": round(up_count / total, 4) if total > 0 else 1.0,
            "feedback": items,
        }

    def status_metadata(self) -> dict[str, Any]:
        attempt_gcp, strict_gcp = self._should_attempt_gcp()
        with self._lock:
            local_cnt = len(self._local_feedback)
            fs_cnt = self._persisted_firestore_count
            last_err = self._last_error
        return {
            "backend_mode": self.config.feedback_backend,
            "live_firestore_enabled": attempt_gcp,
            "strict_gcp": strict_gcp,
            "project_id": self.project_id,
            "database_id": self.database_id,
            "collection": self.collection_id,
            "persisted_firestore_count": fs_cnt,
            "buffered_feedback_count": local_cnt,
            "last_error": last_err,
        }

    def reset_local_buffers(self) -> None:
        with self._lock:
            self._local_feedback.clear()
            self._persisted_firestore_count = 0
            self._last_error = None


def session_firestore_doc_id(user_id: str, session_id: str) -> str:
    """Compute a deterministic, tenant-isolated Firestore document ID for `(user_id, session_id)`."""
    raw = f"{user_id.strip().lower()}::{session_id.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


class FirestoreSessionStore(_TelemetryFirestoreStore):
    """Durable, user-isolated conversational session store backed by Cloud Firestore Native REST v1."""

    def __init__(
        self,
        config: TelemetryConfig | None = None,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
        collection_id: str | None = None,
    ) -> None:
        from edgar_capstone.telemetry import TelemetryConfig as _TelemetryConfig

        self.config = config or _TelemetryConfig.from_env()
        effective_collection = (
            collection_id
            or self.config.firestore_session_collection
            or DEFAULT_FIRESTORE_SESSION_COLLECTION
        )
        super().__init__(
            project_id=self.config.project_id,
            database_id=self.config.firestore_database_id or DEFAULT_FIRESTORE_DATABASE_ID,
            collection_id=effective_collection,
            http_client=http_client,
            credentials=credentials,
            default_timeout=15.0,
        )
        self._reads = 0
        self._writes = 0

    def is_live_enabled(self) -> bool:
        """Return True when session persistence targets live Cloud Firestore."""
        from edgar_capstone.telemetry import _resolve_backend_mode

        override = os.environ.get("EDGAR_SESSION_BACKEND", "").strip()
        mode = _resolve_backend_mode(override) if override else self.config.feedback_backend
        if self._http_client is not None:
            return mode != "local"
        if mode == "local":
            return False
        if mode == "gcp":
            return True
        return self.config.gcp_auto_triggered

    def _doc_url(self, user_id: str, session_id: str) -> str:
        return self.document_url(session_firestore_doc_id(user_id, session_id))

    def load_session(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        """Load a persisted session document for `(user_id, session_id)` (None if 404)."""
        if not self.is_live_enabled():
            return None
        doc_id = session_firestore_doc_id(user_id, session_id)
        try:
            doc = self.get_document(doc_id)
            if doc is not None:
                with self._lock:
                    self._reads += 1
            return doc
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            raise

    def save_session(self, user_id: str, session_id: str, payload: dict[str, Any]) -> None:
        """Upsert a session document via Firestore REST v1 `PATCH`."""
        if not self.is_live_enabled():
            return
        fields = encode_firestore_fields(payload)
        try:
            self._firestore_request_json(
                "PATCH",
                self._doc_url(user_id, session_id),
                json_body={"fields": fields},
            )
            with self._lock:
                self._writes += 1
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            raise

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """Delete a persisted session document for `(user_id, session_id)`."""
        from edgar_capstone.telemetry import TelemetryNotFoundError

        if not self.is_live_enabled():
            return False
        try:
            self._firestore_request_json("DELETE", self._doc_url(user_id, session_id))
            return True
        except TelemetryNotFoundError:
            return False
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            raise

    def list_sessions(self, user_id: str, *, limit: int = 25) -> list[dict[str, Any]]:
        """List recent sessions belonging to `user_id`."""
        if not self.is_live_enabled():
            return []
        structured_query: dict[str, Any] = {
            "from": [{"collectionId": self.collection_id}],
            "where": {
                "fieldFilter": {
                    "field": {"fieldPath": "user_id"},
                    "op": "EQUAL",
                    "value": {"stringValue": user_id.strip().lower()},
                }
            },
            "limit": max(1, min(int(limit), 100)),
        }
        try:
            items = self.run_structured_query(structured_query)
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            raise
        items.sort(key=lambda d: str(d.get("updated_at") or ""), reverse=True)
        return items[:limit]

    def status_metadata(self) -> dict[str, Any]:
        with self._lock:
            reads, writes, last_err = self._reads, self._writes, self._last_error
        return {
            "live_firestore_enabled": self.is_live_enabled(),
            "project_id": self.project_id,
            "database_id": self.database_id,
            "collection": self.collection_id,
            "reads": reads,
            "writes": writes,
            "last_error": last_err,
        }

    def reset_local_buffers(self) -> None:
        with self._lock:
            self._reads = 0
            self._writes = 0
            self._last_error = None


__all__ = [
    "DEFAULT_FIRESTORE_FEEDBACK_COLLECTION",
    "DEFAULT_FIRESTORE_SESSION_COLLECTION",
    "FirestoreFeedbackStore",
    "FirestoreSessionStore",
    "session_firestore_doc_id",
]
