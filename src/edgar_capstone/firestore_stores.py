"""Google Cloud Firestore SDK stores for analyst feedback and multi-instance session memory."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from edgar_capstone.common.formatting import utc_now_iso
from edgar_capstone.gcp.bigquery import validate_bq_identifier
from edgar_capstone.gcp.errors import translate_gcp_sdk_error

if TYPE_CHECKING:
    from edgar_capstone.telemetry import TelemetryConfig

logger = logging.getLogger(__name__)

DEFAULT_FIRESTORE_DATABASE_ID = "(default)"
DEFAULT_FIRESTORE_FEEDBACK_COLLECTION = "analyst_feedback"
DEFAULT_FIRESTORE_SESSION_COLLECTION = "analyst_sessions"
DEFAULT_MAX_LOCAL_RECORDS = 500


def _normalize_firestore_dict(
    data: dict[str, Any], *, doc_name: str | None = None
) -> dict[str, Any]:
    """Normalize Firestore SDK document values (converting ``datetime`` to UTC ISO-8601)."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, datetime):
            dt = v.astimezone(UTC) if v.tzinfo is not None else v.replace(tzinfo=UTC)
            out[str(k)] = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif isinstance(v, dict):
            out[str(k)] = _normalize_firestore_dict(v)
        elif isinstance(v, list):
            out[str(k)] = [
                _normalize_firestore_dict(item) if isinstance(item, dict) else item for item in v
            ]
        else:
            out[str(k)] = v
    if doc_name and "document_name" not in out:
        out["document_name"] = doc_name
    return out


class _FirestoreSdkBase:
    """Shared lazy ``google.cloud.firestore.Client`` lifecycle and error translation base."""

    def __init__(
        self,
        *,
        project_id: str,
        database_id: str = DEFAULT_FIRESTORE_DATABASE_ID,
        collection_id: str,
        firestore_client: Any | None = None,
        credentials: Any | None = None,
        default_timeout: float = 15.0,
    ) -> None:
        self.project_id = validate_bq_identifier(project_id, "project_id")
        self.database_id = (database_id or DEFAULT_FIRESTORE_DATABASE_ID).strip()
        self.collection_id = collection_id.strip()
        self._injected_client = firestore_client
        self._owned_client: Any | None = None
        self._credentials = credentials
        self._default_timeout = float(default_timeout)
        self._client_lock = threading.Lock()
        self._lock = threading.Lock()
        self._last_error: str | None = None

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        with self._client_lock:
            if self._owned_client is None:
                from google.cloud import firestore

                kwargs: dict[str, Any] = {
                    "project": self.project_id,
                    "database": self.database_id,
                }
                if self._credentials is not None:
                    kwargs["credentials"] = self._credentials
                self._owned_client = firestore.Client(**kwargs)
            return self._owned_client

    def close(self) -> None:
        """Close the internally managed ``firestore.Client`` (leaves injected clients open)."""
        with self._client_lock:
            if self._owned_client is not None:
                close_fn = getattr(self._owned_client, "close", None)
                if callable(close_fn):
                    close_fn()
                self._owned_client = None

    def __enter__(self) -> _FirestoreSdkBase:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _collection(self) -> Any:
        return self._get_client().collection(self.collection_id)

    def _doc_path(self, doc_id: str) -> str:
        return (
            f"projects/{self.project_id}/databases/{self.database_id}"
            f"/documents/{self.collection_id}/{doc_id}"
        )

    def _translate_error(self, exc: BaseException, *, action: str) -> Exception:
        from google.auth.exceptions import DefaultCredentialsError, RefreshError

        from edgar_capstone.telemetry import _translate_gcp_error

        if isinstance(exc, (DefaultCredentialsError, RefreshError)):
            return exc
        return _translate_gcp_error(
            translate_gcp_sdk_error(exc, context=f"Firestore {action} ({self.collection_id})")
        )


class FirestoreFeedbackStore(_FirestoreSdkBase):
    """Persists analyst thumbs-up/down ratings and audit comments to Google Cloud Firestore."""

    def __init__(
        self,
        config: TelemetryConfig | None = None,
        *,
        firestore_client: Any | None = None,
        credentials: Any | None = None,
        max_local_records: int = DEFAULT_MAX_LOCAL_RECORDS,
    ) -> None:
        from edgar_capstone.telemetry import TelemetryConfig as _TelemetryConfig

        self.config = config or _TelemetryConfig.from_env()
        super().__init__(
            project_id=self.config.project_id,
            database_id=self.config.firestore_database_id,
            collection_id=self.config.firestore_feedback_collection,
            firestore_client=firestore_client,
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

        now_dt = datetime.now(UTC)
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

        document_name = self._doc_path(feedback_id)
        backend_used = "local_memory"
        attempt_gcp, strict_gcp = self._should_attempt_gcp()

        if attempt_gcp:
            firestore_doc = {**entry, "recorded_at": now_dt}
            try:
                doc_ref = self._collection().document(feedback_id)
                doc_ref.create(firestore_doc, timeout=self._default_timeout)
                backend_used = "firestore"
                with self._lock:
                    self._persisted_firestore_count += 1
                    self._last_error = None
            except Exception as raw_exc:
                exc = self._translate_error(raw_exc, action="create feedback")
                with self._lock:
                    self._last_error = str(exc)
                if not _should_fallback_to_local(exc, strict_gcp=strict_gcp):
                    raise exc from raw_exc
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
        """List persisted analyst feedback entries from Firestore or local buffer."""
        from google.cloud import firestore
        from google.cloud.firestore_v1.base_query import FieldFilter

        from edgar_capstone.telemetry import _should_fallback_to_local

        bounded = max(1, min(int(limit), 500))
        attempt_gcp, strict_gcp = self._should_attempt_gcp()
        items: list[dict[str, Any]] | None = None
        source = "local_memory"

        if attempt_gcp:
            try:
                query = self._collection()
                if rating in ("up", "down"):
                    query = query.where(filter=FieldFilter("rating", "==", rating))
                query = query.order_by(
                    "recorded_at_epoch", direction=firestore.Query.DESCENDING
                ).limit(bounded)
                snapshots = query.stream(timeout=self._default_timeout)
                items = [
                    _normalize_firestore_dict(
                        snap.to_dict() or {},
                        doc_name=self._doc_path(snap.id),
                    )
                    for snap in snapshots
                    if getattr(snap, "exists", True)
                ]
                source = "firestore"
            except Exception as raw_exc:
                exc = self._translate_error(raw_exc, action="query feedback")
                with self._lock:
                    self._last_error = str(exc)
                if not _should_fallback_to_local(exc, strict_gcp=strict_gcp):
                    raise exc from raw_exc
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


def _merge_session_payload(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Merge an incoming session payload against a concurrent Firestore snapshot without losing turns."""
    if not existing:
        return dict(incoming)
    existing_turns = int(existing.get("turn_count") or 0)
    incoming_turns = int(incoming.get("turn_count") or 0)
    if existing_turns > incoming_turns:
        # Preserve the newer concurrent state while unioning remembered tickers/sections if present.
        merged = dict(existing)
        for list_field in ("remembered_tickers", "cited_chunk_ids"):
            if isinstance(existing.get(list_field), list) and isinstance(
                incoming.get(list_field), list
            ):
                merged[list_field] = list(
                    dict.fromkeys([*existing[list_field], *incoming[list_field]])
                )
        return merged
    return dict(incoming)


class FirestoreSessionStore(_FirestoreSdkBase):
    """Durable, user-isolated conversational session store backed by ``google-cloud-firestore``."""

    def __init__(
        self,
        config: TelemetryConfig | None = None,
        *,
        firestore_client: Any | None = None,
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
            firestore_client=firestore_client,
            credentials=credentials,
            default_timeout=15.0,
        )
        self._reads = 0
        self._writes = 0

    def is_live_enabled(self) -> bool:
        """Return True when session persistence targets live Cloud Firestore."""
        mode = self.config.feedback_backend
        if self._injected_client is not None:
            return mode != "local"
        if mode == "local":
            return False
        if mode == "gcp":
            return True
        return self.config.gcp_auto_triggered

    def load_session(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        """Load a persisted session document for `(user_id, session_id)` (None if absent)."""
        if not self.is_live_enabled():
            return None
        doc_id = session_firestore_doc_id(user_id, session_id)
        try:
            snap = self._collection().document(doc_id).get(timeout=self._default_timeout)
            if not getattr(snap, "exists", False):
                return None
            doc = _normalize_firestore_dict(
                snap.to_dict() or {},
                doc_name=self._doc_path(doc_id),
            )
            with self._lock:
                self._reads += 1
            return doc
        except Exception as raw_exc:
            exc = self._translate_error(raw_exc, action="load_session")
            with self._lock:
                self._last_error = str(exc)
            raise exc from raw_exc

    def save_session(self, user_id: str, session_id: str, payload: dict[str, Any]) -> None:
        """Transactionally upsert a session document so concurrent writes never clobber turns."""
        if not self.is_live_enabled():
            return
        doc_id = session_firestore_doc_id(user_id, session_id)
        client = self._get_client()
        doc_ref = self._collection().document(doc_id)
        try:
            from google.cloud import firestore

            transaction = client.transaction()

            @firestore.transactional
            def _upsert(tx: Any) -> None:
                snap = doc_ref.get(transaction=tx, timeout=self._default_timeout)
                existing = (
                    _normalize_firestore_dict(snap.to_dict() or {})
                    if getattr(snap, "exists", False)
                    else None
                )
                merged = _merge_session_payload(existing, payload)
                tx.set(doc_ref, merged)

            _upsert(transaction)
            with self._lock:
                self._writes += 1
        except Exception as raw_exc:
            exc = self._translate_error(raw_exc, action="save_session")
            with self._lock:
                self._last_error = str(exc)
            raise exc from raw_exc

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """Delete a persisted session document for `(user_id, session_id)`."""
        if not self.is_live_enabled():
            return False
        doc_id = session_firestore_doc_id(user_id, session_id)
        try:
            doc_ref = self._collection().document(doc_id)
            snap = doc_ref.get(timeout=self._default_timeout)
            if not getattr(snap, "exists", False):
                return False
            doc_ref.delete(timeout=self._default_timeout)
            return True
        except Exception as raw_exc:
            exc = self._translate_error(raw_exc, action="delete_session")
            with self._lock:
                self._last_error = str(exc)
            raise exc from raw_exc

    def list_sessions(self, user_id: str, *, limit: int = 25) -> list[dict[str, Any]]:
        """List recent sessions belonging to `user_id`."""
        from google.cloud.firestore_v1.base_query import FieldFilter

        if not self.is_live_enabled():
            return []
        bounded = max(1, min(int(limit), 100))
        try:
            query = (
                self._collection()
                .where(filter=FieldFilter("user_id", "==", user_id.strip().lower()))
                .limit(bounded)
            )
            snapshots = query.stream(timeout=self._default_timeout)
            items = [
                _normalize_firestore_dict(
                    snap.to_dict() or {},
                    doc_name=self._doc_path(snap.id),
                )
                for snap in snapshots
                if getattr(snap, "exists", True)
            ]
        except Exception as raw_exc:
            exc = self._translate_error(raw_exc, action="list_sessions")
            with self._lock:
                self._last_error = str(exc)
            raise exc from raw_exc
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


__all__ = [
    "DEFAULT_FIRESTORE_DATABASE_ID",
    "DEFAULT_FIRESTORE_FEEDBACK_COLLECTION",
    "DEFAULT_FIRESTORE_SESSION_COLLECTION",
    "FirestoreFeedbackStore",
    "FirestoreSessionStore",
    "session_firestore_doc_id",
]
