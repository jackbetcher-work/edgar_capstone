"""Conversational Session State, Compaction, Memory Bank, and ADK `App` Assembly.

Configures ADK `ContextCacheConfig`, token-triggered `EventsCompactionConfig`, bounded
`InMemorySessionService` / `BoundedInMemoryMemoryService`, and tenant-isolated multi-turn
conversational memory backed by `SessionStoreProtocol` (Firestore REST v1 in production).
Model tier routing lives in `runtime/model_config.py`; latency/cost attribution lives in
`runtime/latency_tracker.py`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.memory import InMemoryMemoryService
from google.adk.models import Gemini
from google.adk.models.base_llm import BaseLlm
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from edgar_capstone.agents.runtime.model_config import (
    DEFAULT_COMPACTION_RETENTION_SIZE,
    DEFAULT_COMPACTION_TOKEN_THRESHOLD,
    DEFAULT_CONTEXT_CACHE_INTERVALS,
    DEFAULT_CONTEXT_CACHE_MIN_TOKENS,
    DEFAULT_CONTEXT_CACHE_TTL_SECONDS,
    DEFAULT_MAX_ADK_MEMORY_SESSIONS,
    DEFAULT_MAX_LOCAL_SESSIONS,
    EPHEMERAL_AGENT_NAMES,
    is_ephemeral_session_id,
    resolve_model_for_task,
)
from edgar_capstone.common.formatting import utc_now_iso
from edgar_capstone.security.model_armor import ModelArmorSafetyPlugin
from edgar_capstone.telemetry import FirestoreSessionStore
from edgar_capstone.vector_db.chunker import estimate_token_count

logger = logging.getLogger(__name__)


@runtime_checkable
class SessionStoreProtocol(Protocol):
    """Minimal persistence contract for durable conversational session stores."""

    def is_live_enabled(self) -> bool: ...
    def load_session(self, user_id: str, session_id: str) -> dict[str, Any] | None: ...
    def save_session(self, user_id: str, session_id: str, payload: dict[str, Any]) -> None: ...
    def delete_session(self, user_id: str, session_id: str) -> bool: ...
    def list_sessions(self, user_id: str, *, limit: int = 25) -> list[dict[str, Any]]: ...
    def status_metadata(self) -> dict[str, Any]: ...


class BoundedInMemoryMemoryService(InMemoryMemoryService):
    """LRU-bounded ADK `InMemoryMemoryService` that ignores ephemeral utility sessions."""

    def __init__(self, *, max_sessions: int = DEFAULT_MAX_ADK_MEMORY_SESSIONS) -> None:
        super().__init__()
        self._max_sessions = max(1, int(max_sessions))
        self._insertion_order: OrderedDict[tuple[tuple[str, str], str | None], None] = OrderedDict()

    async def add_session_to_memory(self, session: Any) -> None:
        session_id = getattr(session, "id", None)
        if is_ephemeral_session_id(session_id):
            return
        await super().add_session_to_memory(session)
        user_key = (str(getattr(session, "app_name", "")), str(getattr(session, "user_id", "")))
        with self._lock:
            entry_key = (user_key, session_id)
            self._insertion_order.pop(entry_key, None)
            self._insertion_order[entry_key] = None
            while len(self._insertion_order) > self._max_sessions:
                (old_user_key, old_sid), _ = self._insertion_order.popitem(last=False)
                user_map = self._session_events.get(old_user_key)
                if user_map is not None:
                    user_map.pop(old_sid, None)
                    if not user_map:
                        self._session_events.pop(old_user_key, None)

    def drop_session(self, session_id: str, candidate_users: set[str]) -> bool:
        """Drop all cached memory events for `session_id` owned by any user in `candidate_users`."""
        removed = False
        with self._lock:
            for user_key in list(self._session_events.keys()):
                if user_key[1] in candidate_users:
                    user_map = self._session_events.get(user_key)
                    if user_map is not None and session_id in user_map:
                        user_map.pop(session_id, None)
                        self._insertion_order.pop((user_key, session_id), None)
                        removed = True
                        if not user_map:
                            self._session_events.pop(user_key, None)
        return removed

    def tracked_session_count(self) -> int:
        with self._lock:
            return len(self._insertion_order)


@dataclass
class TurnMemoryEntry:
    """Structured memory entry stored per conversational turn."""

    turn_index: int
    timestamp: str
    query: str
    intent: str
    tickers: list[str]
    gics_sector: str
    start_year: int
    end_year: int
    variance_highlights: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    cached_10k_keys: list[str] = field(default_factory=list)
    token_estimate: int = 0


@dataclass
class SessionMemoryState:
    """Active session memory state with automatic token-threshold compaction."""

    session_id: str
    user_id: str = "analyst"
    turns: list[TurnMemoryEntry] = field(default_factory=list)
    compacted_summary: str = ""
    compaction_count: int = 0
    remembered_tickers: list[str] = field(default_factory=list)
    last_start_year: int = 2024
    last_end_year: int = 2026
    last_gics_sector: str = "ALL"
    updated_at: str = field(default_factory=utc_now_iso)

    def to_firestore_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id.strip().lower(),
            "compacted_summary": self.compacted_summary,
            "compaction_count": int(self.compaction_count),
            "remembered_tickers": list(self.remembered_tickers),
            "last_start_year": int(self.last_start_year),
            "last_end_year": int(self.last_end_year),
            "last_gics_sector": self.last_gics_sector,
            "turn_count": len(self.turns),
            "turns_json": json.dumps([asdict(t) for t in self.turns]),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_firestore_dict(cls, doc: dict[str, Any]) -> SessionMemoryState:
        raw_turns = doc.get("turns_json") or "[]"
        try:
            parsed_turns = json.loads(raw_turns) if isinstance(raw_turns, str) else list(raw_turns)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to deserialize turns_json for session '%s': %s",
                doc.get("session_id", ""),
                exc,
            )
            parsed_turns = []
        turns: list[TurnMemoryEntry] = []
        for t in parsed_turns:
            if not isinstance(t, dict):
                continue
            turns.append(
                TurnMemoryEntry(
                    turn_index=int(t.get("turn_index") or len(turns) + 1),
                    timestamp=str(t.get("timestamp") or ""),
                    query=str(t.get("query") or ""),
                    intent=str(t.get("intent") or ""),
                    tickers=[str(x) for x in (t.get("tickers") or [])],
                    gics_sector=str(t.get("gics_sector") or "ALL"),
                    start_year=int(t.get("start_year") or 2024),
                    end_year=int(t.get("end_year") or 2026),
                    variance_highlights=[str(x) for x in (t.get("variance_highlights") or [])],
                    cited_chunk_ids=[str(x) for x in (t.get("cited_chunk_ids") or [])],
                    cached_10k_keys=[str(x) for x in (t.get("cached_10k_keys") or [])],
                    token_estimate=int(t.get("token_estimate") or 0),
                )
            )
        return cls(
            session_id=str(doc.get("session_id") or ""),
            user_id=str(doc.get("user_id") or "analyst"),
            turns=turns,
            compacted_summary=str(doc.get("compacted_summary") or ""),
            compaction_count=int(doc.get("compaction_count") or 0),
            remembered_tickers=[str(x) for x in (doc.get("remembered_tickers") or [])],
            last_start_year=int(doc.get("last_start_year") or 2024),
            last_end_year=int(doc.get("last_end_year") or 2026),
            last_gics_sector=str(doc.get("last_gics_sector") or "ALL"),
            updated_at=str(doc.get("updated_at") or utc_now_iso()),
        )


def _compact_session_turns(
    sess: SessionMemoryState,
    *,
    token_threshold: int,
    retention_turns: int,
) -> bool:
    """Compact older turns into a deterministic digest when session tokens cross `token_threshold`."""
    total_mem_tokens = sum(t.token_estimate for t in sess.turns) + estimate_token_count(
        sess.compacted_summary
    )
    if total_mem_tokens < token_threshold or len(sess.turns) <= retention_turns:
        return False

    older_turns = sess.turns[:-retention_turns]
    sess.turns = sess.turns[-retention_turns:]
    digest_parts: list[str] = [sess.compacted_summary] if sess.compacted_summary else []
    for old in older_turns:
        tk_str = ",".join(old.tickers) if old.tickers else old.gics_sector
        hl_str = "; ".join(old.variance_highlights[:1]) if old.variance_highlights else old.intent
        digest_parts.append(
            f"T{old.turn_index}({tk_str} FY{old.start_year}-{old.end_year}): {hl_str}"
        )
    sess.compacted_summary = " | ".join(digest_parts)[-2000:]
    sess.compaction_count += 1
    return True


class ConversationMemoryManager:
    """Manages durable, tenant-isolated conversational session state and follow-up entity recall.

    Sessions are keyed strictly by `(user_id, session_id)` so two authenticated users supplying
    the same `session_id` never collide. Per-session locking serializes concurrent `record_turn`
    load-mutate-save cycles to prevent lost updates across threads while allowing distinct
    sessions to proceed concurrently.
    """

    def __init__(
        self,
        *,
        token_threshold: int = DEFAULT_COMPACTION_TOKEN_THRESHOLD,
        retention_turns: int = DEFAULT_COMPACTION_RETENTION_SIZE,
        session_store: SessionStoreProtocol | None = None,
        refresh_ttl_seconds: float = 5.0,
        max_local_sessions: int = DEFAULT_MAX_LOCAL_SESSIONS,
        max_adk_memory_sessions: int = DEFAULT_MAX_ADK_MEMORY_SESSIONS,
    ) -> None:
        self.token_threshold = token_threshold
        self.retention_turns = retention_turns
        self.refresh_ttl_seconds = float(refresh_ttl_seconds)
        self._max_local_sessions = max(1, int(max_local_sessions))
        self._max_adk_memory_sessions = max(1, int(max_adk_memory_sessions))
        self._explicit_store = session_store
        self._default_store: SessionStoreProtocol | None = None
        self.adk_session_service = InMemorySessionService()
        self.adk_memory_service = BoundedInMemoryMemoryService(
            max_sessions=self._max_adk_memory_sessions
        )
        self._lock = threading.Lock()
        self._session_locks: dict[tuple[str, str], threading.Lock] = {}
        # LRU-bounded map keyed by (norm_user_id, norm_session_id) -> (loaded_monotonic, SessionMemoryState)
        self._sessions: OrderedDict[tuple[str, str], tuple[float, SessionMemoryState]] = (
            OrderedDict()
        )

    def _get_store(self) -> SessionStoreProtocol:
        if self._explicit_store is not None:
            return self._explicit_store
        with self._lock:
            if self._default_store is None:
                self._default_store = FirestoreSessionStore()
            return self._default_store

    @staticmethod
    def _session_key(session_id: str, user_id: str = "analyst") -> tuple[str, str]:
        norm_user = (user_id or "analyst").strip().lower() or "analyst"
        return (norm_user, session_id.strip())

    def _get_session_lock(self, key: tuple[str, str]) -> threading.Lock:
        with self._lock:
            lock = self._session_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[key] = lock
            return lock

    def _store_local_session_locked(
        self,
        key: tuple[str, str],
        state: SessionMemoryState,
    ) -> None:
        self._sessions[key] = (time.monotonic(), state)
        self._sessions.move_to_end(key)
        while len(self._sessions) > self._max_local_sessions:
            old_key, _ = self._sessions.popitem(last=False)
            self._session_locks.pop(old_key, None)

    def _evict_adk_session_state(self, session_id: str, user_id: str) -> bool:
        """Delete matching `(user_id, session_id)` entries from in-memory ADK session & memory services."""
        norm_user = (user_id or "analyst").strip().lower() or "analyst"
        candidate_users = {user_id.strip(), norm_user}
        removed_session = False
        sessions_by_app = getattr(self.adk_session_service, "sessions", None)
        if isinstance(sessions_by_app, dict):
            for app_sessions in list(sessions_by_app.values()):
                if not isinstance(app_sessions, dict):
                    continue
                for uid in candidate_users:
                    user_sessions = app_sessions.get(uid)
                    if isinstance(user_sessions, dict) and session_id in user_sessions:
                        del user_sessions[session_id]
                        removed_session = True
        removed_memory = self.adk_memory_service.drop_session(session_id, candidate_users)
        return removed_session or removed_memory

    def _load_or_refresh_session(
        self,
        session_id: str,
        user_id: str = "analyst",
        *,
        force_remote: bool = False,
    ) -> SessionMemoryState | None:
        key = self._session_key(session_id, user_id)
        now = time.monotonic()
        with self._lock:
            cached = self._sessions.get(key)
        store = self._get_store()
        live = store.is_live_enabled()
        if cached is not None:
            loaded_at, state = cached
            if (not live) or (not force_remote and (now - loaded_at) <= self.refresh_ttl_seconds):
                return state
        if not live:
            return cached[1] if cached is not None else None
        doc = store.load_session(key[0], key[1])
        if doc is None:
            return cached[1] if cached is not None else None
        hydrated = SessionMemoryState.from_firestore_dict(doc)
        with self._lock:
            self._store_local_session_locked(key, hydrated)
        return hydrated

    @staticmethod
    def _empty_memory_context(
        *,
        session_id: str | None,
        user_id: str,
        backend_label: str,
        started_at: float,
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "user_id": user_id,
            "has_prior_memory": False,
            "turn_count": 0,
            "remembered_tickers": [],
            "last_start_year": None,
            "last_end_year": None,
            "last_gics_sector": "ALL",
            "compaction_count": 0,
            "memory_summary_text": "",
            "memory_tokens": 0,
            "persistence_backend": backend_label,
            "memory_lookup_latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
        }

    def recall_session_context(
        self,
        session_id: str | None,
        user_id: str = "analyst",
    ) -> dict[str, Any]:
        """Retrieve active tenant-isolated session memory and measure context recall latency."""
        t0 = time.perf_counter()
        backend_label = "firestore" if self._get_store().is_live_enabled() else "in_memory"
        norm_user = (user_id or "analyst").strip().lower()
        if not session_id or not session_id.strip():
            return self._empty_memory_context(
                session_id=None,
                user_id=norm_user,
                backend_label=backend_label,
                started_at=t0,
            )

        sess = self._load_or_refresh_session(session_id, user_id)
        if sess is None or (not sess.turns and not sess.compacted_summary):
            return self._empty_memory_context(
                session_id=session_id.strip(),
                user_id=norm_user,
                backend_label=backend_label,
                started_at=t0,
            )

        lines: list[str] = []
        if sess.compacted_summary:
            lines.append(f"[Compacted Prior Context]: {sess.compacted_summary}")
        for t in sess.turns[-self.retention_turns :]:
            highlights = "; ".join(t.variance_highlights[:2]) if t.variance_highlights else "N/A"
            lines.append(
                f"[Turn {t.turn_index} | {','.join(t.tickers) or t.gics_sector} FY{t.start_year}-FY{t.end_year}]: "
                f"Query='{t.query}' -> {highlights}"
            )
        summary_text = "\n".join(lines)
        mem_tokens = estimate_token_count(summary_text)
        return {
            "session_id": sess.session_id,
            "user_id": sess.user_id,
            "has_prior_memory": True,
            "turn_count": len(sess.turns),
            "remembered_tickers": list(sess.remembered_tickers),
            "last_start_year": sess.last_start_year,
            "last_end_year": sess.last_end_year,
            "last_gics_sector": sess.last_gics_sector,
            "compaction_count": sess.compaction_count,
            "memory_summary_text": summary_text,
            "memory_tokens": mem_tokens,
            "persistence_backend": backend_label,
            "memory_lookup_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    def record_turn(
        self,
        *,
        session_id: str | None,
        user_id: str = "analyst",
        query: str,
        intent: str,
        tickers: list[str],
        gics_sector: str,
        start_year: int,
        end_year: int,
        variance_highlights: list[str] | None = None,
        cited_chunk_ids: list[str] | None = None,
        cached_10k_keys: list[str] | None = None,
        answer_markdown: str = "",
    ) -> dict[str, Any]:
        """Persist a completed turn into durable session memory and compact if over token threshold."""
        if not session_id or not session_id.strip():
            return {"persisted": False, "compaction_triggered": False, "turn_count": 0}

        key = self._session_key(session_id, user_id)
        session_lock = self._get_session_lock(key)

        # Serialize load -> mutate -> remote save per `(user_id, session_id)` to prevent lost updates.
        with session_lock:
            existing = self._load_or_refresh_session(session_id, user_id)
            with self._lock:
                sess = existing or SessionMemoryState(session_id=key[1], user_id=key[0])

                now_iso = utc_now_iso()
                turn_text = f"{query}\n{answer_markdown}"
                sess.turns.append(
                    TurnMemoryEntry(
                        turn_index=len(sess.turns) + 1,
                        timestamp=now_iso,
                        query=query,
                        intent=intent,
                        tickers=list(tickers),
                        gics_sector=gics_sector,
                        start_year=start_year,
                        end_year=end_year,
                        variance_highlights=list(variance_highlights or []),
                        cited_chunk_ids=list(cited_chunk_ids or []),
                        cached_10k_keys=list(cached_10k_keys or []),
                        token_estimate=estimate_token_count(turn_text),
                    )
                )
                if tickers:
                    sess.remembered_tickers = list(tickers)
                sess.last_start_year = start_year
                sess.last_end_year = end_year
                if gics_sector and gics_sector != "ALL":
                    sess.last_gics_sector = gics_sector
                sess.updated_at = now_iso

                compaction_triggered = _compact_session_turns(
                    sess,
                    token_threshold=self.token_threshold,
                    retention_turns=self.retention_turns,
                )
                self._store_local_session_locked(key, sess)
                turn_count = len(sess.turns)
                compaction_count = sess.compaction_count
                firestore_payload = sess.to_firestore_dict()

            store = self._get_store()
            backend_label = "in_memory"
            if store.is_live_enabled():
                store.save_session(key[0], key[1], firestore_payload)
                backend_label = "firestore"

        return {
            "persisted": True,
            "persistence_backend": backend_label,
            "session_id": key[1],
            "user_id": key[0],
            "turn_count": turn_count,
            "compaction_triggered": compaction_triggered,
            "compaction_method": "deterministic_digest" if compaction_triggered else None,
            "compaction_count": compaction_count,
        }

    @staticmethod
    def _session_summary_row(
        *,
        session_id: str,
        user_id: str,
        turn_count: int,
        remembered_tickers: list[str],
        last_gics_sector: str,
        last_start_year: int,
        last_end_year: int,
        updated_at: str,
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "user_id": user_id,
            "turn_count": turn_count,
            "remembered_tickers": list(remembered_tickers),
            "last_gics_sector": last_gics_sector,
            "last_start_year": last_start_year,
            "last_end_year": last_end_year,
            "updated_at": updated_at,
        }

    def list_user_sessions(self, user_id: str, *, limit: int = 25) -> list[dict[str, Any]]:
        """List recent sessions owned by `user_id`."""
        norm_user = (user_id or "analyst").strip().lower() or "analyst"
        store = self._get_store()
        if store.is_live_enabled():
            docs = store.list_sessions(norm_user, limit=limit)
            return [
                self._session_summary_row(
                    session_id=str(d.get("session_id") or ""),
                    user_id=str(d.get("user_id") or norm_user),
                    turn_count=int(d.get("turn_count") or 0),
                    remembered_tickers=list(d.get("remembered_tickers") or []),
                    last_gics_sector=str(d.get("last_gics_sector") or "ALL"),
                    last_start_year=int(d.get("last_start_year") or 2024),
                    last_end_year=int(d.get("last_end_year") or 2026),
                    updated_at=str(d.get("updated_at") or ""),
                )
                for d in docs
            ]
        with self._lock:
            items = [
                self._session_summary_row(
                    session_id=s.session_id,
                    user_id=s.user_id,
                    turn_count=len(s.turns),
                    remembered_tickers=s.remembered_tickers,
                    last_gics_sector=s.last_gics_sector,
                    last_start_year=s.last_start_year,
                    last_end_year=s.last_end_year,
                    updated_at=s.updated_at,
                )
                for (u, _), (_, s) in self._sessions.items()
                if u == norm_user
            ]
        items.sort(key=lambda d: str(d.get("updated_at") or ""), reverse=True)
        return items[:limit]

    def delete_session(self, session_id: str, user_id: str = "analyst") -> bool:
        """Delete a session belonging to `(user_id, session_id)` from L1 cache, ADK services, and Firestore."""
        key = self._session_key(session_id, user_id)
        session_lock = self._get_session_lock(key)
        with session_lock:
            with self._lock:
                removed_local = self._sessions.pop(key, None) is not None
                self._session_locks.pop(key, None)
            removed_adk = self._evict_adk_session_state(key[1], user_id)
            store = self._get_store()
            removed_remote = (
                store.delete_session(key[0], key[1]) if store.is_live_enabled() else False
            )
        return removed_local or removed_adk or removed_remote

    def persistence_status(self) -> dict[str, Any]:
        """Return health metadata for session memory storage."""
        return self._get_store().status_metadata()

    def reset(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._session_locks.clear()
            self.adk_session_service = InMemorySessionService()
            self.adk_memory_service = BoundedInMemoryMemoryService(
                max_sessions=self._max_adk_memory_sessions
            )


_GLOBAL_MEMORY_MANAGER = ConversationMemoryManager()


def get_conversation_memory_manager() -> ConversationMemoryManager:
    """Return the process-wide `ConversationMemoryManager` singleton."""
    return _GLOBAL_MEMORY_MANAGER


async def after_agent_memory_callback(
    callback_context: CallbackContext,
) -> genai_types.Content | None:
    """ADK `after_agent_callback` persisting non-ephemeral sessions to Memory Bank & tracking duration."""
    state = callback_context.state
    start_ts = state.get("_agent_start_monotonic")
    if start_ts is not None:
        state["last_agent_total_duration_ms"] = round(
            (time.perf_counter() - float(start_ts)) * 1000.0, 3
        )
    agent_name = str(getattr(callback_context, "agent_name", "") or "")
    adk_session = getattr(callback_context, "session", None)
    sess_id = getattr(adk_session, "id", None) or state.get("session_id")
    if agent_name in EPHEMERAL_AGENT_NAMES or (
        sess_id is not None and is_ephemeral_session_id(str(sess_id))
    ):
        return None
    add_mem = getattr(callback_context, "add_session_to_memory", None)
    if callable(add_mem):
        try:
            await add_mem()
        except Exception as exc:
            logger.debug("Optional ADK Memory Bank sync skipped: %s", exc)
    return None


def create_edgar_adk_app(
    root_agent: BaseAgent,
    *,
    app_name: str = "agents",
    min_cache_tokens: int = DEFAULT_CONTEXT_CACHE_MIN_TOKENS,
    cache_ttl_seconds: int = DEFAULT_CONTEXT_CACHE_TTL_SECONDS,
    compaction_token_threshold: int = DEFAULT_COMPACTION_TOKEN_THRESHOLD,
    compaction_retention_size: int = DEFAULT_COMPACTION_RETENTION_SIZE,
    summarizer_llm: BaseLlm | None = None,
    enable_context_cache: bool = True,
    plugins: list[BasePlugin] | None = None,
) -> App:
    """Create the production ADK `App` with Model Armor safety plugin, Context Caching & Compaction."""
    flash_model = resolve_model_for_task("context_compaction")
    cache_cfg = (
        ContextCacheConfig(
            min_tokens=min_cache_tokens,
            ttl_seconds=cache_ttl_seconds,
            cache_intervals=DEFAULT_CONTEXT_CACHE_INTERVALS,
        )
        if enable_context_cache
        else None
    )
    resolved_plugins: list[BasePlugin] = (
        list(plugins) if plugins is not None else [ModelArmorSafetyPlugin()]
    )
    return App(
        name=app_name,
        root_agent=root_agent,
        plugins=resolved_plugins,
        context_cache_config=cache_cfg,
        events_compaction_config=EventsCompactionConfig(
            token_threshold=compaction_token_threshold,
            event_retention_size=compaction_retention_size,
            summarizer=LlmEventSummarizer(
                llm=summarizer_llm if summarizer_llm is not None else Gemini(model=flash_model)
            ),
        ),
    )
