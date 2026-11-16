"""Vertex AI Context Caching for full SEC 10-K sections (Item 1A Risk Factors / Item 7 MD&A).

`FinancialAnalystAgent` grounds its synthesis in the complete text of the 10-K sections that
the structured ReAct loop queried, not just the top-ranked passages. Those sections are
registered as remote Gemini `CachedContent` resources (`google.genai.Client.caches`) so that:

- repeated questions over the same company-years reuse the cached tokens (cached input
  is billed at a steep discount, and the savings show up in `cached_content_token_count`);
- every Cloud Run instance shares the same cache. Each cache gets a deterministic
  `display_name`, derived from the model, system instruction, tools, and section content.
  Instances discover each other's caches with `caches.list()`, so no extra registry
  infrastructure is needed.

Gemini only accepts `cached_content` on requests that don't also set `system_instruction`,
`tools`, or `tool_config`. Because of that, a cache holds the agent's resolved system
instruction and tool declarations together with the 10-K bundle. A request can reference
only one cache, so the cache unit is the bundle of sections needed for one synthesis
(sorted, so the same scope always maps to the same cache).

Fallbacks:
- the bundle is below the model's minimum cacheable size: the sections are sent inline;
- no genai client, caching disabled, or the cache API fails: passages-only grounding
  (the previous behaviour), with the reason recorded in session state and stats.

Configuration (environment):
- `EDGAR_10K_CONTEXT_CACHE` = `auto` (default) | `off`
- `EDGAR_10K_CONTEXT_CACHE_TTL_SECONDS` (default 1800)
- `EDGAR_10K_CONTEXT_CACHE_MAX_TOKENS` (default 200000): bundle token budget
- `EDGAR_10K_CONTEXT_CACHE_MIN_TOKENS`: override the per-model minimum cacheable size
- `EDGAR_10K_CONTEXT_CACHE_DISCOVERY` = `true` (default) | `false`: cross-instance lookup
- `EDGAR_10K_CONTEXT_CACHE_CREATE_TIMEOUT_SECONDS` (default 20) and
  `EDGAR_10K_CONTEXT_CACHE_DISCOVERY_TIMEOUT_SECONDS` (default 5): keep a slow or failing
  cache API from eating into Time-to-First-Token
- `EDGAR_10K_CONTEXT_CACHE_FAILURE_COOLDOWN_SECONDS` (default 600): after a create failure
  or timeout, the circuit breaker skips remote calls for this long and model calls use
  passages-only grounding without waiting on the cache API
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from edgar_capstone.agents.tools import get_full_10k_section
from edgar_capstone.vector_db.chunker import estimate_token_count

logger = logging.getLogger(__name__)

TENK_CACHE_SECTION_KEYS_STATE = "tenk_context_cache_section_keys"
TENK_CACHE_DECISION_STATE = "tenk_context_cache"

DEFAULT_TENK_CACHE_TTL_SECONDS = 1800
DEFAULT_TENK_CACHE_MAX_BUNDLE_TOKENS = 200_000
DEFAULT_TENK_CACHE_REFRESH_MARGIN_SECONDS = 120
DEFAULT_TENK_CACHE_MAX_L1_ENTRIES = 128
DEFAULT_TENK_CACHE_MAX_DISCOVERY_SCAN = 500
DEFAULT_TENK_CACHE_CREATE_TIMEOUT_SECONDS = 20
DEFAULT_TENK_CACHE_DISCOVERY_TIMEOUT_SECONDS = 5
DEFAULT_TENK_CACHE_FAILURE_COOLDOWN_SECONDS = 600
TENK_CACHE_DISPLAY_NAME_PREFIX = "edgar10k-"

# Explicit-cache minimum sizes (mirrors ADK's GeminiContextCacheManager floors).
_MODEL_MIN_CACHE_TOKENS: tuple[tuple[str, int], ...] = (
    ("gemini-2.5-", 2048),
    ("gemini-3", 4096),
)
_DEFAULT_MIN_CACHE_TOKENS = 4096

_FALSY_ENV_VALUES = frozenset({"0", "false", "no", "off", "disabled"})

CacheMode = Literal["cached", "inline", "skipped"]
CacheSource = Literal["l1", "remote", "created"]


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        return default
    return value if value > 0 else default


def _model_short_name(model: str | None) -> str:
    return (model or "").rsplit("/", maxsplit=1)[-1].strip()


def _jsonable(value: Any) -> Any:
    """Canonical JSON-safe representation of genai config fragments for fingerprinting."""
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_none=True)
    return str(value)


# ---------------------------------------------------------------------------
# 10-K section bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TenKSectionBundle:
    """Full-text 10-K sections assembled for one grounded synthesis call."""

    section_keys: tuple[str, ...]
    dropped_section_keys: tuple[str, ...]
    text: str
    content_sha256: str
    token_estimate: int

    def to_content(self) -> genai_types.Content:
        return genai_types.Content(role="user", parts=[genai_types.Part(text=self.text)])


_SECTION_TITLES = {
    "item_1a": "Item 1A. Risk Factors",
    "item_7": "Item 7. Management's Discussion and Analysis",
}

_BUNDLE_PREAMBLE = (
    "# Authoritative SEC 10-K Source Sections\n"
    "The following are the complete Item 1A (Risk Factors) and Item 7 (MD&A) sections for the "
    "company-years in scope. Ground every narrative claim in this text, cite it inline as "
    "`[TICKER FYxxxx | Section Title | chunk_id](sec_filing_url)`, and never use outside knowledge "
    "for causal explanations of variances.\n"
)


def build_tenk_section_bundle(
    section_keys: Sequence[str],
    *,
    max_tokens: int = DEFAULT_TENK_CACHE_MAX_BUNDLE_TOKENS,
) -> TenKSectionBundle | None:
    """Assemble full 10-K section text for `section_keys` within a token budget.

    Sections are admitted in the given (priority) order until `max_tokens` is reached and then
    sorted by key, so the same scope always yields the same bundle and cache fingerprint.
    """
    seen: set[str] = set()
    admitted: list[tuple[str, dict[str, Any]]] = []
    dropped: list[str] = []
    budget = estimate_token_count(_BUNDLE_PREAMBLE)

    for raw_key in section_keys:
        key = str(raw_key).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        entry = get_full_10k_section(key)
        if entry is None or not str(entry.get("full_text") or "").strip():
            dropped.append(key)
            continue
        tokens = int(entry.get("token_estimate") or estimate_token_count(entry["full_text"]))
        if budget + tokens > max_tokens:
            dropped.append(key)
            continue
        budget += tokens
        admitted.append((key, entry))

    if not admitted:
        return None

    admitted.sort(key=lambda item: item[0])
    blocks = [_BUNDLE_PREAMBLE]
    for key, entry in admitted:
        section_id = str(entry.get("section_id") or "")
        blocks.append(
            f'<10k_section cache_key="{key}" ticker="{entry.get("ticker")}" '
            f'company="{entry.get("company_name")}" fiscal_year="{entry.get("fiscal_year")}" '
            f'section="{_SECTION_TITLES.get(section_id, section_id)}" '
            f'sec_filing_url="{entry.get("sec_filing_url") or ""}">\n'
            f"{str(entry['full_text']).strip()}\n"
            f"</10k_section>"
        )
    text = "\n\n".join(blocks)
    return TenKSectionBundle(
        section_keys=tuple(k for k, _ in admitted),
        dropped_section_keys=tuple(dropped),
        text=text,
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        token_estimate=budget,
    )


# ---------------------------------------------------------------------------
# Cache handles, decisions & manager
# ---------------------------------------------------------------------------


@dataclass
class TenKCacheHandle:
    """A remote `CachedContent` resource registered for a 10-K bundle fingerprint."""

    cache_key: str
    name: str
    model: str
    display_name: str
    expire_time: dt.datetime
    token_estimate: int = 0
    section_keys: tuple[str, ...] = ()

    def remaining_seconds(self, now: dt.datetime | None = None) -> float:
        return (self.expire_time - (now or _utcnow())).total_seconds()


@dataclass(frozen=True)
class TenKCacheDecision:
    """Outcome of resolving 10-K context for one model call."""

    mode: CacheMode
    reason: str
    model: str
    cache_key: str | None = None
    cache_name: str | None = None
    source: CacheSource | None = None
    section_keys: tuple[str, ...] = ()
    dropped_section_keys: tuple[str, ...] = ()
    token_estimate: int = 0
    expire_time: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "model": self.model,
            "cache_key": self.cache_key,
            "cache_name": self.cache_name,
            "source": self.source,
            "section_keys": list(self.section_keys),
            "dropped_section_keys": list(self.dropped_section_keys),
            "token_estimate": self.token_estimate,
            "expire_time": self.expire_time,
            "latency_ms": self.latency_ms,
        }


def _is_min_token_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "token" in text and ("minimum" in text or "min_total_token_count" in text)


def is_cached_content_error(exc: BaseException) -> bool:
    """True when a generate call failed because its `cached_content` is missing or expired."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc).lower()
    mentions_cache = "cachedcontent" in text.replace(" ", "") or "cached content" in text
    return mentions_cache and (code in (400, 403, 404) or "not found" in text or "expired" in text)


@dataclass
class _Counters:
    l1_hits: int = 0
    remote_hits: int = 0
    creates: int = 0
    refreshes: int = 0
    inline: int = 0
    skipped: int = 0
    create_failures: int = 0
    timeouts: int = 0
    circuit_opens: int = 0
    discovery_errors: int = 0
    refresh_errors: int = 0
    invalidations: int = 0
    cached_tokens_registered: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)


class TenKContextCacheManager:
    """Registers 10-K section bundles as remote Vertex AI / Gemini `CachedContent` resources."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        ttl_seconds: int = DEFAULT_TENK_CACHE_TTL_SECONDS,
        max_bundle_tokens: int = DEFAULT_TENK_CACHE_MAX_BUNDLE_TOKENS,
        min_tokens_override: int | None = None,
        refresh_margin_seconds: int = DEFAULT_TENK_CACHE_REFRESH_MARGIN_SECONDS,
        discovery_enabled: bool = True,
        max_l1_entries: int = DEFAULT_TENK_CACHE_MAX_L1_ENTRIES,
        max_discovery_scan: int = DEFAULT_TENK_CACHE_MAX_DISCOVERY_SCAN,
        create_timeout_seconds: float = DEFAULT_TENK_CACHE_CREATE_TIMEOUT_SECONDS,
        discovery_timeout_seconds: float = DEFAULT_TENK_CACHE_DISCOVERY_TIMEOUT_SECONDS,
        failure_cooldown_seconds: float = DEFAULT_TENK_CACHE_FAILURE_COOLDOWN_SECONDS,
        client: Any | None = None,
    ) -> None:
        self.enabled = enabled
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_bundle_tokens = int(max_bundle_tokens)
        self.min_tokens_override = min_tokens_override
        self.refresh_margin_seconds = int(refresh_margin_seconds)
        self.discovery_enabled = discovery_enabled
        self.max_l1_entries = max(1, int(max_l1_entries))
        self.max_discovery_scan = int(max_discovery_scan)
        self.create_timeout_seconds = float(create_timeout_seconds)
        self.discovery_timeout_seconds = float(discovery_timeout_seconds)
        self.failure_cooldown_seconds = float(failure_cooldown_seconds)
        self.client = client
        self._circuit_open_until = 0.0
        self._last_failure: str | None = None
        self._handles: OrderedDict[str, TenKCacheHandle] = OrderedDict()
        self._bundles: OrderedDict[tuple[str, ...], TenKSectionBundle | None] = OrderedDict()
        self._lock = threading.RLock()
        self._counters = _Counters()
        self._last_decision: TenKCacheDecision | None = None

    @classmethod
    def from_env(cls) -> TenKContextCacheManager:
        mode = os.environ.get("EDGAR_10K_CONTEXT_CACHE", "auto").strip().lower()
        raw_min = os.environ.get("EDGAR_10K_CONTEXT_CACHE_MIN_TOKENS", "").strip()
        discovery = os.environ.get("EDGAR_10K_CONTEXT_CACHE_DISCOVERY", "true").strip().lower()
        return cls(
            enabled=mode not in _FALSY_ENV_VALUES,
            ttl_seconds=_env_int(
                "EDGAR_10K_CONTEXT_CACHE_TTL_SECONDS", DEFAULT_TENK_CACHE_TTL_SECONDS
            ),
            max_bundle_tokens=_env_int(
                "EDGAR_10K_CONTEXT_CACHE_MAX_TOKENS", DEFAULT_TENK_CACHE_MAX_BUNDLE_TOKENS
            ),
            min_tokens_override=int(raw_min) if raw_min.isdigit() else None,
            discovery_enabled=discovery not in _FALSY_ENV_VALUES,
            create_timeout_seconds=_env_int(
                "EDGAR_10K_CONTEXT_CACHE_CREATE_TIMEOUT_SECONDS",
                DEFAULT_TENK_CACHE_CREATE_TIMEOUT_SECONDS,
            ),
            discovery_timeout_seconds=_env_int(
                "EDGAR_10K_CONTEXT_CACHE_DISCOVERY_TIMEOUT_SECONDS",
                DEFAULT_TENK_CACHE_DISCOVERY_TIMEOUT_SECONDS,
            ),
            failure_cooldown_seconds=_env_int(
                "EDGAR_10K_CONTEXT_CACHE_FAILURE_COOLDOWN_SECONDS",
                DEFAULT_TENK_CACHE_FAILURE_COOLDOWN_SECONDS,
            ),
        )

    # -- configuration helpers -------------------------------------------------

    def min_tokens_for(self, model: str | None) -> int:
        if self.min_tokens_override is not None:
            return self.min_tokens_override
        short = _model_short_name(model)
        for prefix, floor in _MODEL_MIN_CACHE_TOKENS:
            if short.startswith(prefix):
                return floor
        return _DEFAULT_MIN_CACHE_TOKENS

    @staticmethod
    def fingerprint(
        *,
        model: str,
        system_instruction: Any,
        tools: Any,
        tool_config: Any,
        bundle: TenKSectionBundle,
    ) -> str:
        payload = {
            "model": _model_short_name(model),
            "system_instruction": _jsonable(system_instruction),
            "tools": _jsonable(tools),
            "tool_config": _jsonable(tool_config),
            "bundle_sha256": bundle.content_sha256,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def display_name_for(cache_key: str) -> str:
        return f"{TENK_CACHE_DISPLAY_NAME_PREFIX}{cache_key[:48]}"

    def get_bundle(self, section_keys: Sequence[str]) -> TenKSectionBundle | None:
        """Build (or reuse) the bundle for `section_keys`; memoized across model calls."""
        memo_key = tuple(str(k) for k in section_keys)
        with self._lock:
            if memo_key in self._bundles:
                self._bundles.move_to_end(memo_key)
                return self._bundles[memo_key]
        bundle = build_tenk_section_bundle(memo_key, max_tokens=self.max_bundle_tokens)
        with self._lock:
            self._bundles[memo_key] = bundle
            while len(self._bundles) > 16:
                self._bundles.popitem(last=False)
        return bundle

    # -- L1 handle registry ------------------------------------------------------

    def _l1_get(self, cache_key: str) -> TenKCacheHandle | None:
        with self._lock:
            handle = self._handles.get(cache_key)
            if handle is not None:
                self._handles.move_to_end(cache_key)
            return handle

    def _l1_put(self, handle: TenKCacheHandle) -> None:
        with self._lock:
            self._handles[handle.cache_key] = handle
            self._handles.move_to_end(handle.cache_key)
            while len(self._handles) > self.max_l1_entries:
                self._handles.popitem(last=False)

    def invalidate(self, *, cache_key: str | None = None, cache_name: str | None = None) -> int:
        """Drop L1 handles by fingerprint or resource name (e.g. after a 404 at generate time)."""
        with self._lock:
            doomed = [
                k
                for k, h in self._handles.items()
                if (cache_key is not None and k == cache_key)
                or (cache_name is not None and h.name == cache_name)
                or (cache_key is None and cache_name is None)
            ]
            for k in doomed:
                del self._handles[k]
            self._counters.invalidations += len(doomed)
            return len(doomed)

    def reset(self) -> None:
        with self._lock:
            self._handles.clear()
            self._bundles.clear()
            self._counters = _Counters()
            self._last_decision = None
            self._circuit_open_until = 0.0
            self._last_failure = None

    # -- circuit breaker -----------------------------------------------------

    def circuit_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self._circuit_open_until

    def _trip_circuit(self, failure: str) -> None:
        with self._lock:
            self._circuit_open_until = time.monotonic() + self.failure_cooldown_seconds
            self._last_failure = failure
            self._counters.circuit_opens += 1

    # -- remote operations ---------------------------------------------------

    async def _discover(self, client: Any, cache_key: str, model: str) -> TenKCacheHandle | None:
        display_name = self.display_name_for(cache_key)
        short_model = _model_short_name(model)
        pager = await client.aio.caches.list(
            config=genai_types.ListCachedContentsConfig(page_size=100)
        )
        scanned = 0
        best: TenKCacheHandle | None = None
        async for cached in pager:
            scanned += 1
            if scanned > self.max_discovery_scan:
                break
            if getattr(cached, "display_name", None) != display_name:
                continue
            if _model_short_name(getattr(cached, "model", None)) != short_model:
                continue
            expire_time = getattr(cached, "expire_time", None)
            if expire_time is None:
                continue
            candidate = TenKCacheHandle(
                cache_key=cache_key,
                name=str(cached.name),
                model=short_model,
                display_name=display_name,
                expire_time=expire_time,
            )
            if best is None or candidate.expire_time > best.expire_time:
                best = candidate
        return best

    async def _create(
        self,
        client: Any,
        *,
        cache_key: str,
        model: str,
        system_instruction: Any,
        tools: Any,
        tool_config: Any,
        bundle: TenKSectionBundle,
    ) -> TenKCacheHandle:
        display_name = self.display_name_for(cache_key)
        config = genai_types.CreateCachedContentConfig(
            display_name=display_name,
            contents=[bundle.to_content()],
            ttl=f"{self.ttl_seconds}s",
        )
        if system_instruction:
            config.system_instruction = system_instruction
        if tools:
            config.tools = list(tools)
        if tool_config:
            config.tool_config = tool_config
        cached = await client.aio.caches.create(model=model, config=config)
        expire_time = getattr(cached, "expire_time", None) or (
            _utcnow() + dt.timedelta(seconds=self.ttl_seconds)
        )
        return TenKCacheHandle(
            cache_key=cache_key,
            name=str(cached.name),
            model=_model_short_name(model),
            display_name=display_name,
            expire_time=expire_time,
            token_estimate=bundle.token_estimate,
            section_keys=bundle.section_keys,
        )

    async def _maybe_refresh(self, client: Any, handle: TenKCacheHandle) -> None:
        """Extend the TTL once less than half of it remains (keeps hot caches alive)."""
        if handle.remaining_seconds() >= self.ttl_seconds / 2:
            return
        try:
            updated = await client.aio.caches.update(
                name=handle.name,
                config=genai_types.UpdateCachedContentConfig(ttl=f"{self.ttl_seconds}s"),
            )
            handle.expire_time = getattr(updated, "expire_time", None) or (
                _utcnow() + dt.timedelta(seconds=self.ttl_seconds)
            )
            with self._lock:
                self._counters.refreshes += 1
        except Exception as exc:  # refresh is best-effort
            logger.warning("10-K context cache TTL refresh failed for %s: %s", handle.name, exc)
            with self._lock:
                self._counters.refresh_errors += 1

    # -- main entrypoint -----------------------------------------------------

    def _decide(
        self,
        mode: CacheMode,
        reason: str,
        *,
        model: str,
        t0: float,
        bundle: TenKSectionBundle | None = None,
        handle: TenKCacheHandle | None = None,
        source: CacheSource | None = None,
        cache_key: str | None = None,
    ) -> TenKCacheDecision:
        decision = TenKCacheDecision(
            mode=mode,
            reason=reason,
            model=_model_short_name(model),
            cache_key=cache_key or (handle.cache_key if handle else None),
            cache_name=handle.name if handle else None,
            source=source,
            section_keys=bundle.section_keys if bundle else (),
            dropped_section_keys=bundle.dropped_section_keys if bundle else (),
            token_estimate=bundle.token_estimate if bundle else 0,
            expire_time=handle.expire_time.isoformat() if handle else None,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        )
        with self._lock:
            c = self._counters
            if mode == "cached":
                if source == "l1":
                    c.l1_hits += 1
                elif source == "remote":
                    c.remote_hits += 1
                else:
                    c.creates += 1
                    c.cached_tokens_registered += decision.token_estimate
            elif mode == "inline":
                c.inline += 1
            else:
                c.skipped += 1
                bucket = reason.split(":", 1)[0]
                c.skip_reasons[bucket] = c.skip_reasons.get(bucket, 0) + 1
            self._last_decision = decision
        return decision

    async def acquire(
        self,
        *,
        client: Any | None,
        model: str,
        system_instruction: Any,
        tools: Any,
        tool_config: Any,
        bundle: TenKSectionBundle | None,
    ) -> TenKCacheDecision:
        """Resolve a `CachedContent` for this request: L1 -> cross-instance discovery -> create."""
        t0 = time.perf_counter()
        if not self.enabled:
            return self._decide("skipped", "disabled", model=model, t0=t0, bundle=bundle)
        if bundle is None or not bundle.section_keys:
            return self._decide("skipped", "no_sections", model=model, t0=t0)
        if bundle.token_estimate < self.min_tokens_for(model):
            return self._decide(
                "inline", "below_minimum_cacheable_tokens", model=model, t0=t0, bundle=bundle
            )
        if client is None or getattr(client, "aio", None) is None:
            return self._decide("skipped", "no_genai_client", model=model, t0=t0, bundle=bundle)

        cache_key = self.fingerprint(
            model=model,
            system_instruction=system_instruction,
            tools=tools,
            tool_config=tool_config,
            bundle=bundle,
        )

        handle = self._l1_get(cache_key)
        if handle is not None:
            if handle.remaining_seconds() > self.refresh_margin_seconds:
                await self._maybe_refresh(client, handle)
                return self._decide(
                    "cached",
                    "l1_hit",
                    model=model,
                    t0=t0,
                    bundle=bundle,
                    handle=handle,
                    source="l1",
                )
            self.invalidate(cache_key=cache_key)

        if self.circuit_open():
            return self._decide(
                "skipped", "circuit_open", model=model, t0=t0, bundle=bundle, cache_key=cache_key
            )

        if self.discovery_enabled:
            try:
                remote = await asyncio.wait_for(
                    self._discover(client, cache_key, model),
                    timeout=self.discovery_timeout_seconds,
                )
            except Exception as exc:
                logger.warning("10-K context cache discovery failed: %s", exc)
                with self._lock:
                    self._counters.discovery_errors += 1
                remote = None
            if remote is not None and remote.remaining_seconds() > self.refresh_margin_seconds:
                remote.token_estimate = bundle.token_estimate
                remote.section_keys = bundle.section_keys
                await self._maybe_refresh(client, remote)
                self._l1_put(remote)
                return self._decide(
                    "cached",
                    "remote_hit",
                    model=model,
                    t0=t0,
                    bundle=bundle,
                    handle=remote,
                    source="remote",
                )

        try:
            created = await asyncio.wait_for(
                self._create(
                    client,
                    cache_key=cache_key,
                    model=model,
                    system_instruction=system_instruction,
                    tools=tools,
                    tool_config=tool_config,
                    bundle=bundle,
                ),
                timeout=self.create_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "10-K context cache creation exceeded %.0fs; using passages-only grounding.",
                self.create_timeout_seconds,
            )
            with self._lock:
                self._counters.timeouts += 1
            self._trip_circuit("create_timeout")
            return self._decide(
                "skipped",
                "create_failed:TimeoutError",
                model=model,
                t0=t0,
                bundle=bundle,
                cache_key=cache_key,
            )
        except Exception as exc:
            if _is_min_token_error(exc):
                return self._decide(
                    "inline",
                    "below_minimum_cacheable_tokens_server",
                    model=model,
                    t0=t0,
                    bundle=bundle,
                    cache_key=cache_key,
                )
            logger.warning(
                "10-K context cache creation failed (%s): %s",
                type(exc).__name__,
                str(exc)[:500],
            )
            with self._lock:
                self._counters.create_failures += 1
            self._trip_circuit(f"{type(exc).__name__}: {str(exc)[:300]}")
            return self._decide(
                "skipped",
                f"create_failed:{type(exc).__name__}",
                model=model,
                t0=t0,
                bundle=bundle,
                cache_key=cache_key,
            )

        self._l1_put(created)
        return self._decide(
            "cached", "created", model=model, t0=t0, bundle=bundle, handle=created, source="created"
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            c = self._counters
            cached_total = c.l1_hits + c.remote_hits + c.creates
            reuse = c.l1_hits + c.remote_hits
            now = _utcnow()
            return {
                "enabled": self.enabled,
                "ttl_seconds": self.ttl_seconds,
                "max_bundle_tokens": self.max_bundle_tokens,
                "discovery_enabled": self.discovery_enabled,
                "active_handles": [
                    {
                        "cache_name": h.name,
                        "model": h.model,
                        "section_keys": list(h.section_keys),
                        "token_estimate": h.token_estimate,
                        "remaining_seconds": round(h.remaining_seconds(now), 1),
                    }
                    for h in self._handles.values()
                    if h.remaining_seconds(now) > 0
                ],
                "l1_hits": c.l1_hits,
                "remote_hits": c.remote_hits,
                "creates": c.creates,
                "reuse_rate": round(reuse / cached_total, 4) if cached_total else 0.0,
                "refreshes": c.refreshes,
                "inline": c.inline,
                "skipped": c.skipped,
                "skip_reasons": dict(c.skip_reasons),
                "create_failures": c.create_failures,
                "timeouts": c.timeouts,
                "circuit_opens": c.circuit_opens,
                "circuit_open": time.monotonic() < self._circuit_open_until,
                "circuit_open_remaining_seconds": round(
                    max(0.0, self._circuit_open_until - time.monotonic()), 1
                ),
                "last_failure": self._last_failure,
                "discovery_errors": c.discovery_errors,
                "refresh_errors": c.refresh_errors,
                "invalidations": c.invalidations,
                "cached_tokens_registered": c.cached_tokens_registered,
                "last_decision": self._last_decision.to_dict() if self._last_decision else None,
            }


_MANAGER_LOCK = threading.Lock()
_MANAGER: TenKContextCacheManager | None = None


def get_tenk_context_cache_manager() -> TenKContextCacheManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = TenKContextCacheManager.from_env()
        return _MANAGER


def configure_tenk_context_cache_manager(
    manager: TenKContextCacheManager | None,
) -> None:
    """Install a manager (tests / custom clients); `None` re-reads the environment on next use."""
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = manager


# ---------------------------------------------------------------------------
# ADK before_model callback
# ---------------------------------------------------------------------------


def _resolve_genai_client(callback_context: CallbackContext) -> Any | None:
    """Reuse the agent's own `Gemini` client (Vertex AI or API key, with retry options)."""
    invocation_ctx = getattr(callback_context, "_invocation_context", None)
    agent = getattr(invocation_ctx, "agent", None)
    try:
        llm = getattr(agent, "canonical_model", None)
    except Exception:
        return None
    return getattr(llm, "api_client", None) if llm is not None else None


def apply_tenk_cache_decision(
    llm_request: LlmRequest,
    decision: TenKCacheDecision,
    bundle: TenKSectionBundle | None,
) -> None:
    """Mutate the outgoing request to use the remote cache or the inline bundle."""
    if decision.mode == "cached" and decision.cache_name:
        if llm_request.config is None:
            llm_request.config = genai_types.GenerateContentConfig()
        # System instruction, tools, and tool config live inside the CachedContent.
        llm_request.config.system_instruction = None
        llm_request.config.tools = None
        llm_request.config.tool_config = None
        llm_request.config.cached_content = decision.cache_name
        # Skip ADK's own prefix cache manager for this call; it would replace cached_content.
        llm_request.cache_config = None
    elif decision.mode == "inline" and bundle is not None:
        llm_request.contents = [bundle.to_content(), *list(llm_request.contents or [])]


async def tenk_context_cache_before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Attach the 10-K sections in scope to the model call as a remote `CachedContent`.

    No-op unless the invocation seeded `tenk_context_cache_section_keys` in session state
    (see `run_live_adk_financial_reasoning_async`).
    """
    state = callback_context.state
    section_keys = state.get(TENK_CACHE_SECTION_KEYS_STATE) or []
    if not section_keys:
        return None

    manager = get_tenk_context_cache_manager()
    model = str(llm_request.model or "")
    bundle: TenKSectionBundle | None = None
    if manager.enabled:
        try:
            bundle = await asyncio.to_thread(manager.get_bundle, tuple(section_keys))
        except Exception as exc:
            logger.warning("Unable to assemble 10-K context bundle: %s", exc)
            bundle = None

    config = llm_request.config or genai_types.GenerateContentConfig()
    decision = await manager.acquire(
        client=manager.client or _resolve_genai_client(callback_context),
        model=model,
        system_instruction=config.system_instruction,
        tools=config.tools,
        tool_config=config.tool_config,
        bundle=bundle,
    )
    apply_tenk_cache_decision(llm_request, decision, bundle)

    summary = dict(state.get(TENK_CACHE_DECISION_STATE) or {})
    summary["model_calls"] = int(summary.get("model_calls", 0)) + 1
    summary["cached_calls"] = int(summary.get("cached_calls", 0)) + (
        1 if decision.mode == "cached" else 0
    )
    summary["last_decision"] = decision.to_dict()
    state[TENK_CACHE_DECISION_STATE] = summary
    return None
