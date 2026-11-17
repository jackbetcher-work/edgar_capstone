"""Thread-safe bounded LRU+TTL cache, cache counters, and 10-K cache key parser."""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from edgar_capstone.common.domain import normalize_edgar_ticker

_K = TypeVar("_K")
_V = TypeVar("_V")

_10K_CACHE_KEY_RE = re.compile(
    r"^(?P<ticker>.+)_(?P<fiscal_year>\d{4})_(?P<section_id>item_1a|item_7)$"
)


def parse_10k_cache_key(cache_key: str) -> tuple[str, int, str] | None:
    """Parse ``{TICKER}_{YYYY}_{section_id}`` normalizing dotted tickers (``BRK.B`` -> ``BRK-B``)."""
    match = _10K_CACHE_KEY_RE.match(cache_key.strip())
    if not match:
        return None
    return (
        normalize_edgar_ticker(match.group("ticker")),
        int(match.group("fiscal_year")),
        match.group("section_id"),
    )


class BoundedTTLCache(MutableMapping[_K, _V], Generic[_K, _V]):
    """Thread-safe bounded LRU + TTL cache with atomic ``snapshot()`` reads."""

    def __init__(self, *, max_entries: int = 256, ttl_seconds: float = 3600.0) -> None:
        self._max_entries = max(1, int(max_entries))
        self._ttl_seconds = max(0.0, float(ttl_seconds))
        self._lock = threading.RLock()
        self._data: OrderedDict[_K, tuple[float, _V]] = OrderedDict()

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    def _is_expired(self, stored_at: float, now: float) -> bool:
        return self._ttl_seconds > 0.0 and (now - stored_at) >= self._ttl_seconds

    def _prune_expired_unlocked(self, now: float) -> None:
        if self._ttl_seconds <= 0.0:
            return
        expired = [k for k, (ts, _) in self._data.items() if (now - ts) >= self._ttl_seconds]
        for k in expired:
            self._data.pop(k, None)

    def __getitem__(self, key: _K) -> _V:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                raise KeyError(key)
            stored_at, val = entry
            if self._is_expired(stored_at, now):
                self._data.pop(key, None)
                raise KeyError(key)
            self._data.move_to_end(key)
            return val

    def __setitem__(self, key: _K, value: _V) -> None:
        now = time.monotonic()
        with self._lock:
            self._prune_expired_unlocked(now)
            self._data[key] = (now, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max_entries:
                self._data.popitem(last=False)

    def __delitem__(self, key: _K) -> None:
        with self._lock:
            del self._data[key]

    def __iter__(self) -> Iterator[_K]:
        now = time.monotonic()
        with self._lock:
            self._prune_expired_unlocked(now)
            keys = list(self._data.keys())
        return iter(keys)

    def __len__(self) -> int:
        now = time.monotonic()
        with self._lock:
            self._prune_expired_unlocked(now)
            return len(self._data)

    def get(self, key: _K, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def snapshot(self) -> dict[_K, _V]:
        """Atomically copy all live (non-expired) entries without TOCTOU ``KeyError`` risk."""
        now = time.monotonic()
        with self._lock:
            self._prune_expired_unlocked(now)
            return {k: v for k, (_, v) in self._data.items()}

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


@dataclass(slots=True)
class CacheMetrics:
    """Thread-safe counters tracking context-cache hits, misses, and token savings."""

    hits: int = 0
    misses: int = 0
    tokens_served_from_cache: int = 0
    tokens_loaded_on_miss: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record_hit(self, tokens: int) -> None:
        with self._lock:
            self.hits += 1
            self.tokens_served_from_cache += max(0, int(tokens))

    def record_miss(self, tokens: int) -> None:
        with self._lock:
            self.misses += 1
            self.tokens_loaded_on_miss += max(0, int(tokens))

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "tokens_served_from_cache": self.tokens_served_from_cache,
                "tokens_loaded_on_miss": self.tokens_loaded_on_miss,
            }

    def reset(self) -> None:
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.tokens_served_from_cache = 0
            self.tokens_loaded_on_miss = 0
