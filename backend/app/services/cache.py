"""In-process result cache.

An analysis costs seconds of network I/O and produces rasters far too large to
round-trip through JSON. The API therefore returns layer *descriptors* with URLs,
and the pixels stay here until the client asks for them as PNG.

Keys are deterministic hashes of the request, so re-running the same analysis is
free and a page refresh does not re-download imagery. This is deliberately
process-local: it is a latency cache, not a source of truth, and a multi-worker
deployment should point it at Redis (see ``docs/DEPLOYMENT.md``).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from app.core.config import settings


@dataclass
class _Entry:
    value: Any
    expires_at: float


class TTLCache:
    """Thread-safe LRU with per-entry expiry."""

    def __init__(self, max_entries: int = 128, ttl_seconds: int | None = None) -> None:
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._ttl = ttl_seconds or settings.cache_ttl_seconds

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at < time.time():
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry.value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._entries[key] = _Entry(value, time.time() + self._ttl)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            live = sum(1 for e in self._entries.values() if e.expires_at >= now)
            return {"entries": live, "capacity": self._max_entries, "ttl_seconds": self._ttl}


#: Analysis results, holding NumPy arrays. Capped low because each entry can be
#: tens of megabytes of raster.
RESULTS = TTLCache(max_entries=48)
#: Small JSON payloads — time series, catalog responses.
DATA = TTLCache(max_entries=512)


def token(*parts: Any) -> str:
    """Stable short hash of the arguments, used as a cache key and URL segment."""
    payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]
