# savi/cache.py
import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class ResponseCache:
    """In-memory, per-process cache letting a wrapper replay a prior LLM
    response instead of calling the provider again. Opt-in only (see
    enable_cache on the client constructors) - replaying a cached response
    changes what production code sees, so it's never a silent default.

    Keyed on savi.pii.fingerprint()'s MinHash signature of the *masked*
    prompt, exact match only, no similarity threshold.

    pii_flagged prompts are NEVER cached (read or write): the fingerprint
    is computed on the masked prompt, so two different real prompts that
    mask to the same template (e.g. two different names redacted the same
    way) would otherwise collide and one could return the other's cached
    answer.

    Bounded (max_size, LRU eviction) and TTL-expiring.
    """

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 1000):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, fingerprint: str, pii_flagged: bool) -> Optional[Any]:
        if not fingerprint or pii_flagged:
            return None
        with self._lock:
            entry = self._store.get(fingerprint)
            if entry is None:
                return None
            snapshot, expires_at = entry
            if time.monotonic() >= expires_at:
                del self._store[fingerprint]
                return None
            self._store.move_to_end(fingerprint)
            return snapshot

    def set(self, fingerprint: str, pii_flagged: bool, snapshot: Any) -> None:
        if not fingerprint or pii_flagged:
            return
        with self._lock:
            self._store[fingerprint] = (snapshot, time.monotonic() + self._ttl)
            self._store.move_to_end(fingerprint)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
