"""API-key authentication and request rate limiting.

Both are deliberately small and in-process. A single-tenant document API deployed on one
server does not need Redis, and pretending otherwise would be the kind of premature
infrastructure this portfolio is arguing against. The limits of that choice are documented in
the README rather than hidden.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def hash_api_key(raw_key: str) -> str:
    """SHA-256 hex digest of an API key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def verify_api_key(raw_key: str, allowed_hashes: frozenset[str]) -> bool:
    """Check ``raw_key`` against a set of allowed hashes in constant time per candidate.

    The comparison uses :func:`hmac.compare_digest` on fixed-length hex digests so that the
    time taken does not reveal how many leading characters of a guess were correct. Every
    candidate is compared even after a match is found, so the number of comparisons does not
    leak the matching key's position in the set.
    """
    if not raw_key or not allowed_hashes:
        return False
    candidate = hash_api_key(raw_key)
    matched = False
    for allowed in allowed_hashes:
        if hmac.compare_digest(candidate, allowed):
            matched = True
    return matched


def key_fingerprint(raw_key: str) -> str:
    """Short, non-reversible label for logs. Never log the key itself."""
    return hash_api_key(raw_key)[:12]


#: Number of tracked identities above which a new identity triggers a prune sweep.
_PRUNE_THRESHOLD = 1024


@dataclass(slots=True)
class _Window:
    hits: deque[float]


class SlidingWindowRateLimiter:
    """Per-identity sliding-window rate limiter.

    A sliding window is used rather than a fixed window because a fixed window lets a caller
    send ``2 * limit`` requests across a boundary instant. Memory is bounded by
    ``limit`` timestamps per active identity, and idle identities are pruned on access.

    Thread-safe: FastAPI runs sync endpoints in a worker thread pool, so the state is guarded
    by a lock.
    """

    def __init__(self, *, limit: int, window_seconds: float = 60.0) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._windows: dict[str, _Window] = {}
        self._lock = threading.Lock()

    def check(self, identity: str, *, now: float | None = None) -> tuple[bool, float]:
        """Record a hit for ``identity``.

        Args:
            identity: Opaque caller identifier (an API-key fingerprint, never the key).
            now: Injectable clock for tests.

        Returns:
            ``(allowed, retry_after_seconds)``. ``retry_after_seconds`` is 0.0 when allowed.
        """
        timestamp = now if now is not None else time.monotonic()
        cutoff = timestamp - self.window_seconds
        with self._lock:
            window = self._windows.get(identity)
            if window is None:
                # Prune *before* inserting, and only ever prune entries that already exist.
                # Pruning after insertion would delete the brand-new empty window for this
                # very identity, and the hit would then be appended to a detached object --
                # silently exempting every new caller past the threshold from the limit.
                if len(self._windows) >= _PRUNE_THRESHOLD:
                    self._prune(cutoff)
                window = _Window(hits=deque())
                self._windows[identity] = window
            else:
                while window.hits and window.hits[0] <= cutoff:
                    window.hits.popleft()
            if len(window.hits) >= self.limit:
                retry_after = window.hits[0] + self.window_seconds - timestamp
                return False, max(retry_after, 0.0)
            window.hits.append(timestamp)
            return True, 0.0

    def _prune(self, cutoff: float) -> None:
        """Drop identities with no hits inside the window. Caller must hold the lock."""
        stale = [
            identity
            for identity, window in self._windows.items()
            if not window.hits or window.hits[-1] <= cutoff
        ]
        for identity in stale:
            del self._windows[identity]
