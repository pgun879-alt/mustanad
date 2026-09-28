"""Tests for API-key verification and rate limiting."""

from __future__ import annotations

import pytest

from mustanad.security import (
    SlidingWindowRateLimiter,
    hash_api_key,
    key_fingerprint,
    verify_api_key,
)


def test_hash_is_stable_and_not_the_input() -> None:
    digest = hash_api_key("secret-key")
    assert digest == hash_api_key("secret-key")
    assert "secret-key" not in digest
    assert len(digest) == 64


def test_verify_accepts_a_configured_key() -> None:
    allowed = frozenset({hash_api_key("alpha"), hash_api_key("beta")})
    assert verify_api_key("alpha", allowed)
    assert verify_api_key("beta", allowed)


def test_verify_rejects_unknown_and_empty_keys() -> None:
    allowed = frozenset({hash_api_key("alpha")})
    assert not verify_api_key("gamma", allowed)
    assert not verify_api_key("", allowed)
    assert not verify_api_key("alpha ", allowed), "keys must match exactly, not after trimming"


def test_verify_rejects_everything_when_no_keys_are_configured() -> None:
    """Fail closed: an empty allow-list must not mean "allow anything"."""
    assert not verify_api_key("alpha", frozenset())


def test_verify_is_not_fooled_by_a_prefix() -> None:
    allowed = frozenset({hash_api_key("alpha-long-secret")})
    assert not verify_api_key("alpha", allowed)
    assert not verify_api_key("alpha-long-secre", allowed)


def test_fingerprint_is_short_and_non_reversible() -> None:
    fingerprint = key_fingerprint("secret-key")
    assert len(fingerprint) == 12
    assert "secret" not in fingerprint
    assert fingerprint == key_fingerprint("secret-key")
    assert fingerprint != key_fingerprint("other-key")


# ------------------------------------------------------------------- rate limiter


def test_requests_under_the_limit_are_allowed() -> None:
    limiter = SlidingWindowRateLimiter(limit=3)
    for _ in range(3):
        allowed, retry_after = limiter.check("caller", now=100.0)
        assert allowed
        assert retry_after == 0.0


def test_the_request_over_the_limit_is_refused_with_a_retry_after() -> None:
    limiter = SlidingWindowRateLimiter(limit=2)
    limiter.check("caller", now=100.0)
    limiter.check("caller", now=100.5)
    allowed, retry_after = limiter.check("caller", now=101.0)
    assert not allowed
    assert retry_after == pytest.approx(59.0)


def test_the_window_slides_rather_than_resetting() -> None:
    """A fixed window would let 2*limit requests through across a boundary; this must not."""
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60.0)
    assert limiter.check("caller", now=0.0)[0]
    assert limiter.check("caller", now=30.0)[0]
    assert not limiter.check("caller", now=59.0)[0]
    # The first hit ages out at t=60, freeing exactly one slot -- not the whole quota.
    assert limiter.check("caller", now=61.0)[0]
    assert not limiter.check("caller", now=62.0)[0]
    # The second hit ages out at t=90.
    assert limiter.check("caller", now=91.0)[0]


def test_limits_are_tracked_per_identity() -> None:
    limiter = SlidingWindowRateLimiter(limit=1)
    assert limiter.check("first", now=0.0)[0]
    assert not limiter.check("first", now=1.0)[0]
    assert limiter.check("second", now=1.0)[0], "one caller must not exhaust another's quota"


def test_idle_identities_are_pruned_to_bound_memory() -> None:
    limiter = SlidingWindowRateLimiter(limit=5, window_seconds=10.0)
    for index in range(1500):
        limiter.check(f"caller-{index}", now=0.0)
    # Everything is still inside the window, so nothing is prunable yet.
    assert len(limiter._windows) == 1500
    # A request long after the window makes every old entry stale and triggers the sweep.
    limiter.check("late-caller", now=1000.0)
    assert len(limiter._windows) < 1500


def test_pruning_does_not_exempt_new_callers_from_the_limit() -> None:
    """Regression guard for a real bug in the first implementation.

    Pruning used to run *after* the new identity's window was inserted. Because that window
    was still empty, the sweep deleted it, and the request's timestamp was then appended to a
    dict-detached object. Every caller arriving past the prune threshold was therefore never
    rate limited at all -- the limiter silently stopped limiting under exactly the load it
    exists to handle.
    """
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60.0)
    for index in range(1100):  # push well past the prune threshold
        limiter.check(f"warmup-{index}", now=0.0)

    assert limiter.check("fresh", now=1.0)[0]
    assert limiter.check("fresh", now=2.0)[0]
    allowed, retry_after = limiter.check("fresh", now=3.0)
    assert not allowed, "a caller created after the prune threshold must still be limited"
    assert retry_after > 0


def test_invalid_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        SlidingWindowRateLimiter(limit=0)
