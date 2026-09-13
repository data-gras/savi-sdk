import time
from unittest.mock import patch
import pytest
from savi.cache import ResponseCache


def test_get_returns_none_when_empty():
    cache = ResponseCache()
    assert cache.get("some_fp", pii_flagged=False) is None


def test_set_then_get_returns_stored_value():
    cache = ResponseCache()
    cache.set("fp1", pii_flagged=False, snapshot="the response")
    assert cache.get("fp1", pii_flagged=False) == "the response"


def test_get_never_returns_a_value_for_pii_flagged_lookup():
    """The core safety guarantee: even if a response WAS stored under this
    fingerprint (by an earlier non-PII call that happened to collide), a
    pii_flagged=True lookup must never receive it back — two different
    real prompts masking to the same template must never cross-contaminate."""
    cache = ResponseCache()
    cache.set("fp_shared", pii_flagged=False, snapshot="cached for John")
    assert cache.get("fp_shared", pii_flagged=True) is None


def test_set_never_stores_a_pii_flagged_response():
    cache = ResponseCache()
    cache.set("fp2", pii_flagged=True, snapshot="should not be stored")
    assert cache.get("fp2", pii_flagged=False) is None
    assert len(cache) == 0


def test_get_and_set_are_no_ops_for_empty_fingerprint():
    cache = ResponseCache()
    cache.set("", pii_flagged=False, snapshot="x")
    assert len(cache) == 0
    assert cache.get("", pii_flagged=False) is None


def test_ttl_expiry():
    cache = ResponseCache(ttl_seconds=100.0)
    cache.set("fp3", pii_flagged=False, snapshot="stale soon")
    with patch("time.monotonic", return_value=time.monotonic() + 1000):
        assert cache.get("fp3", pii_flagged=False) is None
    assert len(cache) == 0  # expired entry is evicted on read, not just hidden


def test_lru_eviction_at_max_size():
    cache = ResponseCache(max_size=2)
    cache.set("a", False, "1")
    cache.set("b", False, "2")
    cache.set("c", False, "3")  # evicts "a" (least recently used)
    assert cache.get("a", False) is None
    assert cache.get("b", False) == "2"
    assert cache.get("c", False) == "3"


def test_get_refreshes_lru_order():
    cache = ResponseCache(max_size=2)
    cache.set("a", False, "1")
    cache.set("b", False, "2")
    cache.get("a", False)       # "a" is now most-recently-used
    cache.set("c", False, "3")  # evicts "b", not "a"
    assert cache.get("a", False) == "1"
    assert cache.get("b", False) is None


def test_len_reflects_store_size():
    cache = ResponseCache()
    assert len(cache) == 0
    cache.set("x", False, "1")
    assert len(cache) == 1
