"""Tests for savi/collector.py — AsyncEventEmitter._flush().

Regression coverage for the 2026-08-28 silent-failure bug: _flush() used to
catch httpx.RequestError and otherwise never look at the response status at
all, so a 401/403/422 (dead API key, unauthorized tenant, malformed event)
was silently dropped with zero signal - confirmed live when a real backfill
run printed "done: N ok" for ~200 events while every single one was actually
rejected. These tests assert the fix: both failure shapes now log a
WARNING loud enough to show up even with no logging configuration (Python's
handler-of-last-resort prints WARNING+ to stderr by default), and neither
shape ever raises out of _flush() - it must never block the caller's thread.
"""
import logging
import weakref
from unittest.mock import Mock, patch

import httpx
import pytest

from savi.collector import AsyncEventEmitter


def _emitter() -> AsyncEventEmitter:
    # A very long flush_interval keeps the background daemon thread from
    # ever firing during the test - _flush() is called directly instead,
    # for a deterministic assertion instead of racing a timer.
    return AsyncEventEmitter(endpoint="http://test", savi_key="sk_test", flush_interval_secs=999)


def test_flush_does_nothing_when_buffer_is_empty(caplog):
    emitter = _emitter()
    with patch("savi.collector.httpx.post") as mock_post:
        emitter._flush()
    mock_post.assert_not_called()
    assert caplog.records == []


def test_flush_success_logs_nothing(caplog):
    emitter = _emitter()
    emitter.emit({"event_id": "evt_1"})
    mock_resp = Mock(status_code=200)
    with caplog.at_level(logging.WARNING), patch("savi.collector.httpx.post", return_value=mock_resp) as mock_post:
        emitter._flush()
    mock_post.assert_called_once()
    assert caplog.records == []


def test_flush_logs_warning_on_non_2xx_response_and_does_not_raise(caplog):
    """The exact shape of the real incident: a 401 from a dead API key."""
    emitter = _emitter()
    emitter.emit({"event_id": "evt_1"})
    mock_resp = Mock(status_code=401, text='{"detail":"Invalid API key"}')
    with caplog.at_level(logging.WARNING), patch("savi.collector.httpx.post", return_value=mock_resp):
        emitter._flush()  # must not raise
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "401" in caplog.records[0].message
    assert "Invalid API key" in caplog.records[0].message


def test_flush_logs_warning_on_transport_error_and_does_not_raise(caplog):
    emitter = _emitter()
    emitter.emit({"event_id": "evt_1"})
    with caplog.at_level(logging.WARNING), patch(
        "savi.collector.httpx.post", side_effect=httpx.ConnectError("connection refused")
    ):
        emitter._flush()  # must not raise
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "ConnectError" in caplog.records[0].message


def test_flush_batches_only_up_to_batch_size():
    emitter = AsyncEventEmitter(endpoint="http://test", savi_key="sk_test", batch_size=2, flush_interval_secs=999)
    for i in range(5):
        emitter.emit({"event_id": f"evt_{i}"})
    mock_resp = Mock(status_code=200)
    with patch("savi.collector.httpx.post", return_value=mock_resp) as mock_post:
        emitter._flush()
    sent_events = mock_post.call_args.kwargs["json"]["events"]
    assert len(sent_events) == 2
    assert len(emitter._buffer) == 3


# ── Retry/requeue coverage ─────────────────────────────────────────────────
# A batch is popped off the buffer before the request, so it must go back on
# failure - otherwise a transient SAVI-side or network outage loses it
# permanently, even though nothing about the failure was the caller's fault.
# These tests assert a transient failure requeues the batch, a permanent one
# still drops it (retrying a dead key or malformed event forever would be
# pointless), and order/content survive the round trip.

def test_flush_requeues_batch_on_transport_error():
    emitter = _emitter()
    emitter.emit({"event_id": "evt_1"})
    with patch("savi.collector.httpx.post", side_effect=httpx.ConnectError("connection refused")):
        emitter._flush()
    assert list(emitter._buffer) == [{"event_id": "evt_1"}]


def test_flush_requeues_batch_on_retryable_5xx_status():
    emitter = _emitter()
    emitter.emit({"event_id": "evt_1"})
    mock_resp = Mock(status_code=503, text="Service Unavailable")
    with patch("savi.collector.httpx.post", return_value=mock_resp):
        emitter._flush()
    assert list(emitter._buffer) == [{"event_id": "evt_1"}]


def test_flush_requeues_batch_on_429_rate_limit():
    emitter = _emitter()
    emitter.emit({"event_id": "evt_1"})
    mock_resp = Mock(status_code=429, text="Too Many Requests")
    with patch("savi.collector.httpx.post", return_value=mock_resp):
        emitter._flush()
    assert list(emitter._buffer) == [{"event_id": "evt_1"}]


def test_flush_does_not_requeue_on_permanent_401():
    """A dead API key would fail identically forever - dropped, not retried,
    exactly as before this fix (the 2026-08-28 logging behavior alone)."""
    emitter = _emitter()
    emitter.emit({"event_id": "evt_1"})
    mock_resp = Mock(status_code=401, text="Invalid API key")
    with patch("savi.collector.httpx.post", return_value=mock_resp):
        emitter._flush()
    assert list(emitter._buffer) == []


def test_flush_does_not_requeue_on_permanent_422():
    """A malformed event would fail identically forever - dropped, not
    retried."""
    emitter = _emitter()
    emitter.emit({"event_id": "evt_1"})
    mock_resp = Mock(status_code=422, text="Validation error")
    with patch("savi.collector.httpx.post", return_value=mock_resp):
        emitter._flush()
    assert list(emitter._buffer) == []


def test_flush_requeue_preserves_order_ahead_of_newer_events():
    """A requeued batch goes back at the FRONT of the buffer - first in line
    on the next flush - not the back, where it would wait behind whatever
    the caller emitted while the request was in flight."""
    emitter = AsyncEventEmitter(endpoint="http://test", savi_key="sk_test", batch_size=2, flush_interval_secs=999)
    emitter.emit({"event_id": "evt_1"})
    emitter.emit({"event_id": "evt_2"})
    with patch("savi.collector.httpx.post", side_effect=httpx.ConnectError("connection refused")):
        emitter._flush()  # pops [evt_1, evt_2], fails, requeues both
    emitter.emit({"event_id": "evt_3"})  # emitted after the failed attempt
    assert list(emitter._buffer) == [
        {"event_id": "evt_1"}, {"event_id": "evt_2"}, {"event_id": "evt_3"},
    ]


def test_flush_retry_eventually_succeeds_and_drains_buffer():
    """End-to-end: a transient failure requeues, the next flush (e.g. after
    the endpoint recovers) sends the same batch and it's gone for good."""
    emitter = _emitter()
    emitter.emit({"event_id": "evt_1"})
    with patch("savi.collector.httpx.post", side_effect=httpx.ConnectError("connection refused")):
        emitter._flush()
    assert len(emitter._buffer) == 1
    mock_resp = Mock(status_code=200)
    with patch("savi.collector.httpx.post", return_value=mock_resp) as mock_post:
        emitter._flush()
    sent_events = mock_post.call_args.kwargs["json"]["events"]
    assert sent_events == [{"event_id": "evt_1"}]
    assert len(emitter._buffer) == 0


# ── flush() / process-exit event loss ─────────────────────────────────────
# Without this, a process that exits before the next flush_interval tick (a
# short-lived script, a CLI command, a container shutting down) loses every
# buffered event with zero signal - the same silent-data-loss shape as the
# 2026-08-28/08-30 bugs above, just triggered by exit instead of a response.

def test_flush_public_method_drains_buffer_synchronously():
    emitter = _emitter()
    emitter.emit({"event_id": "evt_1"})
    mock_resp = Mock(status_code=200)
    with patch("savi.collector.httpx.post", return_value=mock_resp) as mock_post:
        emitter.flush()
    mock_post.assert_called_once()
    assert len(emitter._buffer) == 0


def test_flush_public_method_drains_more_than_one_batch():
    emitter = AsyncEventEmitter(endpoint="http://test", savi_key="sk_test", batch_size=2, flush_interval_secs=999)
    for i in range(5):
        emitter.emit({"event_id": f"evt_{i}"})
    mock_resp = Mock(status_code=200)
    with patch("savi.collector.httpx.post", return_value=mock_resp) as mock_post:
        emitter.flush()
    assert len(emitter._buffer) == 0
    assert mock_post.call_count == 3  # 2 + 2 + 1


def test_flush_public_method_returns_on_empty_buffer_without_posting():
    emitter = _emitter()
    with patch("savi.collector.httpx.post") as mock_post:
        emitter.flush()
    mock_post.assert_not_called()


def test_flush_public_method_stops_after_one_failed_attempt_no_progress():
    """Must not busy-spin retrying the same failing batch: _flush() has no
    backoff of its own, and every emitter stays reachable for the life of
    the process (its daemon thread is a standing strong reference), so this
    runs once per still-alive instance at exit - a spin here multiplies
    into a slow, CPU-spinning shutdown across many instances."""
    emitter = _emitter()
    emitter.emit({"event_id": "evt_1"})
    with patch("savi.collector.httpx.post", side_effect=httpx.ConnectError("connection refused")) as mock_post:
        emitter.flush(timeout=5.0)
    mock_post.assert_called_once()  # one attempt, not a retry loop
    assert list(emitter._buffer) == [{"event_id": "evt_1"}]  # requeued, not lost


def test_flush_public_method_respects_timeout_deadline():
    """The timeout is still a real safety net, checked before each attempt."""
    emitter = _emitter()
    emitter.emit({"event_id": "evt_1"})
    mock_resp = Mock(status_code=200)
    with patch("savi.collector.httpx.post", return_value=mock_resp) as mock_post:
        emitter.flush(timeout=0.0)
    mock_post.assert_not_called()


def test_atexit_flush_calls_flush_on_live_emitter():
    from savi.collector import _atexit_flush
    emitter = _emitter()
    emitter.emit({"event_id": "evt_1"})
    mock_resp = Mock(status_code=200)
    with patch("savi.collector.httpx.post", return_value=mock_resp) as mock_post:
        _atexit_flush(weakref.ref(emitter))
    mock_post.assert_called_once()
    assert len(emitter._buffer) == 0


def test_atexit_flush_is_a_no_op_for_a_garbage_collected_emitter():
    from savi.collector import _atexit_flush
    emitter = _emitter()
    ref = weakref.ref(emitter)
    del emitter
    _atexit_flush(ref)  # must not raise


def test_constructing_emitter_registers_an_atexit_callback():
    with patch("savi.collector.atexit.register") as mock_register:
        emitter = _emitter()
    mock_register.assert_called_once()
    args = mock_register.call_args.args
    assert args[0] is __import__("savi.collector", fromlist=["_atexit_flush"])._atexit_flush
    assert args[1]() is emitter  # registered against a weakref, not emitter itself
