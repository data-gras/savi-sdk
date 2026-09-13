# savi/collector.py
import atexit
import logging
import threading
import time
import weakref
from collections import deque
import httpx

# Get a logger, never configure one - the host application decides
# whether/where this goes.
_log = logging.getLogger(__name__)

# Transport-level failures and these status codes are treated as transient -
# the batch is requeued instead of dropped. Everything else (401/403/422/etc.)
# is a permanent rejection (dead key, malformed event) that would just fail
# the same way again, so it's logged and dropped as before.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _atexit_flush(ref: "weakref.ReferenceType") -> None:
    """Registered against a weakref, not a bound method, so an emitter
    built per-request doesn't stay alive for the whole process. No-op if
    it was already garbage collected."""
    emitter = ref()
    if emitter is not None:
        emitter.flush()


class AsyncEventEmitter:
    """Non-blocking emitter. <2ms P99. Ring buffer of 10K events."""
    def __init__(self, endpoint: str, savi_key: str,
                 batch_size: int = 100, flush_interval_secs: float = 5.0):
        self._endpoint       = endpoint
        self._savi_key       = savi_key
        self._batch_size     = batch_size
        self._flush_interval = flush_interval_secs
        self._buffer: deque[dict] = deque(maxlen=10_000)
        self._lock   = threading.Lock()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()
        # Best-effort: without this, a process that exits before the next
        # flush_interval tick loses whatever's still buffered.
        atexit.register(_atexit_flush, weakref.ref(self))

    def emit(self, event: dict) -> None:
        with self._lock:
            self._buffer.append(event)

    def flush(self, timeout: float = 5.0) -> None:
        """Synchronously drain whatever's buffered at call time (not a
        retry loop chasing an outage). Blocks up to `timeout`. Runs
        automatically at process exit; call directly for an explicit
        shutdown hook or in tests.

        Stops after the first attempt that makes no progress, since
        _flush() has no backoff of its own and retrying a real outage in
        a tight loop would just busy-spin for the full timeout."""
        with self._lock:
            starting_len = len(self._buffer)
        attempts = -(-starting_len // self._batch_size)  # ceil division
        deadline = time.monotonic() + timeout
        for _ in range(attempts):
            if time.monotonic() >= deadline:
                return
            with self._lock:
                before = len(self._buffer)
            self._flush()
            with self._lock:
                after = len(self._buffer)
            if after >= before:  # no progress - a batch failed and was requeued
                return

    def _flush_loop(self) -> None:
        while True:
            time.sleep(self._flush_interval)
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            batch = [self._buffer.popleft()
                     for _ in range(min(self._batch_size, len(self._buffer)))]
        # A non-2xx response never raises (this must not block user code),
        # but is always logged so "why is nothing showing up in SAVI"
        # leads back here. Transient failures (transport errors, or
        # _RETRYABLE_STATUS) are requeued; permanent rejections (401/403/
        # 422/...) are dropped since retrying won't fix them. Requeued
        # batches keep their event_id, so a duplicate delivery is a no-op
        # server-side (ingest dedups on event_id).
        try:
            resp = httpx.post(
                f"{self._endpoint}/v1/events/ingest",
                json={"events": batch},
                headers={"Authorization": f"Bearer {self._savi_key}"},
                timeout=5.0,
            )
            if resp.status_code >= 400:
                _log.warning(
                    "savi-sdk: ingest rejected %d event(s) - status=%d endpoint=%s body=%s",
                    len(batch), resp.status_code, self._endpoint, resp.text[:500],
                )
                if resp.status_code in _RETRYABLE_STATUS:
                    self._requeue(batch)
        except httpx.RequestError as exc:
            _log.warning(
                "savi-sdk: failed to reach %s while flushing %d event(s) - %s: %s",
                self._endpoint, len(batch), type(exc).__name__, exc,
            )
            self._requeue(batch)

    def _requeue(self, batch: list[dict]) -> None:
        """Put a transiently-failed batch back at the front of the buffer
        (order-preserving) so it's sent again first. Still bounded by the
        ring buffer's own maxlen, so a prolonged outage can still age
        events out."""
        with self._lock:
            self._buffer.extendleft(reversed(batch))
