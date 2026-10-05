"""One event per streamed model call (troubleshooting pack G12).

A streamed response has no usage up front, so the wrappers used to send nothing for it. This module wraps the stream,
passes every chunk to the caller unchanged, remembers only counts (model, token usage, finish reason, how many chunks
carried text) and emits exactly one event when the stream ends, fails, or is closed early. It never keeps or sends
the text of a chunk: streamed content is not captured in this version.

How a stream ends, and what is sent:
  completed   one normal event, with the usage the provider reported
  failed      one error event (is_error, error_code from the exception); the exception still reaches the caller
  abandoned   the caller closed the stream (or left a `with` block) before the end: one normal event with
              finish_reason "stream_abandoned" and the usage seen so far
A stream that is neither finished nor closed (dropped without close) sends nothing; that is a known gap.

Never raises into the caller: emitting is best effort, like the non-streamed path."""
import logging
import time
import uuid
from datetime import datetime, timezone

from savi.context import attribution_fields

_log = logging.getLogger("savi.streaming")

ABANDONED = "stream_abandoned"


class StreamState:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.tokens_in = 0
        self.tokens_out = 0
        self.tokens_cached = 0
        self.usage_reported = False
        self.text_chunks = 0
        self.finish_reason: "str | None" = None


def build_payload(state: StreamState, *, tenant_id, team_id, latency_ms: int, fingerprint, pii_flagged, pii_types,
                  workload_type=None, outcome: str = "completed", exc: "BaseException | None" = None) -> dict:
    tokens_out = state.tokens_out
    if not state.usage_reported and state.text_chunks:
        # No usage from the provider (OpenAI sends it only when the caller sets stream_options include_usage):
        # a lower-bound estimate from the number of chunks that carried text. tokens_in stays 0.
        tokens_out = state.text_chunks
    payload = {
        "event_id":        f"evt_{uuid.uuid4().hex[:24]}",
        "tenant_id":       tenant_id,
        "team_id":         team_id,
        "provider":        state.provider,
        "model":           state.model,
        "tokens_in":       state.tokens_in,
        "tokens_out":      tokens_out,
        "tokens_cached":   min(state.tokens_cached, state.tokens_in),
        "total_tokens":    state.tokens_in + tokens_out,
        "latency_ms":      latency_ms,
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        **attribution_fields(),
        "lsh_fingerprint": fingerprint,
        "pii_flagged":     pii_flagged,
        "pii_types":       pii_types,
        "is_cache_hit":    False,
        "schema_version":  "1.0",
    }
    if outcome == "failed" and exc is not None:
        status = getattr(exc, "status_code", None)
        payload.update({"is_error": True, "error_code": (str(status) if status is not None else type(exc).__name__)[:64],
                        "error_message": str(exc)[:512]})
    else:
        payload["finish_reason"] = ABANDONED if outcome == "abandoned" else state.finish_reason
    if workload_type is not None:
        payload["workload_type"] = workload_type
    return payload


class _Tracker:
    def __init__(self, stream, state, observe, emit, make_payload):
        self._stream = stream
        self._state = state
        self._observe = observe
        self._emit = emit
        self._make = make_payload
        self._t0 = time.monotonic()
        self._done = False

    def _finish(self, outcome: str, exc: "BaseException | None" = None) -> None:
        if self._done:
            return
        self._done = True
        try:
            latency_ms = int((time.monotonic() - self._t0) * 1000)
            self._emit(self._make(self._state, latency_ms=latency_ms, outcome=outcome, exc=exc))
        except Exception:
            _log.debug("savi.streaming: emit failed", exc_info=True)

    def _see(self, chunk) -> None:
        try:
            self._observe(self._state, chunk)
        except Exception:
            _log.debug("savi.streaming: could not read a chunk", exc_info=True)

    def __getattr__(self, name):
        return getattr(self._stream, name)


class TrackedStream(_Tracker):
    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._finish("completed")
            raise
        except BaseException as exc:
            self._finish("failed", exc)
            raise
        self._see(chunk)
        return chunk

    def close(self):
        self._finish("abandoned")
        close = getattr(self._stream, "close", None)
        return close() if close is not None else None

    def __enter__(self):
        enter = getattr(self._stream, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, *exc_info):
        self._finish("abandoned")
        leave = getattr(self._stream, "__exit__", None)
        return leave(*exc_info) if leave is not None else None


class TrackedAsyncStream(_Tracker):
    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finish("completed")
            raise
        except BaseException as exc:
            self._finish("failed", exc)
            raise
        self._see(chunk)
        return chunk

    async def close(self):
        self._finish("abandoned")
        return await self._stream.close()

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc_info):
        self._finish("abandoned")
        return await self._stream.__aexit__(*exc_info)


def track(stream, *, is_async: bool, state: StreamState, observe, emit, make_payload):
    cls = TrackedAsyncStream if is_async else TrackedStream
    return cls(stream, state, observe, emit, make_payload)


# ── Provider chunk readers: counts only, never text ───────────────────────────

def observe_openai(state: StreamState, chunk) -> None:
    if getattr(chunk, "model", None):
        state.model = chunk.model
    usage = getattr(chunk, "usage", None)
    if usage is not None and getattr(usage, "prompt_tokens", None) is not None:
        state.tokens_in = int(usage.prompt_tokens)
        state.tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        state.tokens_cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
        state.usage_reported = True
    for choice in (getattr(chunk, "choices", None) or []):
        delta = getattr(choice, "delta", None)
        if delta is not None and getattr(delta, "content", None):
            state.text_chunks += 1
        reason = getattr(choice, "finish_reason", None)
        if isinstance(reason, str):
            state.finish_reason = reason[:32]


def observe_anthropic(state: StreamState, event) -> None:
    kind = getattr(event, "type", None)
    if kind == "message_start":
        message = getattr(event, "message", None)
        usage = getattr(message, "usage", None)
        if getattr(message, "model", None):
            state.model = message.model
        if usage is not None:
            state.tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
            state.tokens_cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            state.tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
            state.usage_reported = True
    elif kind == "message_delta":
        usage = getattr(event, "usage", None)
        if usage is not None and getattr(usage, "output_tokens", None) is not None:
            state.tokens_out = int(usage.output_tokens)
        reason = getattr(getattr(event, "delta", None), "stop_reason", None)
        if isinstance(reason, str):
            state.finish_reason = reason[:32]
    elif kind == "content_block_delta":
        if getattr(getattr(event, "delta", None), "text", None):
            state.text_chunks += 1


def track_stream(stream, *, provider: str, model: str, emit, tenant, team, fp, pii_flagged, pii_types, workload_type,
                 is_async: bool, observe):
    """Wrap a provider stream so one metadata-only event is emitted when it ends, fails or is closed."""
    state = StreamState(provider, model)

    def make(state, *, latency_ms, outcome, exc):
        return build_payload(state, tenant_id=tenant, team_id=team, latency_ms=latency_ms, fingerprint=fp,
                             pii_flagged=pii_flagged, pii_types=pii_types, workload_type=workload_type,
                             outcome=outcome, exc=exc)
    return track(stream, is_async=is_async, state=state, observe=observe, emit=emit, make_payload=make)


def observe_mistral(state: StreamState, event) -> None:
    """Mistral wraps each chunk in an event whose `.data` is an OpenAI-shaped chunk."""
    observe_openai(state, getattr(event, "data", event))


def observe_cohere(state: StreamState, event) -> None:
    kind = getattr(event, "type", None)
    if kind == "content-delta":
        text = getattr(getattr(getattr(getattr(event, "delta", None), "message", None), "content", None), "text", None)
        if text:
            state.text_chunks += 1
    elif kind == "message-end":
        delta = getattr(event, "delta", None)
        reason = getattr(delta, "finish_reason", None)
        if reason is not None:
            state.finish_reason = str(getattr(reason, "value", reason)).lower()[:32]
        tokens = getattr(getattr(delta, "usage", None), "tokens", None)
        if tokens is not None:
            state.tokens_in = int(getattr(tokens, "input_tokens", 0) or 0)
            state.tokens_out = int(getattr(tokens, "output_tokens", 0) or 0)
            state.usage_reported = True


def observe_vertex(state: StreamState, chunk) -> None:
    meta = getattr(chunk, "usage_metadata", None)
    if meta is not None and getattr(meta, "prompt_token_count", None):
        state.tokens_in = int(meta.prompt_token_count or 0)
        state.tokens_out = int(getattr(meta, "candidates_token_count", 0) or 0)
        state.tokens_cached = int(getattr(meta, "cached_content_token_count", 0) or 0)
        state.usage_reported = True
    for cand in (getattr(chunk, "candidates", None) or []):
        parts = getattr(getattr(cand, "content", None), "parts", None) or []
        if any(getattr(p, "text", None) for p in parts):
            state.text_chunks += 1
        reason = getattr(cand, "finish_reason", None)
        name = getattr(reason, "name", None)
        if isinstance(name, str) and name not in ("FINISH_REASON_UNSPECIFIED",):
            state.finish_reason = name.lower()[:32]


def observe_bedrock(state: StreamState, event) -> None:
    if not isinstance(event, dict):
        return
    if "contentBlockDelta" in event and (event["contentBlockDelta"].get("delta") or {}).get("text"):
        state.text_chunks += 1
    stop = (event.get("messageStop") or {}).get("stopReason")
    if isinstance(stop, str):
        state.finish_reason = stop[:32]
    usage = (event.get("metadata") or {}).get("usage")
    if usage:
        state.tokens_in = int(usage.get("inputTokens", 0) or 0)
        state.tokens_out = int(usage.get("outputTokens", 0) or 0)
        state.tokens_cached = int(usage.get("cacheReadInputTokens", 0) or 0)
        state.usage_reported = True
