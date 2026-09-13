import logging
import os
import time
import uuid
from datetime import datetime, timezone
from openai import OpenAI as _OpenAI
from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse
from savi.context import attribution_fields
from savi.local import resolve_emitter
from savi.pii import fingerprint
from savi.cache import ResponseCache

try:
    from openai import AsyncOpenAI as _AsyncOpenAI
except ImportError:  # pragma: no cover, openai always ships both; defensive only
    _AsyncOpenAI = None

_log = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = os.environ.get("SAVI_ENDPOINT", "https://api.datagras.com")


def _build_masker(mask_pii, pii_entities, pii_exclude_entities=None):
    # mask_pii is None (the default) means "on if possible" - Presidio missing
    # degrades to unmasked with a warning. mask_pii=True is an explicit ask
    # for the compliance guarantee, so a missing dependency stays a hard
    # failure rather than a silent gap. mask_pii=False skips masking entirely.
    if mask_pii is False:
        return None
    from savi.pii import PiiMasker
    try:
        return PiiMasker(entities=pii_entities, exclude_entities=pii_exclude_entities)
    except ImportError:
        if mask_pii is True:
            raise
        _log.warning(
            "PII masking is on by default but Presidio isn't installed - "
            "continuing without masking. Install with: pip install 'savi-sdk[pii]', "
            "or pass mask_pii=False to silence this warning."
        )
        return None


def _client_kwargs(api_key, timeout, max_retries, base_url=None):
    kwargs = {"api_key": api_key}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    if base_url is not None:
        kwargs["base_url"] = base_url
    return kwargs


class SaviOpenAI:
    def __init__(self, api_key: str, savi_key: "str | None" = None, tenant_id: "str | None" = None,
                 team_id: str = "default",
                 endpoint: str = _DEFAULT_ENDPOINT,
                 batch_size: int = 100,
                 flush_interval_secs: float = 5.0,
                 mask_pii: "bool | None" = None,
                 pii_entities: "list | None" = None,
                 # e.g. pii_exclude_entities=["LOCATION", "DATE_TIME"] to drop
                 # a couple of default entities without hand-copying the rest.
                 pii_exclude_entities: "list | None" = None,
                 workload_type: "str | None" = None,
                 embedding_workload_type: "str | None" = None,
                 timeout: "float | None" = None,
                 max_retries: "int | None" = None,
                 # Base URL for the inner OpenAI client itself (e.g. a local
                 # Ollama endpoint), not SAVI's collector endpoint above.
                 base_url: "str | None" = None,
                 # Opt-in response cache, chat completions only. See
                 # savi/cache.py's ResponseCache for the safety carve-outs.
                 enable_cache: bool = False,
                 cache_ttl_seconds: float = 300.0,
                 cache_max_size: int = 1000,
                 local_mode: bool = False,
                 local_pricing: "dict[str, tuple[float, float]] | None" = None,
                 _collector=None):
        self._inner     = _OpenAI(**_client_kwargs(api_key, timeout, max_retries, base_url))
        self._tenant_id = tenant_id
        self._team_id   = team_id
        self._collector = _collector or resolve_emitter(
            savi_key, tenant_id, local_mode, endpoint,
            batch_size, flush_interval_secs, local_pricing,
        )
        masker = _build_masker(mask_pii, pii_entities, pii_exclude_entities)
        cache  = ResponseCache(cache_ttl_seconds, cache_max_size) if enable_cache else None
        # Prevent auto_instrument from wrapping this client a second time.
        self._inner.chat.completions._savi_instrumented = True
        self.chat       = _ChatProxy(self._inner, self._collector.emit, tenant_id, team_id, masker, workload_type, cache)
        self.embeddings = _EmbeddingsProxy(self._inner, self._collector.emit, tenant_id, team_id, masker, embedding_workload_type)


class _ChatProxy:
    def __init__(self, inner, emit_fn, tenant_id, team_id, masker, workload_type=None, cache=None):
        self.completions = _CompletionsProxy(inner, emit_fn, tenant_id, team_id, masker, workload_type, cache)


class _CompletionsProxy:
    def __init__(self, inner, emit_fn, tenant_id, team_id, masker, workload_type=None, cache=None):
        self._inner         = inner
        self._emit          = emit_fn
        self._tenant        = tenant_id
        self._team          = team_id
        self._masker        = masker
        self._workload_type = workload_type
        self._cache         = cache

    def create(self, model: str, messages: list, **kwargs) -> ChatCompletion:
        masked_msgs, pii_flagged, pii_types = _mask(self._masker, messages)
        fp = _cache_key(masked_msgs, model, kwargs)
        cacheable = self._cache is not None and not kwargs.get("stream")

        if cacheable:
            cached = self._cache.get(fp, pii_flagged)
            if cached is not None:
                try:
                    self._emit(_build_payload(
                        self._tenant, self._team, cached, 0,
                        fp, pii_flagged, pii_types, self._workload_type,
                        is_cache_hit=True,
                    ))
                except Exception:
                    _log.debug("savi.openai: emit failed", exc_info=True)
                return cached

        t0 = time.monotonic()
        try:
            response = self._inner.chat.completions.create(model=model, messages=messages, **kwargs)
        except Exception as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            try:
                self._emit(_build_error_payload(
                    self._tenant, self._team, model, latency_ms,
                    fp, pii_flagged, pii_types, self._workload_type, exc,
                ))
            except Exception:
                _log.debug("savi.openai: emit failed", exc_info=True)
            raise
        if kwargs.get("stream"):
            # A streamed response doesn't carry usage/finish_reason up front
            # the way a full ChatCompletion does, so there's nothing to emit yet.
            return response
        latency_ms = int((time.monotonic() - t0) * 1000)

        if cacheable:
            self._cache.set(fp, pii_flagged, response)

        try:
            self._emit(_build_payload(
                self._tenant, self._team, response, latency_ms,
                fp, pii_flagged, pii_types, self._workload_type,
            ))
        except Exception:
            _log.debug("savi.openai: emit failed", exc_info=True)
        return response


def _mask(masker, messages: list):
    if masker is not None:
        return masker.mask_messages(messages)
    return messages, False, None


def _cache_key(masked_msgs, model: str, kwargs: dict) -> str:
    # model + kwargs (temperature, tools, etc.) are part of the key too,
    # not just message content.
    return fingerprint((masked_msgs, model, kwargs))


class _EmbeddingsProxy:
    def __init__(self, inner, emit_fn, tenant_id, team_id, masker, workload_type=None):
        self._inner         = inner
        self._emit          = emit_fn
        self._tenant        = tenant_id
        self._team          = team_id
        self._masker        = masker
        self._workload_type = workload_type

    def create(self, model: str, input, **kwargs) -> CreateEmbeddingResponse:
        texts = input if isinstance(input, list) else [input]
        masked_texts, pii_flagged, pii_types = _mask_texts(self._masker, texts)
        fp = fingerprint(masked_texts)

        t0         = time.monotonic()
        response   = self._inner.embeddings.create(model=model, input=input, **kwargs)
        latency_ms = int((time.monotonic() - t0) * 1000)

        try:
            self._emit(_build_embedding_payload(
                self._tenant, self._team, response, latency_ms,
                fp, pii_flagged, pii_types, self._workload_type,
            ))
        except Exception:
            _log.debug("savi.openai: emit failed", exc_info=True)
        return response


def _mask_texts(masker, texts: list):
    """Like _mask(), but for a flat list of input strings (embeddings)
    rather than chat-style {role, content} message dicts."""
    if masker is None:
        return texts, False, None
    masked = []
    all_counts: dict = {}
    for t in texts:
        if isinstance(t, str):
            new_t, counts = masker.mask(t)
            for entity_type, count in counts.items():
                all_counts[entity_type] = all_counts.get(entity_type, 0) + count
            masked.append(new_t)
        else:
            masked.append(t)
    pii_types = all_counts if all_counts else None
    return masked, bool(all_counts), pii_types


def _build_embedding_payload(tenant_id, team_id, response, latency_ms, fp, pii_flagged, pii_types, workload_type):
    # Embeddings report a single total_tokens figure, no in/out split.
    payload = {
        "event_id":        f"evt_{uuid.uuid4().hex[:24]}",
        "tenant_id":       tenant_id,
        "team_id":         team_id,
        "provider":        "openai",
        "model":           response.model,
        "tokens_in":       response.usage.total_tokens,
        "tokens_out":      0,
        "tokens_cached":   0,
        "total_tokens":    response.usage.total_tokens,
        "latency_ms":      latency_ms,
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        **attribution_fields(),
        "lsh_fingerprint": fp,
        "pii_flagged":     pii_flagged,
        "pii_types":       pii_types,
        "schema_version":  "1.0",
    }
    if workload_type is not None:
        payload["workload_type"] = workload_type
    return payload


def _finish_reason(response) -> "str | None":
    """choices[0].finish_reason, or None if the response shape is
    unexpected (never raises)."""
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        value = getattr(choices[0], "finish_reason", None)
        return value[:32] if isinstance(value, str) else None
    except Exception:
        return None


def _build_payload(tenant_id, team_id, response, latency_ms, fp, pii_flagged, pii_types, workload_type, is_cache_hit=False):
    payload = {
        "event_id":        f"evt_{uuid.uuid4().hex[:24]}",
        "tenant_id":       tenant_id,
        "team_id":         team_id,
        "provider":        "openai",
        "model":           response.model,
        "tokens_in":       response.usage.prompt_tokens,
        "tokens_out":      response.usage.completion_tokens,
        "tokens_cached":   getattr(response.usage, "cached_tokens", 0),
        "total_tokens":    response.usage.total_tokens,
        "latency_ms":      latency_ms,
        "finish_reason":   _finish_reason(response),
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        **attribution_fields(),
        "lsh_fingerprint": fp,
        "pii_flagged":     pii_flagged,
        "pii_types":       pii_types,
        # Cache hits report the original call's real token counts, not
        # zero. cost_usd downstream is the cost the hit avoided.
        "is_cache_hit":    is_cache_hit,
        "schema_version":  "1.0",
    }
    if workload_type is not None:
        payload["workload_type"] = workload_type
    return payload


def _extract_error_info(exc: BaseException) -> tuple[str, str]:
    """(error_code, message) for any exception the provider client raises.
    Uses status_code when available (e.g. "429"), else the exception's
    class name. Never raises."""
    try:
        status = getattr(exc, "status_code", None)
        error_code = str(status) if status is not None else type(exc).__name__
        return error_code, str(exc)[:512]
    except Exception:
        return "unknown_error", "error details unavailable"


def _build_error_payload(tenant_id, team_id, model, latency_ms, fp, pii_flagged, pii_types, workload_type, exc):
    """Emitted when the provider call itself raises, so failures are still
    visible to SAVI instead of vanishing silently."""
    error_code, error_message = _extract_error_info(exc)
    payload = {
        "event_id":        f"evt_{uuid.uuid4().hex[:24]}",
        "tenant_id":       tenant_id,
        "team_id":         team_id,
        "provider":        "openai",
        "model":           model,
        "tokens_in":       0,
        "tokens_out":      0,
        "tokens_cached":   0,
        "latency_ms":      latency_ms,
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        **attribution_fields(),
        "lsh_fingerprint": fp,
        "pii_flagged":     pii_flagged,
        "pii_types":       pii_types,
        "is_cache_hit":    False,
        "is_error":        True,
        "error_code":      error_code,
        "error_message":   error_message,
        "schema_version":  "1.0",
    }
    if workload_type is not None:
        payload["workload_type"] = workload_type
    return payload


class SaviAsyncOpenAI:
    """Async-native equivalent of SaviOpenAI, wraps openai.AsyncOpenAI so it can
    be awaited from async application code (SaviOpenAI wraps the *synchronous*
    openai.OpenAI client and cannot be awaited; calling its .create() from an
    async call site would block the event loop). Same constructor shape and
    emitted-event contract as SaviOpenAI, only the client + call semantics
    differ.

    Usage:
        client = SaviAsyncOpenAI(api_key=..., savi_key=..., tenant_id=...)
        response = await client.chat.completions.create(model=..., messages=...)
    """
    def __init__(self, api_key: str, savi_key: "str | None" = None, tenant_id: "str | None" = None,
                 team_id: str = "default",
                 endpoint: str = _DEFAULT_ENDPOINT,
                 batch_size: int = 100,
                 flush_interval_secs: float = 5.0,
                 mask_pii: "bool | None" = None,
                 pii_entities: "list | None" = None,
                 pii_exclude_entities: "list | None" = None,
                 workload_type: "str | None" = None,
                 embedding_workload_type: "str | None" = None,
                 timeout: "float | None" = None,
                 max_retries: "int | None" = None,
                 # Base URL for the inner OpenAI client itself, not SAVI's
                 # collector endpoint above.
                 base_url: "str | None" = None,
                 enable_cache: bool = False,
                 cache_ttl_seconds: float = 300.0,
                 cache_max_size: int = 1000,
                 local_mode: bool = False,
                 local_pricing: "dict[str, tuple[float, float]] | None" = None,
                 _collector=None):
        if _AsyncOpenAI is None:
            raise ImportError("openai>=1.0 with AsyncOpenAI support is required for SaviAsyncOpenAI")
        self._inner     = _AsyncOpenAI(**_client_kwargs(api_key, timeout, max_retries, base_url))
        self._tenant_id = tenant_id
        self._team_id   = team_id
        self._collector = _collector or resolve_emitter(
            savi_key, tenant_id, local_mode, endpoint,
            batch_size, flush_interval_secs, local_pricing,
        )
        masker = _build_masker(mask_pii, pii_entities, pii_exclude_entities)
        cache  = ResponseCache(cache_ttl_seconds, cache_max_size) if enable_cache else None
        # Prevent auto_instrument from wrapping this client a second time.
        self._inner.chat.completions._savi_instrumented = True
        self.chat       = _AsyncChatProxy(self._inner, self._collector.emit, tenant_id, team_id, masker, workload_type, cache)
        self.embeddings = _AsyncEmbeddingsProxy(self._inner, self._collector.emit, tenant_id, team_id, masker, embedding_workload_type)


class _AsyncChatProxy:
    def __init__(self, inner, emit_fn, tenant_id, team_id, masker, workload_type=None, cache=None):
        self.completions = _AsyncCompletionsProxy(inner, emit_fn, tenant_id, team_id, masker, workload_type, cache)


class _AsyncCompletionsProxy:
    def __init__(self, inner, emit_fn, tenant_id, team_id, masker, workload_type=None, cache=None):
        self._inner         = inner
        self._emit          = emit_fn
        self._tenant        = tenant_id
        self._team          = team_id
        self._masker        = masker
        self._workload_type = workload_type
        self._cache         = cache

    async def create(self, model: str, messages: list, **kwargs) -> ChatCompletion:
        masked_msgs, pii_flagged, pii_types = _mask(self._masker, messages)
        fp = _cache_key(masked_msgs, model, kwargs)
        cacheable = self._cache is not None and not kwargs.get("stream")

        if cacheable:
            cached = self._cache.get(fp, pii_flagged)
            if cached is not None:
                try:
                    self._emit(_build_payload(
                        self._tenant, self._team, cached, 0,
                        fp, pii_flagged, pii_types, self._workload_type,
                        is_cache_hit=True,
                    ))
                except Exception:
                    _log.debug("savi.openai: emit failed", exc_info=True)
                return cached

        t0 = time.monotonic()
        try:
            response = await self._inner.chat.completions.create(model=model, messages=messages, **kwargs)
        except Exception as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            try:
                self._emit(_build_error_payload(
                    self._tenant, self._team, model, latency_ms,
                    fp, pii_flagged, pii_types, self._workload_type, exc,
                ))
            except Exception:
                _log.debug("savi.openai: emit failed", exc_info=True)
            raise
        if kwargs.get("stream"):
            # A streamed response doesn't carry usage/finish_reason up front
            # the way a full ChatCompletion does, so there's nothing to emit yet.
            return response
        latency_ms = int((time.monotonic() - t0) * 1000)

        if cacheable:
            self._cache.set(fp, pii_flagged, response)

        # emit() is synchronous (a plain buffer append), so it's called
        # directly here rather than awaited.
        try:
            self._emit(_build_payload(
                self._tenant, self._team, response, latency_ms,
                fp, pii_flagged, pii_types, self._workload_type,
            ))
        except Exception:
            _log.debug("savi.openai: emit failed", exc_info=True)
        return response


class _AsyncEmbeddingsProxy:
    def __init__(self, inner, emit_fn, tenant_id, team_id, masker, workload_type=None):
        self._inner         = inner
        self._emit          = emit_fn
        self._tenant        = tenant_id
        self._team          = team_id
        self._masker        = masker
        self._workload_type = workload_type

    async def create(self, model: str, input, **kwargs) -> CreateEmbeddingResponse:
        texts = input if isinstance(input, list) else [input]
        masked_texts, pii_flagged, pii_types = _mask_texts(self._masker, texts)
        fp = fingerprint(masked_texts)

        t0         = time.monotonic()
        response   = await self._inner.embeddings.create(model=model, input=input, **kwargs)
        latency_ms = int((time.monotonic() - t0) * 1000)

        try:
            self._emit(_build_embedding_payload(
                self._tenant, self._team, response, latency_ms,
                fp, pii_flagged, pii_types, self._workload_type,
            ))
        except Exception:
            _log.debug("savi.openai: emit failed", exc_info=True)
        return response
