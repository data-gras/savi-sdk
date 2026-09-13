import logging
import os
import time
import uuid
from datetime import datetime, timezone
from openai.types.chat import ChatCompletion
from savi.pii import fingerprint
from savi.cache import ResponseCache

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


class SaviAzureOpenAI:
    """Drop-in replacement for openai.AzureOpenAI with SAVI observability.

    Usage:
        client = SaviAzureOpenAI(
            azure_endpoint="https://my-resource.openai.azure.com/",
            api_key="...",
            api_version="2024-02-01",
            savi_key="sk_...",
            tenant_id="ten_acme",
        )
        resp = client.chat.completions.create(
            model="gpt-4o",          # deployment name
            messages=[{"role": "user", "content": "Hello"}],
        )

    Token field mapping (Azure OpenAI = standard OpenAI):
        prompt_tokens     → tokens_in
        completion_tokens → tokens_out
        cached_tokens     → tokens_cached (provider-side KV cache, reduces billed input)
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        api_version: str,
        savi_key: "str | None" = None,
        tenant_id: "str | None" = None,
        team_id: str = "default",
        endpoint: str = _DEFAULT_ENDPOINT,
        batch_size: int = 100,
        flush_interval_secs: float = 5.0,
        mask_pii: "bool | None" = None,
        pii_entities: "list | None" = None,
        pii_exclude_entities: "list | None" = None,
        workload_type: "str | None" = None,
        enable_cache: bool = False,
        cache_ttl_seconds: float = 300.0,
        cache_max_size: int = 1000,
        local_mode: bool = False,
        local_pricing: "dict[str, tuple[float, float]] | None" = None,
        _collector=None,
    ):
        from openai import AzureOpenAI as _AzureOpenAI
        self._inner     = _AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self._tenant_id = tenant_id
        self._team_id   = team_id
        from savi.local import resolve_emitter
        self._collector = _collector or resolve_emitter(
            savi_key, tenant_id, local_mode, endpoint,
            batch_size, flush_interval_secs, local_pricing,
        )
        masker = _build_masker(mask_pii, pii_entities, pii_exclude_entities)
        cache  = ResponseCache(cache_ttl_seconds, cache_max_size) if enable_cache else None
        # Prevent auto_instrument from wrapping this client a second time
        # (AzureOpenAI shares the same completions class OpenAI uses).
        self._inner.chat.completions._savi_instrumented = True
        self.chat = _ChatProxy(self._inner, self._collector.emit, tenant_id, team_id, masker, workload_type, cache)


class _ChatProxy:
    def __init__(self, inner, emit_fn, tenant_id, team_id, masker, workload_type=None, cache=None):
        self.completions = _CompletionsProxy(inner, emit_fn, tenant_id, team_id, masker, workload_type, cache)


class _CompletionsProxy:
    def __init__(self, inner, emit_fn, tenant_id, team_id, masker, workload_type=None, cache=None):
        self._inner        = inner
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
                        self._tenant, self._team, cached, 0, fp, pii_flagged, pii_types,
                        self._workload_type, is_cache_hit=True,
                    ))
                except Exception:
                    _log.debug("savi.azure_openai: emit failed", exc_info=True)
                return cached

        t0         = time.monotonic()
        response   = self._inner.chat.completions.create(model=model, messages=messages, **kwargs)
        if kwargs.get("stream"):
            # A streamed response doesn't carry usage/finish_reason up front
            # the way a full ChatCompletion does, so there's nothing to emit yet.
            return response
        latency_ms = int((time.monotonic() - t0) * 1000)

        if cacheable:
            self._cache.set(fp, pii_flagged, response)

        try:
            self._emit(_build_payload(
                self._tenant, self._team, response, latency_ms, fp, pii_flagged, pii_types,
                self._workload_type,
            ))
        except Exception:
            _log.debug("savi.azure_openai: emit failed", exc_info=True)
        return response


def _mask(masker, messages: list):
    if masker is not None:
        return masker.mask_messages(messages)
    return messages, False, None


def _cache_key(masked_msgs, model: str, kwargs: dict) -> str:
    # model + kwargs are part of the key too, not just message content.
    return fingerprint((masked_msgs, model, kwargs))


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


def _build_payload(tenant_id, team_id, response, latency_ms, fp, pii_flagged, pii_types,
                    workload_type=None, is_cache_hit=False):
    from savi.context import attribution_fields
    payload = {
        "event_id":        f"evt_{uuid.uuid4().hex[:24]}",
        "tenant_id":       tenant_id,
        "team_id":         team_id,
        "provider":        "azure",
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
        "is_cache_hit":    is_cache_hit,
        "schema_version":  "1.0",
    }
    if workload_type is not None:
        payload["workload_type"] = workload_type
    return payload


class SaviAsyncAzureOpenAI:
    """Async-native equivalent of SaviAzureOpenAI, wraps openai.AsyncAzureOpenAI
    so it can be awaited from async application code. Same constructor shape
    and emitted-event contract as SaviAzureOpenAI.

    Usage:
        client = SaviAsyncAzureOpenAI(azure_endpoint=..., api_key=...,
                                       api_version=..., savi_key=..., tenant_id=...)
        response = await client.chat.completions.create(model=..., messages=...)
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        api_version: str,
        savi_key: "str | None" = None,
        tenant_id: "str | None" = None,
        team_id: str = "default",
        endpoint: str = _DEFAULT_ENDPOINT,
        batch_size: int = 100,
        flush_interval_secs: float = 5.0,
        mask_pii: "bool | None" = None,
        pii_entities: "list | None" = None,
        pii_exclude_entities: "list | None" = None,
        workload_type: "str | None" = None,
        enable_cache: bool = False,
        cache_ttl_seconds: float = 300.0,
        cache_max_size: int = 1000,
        local_mode: bool = False,
        local_pricing: "dict[str, tuple[float, float]] | None" = None,
        _collector=None,
    ):
        from openai import AsyncAzureOpenAI as _AsyncAzureOpenAI
        self._inner     = _AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self._tenant_id = tenant_id
        self._team_id   = team_id
        from savi.local import resolve_emitter
        self._collector = _collector or resolve_emitter(
            savi_key, tenant_id, local_mode, endpoint,
            batch_size, flush_interval_secs, local_pricing,
        )
        masker = _build_masker(mask_pii, pii_entities, pii_exclude_entities)
        cache  = ResponseCache(cache_ttl_seconds, cache_max_size) if enable_cache else None
        # Prevent auto_instrument from wrapping this client a second time.
        self._inner.chat.completions._savi_instrumented = True
        self.chat = _AsyncChatProxy(self._inner, self._collector.emit, tenant_id, team_id, masker, workload_type, cache)


class _AsyncChatProxy:
    def __init__(self, inner, emit_fn, tenant_id, team_id, masker, workload_type=None, cache=None):
        self.completions = _AsyncCompletionsProxy(inner, emit_fn, tenant_id, team_id, masker, workload_type, cache)


class _AsyncCompletionsProxy:
    def __init__(self, inner, emit_fn, tenant_id, team_id, masker, workload_type=None, cache=None):
        self._inner        = inner
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
                        self._tenant, self._team, cached, 0, fp, pii_flagged, pii_types,
                        self._workload_type, is_cache_hit=True,
                    ))
                except Exception:
                    _log.debug("savi.azure_openai: emit failed", exc_info=True)
                return cached

        t0         = time.monotonic()
        response   = await self._inner.chat.completions.create(model=model, messages=messages, **kwargs)
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
                self._tenant, self._team, response, latency_ms, fp, pii_flagged, pii_types,
                self._workload_type,
            ))
        except Exception:
            _log.debug("savi.azure_openai: emit failed", exc_info=True)
        return response
