import logging
import os
import time
import uuid
from datetime import datetime, timezone
from anthropic import Anthropic as _Anthropic
from anthropic.types import Message
from savi.context import attribution_fields
from savi.local import resolve_emitter
from savi.pii import fingerprint
from savi.cache import ResponseCache

try:
    from anthropic import AsyncAnthropic as _AsyncAnthropic
except ImportError:  # pragma: no cover, anthropic always ships both; defensive only
    _AsyncAnthropic = None

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


def _client_kwargs(api_key, timeout, max_retries):
    kwargs = {"api_key": api_key}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return kwargs


class SaviAnthropic:
    """Drop-in Anthropic replacement. Tracks cache_read_input_tokens for
    accurate cost, and (with mask_pii, on by default) detects PII in prompts
    so SAVI's own telemetry never stores it. The request Anthropic itself
    receives is never modified by this; see _scan_system_for_pii below."""
    def __init__(self, api_key: str, savi_key: "str | None" = None, tenant_id: "str | None" = None,
                 team_id: str = "default",
                 endpoint: str = _DEFAULT_ENDPOINT,
                 batch_size: int = 100,
                 flush_interval_secs: float = 5.0,
                 mask_pii: "bool | None" = None,
                 pii_entities: "list | None" = None,
                 pii_exclude_entities: "list | None" = None,
                 workload_type: "str | None" = None,
                 timeout: "float | None" = None,
                 max_retries: "int | None" = None,
                 enable_cache: bool = False,
                 cache_ttl_seconds: float = 300.0,
                 cache_max_size: int = 1000,
                 local_mode: bool = False,
                 local_pricing: "dict[str, tuple[float, float]] | None" = None,
                 _collector=None):
        self._inner     = _Anthropic(**_client_kwargs(api_key, timeout, max_retries))
        self._tenant_id = tenant_id
        self._team_id   = team_id
        self._collector = _collector or resolve_emitter(
            savi_key, tenant_id, local_mode, endpoint,
            batch_size, flush_interval_secs, local_pricing,
        )
        masker = _build_masker(mask_pii, pii_entities, pii_exclude_entities)
        cache  = ResponseCache(cache_ttl_seconds, cache_max_size) if enable_cache else None
        # Prevent auto_instrument from wrapping this client a second time.
        self._inner.messages._savi_instrumented = True
        self.messages = _MessagesProxy(self._inner, self._collector.emit, tenant_id, team_id, masker, workload_type, cache)


class _MessagesProxy:
    def __init__(self, inner, emit_fn, tenant_id, team_id, masker, workload_type=None, cache=None):
        self._inner        = inner
        self._emit         = emit_fn
        self._tenant       = tenant_id
        self._team         = team_id
        self._masker       = masker
        self._workload_type = workload_type
        self._cache         = cache

    def create(self, model: str, messages: list, max_tokens: int, **kwargs) -> Message:
        masked_msgs, pii_flagged, pii_types = _mask(self._masker, messages)
        pii_flagged, pii_types = _merge_pii(
            pii_flagged, pii_types, _scan_system_for_pii(self._masker, kwargs.get("system"))
        )
        fp = _cache_key(masked_msgs, model, max_tokens, kwargs)
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
                    _log.debug("savi.anthropic: emit failed", exc_info=True)
                return cached

        t0         = time.monotonic()
        response   = self._inner.messages.create(
            model=model, messages=messages, max_tokens=max_tokens, **kwargs
        )
        if kwargs.get("stream"):
            # A streamed response doesn't carry usage/stop_reason up front
            # the way a full Message does, so there's nothing to emit yet.
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
            _log.debug("savi.anthropic: emit failed", exc_info=True)
        return response


def _mask(masker, messages: list):
    if masker is not None:
        return masker.mask_messages(messages)
    return messages, False, None


def _cache_key(masked_msgs, model: str, max_tokens: int, kwargs: dict) -> str:
    # kwargs is part of the key too: Anthropic's `system` prompt (and tools,
    # temperature, etc.) lives there, not in `messages`.
    return fingerprint((masked_msgs, model, max_tokens, kwargs))


def _scan_system_for_pii(masker, system) -> "dict | None":
    """Count PII entities in Anthropic's separate system prompt (not part of
    `messages`, so mask_messages() never sees it).

    Like message masking, this only affects what SAVI's own telemetry
    receives: `system` is always sent to Anthropic unmodified, since
    redacting a customer's own prompt before the model sees it would break
    the call. Accepts a plain string or a list of content blocks
    ([{"type": "text", "text": "..."}, ...])."""
    if masker is None or not system:
        return None
    counts: dict = {}

    def _accumulate(block_counts: dict) -> None:
        for entity_type, count in block_counts.items():
            counts[entity_type] = counts.get(entity_type, 0) + count

    if isinstance(system, str):
        _, block_counts = masker.mask(system)
        _accumulate(block_counts)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                _, block_counts = masker.mask(block["text"])
                _accumulate(block_counts)
    return counts or None


def _merge_pii(pii_flagged: bool, pii_types: "dict | None", extra_types: "dict | None"):
    if not extra_types:
        return pii_flagged, pii_types
    merged = dict(pii_types or {})
    for entity_type, count in extra_types.items():
        merged[entity_type] = merged.get(entity_type, 0) + count
    return True, merged


def _build_payload(tenant_id, team_id, response, latency_ms, fp, pii_flagged, pii_types, workload_type=None, is_cache_hit=False):
    # Anthropic exposes stop_reason directly on the response.
    _raw_stop_reason = getattr(response, "stop_reason", None)
    finish_reason = _raw_stop_reason[:32] if isinstance(_raw_stop_reason, str) else None
    payload = {
        "event_id":        f"evt_{uuid.uuid4().hex[:24]}",
        "tenant_id":       tenant_id,
        "team_id":         team_id,
        "provider":        "anthropic",
        "model":           response.model,
        # Cache hits report the original call's real token counts, not zero.
        # cost_usd downstream is the cost the hit avoided.
        "tokens_in":       response.usage.input_tokens,
        "tokens_out":      response.usage.output_tokens,
        "tokens_cached":   getattr(response.usage, "cache_read_input_tokens", 0),
        "total_tokens":    response.usage.input_tokens + response.usage.output_tokens,
        "latency_ms":      latency_ms,
        "finish_reason":   finish_reason,
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


class SaviAsyncAnthropic:
    """Async-native equivalent of SaviAnthropic, wraps anthropic.AsyncAnthropic
    so it can be awaited from async application code (SaviAnthropic wraps the
    *synchronous* anthropic.Anthropic client and cannot be awaited). Same
    constructor shape and emitted-event contract as SaviAnthropic.

    Usage:
        client = SaviAsyncAnthropic(api_key=..., savi_key=..., tenant_id=...)
        response = await client.messages.create(model=..., messages=..., max_tokens=...)
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
                 timeout: "float | None" = None,
                 max_retries: "int | None" = None,
                 enable_cache: bool = False,
                 cache_ttl_seconds: float = 300.0,
                 cache_max_size: int = 1000,
                 local_mode: bool = False,
                 local_pricing: "dict[str, tuple[float, float]] | None" = None,
                 _collector=None):
        if _AsyncAnthropic is None:
            raise ImportError("anthropic>=0.40 with AsyncAnthropic support is required for SaviAsyncAnthropic")
        self._inner     = _AsyncAnthropic(**_client_kwargs(api_key, timeout, max_retries))
        self._tenant_id = tenant_id
        self._team_id   = team_id
        self._collector = _collector or resolve_emitter(
            savi_key, tenant_id, local_mode, endpoint,
            batch_size, flush_interval_secs, local_pricing,
        )
        masker = _build_masker(mask_pii, pii_entities, pii_exclude_entities)
        cache  = ResponseCache(cache_ttl_seconds, cache_max_size) if enable_cache else None
        # Prevent auto_instrument from wrapping this client a second time.
        self._inner.messages._savi_instrumented = True
        self.messages = _AsyncMessagesProxy(self._inner, self._collector.emit, tenant_id, team_id, masker, workload_type, cache)


class _AsyncMessagesProxy:
    def __init__(self, inner, emit_fn, tenant_id, team_id, masker, workload_type=None, cache=None):
        self._inner        = inner
        self._emit         = emit_fn
        self._tenant       = tenant_id
        self._team         = team_id
        self._masker       = masker
        self._workload_type = workload_type
        self._cache         = cache

    async def create(self, model: str, messages: list, max_tokens: int, **kwargs) -> Message:
        masked_msgs, pii_flagged, pii_types = _mask(self._masker, messages)
        pii_flagged, pii_types = _merge_pii(
            pii_flagged, pii_types, _scan_system_for_pii(self._masker, kwargs.get("system"))
        )
        fp = _cache_key(masked_msgs, model, max_tokens, kwargs)
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
                    _log.debug("savi.anthropic: emit failed", exc_info=True)
                return cached

        t0         = time.monotonic()
        response   = await self._inner.messages.create(
            model=model, messages=messages, max_tokens=max_tokens, **kwargs
        )
        if kwargs.get("stream"):
            # A streamed response doesn't carry usage/stop_reason up front
            # the way a full Message does, so there's nothing to emit yet.
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
            _log.debug("savi.anthropic: emit failed", exc_info=True)
        return response
