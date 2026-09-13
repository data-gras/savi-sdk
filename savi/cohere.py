import logging
import os
import time
import uuid
from datetime import datetime, timezone
from savi.context import attribution_fields
from savi.local import resolve_emitter
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


class SaviCohere:
    """Drop-in observability wrapper for cohere.ClientV2.chat().

    Uses the v2 messages API (OpenAI-compatible format). Tokens are read from
    response.usage.billed_units, what Cohere actually charges for.

    Usage:
        savi = SaviCohere(api_key="...", savi_key="sk_...", tenant_id="ten_acme")
        response = savi.chat(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": "Summarise this contract."}],
        )
    """

    def __init__(
        self,
        api_key: str,
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
        _client=None,
    ):
        if _client is not None:
            self._inner = _client
        else:
            import cohere
            self._inner = cohere.ClientV2(api_key=api_key)
        self._tenant    = tenant_id
        self._team      = team_id
        self._workload_type = workload_type
        self._collector = _collector or resolve_emitter(
            savi_key, tenant_id, local_mode, endpoint,
            batch_size, flush_interval_secs, local_pricing,
        )
        self._masker = _build_masker(mask_pii, pii_entities, pii_exclude_entities)
        self._cache  = ResponseCache(cache_ttl_seconds, cache_max_size) if enable_cache else None

    def chat(self, model: str, messages: list, **kwargs) -> object:
        """Wrap ClientV2.chat(). Provider always receives original messages."""
        masked_msgs, pii_flagged, pii_types = _mask(self._masker, messages)
        fp = _cache_key(masked_msgs, model, kwargs)
        cacheable = self._cache is not None and not kwargs.get("stream")

        if cacheable:
            cached = self._cache.get(fp, pii_flagged)
            if cached is not None:
                try:
                    self._collector.emit(_build_payload(
                        self._tenant, self._team, model, cached, 0, fp, pii_flagged, pii_types,
                        self._workload_type, is_cache_hit=True,
                    ))
                except Exception:
                    _log.debug("savi.cohere: emit failed", exc_info=True)
                return cached

        t0         = time.monotonic()
        response   = self._inner.chat(model=model, messages=messages, **kwargs)
        if kwargs.get("stream"):
            # A streamed response doesn't carry usage/finish_reason up front
            # the way a full ChatResponse does, so there's nothing to emit yet.
            return response
        latency_ms = int((time.monotonic() - t0) * 1000)

        if cacheable:
            self._cache.set(fp, pii_flagged, response)

        try:
            self._collector.emit(_build_payload(
                self._tenant, self._team, model, response, latency_ms, fp, pii_flagged, pii_types,
                self._workload_type,
            ))
        except Exception:
            _log.debug("savi.cohere: emit failed", exc_info=True)
        return response


def _mask(masker, messages: list):
    if masker is not None:
        return masker.mask_messages(messages)
    return messages, False, None


def _cache_key(masked_msgs, model: str, kwargs: dict) -> str:
    # model + kwargs are part of the key too, not just message content.
    return fingerprint((masked_msgs, model, kwargs))


def _build_payload(tenant_id, team_id, model, response, latency_ms, fp, pii_flagged, pii_types,
                    workload_type=None, is_cache_hit=False):
    # billed_units = what Cohere charges; may be None on edge cases (e.g. errors)
    billed     = getattr(response.usage, "billed_units", None) if response.usage else None
    tokens_in  = int(getattr(billed, "input_tokens",  0) or 0) if billed else 0
    tokens_out = int(getattr(billed, "output_tokens", 0) or 0) if billed else 0
    # Cohere reports finish_reason UPPERCASE; lowercased to match the other
    # providers' vocabulary in the stored column.
    _raw_finish_reason = getattr(response, "finish_reason", None)
    finish_reason = _raw_finish_reason.lower()[:32] if isinstance(_raw_finish_reason, str) else None
    payload = {
        "event_id":        f"evt_{uuid.uuid4().hex[:24]}",
        "tenant_id":       tenant_id,
        "team_id":         team_id,
        "provider":        "cohere",
        "model":           getattr(response, "model", model),
        "tokens_in":       tokens_in,
        "tokens_out":      tokens_out,
        "tokens_cached":   0,  # Cohere's Chat v2 API doesn't report a cached-token count
        "total_tokens":    tokens_in + tokens_out,
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


class SaviAsyncCohere:
    """Async-native equivalent of SaviCohere, wraps cohere.AsyncClientV2 so it
    can be awaited from async application code. Same constructor shape and
    emitted-event contract as SaviCohere.

    Usage:
        client = SaviAsyncCohere(api_key=..., savi_key=..., tenant_id=...)
        response = await client.chat(model=..., messages=...)
    """

    def __init__(
        self,
        api_key: str,
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
        _client=None,
    ):
        if _client is not None:
            self._inner = _client
        else:
            import cohere
            self._inner = cohere.AsyncClientV2(api_key=api_key)
        self._tenant    = tenant_id
        self._team      = team_id
        self._workload_type = workload_type
        self._collector = _collector or resolve_emitter(
            savi_key, tenant_id, local_mode, endpoint,
            batch_size, flush_interval_secs, local_pricing,
        )
        self._masker = _build_masker(mask_pii, pii_entities, pii_exclude_entities)
        self._cache  = ResponseCache(cache_ttl_seconds, cache_max_size) if enable_cache else None

    async def chat(self, model: str, messages: list, **kwargs) -> object:
        """Wrap AsyncClientV2.chat(). Provider always receives original messages."""
        masked_msgs, pii_flagged, pii_types = _mask(self._masker, messages)
        fp = _cache_key(masked_msgs, model, kwargs)
        cacheable = self._cache is not None and not kwargs.get("stream")

        if cacheable:
            cached = self._cache.get(fp, pii_flagged)
            if cached is not None:
                try:
                    self._collector.emit(_build_payload(
                        self._tenant, self._team, model, cached, 0, fp, pii_flagged, pii_types,
                        self._workload_type, is_cache_hit=True,
                    ))
                except Exception:
                    _log.debug("savi.cohere: emit failed", exc_info=True)
                return cached

        t0         = time.monotonic()
        response   = await self._inner.chat(model=model, messages=messages, **kwargs)
        if kwargs.get("stream"):
            # A streamed response doesn't carry usage/finish_reason up front
            # the way a full ChatResponse does, so there's nothing to emit yet.
            return response
        latency_ms = int((time.monotonic() - t0) * 1000)

        if cacheable:
            self._cache.set(fp, pii_flagged, response)

        # emit() is synchronous (a plain buffer append), so it's called
        # directly here rather than awaited.
        try:
            self._collector.emit(_build_payload(
                self._tenant, self._team, model, response, latency_ms, fp, pii_flagged, pii_types,
                self._workload_type,
            ))
        except Exception:
            _log.debug("savi.cohere: emit failed", exc_info=True)
        return response
