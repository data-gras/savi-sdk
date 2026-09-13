import logging
import os
import time
import uuid
from datetime import datetime, timezone
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


class SaviVertexAI:
    """Drop-in wrapper for Vertex AI GenerativeModel with SAVI observability.

    Usage:
        savi = SaviVertexAI(project="my-project", location="us-central1",
                             savi_key="sk_...", tenant_id="ten_acme")
        model = savi.GenerativeModel("gemini-1.5-pro-002")
        response = model.generate_content("Summarise this contract: ...")
    """

    def __init__(
        self,
        project: str,
        location: str,
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
        import vertexai
        vertexai.init(project=project, location=location)
        self._project   = project
        self._location  = location
        self._tenant_id = tenant_id
        self._team_id   = team_id
        self._workload_type = workload_type
        from savi.local import resolve_emitter
        self._collector = _collector or resolve_emitter(
            savi_key, tenant_id, local_mode, endpoint,
            batch_size, flush_interval_secs, local_pricing,
        )
        self._masker = _build_masker(mask_pii, pii_entities, pii_exclude_entities)
        self._cache  = ResponseCache(cache_ttl_seconds, cache_max_size) if enable_cache else None

    def GenerativeModel(self, model_name: str) -> "_WrappedModel":
        import vertexai.generative_models as _vg
        inner = _vg.GenerativeModel(model_name=model_name)
        return _WrappedModel(inner, model_name, self._collector.emit,
                             self._tenant_id, self._team_id, self._masker,
                             self._workload_type, self._cache)


class _WrappedModel:
    def __init__(self, inner, model_name: str, emit_fn, tenant_id: str, team_id: str, masker,
                 workload_type=None, cache=None):
        self._inner        = inner
        self._model_name    = model_name
        self._emit          = emit_fn
        self._tenant        = tenant_id
        self._team          = team_id
        self._masker        = masker
        self._workload_type = workload_type
        self._cache         = cache

    def generate_content(self, contents, **kwargs) -> object:
        # Mask string prompts; complex Content objects are fingerprinted
        # but not masked. model doesn't need to be in the cache key (it's
        # fixed per wrapped instance), but kwargs does.
        if self._masker is not None and isinstance(contents, str):
            masked_text, counts = self._masker.mask(contents)
            pii_flagged = bool(counts)
            pii_types   = counts if counts else None
            fp = fingerprint((masked_text, kwargs))
        else:
            pii_flagged = False
            pii_types   = None
            fp = fingerprint((contents, kwargs))

        cacheable = self._cache is not None and not kwargs.get("stream")
        if cacheable:
            cached = self._cache.get(fp, pii_flagged)
            if cached is not None:
                try:
                    self._emit(self._build_payload(cached, 0, fp, pii_flagged, pii_types, is_cache_hit=True))
                except Exception:
                    _log.debug("savi.google_vertex: emit failed", exc_info=True)
                return cached

        t0         = time.monotonic()
        response   = self._inner.generate_content(contents, **kwargs)
        if kwargs.get("stream"):
            # A streamed response doesn't carry usage_metadata up front the
            # way a full GenerationResponse does, so there's nothing to emit yet.
            return response
        latency_ms = int((time.monotonic() - t0) * 1000)

        if cacheable:
            self._cache.set(fp, pii_flagged, response)

        try:
            self._emit(self._build_payload(response, latency_ms, fp, pii_flagged, pii_types))
        except Exception:
            _log.debug("savi.google_vertex: emit failed", exc_info=True)
        return response

    @staticmethod
    def _finish_reason(response) -> "str | None":
        """Vertex reports finish_reason as an enum on the first candidate;
        .name lowercased matches the other providers' vocabulary. Never
        raises."""
        try:
            candidates = getattr(response, "candidates", None)
            if not candidates:
                return None
            raw = getattr(candidates[0], "finish_reason", None)
            if raw is None:
                return None
            name = getattr(raw, "name", None) or str(raw)
            return name.lower()[:32]
        except Exception:
            return None

    def _build_payload(self, response, latency_ms, fp, pii_flagged, pii_types, is_cache_hit=False):
        from savi.context import attribution_fields

        meta          = response.usage_metadata
        tokens_in     = getattr(meta, "prompt_token_count",          0) or 0
        tokens_out    = getattr(meta, "candidates_token_count",      0) or 0
        tokens_cached = getattr(meta, "cached_content_token_count",  0) or 0
        # Prefer Vertex's own reported total; fall back to prompt+candidates
        # only if a response shape omits it.
        tokens_total  = getattr(meta, "total_token_count", None) or (tokens_in + tokens_out)

        payload = {
            "event_id":        f"evt_{uuid.uuid4().hex[:24]}",
            "tenant_id":       self._tenant,
            "team_id":         self._team,
            "provider":        "google",
            "model":           self._model_name,
            "tokens_in":       tokens_in,
            "tokens_out":      tokens_out,
            "tokens_cached":   tokens_cached,
            "total_tokens":    tokens_total,
            "latency_ms":      latency_ms,
            "finish_reason":   self._finish_reason(response),
            "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
            **attribution_fields(),
            "lsh_fingerprint": fp,
            "pii_flagged":     pii_flagged,
            "pii_types":       pii_types,
            "is_cache_hit":    is_cache_hit,
            "schema_version":  "1.0",
        }
        if self._workload_type is not None:
            payload["workload_type"] = self._workload_type
        return payload

    async def generate_content_async(self, contents, **kwargs) -> object:
        """Async variant, wraps the sync call since Vertex SDK has limited async support."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate_content(contents, **kwargs)
        )
