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


def _mask_converse_messages(masker, messages: list) -> tuple[list, bool, "dict | None"]:
    """Masked copies of Bedrock converse messages ([{"text": "..."}, ...]
    content blocks) for fingerprinting; the provider still gets the
    originals. Returns (masked_messages, pii_flagged, pii_types)."""
    masked_messages = []
    all_counts: dict = {}
    pii_flagged = False

    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            masked_blocks = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    masked_text, counts = masker.mask(block["text"])
                    masked_blocks.append({**block, "text": masked_text})
                    if counts:
                        pii_flagged = True
                        for entity_type, count in counts.items():
                            all_counts[entity_type] = all_counts.get(entity_type, 0) + count
                else:
                    masked_blocks.append(block)
            masked_messages.append({**msg, "content": masked_blocks})
        elif isinstance(content, str):
            masked_text, counts = masker.mask(content)
            masked_messages.append({**msg, "content": masked_text})
            if counts:
                pii_flagged = True
                for entity_type, count in counts.items():
                    all_counts[entity_type] = all_counts.get(entity_type, 0) + count
        else:
            masked_messages.append(msg)

    return masked_messages, pii_flagged, (all_counts if all_counts else None)


class SaviBedrockRuntime:
    """Drop-in observability wrapper for boto3 bedrock-runtime.converse().

    Uses the Converse API (not InvokeModel), unified token schema across all
    Bedrock-hosted models (Claude, Titan, Llama, Mistral, Cohere on Bedrock).

    Usage:
        savi = SaviBedrockRuntime(
            savi_key="sk_...", tenant_id="ten_acme",
            region_name="ap-southeast-2",
        )
        response = savi.converse(
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": [{"text": "Hello"}]}],
        )
    """

    def __init__(
        self,
        savi_key: "str | None" = None,
        tenant_id: "str | None" = None,
        region_name: str = "us-east-1",
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
        **boto3_kwargs,
    ):
        if _client is not None:
            self._client = _client
        else:
            import boto3
            self._client = boto3.client(
                "bedrock-runtime", region_name=region_name, **boto3_kwargs
            )
        self._tenant    = tenant_id
        self._team      = team_id
        self._workload_type = workload_type
        self._collector = _collector or resolve_emitter(
            savi_key, tenant_id, local_mode, endpoint,
            batch_size, flush_interval_secs, local_pricing,
        )
        self._masker = _build_masker(mask_pii, pii_entities, pii_exclude_entities)
        self._cache  = ResponseCache(cache_ttl_seconds, cache_max_size) if enable_cache else None

    def converse(self, model_id: str, messages: list, **kwargs) -> dict:
        """Wrap bedrock-runtime converse(). Provider always receives original messages."""
        fp, pii_flagged, pii_types = self._fingerprint_and_mask(model_id, messages, **kwargs)
        cacheable = self._cache is not None

        if cacheable:
            cached = self._cache.get(fp, pii_flagged)
            if cached is not None:
                try:
                    self._collector.emit(self._build_payload(
                        model_id, cached, 0, fp, pii_flagged, pii_types, is_cache_hit=True,
                    ))
                except Exception:
                    _log.debug("savi.bedrock: emit failed", exc_info=True)
                return cached

        t0         = time.monotonic()
        response   = self._client.converse(modelId=model_id, messages=messages, **kwargs)
        latency_ms = int((time.monotonic() - t0) * 1000)

        if cacheable:
            self._cache.set(fp, pii_flagged, response)

        try:
            self._collector.emit(self._build_payload(model_id, response, latency_ms, fp, pii_flagged, pii_types))
        except Exception:
            _log.debug("savi.bedrock: emit failed", exc_info=True)
        return response

    async def converse_async(self, model_id: str, messages: list, **kwargs) -> dict:
        """Async variant. boto3 has no native async client, so this runs the
        sync call in a thread executor (same approach as
        SaviVertexAI.generate_content_async)."""
        import asyncio
        fp, pii_flagged, pii_types = self._fingerprint_and_mask(model_id, messages, **kwargs)
        cacheable = self._cache is not None

        if cacheable:
            cached = self._cache.get(fp, pii_flagged)
            if cached is not None:
                try:
                    self._collector.emit(self._build_payload(
                        model_id, cached, 0, fp, pii_flagged, pii_types, is_cache_hit=True,
                    ))
                except Exception:
                    _log.debug("savi.bedrock: emit failed", exc_info=True)
                return cached

        loop       = asyncio.get_event_loop()
        t0         = time.monotonic()
        response   = await loop.run_in_executor(
            None, lambda: self._client.converse(modelId=model_id, messages=messages, **kwargs)
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        if cacheable:
            self._cache.set(fp, pii_flagged, response)

        try:
            self._collector.emit(self._build_payload(model_id, response, latency_ms, fp, pii_flagged, pii_types))
        except Exception:
            _log.debug("savi.bedrock: emit failed", exc_info=True)
        return response

    def _fingerprint_and_mask(self, model_id: str, messages: list, **kwargs):
        # model_id + kwargs (inferenceConfig, toolConfig, etc.) are in the
        # cache key too, not just message content.
        if self._masker is not None:
            masked_msgs, pii_flagged, pii_types = _mask_converse_messages(self._masker, messages)
            # Bedrock's `system` is a separate top-level kwarg, same content-
            # block shape as a message: scanned here for telemetry signal
            # only, still sent to the provider untouched via **kwargs.
            system = kwargs.get("system")
            if system:
                _, sys_flagged, sys_types = _mask_converse_messages(self._masker, [{"content": system}])
                if sys_flagged:
                    pii_flagged = True
                    merged = dict(pii_types or {})
                    for entity_type, count in (sys_types or {}).items():
                        merged[entity_type] = merged.get(entity_type, 0) + count
                    pii_types = merged
            return fingerprint((masked_msgs, model_id, kwargs)), pii_flagged, pii_types
        return fingerprint((messages, model_id, kwargs)), False, None

    def _build_payload(self, model_id, response, latency_ms, fp, pii_flagged, pii_types, is_cache_hit=False):
        usage = response.get("usage", {})
        # Bedrock's Converse API reports stopReason at the top level,
        # already using the shared vocabulary ("end_turn", "max_tokens", etc).
        _raw_stop_reason = response.get("stopReason") if isinstance(response, dict) else None
        finish_reason = _raw_stop_reason[:32] if isinstance(_raw_stop_reason, str) else None
        payload = {
            "event_id":        f"evt_{uuid.uuid4().hex[:24]}",
            "tenant_id":       self._tenant,
            "team_id":         self._team,
            "provider":        "bedrock",
            "model":           model_id,
            "tokens_in":       usage.get("inputTokens", 0),
            "tokens_out":      usage.get("outputTokens", 0),
            "tokens_cached":   usage.get("cacheReadInputTokens", 0),
            "total_tokens":    usage.get("totalTokens", 0),
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
        if self._workload_type is not None:
            payload["workload_type"] = self._workload_type
        return payload
