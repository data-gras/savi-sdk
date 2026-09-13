"""Process-wide auto-instrumentation for openai/anthropic clients.

SaviOpenAI/SaviAnthropic only intercept calls through their own wrapped
client, so a plain openai.OpenAI() or anthropic.Anthropic() constructed
elsewhere (a sub-agent framework, a missed wrap) is invisible to SAVI.
This patches the provider SDK classes directly -
openai.resources.chat.completions.{Completions,AsyncCompletions}.create
and anthropic.resources.messages.{Messages,AsyncMessages}.create - so
every client built in this process after enable_auto_instrumentation()
is covered, including ones SAVI's own code never sees. Only providers
whose package is installed get patched.

Coexists safely with SaviOpenAI/SaviAnthropic: each wrapper tags its own
resource object (_savi_instrumented = True) so the patched method calls
straight through instead of masking+emitting a second time (which would
double-count spend).

Real, unmasked messages always reach the provider - masking only shapes
what SAVI's own telemetry captures.

Must be called before constructing any client you want covered.

Scope is narrower than SaviOpenAI/SaviAnthropic: chat/messages
completions only, no embeddings, no response caching.

Same local_mode option as every direct wrapper (see resolve_emitter() in
savi/local.py): pass local_mode=True for the same zero-account,
zero-network-call behavior, applied process-wide instead of per client.
"""
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone

from savi.context import attribution_fields
from savi.local import resolve_emitter
from savi.pii import fingerprint

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = os.environ.get("SAVI_ENDPOINT", "https://api.datagras.com")

_state: dict = {
    "enabled": False,
    "collector": None,
    "tenant_id": None,
    "team_id": None,
    "masker": None,
    "workload_type": None,
}
# (cls, method_name) -> original unbound function, for disable_auto_instrumentation().
_originals: dict = {}
# Only guards the enable/patch sequence itself, not every _state read -
# prevents two threads racing enable_auto_instrumentation() at startup
# from both patching.
_init_lock = threading.Lock()


def enable_auto_instrumentation(
    savi_key: "str | None" = None,
    tenant_id: "str | None" = None,
    team_id: str = "default",
    endpoint: "str | None" = None,
    mask_pii: "bool | None" = None,
    pii_entities: "list | None" = None,
    pii_exclude_entities: "list | None" = None,
    workload_type: "str | None" = None,
    batch_size: int = 100,
    flush_interval_secs: float = 5.0,
    local_mode: bool = False,
    local_pricing: "dict[str, tuple[float, float]] | None" = None,
    _collector=None,
) -> None:
    """Patch openai/anthropic chat/message completion calls process-wide.

    Call once, before constructing any client you want covered - typically
    the first line of your entrypoint, before any framework/sub-agent
    import that might construct its own client.

    Same savi_key/tenant_id-or-local_mode choice as every direct wrapper
    (SaviOpenAI, etc.) - see resolve_emitter() in savi/local.py.
    """
    with _init_lock:
        if _state["enabled"]:
            logger.warning(
                "savi.auto_instrument: already enabled - call "
                "disable_auto_instrumentation() first to reconfigure"
            )
            return

        collector = _collector or resolve_emitter(
            savi_key, tenant_id, local_mode, endpoint or _DEFAULT_ENDPOINT,
            batch_size, flush_interval_secs, local_pricing,
        )
        # mask_pii is None (the default) means "on if possible" - Presidio
        # missing degrades to unmasked with a warning. mask_pii=True is an
        # explicit ask for the compliance guarantee, so a missing dependency
        # stays a hard failure rather than a silent gap.
        masker = None
        if mask_pii is not False:
            from savi.pii import PiiMasker
            try:
                masker = PiiMasker(entities=pii_entities, exclude_entities=pii_exclude_entities)
            except ImportError:
                if mask_pii is True:
                    raise
                logger.warning(
                    "PII masking is on by default but Presidio isn't installed - "
                    "continuing without masking. Install with: pip install 'savi-sdk[pii]', "
                    "or pass mask_pii=False to silence this warning."
                )

        _state.update(
            enabled=True, collector=collector, tenant_id=tenant_id, team_id=team_id,
            masker=masker, workload_type=workload_type,
        )

        _patch_openai()
        _patch_anthropic()


def disable_auto_instrumentation() -> None:
    """Restore original methods and clear state. For tests/cleanup - not
    normal customer usage (there's no legitimate reason to turn PII
    masking off mid-process in production)."""
    with _init_lock:
        for (cls, method_name), original in _originals.items():
            setattr(cls, method_name, original)
        _originals.clear()
        _state.update(
            enabled=False, collector=None, tenant_id=None, team_id=None,
            masker=None, workload_type=None,
        )


def _patch_method(cls, method_name: str, make_wrapper) -> None:
    key = (cls, method_name)
    if key in _originals:
        return  # already patched this exact class/method
    original = getattr(cls, method_name)
    _originals[key] = original
    setattr(cls, method_name, make_wrapper(original))


def _patch_openai() -> None:
    try:
        from openai.resources.chat.completions import Completions, AsyncCompletions
    except ImportError:
        logger.debug("savi.auto_instrument: openai not installed - skipping")
        return
    _patch_method(Completions, "create", _make_sync_wrapper("openai"))
    _patch_method(AsyncCompletions, "create", _make_async_wrapper("openai"))


def _patch_anthropic() -> None:
    try:
        from anthropic.resources.messages import Messages, AsyncMessages
    except ImportError:
        logger.debug("savi.auto_instrument: anthropic not installed - skipping")
        return
    _patch_method(Messages, "create", _make_sync_wrapper("anthropic"))
    _patch_method(AsyncMessages, "create", _make_async_wrapper("anthropic"))


def _mask_messages(messages):
    masker = _state["masker"]
    if masker is None or messages is None:
        return messages, False, None
    return masker.mask_messages(messages)


def _emit(provider, response, latency_ms, fp, pii_flagged, pii_types) -> None:
    try:
        payload = _build_payload(provider, response, latency_ms, fp, pii_flagged, pii_types)
        _state["collector"].emit(payload)
    except Exception:
        # Telemetry must never break the customer's actual LLM call.
        logger.debug("savi.auto_instrument: failed to emit event", exc_info=True)


def _build_payload(provider, response, latency_ms, fp, pii_flagged, pii_types) -> dict:
    if provider == "openai":
        tokens_in     = response.usage.prompt_tokens
        tokens_out    = response.usage.completion_tokens
        tokens_cached = getattr(response.usage, "cached_tokens", 0)
        total_tokens  = response.usage.total_tokens
        _choices = getattr(response, "choices", None)
        _raw_stop_reason = getattr(_choices[0], "finish_reason", None) if _choices else None
    else:  # anthropic
        tokens_in     = response.usage.input_tokens
        tokens_out    = response.usage.output_tokens
        tokens_cached = getattr(response.usage, "cache_read_input_tokens", 0)
        total_tokens  = tokens_in + tokens_out
        # anthropic reports stop_reason directly on the response, no
        # choices[] wrapper.
        _raw_stop_reason = getattr(response, "stop_reason", None)
    finish_reason = _raw_stop_reason[:32] if isinstance(_raw_stop_reason, str) else None

    payload = {
        "event_id":        f"evt_{uuid.uuid4().hex[:24]}",
        "tenant_id":       _state["tenant_id"],
        "team_id":         _state["team_id"],
        "provider":        provider,
        "model":           response.model,
        "tokens_in":       tokens_in,
        "tokens_out":      tokens_out,
        "tokens_cached":   tokens_cached,
        "total_tokens":    total_tokens,
        "latency_ms":      latency_ms,
        "finish_reason":   finish_reason,
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        **attribution_fields(),
        "lsh_fingerprint": fp,
        "pii_flagged":     pii_flagged,
        "pii_types":       pii_types,
        "is_cache_hit":    False,  # auto-instrumentation has no response cache (see module docstring)
        "schema_version":  "1.0",
    }
    if _state["workload_type"] is not None:
        payload["workload_type"] = _state["workload_type"]
    return payload


def _make_sync_wrapper(provider: str):
    def make(original):
        def wrapper(self, *args, **kwargs):
            if getattr(self, "_savi_instrumented", False) or not _state["enabled"]:
                return original(self, *args, **kwargs)

            # kwargs["messages"] stays untouched; masked_msgs is only used
            # below for SAVI's own telemetry.
            messages = kwargs.get("messages")
            masked_msgs, pii_flagged, pii_types = _mask_messages(messages)
            fp = fingerprint(masked_msgs) if masked_msgs is not None else ""

            t0 = time.monotonic()
            response = original(self, *args, **kwargs)
            latency_ms = int((time.monotonic() - t0) * 1000)

            _emit(provider, response, latency_ms, fp, pii_flagged, pii_types)
            return response
        return wrapper
    return make


def _make_async_wrapper(provider: str):
    def make(original):
        async def wrapper(self, *args, **kwargs):
            if getattr(self, "_savi_instrumented", False) or not _state["enabled"]:
                return await original(self, *args, **kwargs)

            # kwargs["messages"] stays untouched; masked_msgs is only used
            # below for SAVI's own telemetry.
            messages = kwargs.get("messages")
            masked_msgs, pii_flagged, pii_types = _mask_messages(messages)
            fp = fingerprint(masked_msgs) if masked_msgs is not None else ""

            t0 = time.monotonic()
            response = await original(self, *args, **kwargs)
            latency_ms = int((time.monotonic() - t0) * 1000)

            # emit() is a plain buffer append, not a coroutine - fine to
            # call directly here without awaiting it.
            _emit(provider, response, latency_ms, fp, pii_flagged, pii_types)
            return response
        return wrapper
    return make
