"""MCP tool-call correlation instrumentation.

SAVI's MCP interceptor logs a governance event for every gated MCP tool
call, but without this, that event carries nothing to join it back to
the calling agent's own run.

Patches mcp.client.session.ClientSession.call_tool at the class level
(same approach as savi/auto_instrument.py) to stamp the outgoing call's
`_meta` with the active SpanContext's agent_run_id/parent_span_id, using
MCP's own `_meta` extensibility point under a namespaced key,
`io.savi/context`.

Reads from the same contextvar-based SpanContext (savi/context.py) that
already stamps every LLM call. Outside any active context, both fields
are None and nothing beyond the caller's own `meta` is injected.

Never raises: a correlation failure must never break the tool call.
"""
import logging

from savi.context import attribution_fields

logger = logging.getLogger(__name__)

# (cls, method_name) -> original unbound function, for disable_mcp_instrumentation().
_originals: dict = {}


def enable_mcp_instrumentation() -> None:
    """Patch mcp.client.session.ClientSession.call_tool process-wide.

    Call once, before constructing any MCP ClientSession you want covered.
    A no-op if the `mcp` package isn't installed, or already enabled.
    """
    try:
        from mcp.client.session import ClientSession
    except ImportError:
        logger.debug("savi.mcp_instrument: mcp package not installed - skipping")
        return

    key = (ClientSession, "call_tool")
    if key in _originals:
        return  # already patched
    original = ClientSession.call_tool
    _originals[key] = original
    ClientSession.call_tool = _make_wrapper(original)


def disable_mcp_instrumentation() -> None:
    """Restore the original call_tool. For tests/cleanup, not normal
    customer usage."""
    for (cls, method_name), original in _originals.items():
        setattr(cls, method_name, original)
    _originals.clear()


def _make_wrapper(original):
    # *args/**kwargs rather than redeclaring call_tool's own signature, so
    # this wrapper doesn't drift out of sync with it across mcp versions.
    async def wrapper(self, *args, **kwargs):
        try:
            fields = attribution_fields()
            agent_run_id = fields.get("agent_run_id")
            parent_span_id = fields.get("parent_span_id")
            if agent_run_id or parent_span_id:
                existing_meta = kwargs.get("meta") or {}
                kwargs["meta"] = {
                    **existing_meta,
                    "io.savi/context": {
                        "agent_run_id": agent_run_id,
                        "parent_span_id": parent_span_id,
                    },
                }
        except Exception:
            logger.debug("savi.mcp_instrument: failed to stamp correlation meta", exc_info=True)
        return await original(self, *args, **kwargs)
    return wrapper
