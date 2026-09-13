"""
savi/local.py, local-only event output for developers with no SAVI account.

Nothing in this module ever leaves the customer's machine; local mode
just swaps AsyncEventEmitter's background HTTP POST for a local log line.

No cost_usd is computed by default - SAVI's real pricing table is kept in
sync server-side, and a bundled copy here would go stale with no such
mechanism. Pass local_pricing={"gpt-4o": (usd_per_1m_in, usd_per_1m_out)}
to populate it yourself if you want cost_usd locally.
"""
import json
import logging

_log = logging.getLogger("savi.local")
if not _log.handlers:
    # Attach our own handler so output is visible with zero setup.
    # Propagation stays on, so a caller's own root handler (or caplog in
    # tests) still sees these records too.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)


class LocalEventEmitter:
    """
    Drop-in replacement for AsyncEventEmitter's emit() interface - never
    makes a network call. Every event is logged locally via the standard
    `logging` module (logger name "savi.local") instead of being buffered
    for an HTTP POST.
    """

    def __init__(self, local_pricing: "dict[str, tuple[float, float]] | None" = None):
        # Copied, not aliased - the caller's dict is theirs to keep mutating
        # after construction without silently changing this instance's
        # pricing out from under it.
        self._local_pricing = dict(local_pricing) if local_pricing else {}

    def emit(self, event: dict) -> None:
        event = dict(event)
        cost_usd = _compute_local_cost(event, self._local_pricing)
        if cost_usd is not None:
            event["cost_usd"] = cost_usd
        _log.info("[savi:local] %s", json.dumps(event, default=str))


def _compute_local_cost(event: dict, local_pricing: dict) -> "float | None":
    rates = local_pricing.get(event.get("model"))
    if rates is None:
        return None
    usd_per_1m_in, usd_per_1m_out = rates
    tokens_in  = event.get("tokens_in") or 0
    tokens_out = event.get("tokens_out") or 0
    return (tokens_in / 1_000_000) * usd_per_1m_in + (tokens_out / 1_000_000) * usd_per_1m_out


def resolve_emitter(
    savi_key: "str | None",
    tenant_id: "str | None",
    local_mode: bool,
    endpoint: str,
    batch_size: int,
    flush_interval_secs: float,
    local_pricing: "dict[str, tuple[float, float]] | None" = None,
):
    """
    Single validation point shared by every provider wrapper's constructor.

    Returns a real AsyncEventEmitter when savi_key+tenant_id are given, a
    LocalEventEmitter when local_mode=True, and raises ValueError if
    neither is provided - no silent default, matching the SDK's existing
    "explicit opt-in, no magic defaults" convention.
    """
    if local_mode:
        return LocalEventEmitter(local_pricing=local_pricing)
    if savi_key and tenant_id:
        from savi.collector import AsyncEventEmitter
        return AsyncEventEmitter(
            endpoint, savi_key,
            batch_size=batch_size,
            flush_interval_secs=flush_interval_secs,
        )
    raise ValueError(
        "Pass savi_key and tenant_id for a real SAVI account, or set "
        "local_mode=True to run with zero network calls and no account. "
        "See the SDK README's Local Mode section."
    )
