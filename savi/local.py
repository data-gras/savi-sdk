"""
savi/local.py, local-only event output for developers with no SAVI account.

Nothing in this module ever leaves the customer's machine; local mode
just swaps AsyncEventEmitter's background HTTP POST for a local log line.

No cost_usd is computed by default - SAVI's real pricing table is kept in
sync server-side, and a bundled copy here would go stale with no such
mechanism. Pass local_pricing={"gpt-4o": (usd_per_1m_in, usd_per_1m_out)}
to populate it yourself if you want cost_usd locally.

A key matches the model name exactly, or any model name that starts with it
(the longest key wins), because providers often return a dated name such as
"gpt-4o-mini-2024-07-18" for a request made as "gpt-4o-mini". Costs built from
your own rates carry cost_estimated=True. A cache hit costs nothing (cost_usd
is 0.0 and cost_saved_usd shows what the call would have cost), and a failed
call gets no cost.
"""
import json
import logging
import os

from savi.content import strip_content

_log = logging.getLogger("savi.local")
if not _log.handlers:
    # Attach our own handler so output is visible with zero setup.
    # Propagation stays on, so a caller's own root handler (or caplog in
    # tests) still sees these records too.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)


# Hooks for savi.local_report: it listens for events and can turn the raw console line off.
_listeners: list = []
_console = True
_default_pricing: dict = {}
_env_pricing_cache: "tuple[str, dict] | None" = None


def add_listener(fn) -> None:
    """Call fn(event) for every local event, after cost is filled in."""
    if fn not in _listeners:
        _listeners.append(fn)


def remove_listener(fn) -> None:
    if fn in _listeners:
        _listeners.remove(fn)


def set_console(on: bool) -> None:
    """Turn the raw JSON console line on or off. Listeners and other handlers still get every event."""
    global _console
    _console = bool(on)


def _clean_rates(rates: dict, where: str) -> dict:
    out = {}
    for key, value in rates.items():
        if str(key).startswith("_"):
            continue
        try:
            r_in, r_out = value
            r_in, r_out = float(r_in), float(r_out)
        except (TypeError, ValueError):
            raise ValueError(f'{where}: the price for "{key}" must be two numbers, like [0.15, 0.60] (input and output price per 1M tokens)') from None
        if r_in < 0 or r_out < 0:
            raise ValueError(f'{where}: the price for "{key}" cannot be negative')
        out[str(key)] = (r_in, r_out)
    return out


def load_pricing_file(path: str) -> dict:
    """Read prices from a JSON file: {"gpt-4o-mini": [0.15, 0.60]}, in USD per 1M tokens (input, output)."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Prices file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON ({exc.msg} at line {exc.lineno}). Check commas and quotes.") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold one JSON object, like {{\"gpt-4o-mini\": [0.15, 0.60]}}")
    return _clean_rates(data, path)


def set_default_pricing(rates: "dict | str | None") -> None:
    """Prices used by every local client that has no local_pricing of its own. Pass a dict, a file path, or None to clear."""
    global _default_pricing
    if rates is None:
        _default_pricing = {}
    elif isinstance(rates, (str, os.PathLike)):
        _default_pricing = load_pricing_file(os.fspath(rates))
    else:
        _default_pricing = _clean_rates(dict(rates), "rates")


def _env_pricing() -> dict:
    """Prices from the file named in SAVI_LOCAL_PRICING, re-read only when the path changes."""
    global _env_pricing_cache
    path = os.environ.get("SAVI_LOCAL_PRICING")
    if not path:
        return {}
    if _env_pricing_cache is None or _env_pricing_cache[0] != path:
        _env_pricing_cache = (path, load_pricing_file(path))
    return _env_pricing_cache[1]


def active_pricing(own: "dict | None" = None) -> dict:
    """Every price in force: the environment file, then the default set by set_default_pricing, then the client's own."""
    return {**_env_pricing(), **_default_pricing, **(own or {})}


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
        # Content fields never reach a log line, even locally.
        event = strip_content(event)
        if not event.get("is_error"):
            pricing = active_pricing(self._local_pricing)
            key = _find_key(pricing, event.get("model"))
            if key is not None:
                cost_usd = _cost_from_rates(event, pricing[key])
                if event.get("is_cache_hit"):
                    # The provider was never called, so this call cost nothing; show what it saved.
                    event["cost_saved_usd"] = cost_usd
                    cost_usd = 0.0
                event["cost_usd"] = cost_usd
                event["cost_estimated"] = True
                event["cost_rate_key"] = key
                event["cost_rates"] = list(pricing[key])
        for fn in list(_listeners):
            try:
                fn(event)
            except Exception:
                _log.debug("savi.local: listener failed", exc_info=True)
        if _console:
            _log.info("[savi:local] %s", json.dumps(event, default=str))


def _find_key(local_pricing: dict, model: "str | None"):
    """Exact model name first, otherwise the longest key the model name starts with."""
    if not model:
        return None
    if model in local_pricing:
        return model
    keys = [k for k in local_pricing if model.startswith(k)]
    return max(keys, key=len) if keys else None


def _find_rates(local_pricing: dict, model: "str | None"):
    key = _find_key(local_pricing, model)
    return local_pricing[key] if key is not None else None


def _cost_from_rates(event: dict, rates) -> float:
    usd_per_1m_in, usd_per_1m_out = rates
    tokens_in  = event.get("tokens_in") or 0
    tokens_out = event.get("tokens_out") or 0
    return (tokens_in / 1_000_000) * usd_per_1m_in + (tokens_out / 1_000_000) * usd_per_1m_out


def _compute_local_cost(event: dict, local_pricing: dict) -> "float | None":
    rates = _find_rates(local_pricing, event.get("model"))
    return None if rates is None else _cost_from_rates(event, rates)


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
        emitter = AsyncEventEmitter(
            endpoint, savi_key,
            batch_size=batch_size,
            flush_interval_secs=flush_interval_secs,
        )
        return emitter
    raise ValueError(
        "Pass savi_key and tenant_id for a real SAVI account, or set "
        "local_mode=True to run with zero network calls and no account. "
        "See the SDK README's Local Mode section."
    )
