import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from savi.local import LocalEventEmitter, resolve_emitter
from savi.openai import SaviOpenAI


# ---------------------------------------------------------------------------
# resolve_emitter() — the single validation point every provider wrapper uses
# ---------------------------------------------------------------------------

def test_resolve_emitter_raises_when_neither_credentials_nor_local_mode():
    with pytest.raises(ValueError, match="local_mode=True"):
        resolve_emitter(None, None, False, "https://api.savi.io", 100, 5.0)


def test_resolve_emitter_returns_local_emitter_when_local_mode_true():
    emitter = resolve_emitter(None, None, True, "https://api.savi.io", 100, 5.0)
    assert isinstance(emitter, LocalEventEmitter)


def test_resolve_emitter_returns_local_emitter_even_if_credentials_also_given():
    # local_mode=True wins regardless of what else was passed — explicit
    # opt-in, no ambiguity about which path is taken.
    emitter = resolve_emitter("sk_real", "ten_real", True, "https://api.savi.io", 100, 5.0)
    assert isinstance(emitter, LocalEventEmitter)


def test_resolve_emitter_returns_real_emitter_with_credentials():
    from savi.collector import AsyncEventEmitter
    emitter = resolve_emitter("sk_real", "ten_real", False, "https://api.savi.io", 100, 5.0)
    assert isinstance(emitter, AsyncEventEmitter)


# ---------------------------------------------------------------------------
# LocalEventEmitter — never touches the network, logs locally
# ---------------------------------------------------------------------------

def test_local_emitter_never_calls_httpx_post():
    with patch("httpx.post") as mock_post:
        emitter = LocalEventEmitter()
        emitter.emit({"event_id": "evt_1", "model": "gpt-4o", "tokens_in": 10, "tokens_out": 5})
        mock_post.assert_not_called()


def test_local_emitter_logs_event_as_json(caplog):
    with caplog.at_level(logging.INFO, logger="savi.local"):
        LocalEventEmitter().emit({"event_id": "evt_1", "model": "gpt-4o", "tokens_in": 10})

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert message.startswith("[savi:local] ")
    logged_event = json.loads(message.removeprefix("[savi:local] "))
    assert logged_event["event_id"] == "evt_1"
    assert logged_event["model"] == "gpt-4o"


def test_local_emitter_no_cost_usd_without_local_pricing(caplog):
    with caplog.at_level(logging.INFO, logger="savi.local"):
        LocalEventEmitter().emit({"model": "gpt-4o", "tokens_in": 1_000_000, "tokens_out": 1_000_000})

    logged_event = json.loads(caplog.records[0].getMessage().removeprefix("[savi:local] "))
    assert "cost_usd" not in logged_event


def test_local_emitter_computes_cost_usd_from_supplied_local_pricing(caplog):
    emitter = LocalEventEmitter(local_pricing={"gpt-4o": (2.50, 10.00)})
    with caplog.at_level(logging.INFO, logger="savi.local"):
        emitter.emit({"model": "gpt-4o", "tokens_in": 1_000_000, "tokens_out": 1_000_000})

    logged_event = json.loads(caplog.records[0].getMessage().removeprefix("[savi:local] "))
    assert logged_event["cost_usd"] == pytest.approx(12.50)


def test_local_emitter_no_cost_usd_for_unmapped_model(caplog):
    emitter = LocalEventEmitter(local_pricing={"gpt-4o": (2.50, 10.00)})
    with caplog.at_level(logging.INFO, logger="savi.local"):
        emitter.emit({"model": "claude-sonnet-5", "tokens_in": 100, "tokens_out": 50})

    logged_event = json.loads(caplog.records[0].getMessage().removeprefix("[savi:local] "))
    assert "cost_usd" not in logged_event


def test_local_emitter_copies_local_pricing_not_aliased(caplog):
    """The caller's dict is theirs to keep mutating after construction -
    must not silently change this instance's pricing out from under it."""
    pricing = {"gpt-4o": (2.50, 10.00)}
    emitter = LocalEventEmitter(local_pricing=pricing)
    pricing["gpt-4o"] = (0.0, 0.0)  # mutate the caller's own dict after the fact

    with caplog.at_level(logging.INFO, logger="savi.local"):
        emitter.emit({"model": "gpt-4o", "tokens_in": 1_000_000, "tokens_out": 1_000_000})

    logged_event = json.loads(caplog.records[0].getMessage().removeprefix("[savi:local] "))
    assert logged_event["cost_usd"] == pytest.approx(12.50)


# ---------------------------------------------------------------------------
# End-to-end: SaviOpenAI(local_mode=True) — the actual feature a developer uses
# ---------------------------------------------------------------------------

def test_savi_openai_local_mode_requires_no_credentials():
    # Would raise TypeError before this change (savi_key/tenant_id were
    # required positional params) — now succeeds with neither.
    client = SaviOpenAI(api_key="test-key", local_mode=True, mask_pii=False)
    assert isinstance(client._collector, LocalEventEmitter)


def test_savi_openai_local_mode_never_calls_httpx_post(caplog):
    client = SaviOpenAI(api_key="test-key", local_mode=True, mask_pii=False)

    mock_response = MagicMock()
    mock_response.usage.prompt_tokens     = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens      = 150
    mock_response.model                   = "gpt-4o"

    with patch("httpx.post") as mock_post, \
         patch.object(client._inner.chat.completions, "create", return_value=mock_response), \
         caplog.at_level(logging.INFO, logger="savi.local"):
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    mock_post.assert_not_called()
    assert len(caplog.records) == 1
    logged_event = json.loads(caplog.records[0].getMessage().removeprefix("[savi:local] "))
    assert logged_event["provider"]   == "openai"
    assert logged_event["tokens_in"]  == 100
    assert logged_event["tokens_out"] == 50


def test_savi_openai_without_credentials_and_without_local_mode_raises():
    with pytest.raises(ValueError, match="local_mode=True"):
        SaviOpenAI(api_key="test-key", mask_pii=False)


def test_savi_openai_with_real_credentials_still_works_unchanged():
    # Backward-compat guard: existing callers passing savi_key/tenant_id
    # positionally must keep working exactly as before this change.
    mock_collector = MagicMock()
    client = SaviOpenAI("test-key", "sk_test", "ten_test", _collector=mock_collector, mask_pii=False)
    assert client._collector is mock_collector


# ---------------------------------------------------------------------------
# Constructor-level check across other wrappers — local_mode=True needs no
# credentials and no provider package import for wrappers that accept a
# _client injection (bedrock/cohere/mistral bypass their real SDK import
# entirely when _client is given). SaviVertexAI is excluded — it calls
# vertexai.init() unconditionally with no _client bypass, so exercising it
# here would require the vertexai package, which isn't a dev dependency;
# it goes through the identical resolve_emitter() call as every other
# wrapper, so it isn't a separately novel code path.
# ---------------------------------------------------------------------------

def test_anthropic_local_mode_requires_no_credentials():
    from savi.anthropic import SaviAnthropic
    client = SaviAnthropic(api_key="test-key", local_mode=True, mask_pii=False)
    assert isinstance(client._collector, LocalEventEmitter)


def test_azure_openai_local_mode_requires_no_credentials():
    from savi.azure_openai import SaviAzureOpenAI
    client = SaviAzureOpenAI(
        azure_endpoint="https://x.openai.azure.com/", api_key="test-key",
        api_version="2024-02-01", local_mode=True, mask_pii=False,
    )
    assert isinstance(client._collector, LocalEventEmitter)


def test_bedrock_local_mode_requires_no_credentials():
    from savi.bedrock import SaviBedrockRuntime
    client = SaviBedrockRuntime(local_mode=True, mask_pii=False, _client=MagicMock())
    assert isinstance(client._collector, LocalEventEmitter)


def test_cohere_local_mode_requires_no_credentials():
    from savi.cohere import SaviCohere
    client = SaviCohere(api_key="test-key", local_mode=True, mask_pii=False, _client=MagicMock())
    assert isinstance(client._collector, LocalEventEmitter)


def test_mistral_local_mode_requires_no_credentials():
    from savi.mistral import SaviMistral
    client = SaviMistral(api_key="test-key", local_mode=True, mask_pii=False, _client=MagicMock())
    assert isinstance(client._collector, LocalEventEmitter)
