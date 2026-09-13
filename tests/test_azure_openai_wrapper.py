"""Unit tests for SaviAzureOpenAI / SaviAsyncAzureOpenAI.

No prior test coverage existed for this module (sync or async) before this
file — added alongside the async-support fix rather than left as a gap.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from savi.azure_openai import SaviAzureOpenAI, SaviAsyncAzureOpenAI
from savi.context import SpanContext

_MODEL = "gpt-4o"  # Azure deployment name


def _make_response(prompt_tokens=100, completion_tokens=50, total_tokens=150, model=_MODEL):
    resp = MagicMock()
    resp.model = model
    resp.usage = MagicMock(spec=["prompt_tokens", "completion_tokens", "total_tokens"])
    resp.usage.prompt_tokens     = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens      = total_tokens
    return resp


@pytest.fixture
def mock_collector():
    return MagicMock()


# ── SaviAzureOpenAI (sync) ───────────────────────────────────────────────────

@pytest.fixture
def client(mock_collector):
    return SaviAzureOpenAI(
        azure_endpoint="https://my-resource.openai.azure.com/",
        api_key="test-key",
        api_version="2024-02-01",
        savi_key="sk_test",
        tenant_id="ten_test",
        team_id="engineering",
        mask_pii=False,
        _collector=mock_collector,
    )


def test_emits_event_on_chat_completion(client, mock_collector):
    mock_response = _make_response()
    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        response = client.chat.completions.create(
            model=_MODEL, messages=[{"role": "user", "content": "Hi"}]
        )

    assert response is mock_response
    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"]   == "azure"
    assert event["model"]      == _MODEL
    assert event["tokens_in"]  == 100
    assert event["tokens_out"] == 50
    assert event["tenant_id"]  == "ten_test"
    assert event["team_id"]    == "engineering"


def test_tokens_cached_defaults_to_zero_when_absent(client, mock_collector):
    mock_response = _make_response()
    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model=_MODEL, messages=[])

    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_cached"] == 0


def test_fingerprint_and_pii_fields_present(client, mock_collector):
    mock_response = _make_response()
    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(
            model=_MODEL, messages=[{"role": "user", "content": "Summarise the contract"}]
        )

    event = mock_collector.emit.call_args[0][0]
    assert "lsh_fingerprint" in event
    assert len(event["lsh_fingerprint"]) == 256  # MinHash: 32 x 8-hex
    assert event["pii_flagged"] is False


def test_span_id_propagated_from_context(client, mock_collector):
    mock_response = _make_response()
    with SpanContext(workflow_id="test-wf", agent_id="agent-001") as span:
        with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
            client.chat.completions.create(model=_MODEL, messages=[])

    event = mock_collector.emit.call_args[0][0]
    assert event["parent_span_id"] == span.span_id
    assert event["agent_id"]       == "agent-001"


def test_timestamp_utc_present(client, mock_collector):
    # Required (non-nullable) by LLMEventIn - was missing entirely, so every
    # event this wrapper emitted 422'd silently against the ingest endpoint.
    mock_response = _make_response()
    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model=_MODEL, messages=[])

    event = mock_collector.emit.call_args[0][0]
    assert "timestamp_utc" in event
    assert event["timestamp_utc"] is not None


def test_workload_type_included_when_set(mock_collector):
    client = SaviAzureOpenAI(
        azure_endpoint="https://my-resource.openai.azure.com/",
        api_key="test-key", api_version="2024-02-01",
        savi_key="sk_test", tenant_id="ten_test", mask_pii=False,
        workload_type="summarization", _collector=mock_collector,
    )
    mock_response = _make_response()
    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model=_MODEL, messages=[])

    assert mock_collector.emit.call_args[0][0]["workload_type"] == "summarization"


def test_workload_type_omitted_when_not_set(client, mock_collector):
    mock_response = _make_response()
    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model=_MODEL, messages=[])

    assert "workload_type" not in mock_collector.emit.call_args[0][0]


def test_emit_failure_does_not_break_the_call(client, mock_collector):
    # Telemetry must never break the customer's actual LLM call - same
    # contract SaviOpenAI/SaviAnthropic already give their own callers.
    mock_collector.emit.side_effect = RuntimeError("collector exploded")
    mock_response = _make_response()
    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        response = client.chat.completions.create(model=_MODEL, messages=[])
    assert response is mock_response  # did not raise


def test_savi_instrumented_flag_set_to_prevent_double_instrumentation(client):
    # AzureOpenAI's chat.completions is the SAME class
    # (openai.resources.chat.completions.Completions) that
    # auto_instrument.py patches process-wide - without this flag, using
    # both together would double-count spend on every Azure call.
    assert client._inner.chat.completions._savi_instrumented is True


def test_no_double_emit_with_auto_instrumentation_enabled(mock_collector):
    from savi import auto_instrument
    from openai.resources.chat.completions import Completions

    mock_response = _make_response()
    Completions.create = MagicMock(return_value=mock_response)
    try:
        auto_instrument.enable_auto_instrumentation(
            savi_key="sk_test", tenant_id="ten_auto", mask_pii=False, _collector=MagicMock(),
        )
        client = SaviAzureOpenAI(
            azure_endpoint="https://my-resource.openai.azure.com/",
            api_key="test-key", api_version="2024-02-01",
            savi_key="sk_test", tenant_id="ten_wrapped", mask_pii=False,
            _collector=mock_collector,
        )
        client.chat.completions.create(model=_MODEL, messages=[{"role": "user", "content": "Hi"}])
    finally:
        auto_instrument.disable_auto_instrumentation()

    # Exactly one emit - from SaviAzureOpenAI's own proxy, not the
    # process-wide patch too.
    mock_collector.emit.assert_called_once()
    assert mock_collector.emit.call_args[0][0]["tenant_id"] == "ten_wrapped"


# ── Response caching (backlog item 26) ──────────────────────────────────────

def test_cache_hit_avoids_second_provider_call(mock_collector):
    client = SaviAzureOpenAI(
        azure_endpoint="https://my-resource.openai.azure.com/", api_key="test-key",
        api_version="2024-02-01", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    with patch.object(client._inner.chat.completions, "create", return_value=_make_response()) as mock_create:
        r1 = client.chat.completions.create(model=_MODEL, messages=[{"role": "user", "content": "same"}])
        r2 = client.chat.completions.create(model=_MODEL, messages=[{"role": "user", "content": "same"}])
    assert mock_create.call_count == 1
    assert r2 is r1
    events = [c[0][0] for c in mock_collector.emit.call_args_list]
    assert events[0]["is_cache_hit"] is False
    assert events[1]["is_cache_hit"] is True


def test_cache_disabled_by_default(client, mock_collector):
    with patch.object(client._inner.chat.completions, "create", return_value=_make_response()) as mock_create:
        client.chat.completions.create(model=_MODEL, messages=[{"role": "user", "content": "same"}])
        client.chat.completions.create(model=_MODEL, messages=[{"role": "user", "content": "same"}])
    assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_async_cache_hit_avoids_second_provider_call(mock_collector):
    client = SaviAsyncAzureOpenAI(
        azure_endpoint="https://my-resource.openai.azure.com/", api_key="test-key",
        api_version="2024-02-01", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    with patch.object(
        client._inner.chat.completions, "create", new=AsyncMock(return_value=_make_response()),
    ) as mock_create:
        r1 = await client.chat.completions.create(model=_MODEL, messages=[{"role": "user", "content": "same"}])
        r2 = await client.chat.completions.create(model=_MODEL, messages=[{"role": "user", "content": "same"}])
    assert mock_create.call_count == 1
    assert r2 is r1


# ── SaviAsyncAzureOpenAI ─────────────────────────────────────────────────────
# Async-native equivalent, wraps openai.AsyncAzureOpenAI (SaviAzureOpenAI
# wraps the synchronous AzureOpenAI client and cannot be awaited).

@pytest.fixture
def async_client(mock_collector):
    return SaviAsyncAzureOpenAI(
        azure_endpoint="https://my-resource.openai.azure.com/",
        api_key="test-key",
        api_version="2024-02-01",
        savi_key="sk_test",
        tenant_id="ten_test",
        team_id="engineering",
        mask_pii=False,
        _collector=mock_collector,
    )


@pytest.mark.asyncio
async def test_async_emits_event_on_chat_completion(async_client, mock_collector):
    mock_response = _make_response()
    with patch.object(
        async_client._inner.chat.completions, "create",
        new=AsyncMock(return_value=mock_response),
    ):
        response = await async_client.chat.completions.create(
            model=_MODEL, messages=[{"role": "user", "content": "Hi"}]
        )

    assert response is mock_response
    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"]   == "azure"
    assert event["tokens_in"]  == 100
    assert event["tokens_out"] == 50


@pytest.mark.asyncio
async def test_async_span_id_propagated_from_context(async_client, mock_collector):
    mock_response = _make_response()
    with SpanContext(workflow_id="test-wf", agent_id="agent-001") as span:
        with patch.object(
            async_client._inner.chat.completions, "create",
            new=AsyncMock(return_value=mock_response),
        ):
            await async_client.chat.completions.create(model=_MODEL, messages=[])

    event = mock_collector.emit.call_args[0][0]
    assert event["parent_span_id"] == span.span_id
    assert event["agent_id"]       == "agent-001"


def test_finish_reason_captured_from_choices(client, mock_collector):
    mock_response = _make_response()
    mock_response.choices = [MagicMock(finish_reason="content_filter")]

    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model=_MODEL, messages=[{"role": "user", "content": "Hi"}])

    event = mock_collector.emit.call_args[0][0]
    assert event["finish_reason"] == "content_filter"
