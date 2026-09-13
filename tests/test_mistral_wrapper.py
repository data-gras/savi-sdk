"""Unit tests for SaviMistral.

mistralai is mocked throughout — no real API calls or credentials required.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from savi.mistral import SaviMistral
from savi.context import SpanContext


_MODEL    = "mistral-large-latest"
_MESSAGES = [{"role": "user", "content": "Summarise this contract."}]


def _make_response(prompt_tokens=150, completion_tokens=60, total_tokens=210,
                   model=_MODEL, cached_tokens=0):
    resp = MagicMock()
    resp.model = model
    resp.usage = MagicMock()
    resp.usage.prompt_tokens     = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens      = total_tokens
    resp.usage.cached_tokens     = cached_tokens
    return resp


@pytest.fixture
def mock_collector():
    return MagicMock()


@pytest.fixture
def mock_mistral_client():
    client = MagicMock()
    client.chat.complete.return_value = _make_response()
    return client


@pytest.fixture
def savi(mock_collector, mock_mistral_client):
    return SaviMistral(
        api_key="test-key",
        savi_key="sk_test",
        tenant_id="ten_test",
        team_id="engineering",
        mask_pii=False,
        _collector=mock_collector,
        _client=mock_mistral_client,
    )


# ── Basic emission ────────────────────────────────────────────────────────────

def test_emits_event_on_chat_complete(savi, mock_collector):
    savi.chat.complete(model=_MODEL, messages=_MESSAGES)

    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"]  == "mistral"
    assert event["tenant_id"] == "ten_test"
    assert event["team_id"]   == "engineering"
    assert event["model"]     == _MODEL


def test_timestamp_utc_present(savi, mock_collector):
    # Required (non-nullable) by LLMEventIn - was missing entirely, so
    # every event this wrapper emitted 422'd silently against the ingest
    # endpoint.
    savi.chat.complete(model=_MODEL, messages=_MESSAGES)
    event = mock_collector.emit.call_args[0][0]
    assert "timestamp_utc" in event
    assert event["timestamp_utc"] is not None


def test_workload_type_included_when_set(mock_collector, mock_mistral_client):
    savi = SaviMistral(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_test", mask_pii=False,
        workload_type="summarization", _collector=mock_collector, _client=mock_mistral_client,
    )
    savi.chat.complete(model=_MODEL, messages=_MESSAGES)
    assert mock_collector.emit.call_args[0][0]["workload_type"] == "summarization"


def test_workload_type_omitted_when_not_set(savi, mock_collector):
    savi.chat.complete(model=_MODEL, messages=_MESSAGES)
    assert "workload_type" not in mock_collector.emit.call_args[0][0]


def test_emit_failure_does_not_break_the_call(savi, mock_collector):
    mock_collector.emit.side_effect = RuntimeError("collector exploded")
    response = savi.chat.complete(model=_MODEL, messages=_MESSAGES)
    assert response is not None  # did not raise


# ── Token mapping ─────────────────────────────────────────────────────────────

def test_tokens_mapped_from_usage(savi, mock_collector):
    savi.chat.complete(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_in"]    == 150
    assert event["tokens_out"]   == 60
    assert event["total_tokens"] == 210


def test_cache_tokens_read_when_present(mock_collector, mock_mistral_client):
    mock_mistral_client.chat.complete.return_value = _make_response(
        prompt_tokens=200, completion_tokens=80, total_tokens=280, cached_tokens=120
    )
    savi = SaviMistral(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector, _client=mock_mistral_client,
    )
    savi.chat.complete(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_cached"] == 120


def test_cache_tokens_default_zero_when_absent(mock_collector, mock_mistral_client):
    resp = MagicMock()
    resp.model = _MODEL
    resp.usage = MagicMock(spec=["prompt_tokens", "completion_tokens", "total_tokens"])
    resp.usage.prompt_tokens     = 100
    resp.usage.completion_tokens = 40
    resp.usage.total_tokens      = 140
    mock_mistral_client.chat.complete.return_value = resp

    savi = SaviMistral(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector, _client=mock_mistral_client,
    )
    savi.chat.complete(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_cached"] == 0


# ── Fingerprint + PII fields ──────────────────────────────────────────────────

def test_fingerprint_and_pii_fields_present(savi, mock_collector):
    savi.chat.complete(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert "lsh_fingerprint" in event
    assert len(event["lsh_fingerprint"]) == 256  # MinHash: 32 x 8-hex
    assert event["pii_flagged"] is False
    assert event["pii_types"] is None


def test_original_messages_sent_to_provider_not_masked(mock_collector, mock_mistral_client):
    """Provider must receive original messages even when PII masking is enabled."""
    with patch("savi.mistral._build_masker") as mock_build:
        mock_masker = MagicMock()
        mock_masker.mask_messages.return_value = (
            [{"role": "user", "content": "<REDACTED>"}], True, {"PERSON": 1}
        )
        mock_build.return_value = mock_masker

        savi = SaviMistral(
            api_key="k", savi_key="sk_test", tenant_id="ten_test",
            mask_pii=True,
            _collector=mock_collector,
            _client=mock_mistral_client,
        )

    original_messages = [{"role": "user", "content": "Hello Jane Smith"}]
    savi.chat.complete(model=_MODEL, messages=original_messages)

    call_kwargs = mock_mistral_client.chat.complete.call_args
    assert call_kwargs.kwargs["messages"] is original_messages


# ── SpanContext propagation ───────────────────────────────────────────────────

def test_agent_id_propagated_from_span_context(savi, mock_collector):
    with SpanContext(workflow_id="wf-legal", agent_id="contract-agent") as span:
        savi.chat.complete(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["agent_id"]       == "contract-agent"
    assert event["parent_span_id"] == span.span_id


# ── complete_async ───────────────────────────────────────────────────────────
# mistralai exposes async as an `_async`-suffixed method on the same client
# (not a separate async client class like openai/anthropic).

@pytest.fixture
def mock_mistral_client_async(mock_mistral_client):
    mock_mistral_client.chat.complete_async = AsyncMock(return_value=_make_response())
    return mock_mistral_client


@pytest.fixture
def savi_async(mock_collector, mock_mistral_client_async):
    return SaviMistral(
        api_key="test-key",
        savi_key="sk_test",
        tenant_id="ten_test",
        team_id="engineering",
        mask_pii=False,
        _collector=mock_collector,
        _client=mock_mistral_client_async,
    )


@pytest.mark.asyncio
async def test_async_emits_event_on_chat_complete(savi_async, mock_collector):
    response = await savi_async.chat.complete_async(model=_MODEL, messages=_MESSAGES)

    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"]  == "mistral"
    assert event["tenant_id"] == "ten_test"
    assert event["team_id"]   == "engineering"
    assert event["model"]     == _MODEL


@pytest.mark.asyncio
async def test_async_tokens_mapped_from_usage(savi_async, mock_collector):
    await savi_async.chat.complete_async(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_in"]    == 150
    assert event["tokens_out"]   == 60
    assert event["total_tokens"] == 210


@pytest.mark.asyncio
async def test_async_agent_id_propagated_from_span_context(savi_async, mock_collector):
    with SpanContext(workflow_id="wf-legal", agent_id="contract-agent") as span:
        await savi_async.chat.complete_async(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["agent_id"]       == "contract-agent"
    assert event["parent_span_id"] == span.span_id


# ── Response caching (backlog item 26) ──────────────────────────────────────

def test_cache_hit_avoids_second_provider_call(mock_collector, mock_mistral_client):
    savi = SaviMistral(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_test", mask_pii=False,
        enable_cache=True, _collector=mock_collector, _client=mock_mistral_client,
    )
    r1 = savi.chat.complete(model=_MODEL, messages=_MESSAGES)
    r2 = savi.chat.complete(model=_MODEL, messages=_MESSAGES)
    assert mock_mistral_client.chat.complete.call_count == 1
    assert r2 is r1
    events = [c[0][0] for c in mock_collector.emit.call_args_list]
    assert events[0]["is_cache_hit"] is False
    assert events[1]["is_cache_hit"] is True


def test_cache_disabled_by_default(savi, mock_collector, mock_mistral_client):
    savi.chat.complete(model=_MODEL, messages=_MESSAGES)
    savi.chat.complete(model=_MODEL, messages=_MESSAGES)
    assert mock_mistral_client.chat.complete.call_count == 2


def test_finish_reason_captured_from_choices(savi, mock_collector, mock_mistral_client):
    response = _make_response()
    response.choices = [MagicMock(finish_reason="length")]
    mock_mistral_client.chat.complete.return_value = response

    savi.chat.complete(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_async_cache_hit_avoids_second_provider_call(mock_collector, mock_mistral_client_async):
    savi_async = SaviMistral(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_test", mask_pii=False,
        enable_cache=True, _collector=mock_collector, _client=mock_mistral_client_async,
    )
    r1 = await savi_async.chat.complete_async(model=_MODEL, messages=_MESSAGES)
    r2 = await savi_async.chat.complete_async(model=_MODEL, messages=_MESSAGES)
    assert mock_mistral_client_async.chat.complete_async.call_count == 1
    assert r2 is r1
