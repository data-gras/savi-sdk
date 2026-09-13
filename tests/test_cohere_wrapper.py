"""Unit tests for SaviCohere.

cohere is mocked throughout — no real API calls or credentials required.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from savi.cohere import SaviCohere, SaviAsyncCohere
from savi.context import SpanContext


_MODEL = "command-r-plus-08-2024"
_MESSAGES = [{"role": "user", "content": "Summarise this contract."}]


def _make_response(input_tokens=150, output_tokens=60, model=_MODEL, billed_units=True):
    resp = MagicMock()
    resp.model = model
    if billed_units:
        resp.usage = MagicMock()
        resp.usage.billed_units = MagicMock()
        resp.usage.billed_units.input_tokens  = input_tokens
        resp.usage.billed_units.output_tokens = output_tokens
    else:
        resp.usage = None
    return resp


@pytest.fixture
def mock_collector():
    return MagicMock()


@pytest.fixture
def mock_cohere_client():
    client = MagicMock()
    client.chat.return_value = _make_response()
    return client


@pytest.fixture
def savi(mock_collector, mock_cohere_client):
    return SaviCohere(
        api_key="test-key",
        savi_key="sk_test",
        tenant_id="ten_test",
        team_id="engineering",
        mask_pii=False,
        _collector=mock_collector,
        _client=mock_cohere_client,
    )


# ── Basic emission ────────────────────────────────────────────────────────────

def test_emits_event_on_chat(savi, mock_collector):
    savi.chat(model=_MODEL, messages=_MESSAGES)

    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"]  == "cohere"
    assert event["tenant_id"] == "ten_test"
    assert event["team_id"]   == "engineering"
    assert event["model"]     == _MODEL


def test_timestamp_utc_present(savi, mock_collector):
    # Required (non-nullable) by LLMEventIn - was missing entirely, so
    # every event this wrapper emitted 422'd silently against the ingest
    # endpoint.
    savi.chat(model=_MODEL, messages=_MESSAGES)
    event = mock_collector.emit.call_args[0][0]
    assert "timestamp_utc" in event
    assert event["timestamp_utc"] is not None


def test_workload_type_included_when_set(mock_collector, mock_cohere_client):
    savi = SaviCohere(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_test", mask_pii=False,
        workload_type="summarization", _collector=mock_collector, _client=mock_cohere_client,
    )
    savi.chat(model=_MODEL, messages=_MESSAGES)
    assert mock_collector.emit.call_args[0][0]["workload_type"] == "summarization"


def test_workload_type_omitted_when_not_set(savi, mock_collector):
    savi.chat(model=_MODEL, messages=_MESSAGES)
    assert "workload_type" not in mock_collector.emit.call_args[0][0]


def test_emit_failure_does_not_break_the_call(savi, mock_collector):
    mock_collector.emit.side_effect = RuntimeError("collector exploded")
    response = savi.chat(model=_MODEL, messages=_MESSAGES)
    assert response is not None  # did not raise


# ── Token mapping ─────────────────────────────────────────────────────────────

def test_tokens_mapped_from_billed_units(savi, mock_collector):
    savi.chat(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_in"]    == 150
    assert event["tokens_out"]   == 60
    assert event["total_tokens"] == 210


def test_total_tokens_is_sum_of_in_and_out(savi, mock_collector, mock_cohere_client):
    mock_cohere_client.chat.return_value = _make_response(input_tokens=300, output_tokens=120)
    savi.chat(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["total_tokens"] == 420


def test_tokens_default_zero_when_usage_none(mock_collector, mock_cohere_client):
    mock_cohere_client.chat.return_value = _make_response(billed_units=False)
    savi = SaviCohere(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector, _client=mock_cohere_client,
    )
    savi.chat(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_in"]  == 0
    assert event["tokens_out"] == 0


def test_tokens_cached_always_zero(savi, mock_collector):
    savi.chat(model=_MODEL, messages=_MESSAGES)
    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_cached"] == 0


# ── Fingerprint + PII fields ──────────────────────────────────────────────────

def test_fingerprint_and_pii_fields_present(savi, mock_collector):
    savi.chat(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert "lsh_fingerprint" in event
    assert len(event["lsh_fingerprint"]) == 256  # MinHash: 32 x 8-hex
    assert event["pii_flagged"] is False
    assert event["pii_types"] is None


def test_original_messages_sent_to_provider_not_masked(mock_collector, mock_cohere_client):
    """Provider must receive original messages even when PII masking is enabled."""
    with patch("savi.cohere._build_masker") as mock_build:
        mock_masker = MagicMock()
        mock_masker.mask_messages.return_value = (
            [{"role": "user", "content": "<REDACTED>"}], True, {"PERSON": 1}
        )
        mock_build.return_value = mock_masker

        savi = SaviCohere(
            api_key="k", savi_key="sk_test", tenant_id="ten_test",
            mask_pii=True,
            _collector=mock_collector,
            _client=mock_cohere_client,
        )

    original_messages = [{"role": "user", "content": "Hello Jane Smith"}]
    savi.chat(model=_MODEL, messages=original_messages)

    call_kwargs = mock_cohere_client.chat.call_args
    assert call_kwargs.kwargs["messages"] is original_messages


# ── SpanContext propagation ───────────────────────────────────────────────────

def test_agent_id_propagated_from_span_context(savi, mock_collector):
    with SpanContext(workflow_id="wf-contracts", agent_id="review-agent") as span:
        savi.chat(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["agent_id"]       == "review-agent"
    assert event["parent_span_id"] == span.span_id


# ── SaviAsyncCohere ──────────────────────────────────────────────────────────
# Async-native equivalent, wraps cohere.AsyncClientV2 (SaviCohere wraps the
# synchronous ClientV2 and cannot be awaited).

@pytest.fixture
def mock_async_cohere_client():
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_response())
    return client


@pytest.fixture
def async_savi(mock_collector, mock_async_cohere_client):
    return SaviAsyncCohere(
        api_key="test-key",
        savi_key="sk_test",
        tenant_id="ten_test",
        team_id="engineering",
        mask_pii=False,
        _collector=mock_collector,
        _client=mock_async_cohere_client,
    )


@pytest.mark.asyncio
async def test_async_emits_event_on_chat(async_savi, mock_collector):
    response = await async_savi.chat(model=_MODEL, messages=_MESSAGES)

    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"]  == "cohere"
    assert event["tenant_id"] == "ten_test"
    assert event["team_id"]   == "engineering"
    assert event["model"]     == _MODEL


@pytest.mark.asyncio
async def test_async_tokens_mapped_from_billed_units(async_savi, mock_collector):
    await async_savi.chat(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_in"]    == 150
    assert event["tokens_out"]   == 60
    assert event["total_tokens"] == 210


@pytest.mark.asyncio
async def test_async_agent_id_propagated_from_span_context(async_savi, mock_collector):
    with SpanContext(workflow_id="wf-contracts", agent_id="review-agent") as span:
        await async_savi.chat(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["agent_id"]       == "review-agent"
    assert event["parent_span_id"] == span.span_id


# ── Response caching (backlog item 26) ──────────────────────────────────────

def test_cache_hit_avoids_second_provider_call(mock_collector, mock_cohere_client):
    savi = SaviCohere(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_test", mask_pii=False,
        enable_cache=True, _collector=mock_collector, _client=mock_cohere_client,
    )
    r1 = savi.chat(model=_MODEL, messages=_MESSAGES)
    r2 = savi.chat(model=_MODEL, messages=_MESSAGES)
    assert mock_cohere_client.chat.call_count == 1
    assert r2 is r1
    events = [c[0][0] for c in mock_collector.emit.call_args_list]
    assert events[0]["is_cache_hit"] is False
    assert events[1]["is_cache_hit"] is True


def test_cache_disabled_by_default(savi, mock_collector, mock_cohere_client):
    savi.chat(model=_MODEL, messages=_MESSAGES)
    savi.chat(model=_MODEL, messages=_MESSAGES)
    assert mock_cohere_client.chat.call_count == 2


def test_finish_reason_lowercased(savi, mock_collector, mock_cohere_client):
    # Cohere reports finish_reason UPPERCASE — normalized to lowercase so it
    # matches every other provider's vocabulary in the stored column.
    response = _make_response()
    response.finish_reason = "MAX_TOKENS"
    mock_cohere_client.chat.return_value = response

    savi.chat(model=_MODEL, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["finish_reason"] == "max_tokens"


@pytest.mark.asyncio
async def test_async_cache_hit_avoids_second_provider_call(mock_collector, mock_async_cohere_client):
    async_savi = SaviAsyncCohere(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_test", mask_pii=False,
        enable_cache=True, _collector=mock_collector, _client=mock_async_cohere_client,
    )
    r1 = await async_savi.chat(model=_MODEL, messages=_MESSAGES)
    r2 = await async_savi.chat(model=_MODEL, messages=_MESSAGES)
    assert mock_async_cohere_client.chat.call_count == 1
    assert r2 is r1
