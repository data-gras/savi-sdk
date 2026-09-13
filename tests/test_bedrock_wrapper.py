"""Unit tests for SaviBedrockRuntime.

boto3 is mocked throughout — no real AWS calls or credentials required.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from savi.bedrock import SaviBedrockRuntime, _mask_converse_messages
from savi.context import SpanContext


_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

_MESSAGES = [{"role": "user", "content": [{"text": "Summarise this contract."}]}]

_BEDROCK_RESPONSE = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "Here is a summary."}],
        }
    },
    "stopReason": "end_turn",
    "usage": {
        "inputTokens":  150,
        "outputTokens":  60,
        "totalTokens":  210,
    },
}


@pytest.fixture
def mock_collector():
    return MagicMock()


@pytest.fixture
def mock_boto_client():
    client = MagicMock()
    client.converse.return_value = _BEDROCK_RESPONSE
    return client


@pytest.fixture
def savi(mock_collector, mock_boto_client):
    return SaviBedrockRuntime(
        savi_key="sk_test",
        tenant_id="ten_test",
        team_id="engineering",
        mask_pii=False,
        _collector=mock_collector,
        _client=mock_boto_client,
    )


# ── Basic emission ────────────────────────────────────────────────────────────

def test_emits_event_on_converse(savi, mock_collector):
    savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)

    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"]   == "bedrock"
    assert event["tenant_id"]  == "ten_test"
    assert event["team_id"]    == "engineering"
    assert event["model"]      == _MODEL_ID


def test_timestamp_utc_present(savi, mock_collector):
    # Required (non-nullable) by LLMEventIn - was missing entirely, so
    # every event this wrapper emitted 422'd silently against the ingest
    # endpoint.
    savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)
    event = mock_collector.emit.call_args[0][0]
    assert "timestamp_utc" in event
    assert event["timestamp_utc"] is not None


def test_workload_type_included_when_set(mock_collector, mock_boto_client):
    savi = SaviBedrockRuntime(
        savi_key="sk_test", tenant_id="ten_test", mask_pii=False,
        workload_type="summarization", _collector=mock_collector, _client=mock_boto_client,
    )
    savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)
    assert mock_collector.emit.call_args[0][0]["workload_type"] == "summarization"


def test_workload_type_omitted_when_not_set(savi, mock_collector):
    savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)
    assert "workload_type" not in mock_collector.emit.call_args[0][0]


def test_emit_failure_does_not_break_the_call(savi, mock_collector):
    mock_collector.emit.side_effect = RuntimeError("collector exploded")
    response = savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)
    assert response is _BEDROCK_RESPONSE  # did not raise


# ── Token mapping ─────────────────────────────────────────────────────────────

def test_tokens_mapped_from_bedrock_usage_dict(savi, mock_collector):
    savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_in"]    == 150
    assert event["tokens_out"]   == 60
    assert event["total_tokens"] == 210


def test_cache_tokens_default_zero_when_absent(savi, mock_collector, mock_boto_client):
    mock_boto_client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "Hi"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 50, "outputTokens": 20, "totalTokens": 70},
    }
    savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_cached"] == 0


def test_cache_tokens_read_when_present(savi, mock_collector, mock_boto_client):
    mock_boto_client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": []}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 200, "outputTokens": 80, "totalTokens": 280,
            "cacheReadInputTokens": 120,
        },
    }
    savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_cached"] == 120


# ── Fingerprint + PII fields ──────────────────────────────────────────────────

def test_fingerprint_and_pii_fields_present(savi, mock_collector):
    savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert "lsh_fingerprint" in event
    assert len(event["lsh_fingerprint"]) == 256  # MinHash: 32 x 8-hex
    assert event["pii_flagged"] is False
    assert event["pii_types"] is None


def test_original_messages_sent_to_provider_not_masked(mock_collector, mock_boto_client):
    """Provider must receive original messages even when PII masking is enabled."""
    with patch("savi.bedrock._build_masker") as mock_build:
        mock_masker = MagicMock()
        mock_masker.mask.return_value = ("<REDACTED>", {"PERSON": 1})
        mock_build.return_value = mock_masker

        savi = SaviBedrockRuntime(
            savi_key="sk_test", tenant_id="ten_test",
            mask_pii=True,
            _collector=mock_collector,
            _client=mock_boto_client,
        )

    original_messages = [
        {"role": "user", "content": [{"text": "Hello Jane Smith"}]}
    ]
    savi.converse(model_id=_MODEL_ID, messages=original_messages)

    call_kwargs = mock_boto_client.converse.call_args
    assert call_kwargs.kwargs["messages"] is original_messages


# ── SpanContext propagation ───────────────────────────────────────────────────

def test_agent_id_propagated_from_span_context(savi, mock_collector):
    with SpanContext(workflow_id="wf-order", agent_id="contracts-agent") as span:
        savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["agent_id"]       == "contracts-agent"
    assert event["parent_span_id"] == span.span_id


# ── _mask_converse_messages unit tests ────────────────────────────────────────

def test_mask_converse_messages_handles_content_blocks():
    masker = MagicMock()
    masker.mask.return_value = ("<REDACTED>", {"PERSON": 1})

    messages = [
        {"role": "user", "content": [{"text": "Hello Jane Smith"}, {"text": "How are you?"}]}
    ]
    masked, pii_flagged, pii_types = _mask_converse_messages(masker, messages)

    assert pii_flagged is True
    assert pii_types == {"PERSON": 2}
    assert masked[0]["content"][0]["text"] == "<REDACTED>"
    assert masked[0]["content"][1]["text"] == "<REDACTED>"
    assert masker.mask.call_count == 2


def test_mask_converse_messages_no_pii_returns_false():
    masker = MagicMock()
    masker.mask.return_value = ("Hello world", {})

    messages = [{"role": "user", "content": [{"text": "Hello world"}]}]
    masked, pii_flagged, pii_types = _mask_converse_messages(masker, messages)

    assert pii_flagged is False
    assert pii_types is None
    assert masked[0]["content"][0]["text"] == "Hello world"


# ── converse_async ───────────────────────────────────────────────────────────
# boto3 has no native async client (unlike openai/anthropic's official async
# clients) — converse_async runs the sync boto3 call in a thread executor, same
# pattern SaviVertexAI.generate_content_async already used before this change.

@pytest.mark.asyncio
async def test_async_emits_event_on_converse(savi, mock_collector):
    response = await savi.converse_async(model_id=_MODEL_ID, messages=_MESSAGES)

    assert response == _BEDROCK_RESPONSE
    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"]     == "bedrock"
    assert event["model"]        == _MODEL_ID
    assert event["tokens_in"]    == 150
    assert event["tokens_out"]   == 60
    assert event["total_tokens"] == 210


@pytest.mark.asyncio
async def test_async_agent_id_propagated_from_span_context(savi, mock_collector):
    with SpanContext(workflow_id="wf-order", agent_id="contracts-agent") as span:
        await savi.converse_async(model_id=_MODEL_ID, messages=_MESSAGES)

    event = mock_collector.emit.call_args[0][0]
    assert event["agent_id"]       == "contracts-agent"
    assert event["parent_span_id"] == span.span_id


# ── Response caching (backlog item 26) ──────────────────────────────────────

def test_cache_hit_avoids_second_provider_call(mock_collector, mock_boto_client):
    savi = SaviBedrockRuntime(
        savi_key="sk_test", tenant_id="ten_test", mask_pii=False,
        enable_cache=True, _collector=mock_collector, _client=mock_boto_client,
    )
    r1 = savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)
    r2 = savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)
    assert mock_boto_client.converse.call_count == 1
    assert r2 is r1
    events = [c[0][0] for c in mock_collector.emit.call_args_list]
    assert events[0]["is_cache_hit"] is False
    assert events[1]["is_cache_hit"] is True


def test_cache_disabled_by_default(savi, mock_collector, mock_boto_client):
    savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)
    savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)
    assert mock_boto_client.converse.call_count == 2


def test_finish_reason_captured_from_stop_reason(savi, mock_collector, mock_boto_client):
    # _BEDROCK_RESPONSE already carries "stopReason": "end_turn" — matches
    # the shared vocabulary directly, no per-provider remapping.
    savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)
    event = mock_collector.emit.call_args[0][0]
    assert event["finish_reason"] == "end_turn"


def test_finish_reason_truncated_by_length(savi, mock_collector, mock_boto_client):
    truncated_response = {**_BEDROCK_RESPONSE, "stopReason": "max_tokens"}
    mock_boto_client.converse.return_value = truncated_response
    savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)
    event = mock_collector.emit.call_args[0][0]
    assert event["finish_reason"] == "max_tokens"


@pytest.mark.asyncio
async def test_async_cache_hit_avoids_second_provider_call(mock_collector, mock_boto_client):
    savi = SaviBedrockRuntime(
        savi_key="sk_test", tenant_id="ten_test", mask_pii=False,
        enable_cache=True, _collector=mock_collector, _client=mock_boto_client,
    )
    r1 = await savi.converse_async(model_id=_MODEL_ID, messages=_MESSAGES)
    r2 = await savi.converse_async(model_id=_MODEL_ID, messages=_MESSAGES)
    assert mock_boto_client.converse.call_count == 1
    assert r2 is r1


# ── system-prompt PII scanning ────────────────────────────────────────────
# Bedrock's `system` is a separate top-level converse() kwarg
# ([{"text": "..."}] content blocks), not part of `messages` - was never
# scanned, so PII living only in a system prompt was silently invisible to
# pii_flagged/pii_types telemetry.

def test_pii_in_system_prompt_is_flagged(mock_collector, mock_boto_client):
    # _MESSAGES' own content also passes through mask() here (it's the same
    # mock for both paths) — the assertion below reflects both messages and
    # system contributing PERSON:1 each; test_system_prompt_pii_merges_with_
    # message_pii below isolates the system-only contribution more directly.
    with patch("savi.bedrock._build_masker") as mock_build:
        mock_masker = MagicMock()
        mock_masker.mask.return_value = ("<REDACTED>", {"PERSON": 1})
        mock_build.return_value = mock_masker

        savi = SaviBedrockRuntime(
            savi_key="sk_test", tenant_id="ten_test", mask_pii=True,
            _collector=mock_collector, _client=mock_boto_client,
        )

    savi.converse(
        model_id=_MODEL_ID, messages=_MESSAGES,
        system=[{"text": "You are Jane Smith's assistant."}],
    )

    event = mock_collector.emit.call_args[0][0]
    assert event["pii_flagged"] is True
    assert event["pii_types"] == {"PERSON": 2}


def test_system_prompt_pii_merges_with_message_pii(mock_collector, mock_boto_client):
    with patch("savi.bedrock._build_masker") as mock_build:
        mock_masker = MagicMock()

        def _mask(text):
            if "jane@example.com" in text:
                return "<REDACTED>", {"EMAIL_ADDRESS": 1}
            return "<REDACTED>", {"PERSON": 1}

        mock_masker.mask.side_effect = _mask
        mock_build.return_value = mock_masker

        savi = SaviBedrockRuntime(
            savi_key="sk_test", tenant_id="ten_test", mask_pii=True,
            _collector=mock_collector, _client=mock_boto_client,
        )

    savi.converse(
        model_id=_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": "Hello Jane Smith"}]}],
        system=[{"text": "Contact jane@example.com for help."}],
    )

    event = mock_collector.emit.call_args[0][0]
    assert event["pii_flagged"] is True
    assert event["pii_types"] == {"PERSON": 1, "EMAIL_ADDRESS": 1}


def test_system_prompt_sent_to_provider_unmodified(mock_collector, mock_boto_client):
    with patch("savi.bedrock._build_masker") as mock_build:
        mock_masker = MagicMock()
        mock_masker.mask.return_value = ("<REDACTED>", {"PERSON": 1})
        mock_build.return_value = mock_masker

        savi = SaviBedrockRuntime(
            savi_key="sk_test", tenant_id="ten_test", mask_pii=True,
            _collector=mock_collector, _client=mock_boto_client,
        )

    original_system = [{"text": "You are Jane Smith's assistant."}]
    savi.converse(model_id=_MODEL_ID, messages=_MESSAGES, system=original_system)

    call_kwargs = mock_boto_client.converse.call_args
    assert call_kwargs.kwargs["system"] is original_system


def test_no_system_kwarg_does_not_break_masking(savi, mock_collector):
    # converse() without a `system` kwarg at all must not raise.
    savi.converse(model_id=_MODEL_ID, messages=_MESSAGES)
    event = mock_collector.emit.call_args[0][0]
    assert event["pii_flagged"] is False
