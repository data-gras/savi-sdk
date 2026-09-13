import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from savi.anthropic import SaviAnthropic, SaviAsyncAnthropic
from savi.context import SpanContext

@pytest.fixture
def mock_collector():
    return MagicMock()

@pytest.fixture
def client(mock_collector):
    return SaviAnthropic(
        api_key="test-key",
        savi_key="sk_test",
        tenant_id="ten_test",
        team_id="engineering",
        mask_pii=False,
        _collector=mock_collector,
    )

def test_emits_event_on_message_create(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.input_tokens            = 200
    mock_response.usage.output_tokens           = 80
    mock_response.usage.cache_read_input_tokens = 50
    mock_response.model                         = "claude-sonnet-4-6"

    with patch.object(client._inner.messages, "create", return_value=mock_response):
        client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
        )

    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"]      == "anthropic"
    assert event["tokens_in"]     == 200
    assert event["tokens_out"]    == 80
    assert event["tokens_cached"] == 50
    assert event["tenant_id"]     == "ten_test"

def test_finish_reason_captured_from_stop_reason(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.input_tokens            = 200
    mock_response.usage.output_tokens           = 80
    mock_response.usage.cache_read_input_tokens = 50
    mock_response.model                         = "claude-sonnet-4-6"
    mock_response.stop_reason                   = "max_tokens"

    with patch.object(client._inner.messages, "create", return_value=mock_response):
        client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
        )

    event = mock_collector.emit.call_args[0][0]
    assert event["finish_reason"] == "max_tokens"

def test_fingerprint_and_pii_fields_present(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.input_tokens            = 100
    mock_response.usage.output_tokens           = 40
    mock_response.usage.cache_read_input_tokens = 0
    mock_response.model                         = "claude-sonnet-4-6"

    with patch.object(client._inner.messages, "create", return_value=mock_response):
        client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Classify this document"}],
            max_tokens=256,
        )

    event = mock_collector.emit.call_args[0][0]
    assert "lsh_fingerprint" in event
    assert len(event["lsh_fingerprint"]) == 256  # MinHash: 32 x 8-hex
    assert event["pii_flagged"] is False
    assert event["pii_types"] is None

def test_cache_tokens_default_to_zero_when_absent(client, mock_collector):
    mock_response = MagicMock(spec=["usage", "model"])
    mock_response.usage = MagicMock(spec=["input_tokens", "output_tokens"])
    mock_response.usage.input_tokens  = 100
    mock_response.usage.output_tokens = 40
    mock_response.model               = "claude-haiku-4-5-20251001"

    with patch.object(client._inner.messages, "create", return_value=mock_response):
        client.messages.create(model="claude-haiku-4-5-20251001", messages=[], max_tokens=512)

    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_cached"] == 0

def test_total_tokens_is_sum_of_input_and_output(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.input_tokens            = 300
    mock_response.usage.output_tokens           = 120
    mock_response.usage.cache_read_input_tokens = 0
    mock_response.model                         = "claude-sonnet-4-6"

    with patch.object(client._inner.messages, "create", return_value=mock_response):
        client.messages.create(model="claude-sonnet-4-6", messages=[], max_tokens=1024)

    event = mock_collector.emit.call_args[0][0]
    assert event["total_tokens"] == 420

def test_timestamp_utc_present(client, mock_collector):
    """LLMEventIn (backend schema) requires timestamp_utc - confirmed live
    during a production pilot that its absence 422s the ingest endpoint
    silently (collector._flush() never checks the response status)."""
    mock_response = MagicMock()
    mock_response.usage.input_tokens            = 50
    mock_response.usage.output_tokens           = 20
    mock_response.usage.cache_read_input_tokens = 0
    mock_response.model                         = "claude-sonnet-5"

    with patch.object(client._inner.messages, "create", return_value=mock_response):
        client.messages.create(model="claude-sonnet-5", messages=[], max_tokens=256)

    event = mock_collector.emit.call_args[0][0]
    assert event["timestamp_utc"]


def test_workload_type_included_in_telemetry(mock_collector):
    client = SaviAnthropic(
        api_key="test-key", savi_key="sk_test",
        tenant_id="ten_test", mask_pii=False,
        workload_type="sql-generation", _collector=mock_collector,
    )
    mock_response = MagicMock()
    mock_response.usage.input_tokens            = 50
    mock_response.usage.output_tokens           = 20
    mock_response.usage.cache_read_input_tokens = 0
    mock_response.model                         = "claude-sonnet-5"

    with patch.object(client._inner.messages, "create", return_value=mock_response):
        client.messages.create(model="claude-sonnet-5", messages=[], max_tokens=256)

    event = mock_collector.emit.call_args[0][0]
    assert event["workload_type"] == "sql-generation"


def test_workload_type_absent_when_not_set(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.input_tokens            = 50
    mock_response.usage.output_tokens           = 20
    mock_response.usage.cache_read_input_tokens = 0
    mock_response.model                         = "claude-sonnet-5"

    with patch.object(client._inner.messages, "create", return_value=mock_response):
        client.messages.create(model="claude-sonnet-5", messages=[], max_tokens=256)

    event = mock_collector.emit.call_args[0][0]
    assert event.get("workload_type") is None


def test_emit_failure_does_not_break_caller(client, mock_collector):
    """A bug in _build_payload/emit must not turn a successful provider call into
    a failed one — the real response is already returned before emit runs."""
    mock_collector.emit.side_effect = RuntimeError("boom")
    mock_response = MagicMock()
    mock_response.usage.input_tokens            = 50
    mock_response.usage.output_tokens           = 20
    mock_response.usage.cache_read_input_tokens = 0
    mock_response.model                         = "claude-sonnet-5"

    with patch.object(client._inner.messages, "create", return_value=mock_response):
        response = client.messages.create(model="claude-sonnet-5", messages=[], max_tokens=256)

    assert response is mock_response


def test_timeout_and_max_retries_forwarded_to_inner_client(mock_collector):
    with patch("savi.anthropic._Anthropic") as mock_anthropic_cls:
        SaviAnthropic(
            api_key="test-key", savi_key="sk_test", tenant_id="ten_test",
            timeout=90.0, max_retries=2, mask_pii=False, _collector=mock_collector,
        )
    mock_anthropic_cls.assert_called_once_with(api_key="test-key", timeout=90.0, max_retries=2)


def test_timeout_and_max_retries_omitted_when_not_set(mock_collector):
    """Passing None explicitly to the inner Anthropic client would disable its
    default timeout rather than use it — omit the kwargs instead."""
    with patch("savi.anthropic._Anthropic") as mock_anthropic_cls:
        SaviAnthropic(
            api_key="test-key", savi_key="sk_test", tenant_id="ten_test",
            mask_pii=False, _collector=mock_collector,
        )
    mock_anthropic_cls.assert_called_once_with(api_key="test-key")


def test_agent_id_propagated_from_span_context(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.input_tokens            = 50
    mock_response.usage.output_tokens           = 20
    mock_response.usage.cache_read_input_tokens = 0
    mock_response.model                         = "claude-haiku-4-5-20251001"

    with SpanContext(workflow_id="wf", agent_id="kyc-agent") as span:
        with patch.object(client._inner.messages, "create", return_value=mock_response):
            client.messages.create(model="claude-haiku-4-5-20251001", messages=[], max_tokens=256)

    event = mock_collector.emit.call_args[0][0]
    assert event["agent_id"]       == "kyc-agent"
    assert event["parent_span_id"] == span.span_id


# ── Response caching (backlog item 26) ──────────────────────────────────────

def _mock_message(model="claude-sonnet-4-6"):
    r = MagicMock()
    r.usage.input_tokens            = 100
    r.usage.output_tokens           = 40
    r.usage.cache_read_input_tokens = 0
    r.model                         = model
    return r


def test_cache_hit_avoids_second_provider_call(mock_collector):
    client = SaviAnthropic(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    with patch.object(client._inner.messages, "create", return_value=_mock_message()) as mock_create:
        r1 = client.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "same"}], max_tokens=256)
        r2 = client.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "same"}], max_tokens=256)
    assert mock_create.call_count == 1
    assert r2 is r1
    events = [c[0][0] for c in mock_collector.emit.call_args_list]
    assert events[0]["is_cache_hit"] is False
    assert events[1]["is_cache_hit"] is True


def test_cache_does_not_collide_across_different_models(mock_collector):
    # Same messages, different model - must never serve the other model's
    # cached response just because the fingerprint used to ignore model.
    client = SaviAnthropic(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    with patch.object(client._inner.messages, "create", return_value=_mock_message()) as mock_create:
        client.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "same"}], max_tokens=256)
        client.messages.create(model="claude-haiku-4-6", messages=[{"role": "user", "content": "same"}], max_tokens=256)
    assert mock_create.call_count == 2


def test_cache_does_not_collide_across_different_max_tokens(mock_collector):
    client = SaviAnthropic(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    with patch.object(client._inner.messages, "create", return_value=_mock_message()) as mock_create:
        client.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "same"}], max_tokens=100)
        client.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "same"}], max_tokens=1000)
    assert mock_create.call_count == 2


def test_cache_does_not_collide_across_different_system_prompts(mock_collector):
    # Anthropic-specific: `system` is a top-level parameter, not part of
    # `messages` the way OpenAI-style system messages are, so it would
    # otherwise be invisible to the cache key entirely.
    client = SaviAnthropic(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    with patch.object(client._inner.messages, "create", return_value=_mock_message()) as mock_create:
        client.messages.create(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "same"}],
            max_tokens=256, system="You are a doctor.",
        )
        client.messages.create(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "same"}],
            max_tokens=256, system="You are a lawyer.",
        )
    assert mock_create.call_count == 2


def test_cache_disabled_by_default(mock_collector):
    client = SaviAnthropic(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector,
    )
    with patch.object(client._inner.messages, "create", return_value=_mock_message()) as mock_create:
        client.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "same"}], max_tokens=256)
        client.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "same"}], max_tokens=256)
    assert mock_create.call_count == 2


def test_cache_never_replays_pii_flagged_prompts(mock_collector):
    from savi.pii import PiiMasker
    masker = MagicMock(spec=PiiMasker)
    masker.mask_messages.return_value = (
        [{"role": "user", "content": "My name is <REDACTED>, what's my balance?"}], True, {"PERSON": 1},
    )
    client = SaviAnthropic(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    client.messages._masker = masker
    with patch.object(client._inner.messages, "create", return_value=_mock_message()) as mock_create:
        client.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "My name is John, what's my balance?"}], max_tokens=256)
        client.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "My name is Jane, what's my balance?"}], max_tokens=256)
    assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_async_cache_hit_avoids_second_provider_call(mock_collector):
    client = SaviAsyncAnthropic(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    with patch.object(
        client._inner.messages, "create", new=AsyncMock(return_value=_mock_message()),
    ) as mock_create:
        r1 = await client.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "same"}], max_tokens=256)
        r2 = await client.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "same"}], max_tokens=256)
    assert mock_create.call_count == 1
    assert r2 is r1


# ── system-prompt PII scanning (`system` is a top-level param, not part of
# `messages` — mask_messages() never sees it, so PII living only in a
# system prompt was silently invisible to pii_flagged/pii_types) ───────────

def test_pii_in_string_system_prompt_is_flagged(mock_collector):
    from savi.pii import PiiMasker
    masker = MagicMock(spec=PiiMasker)
    masker.mask_messages.return_value = ([{"role": "user", "content": "Hello"}], False, None)
    masker.mask.return_value = ("My name is <REDACTED>", {"PERSON": 1})

    client = SaviAnthropic(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector,
    )
    client.messages._masker = masker
    with patch.object(client._inner.messages, "create", return_value=_mock_message()):
        client.messages.create(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "Hello"}],
            max_tokens=256, system="My name is Jane Smith",
        )

    event = mock_collector.emit.call_args[0][0]
    assert event["pii_flagged"] is True
    assert event["pii_types"] == {"PERSON": 1}


def test_pii_in_content_block_system_prompt_is_flagged(mock_collector):
    from savi.pii import PiiMasker
    masker = MagicMock(spec=PiiMasker)
    masker.mask_messages.return_value = ([{"role": "user", "content": "Hello"}], False, None)
    masker.mask.return_value = ("Contact <REDACTED>", {"EMAIL_ADDRESS": 1})

    client = SaviAnthropic(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector,
    )
    client.messages._masker = masker
    with patch.object(client._inner.messages, "create", return_value=_mock_message()):
        client.messages.create(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "Hello"}],
            max_tokens=256, system=[{"type": "text", "text": "Contact jane@example.com"}],
        )

    event = mock_collector.emit.call_args[0][0]
    assert event["pii_flagged"] is True
    assert event["pii_types"] == {"EMAIL_ADDRESS": 1}


def test_system_prompt_pii_merges_with_message_pii(mock_collector):
    from savi.pii import PiiMasker
    masker = MagicMock(spec=PiiMasker)
    masker.mask_messages.return_value = (
        [{"role": "user", "content": "<REDACTED>"}], True, {"PERSON": 1},
    )
    masker.mask.return_value = ("<REDACTED>", {"EMAIL_ADDRESS": 1})

    client = SaviAnthropic(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector,
    )
    client.messages._masker = masker
    with patch.object(client._inner.messages, "create", return_value=_mock_message()):
        client.messages.create(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "John Smith"}],
            max_tokens=256, system="jane@example.com",
        )

    event = mock_collector.emit.call_args[0][0]
    assert event["pii_flagged"] is True
    assert event["pii_types"] == {"PERSON": 1, "EMAIL_ADDRESS": 1}


def test_system_prompt_not_scanned_when_masking_disabled(mock_collector):
    client = SaviAnthropic(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector,
    )
    with patch.object(client._inner.messages, "create", return_value=_mock_message()):
        client.messages.create(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "Hello"}],
            max_tokens=256, system="My name is Jane Smith",
        )

    event = mock_collector.emit.call_args[0][0]
    assert event["pii_flagged"] is False
    assert event["pii_types"] is None


def test_system_prompt_sent_to_provider_unmodified(mock_collector):
    """The scan is for telemetry signal only - the provider must still
    receive the original, unmasked system prompt (matches every other
    content path in this file)."""
    from savi.pii import PiiMasker
    masker = MagicMock(spec=PiiMasker)
    masker.mask_messages.return_value = ([{"role": "user", "content": "Hello"}], False, None)
    masker.mask.return_value = ("My name is <REDACTED>", {"PERSON": 1})

    client = SaviAnthropic(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector,
    )
    client.messages._masker = masker
    with patch.object(client._inner.messages, "create", return_value=_mock_message()) as mock_create:
        client.messages.create(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "Hello"}],
            max_tokens=256, system="My name is Jane Smith",
        )

    assert mock_create.call_args.kwargs["system"] == "My name is Jane Smith"


@pytest.mark.asyncio
async def test_async_pii_in_system_prompt_is_flagged(mock_collector):
    from savi.pii import PiiMasker
    masker = MagicMock(spec=PiiMasker)
    masker.mask_messages.return_value = ([{"role": "user", "content": "Hello"}], False, None)
    masker.mask.return_value = ("My name is <REDACTED>", {"PERSON": 1})

    client = SaviAsyncAnthropic(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector,
    )
    client.messages._masker = masker
    with patch.object(
        client._inner.messages, "create", new=AsyncMock(return_value=_mock_message()),
    ):
        await client.messages.create(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "Hello"}],
            max_tokens=256, system="My name is Jane Smith",
        )

    event = mock_collector.emit.call_args[0][0]
    assert event["pii_flagged"] is True
    assert event["pii_types"] == {"PERSON": 1}


# ── SaviAsyncAnthropic ───────────────────────────────────────────────────────
# Async-native equivalent, added because SaviAnthropic wraps the *synchronous*
# anthropic.Anthropic client and cannot be awaited from async application code.

@pytest.fixture
def async_client(mock_collector):
    return SaviAsyncAnthropic(
        api_key="test-key",
        savi_key="sk_test",
        tenant_id="ten_test",
        team_id="engineering",
        mask_pii=False,
        _collector=mock_collector,
    )


@pytest.mark.asyncio
async def test_async_emits_event_on_message_create(async_client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.input_tokens            = 200
    mock_response.usage.output_tokens           = 80
    mock_response.usage.cache_read_input_tokens  = 50
    mock_response.model                          = "claude-sonnet-4-6"

    with patch.object(
        async_client._inner.messages, "create",
        new=AsyncMock(return_value=mock_response),
    ):
        response = await async_client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
        )

    assert response is mock_response
    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"]      == "anthropic"
    assert event["tokens_in"]     == 200
    assert event["tokens_out"]    == 80
    assert event["tokens_cached"] == 50
    assert event["tenant_id"]     == "ten_test"


@pytest.mark.asyncio
async def test_async_agent_id_propagated_from_span_context(async_client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.input_tokens            = 50
    mock_response.usage.output_tokens           = 20
    mock_response.usage.cache_read_input_tokens = 0
    mock_response.model                         = "claude-haiku-4-5-20251001"

    with SpanContext(workflow_id="wf", agent_id="kyc-agent") as span:
        with patch.object(
            async_client._inner.messages, "create",
            new=AsyncMock(return_value=mock_response),
        ):
            await async_client.messages.create(
                model="claude-haiku-4-5-20251001", messages=[], max_tokens=256
            )

    event = mock_collector.emit.call_args[0][0]
    assert event["agent_id"]       == "kyc-agent"
    assert event["parent_span_id"] == span.span_id


@pytest.mark.asyncio
async def test_async_emit_failure_does_not_break_caller(async_client, mock_collector):
    mock_collector.emit.side_effect = RuntimeError("boom")
    mock_response = MagicMock()
    mock_response.usage.input_tokens            = 50
    mock_response.usage.output_tokens           = 20
    mock_response.usage.cache_read_input_tokens = 0
    mock_response.model                         = "claude-sonnet-5"

    with patch.object(
        async_client._inner.messages, "create",
        new=AsyncMock(return_value=mock_response),
    ):
        response = await async_client.messages.create(
            model="claude-sonnet-5", messages=[], max_tokens=256
        )

    assert response is mock_response


def test_async_client_raises_clear_error_when_anthropic_async_unavailable(mock_collector):
    import savi.anthropic as anthropic_mod
    with patch.object(anthropic_mod, "_AsyncAnthropic", None):
        with pytest.raises(ImportError, match="AsyncAnthropic"):
            SaviAsyncAnthropic(
                api_key="k", savi_key="sk_test", tenant_id="ten_test",
                _collector=mock_collector,
            )
