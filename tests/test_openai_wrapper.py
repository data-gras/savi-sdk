import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from savi.openai import SaviOpenAI, SaviAsyncOpenAI
from savi.context import SpanContext
from savi.pii import PiiMasker

@pytest.fixture
def mock_collector():
    return MagicMock()

@pytest.fixture
def client(mock_collector):
    return SaviOpenAI(
        api_key="test-key",
        savi_key="sk_test",
        tenant_id="ten_test",
        team_id="engineering",
        mask_pii=False,
        _collector=mock_collector,
    )

def test_emits_event_on_chat_completion(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens    = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens     = 150
    mock_response.model                  = "gpt-4o"

    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"]  == "openai"
    assert event["model"]     == "gpt-4o"
    assert event["tokens_in"] == 100
    assert event["tokens_out"] == 50
    assert event["tenant_id"] == "ten_test"
    assert event["team_id"]   == "engineering"

def test_finish_reason_captured_from_choices(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens    = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens     = 15
    mock_response.model                  = "gpt-4o"
    mock_response.choices = [MagicMock(finish_reason="length")]

    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    event = mock_collector.emit.call_args[0][0]
    assert event["finish_reason"] == "length"

def test_finish_reason_none_when_choices_empty(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens    = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens     = 15
    mock_response.model                  = "gpt-4o"
    mock_response.choices = []

    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    event = mock_collector.emit.call_args[0][0]
    assert event["finish_reason"] is None

def test_fingerprint_and_pii_fields_present(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens    = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens     = 15
    mock_response.model                  = "gpt-4o"

    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Summarise the contract"}]
        )

    event = mock_collector.emit.call_args[0][0]
    assert "lsh_fingerprint" in event
    assert len(event["lsh_fingerprint"]) == 256  # MinHash: 32 x 8-hex
    assert event["pii_flagged"] is False
    assert event["pii_types"] is None

def test_fingerprint_stable_for_same_messages(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens    = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens     = 15
    mock_response.model                  = "gpt-4o"

    messages = [{"role": "user", "content": "Classify this document"}]
    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model="gpt-4o", messages=messages)
        client.chat.completions.create(model="gpt-4o", messages=messages)

    calls = mock_collector.emit.call_args_list
    assert calls[0][0][0]["lsh_fingerprint"] == calls[1][0][0]["lsh_fingerprint"]

def test_pii_masking_sets_flagged_and_types(mock_collector):
    masker = MagicMock(spec=PiiMasker)
    masked_msgs = [{"role": "user", "content": "Hello <REDACTED>"}]
    masker.mask_messages.return_value = (masked_msgs, True, {"PERSON": 1})

    client = SaviOpenAI(
        api_key="test-key",
        savi_key="sk_test",
        tenant_id="ten_test",
        mask_pii=False,
        _collector=mock_collector,
    )
    client.chat.completions._masker = masker

    mock_response = MagicMock()
    mock_response.usage.prompt_tokens    = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens     = 15
    mock_response.model                  = "gpt-4o"

    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hello John Smith"}]
        )

    event = mock_collector.emit.call_args[0][0]
    assert event["pii_flagged"] is True
    assert event["pii_types"] == {"PERSON": 1}

def test_original_messages_sent_to_provider_not_masked(mock_collector):
    """Provider API always receives original messages; masking is for SAVI metadata only."""
    masker = MagicMock(spec=PiiMasker)
    masked_msgs = [{"role": "user", "content": "Hello <REDACTED>"}]
    masker.mask_messages.return_value = (masked_msgs, True, {"PERSON": 1})

    client = SaviOpenAI(
        api_key="test-key",
        savi_key="sk_test",
        tenant_id="ten_test",
        mask_pii=False,
        _collector=mock_collector,
    )
    client.chat.completions._masker = masker

    original_messages = [{"role": "user", "content": "Hello John Smith"}]
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens    = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens     = 15
    mock_response.model                  = "gpt-4o"

    with patch.object(client._inner.chat.completions, "create", return_value=mock_response) as mock_create:
        client.chat.completions.create(model="gpt-4o", messages=original_messages)

    # Provider must receive the original unmasked messages
    called_messages = mock_create.call_args[1]["messages"]
    assert called_messages[0]["content"] == "Hello John Smith"

def test_span_id_propagated_from_context(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens    = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens     = 15
    mock_response.model                  = "gpt-4o"

    with SpanContext(workflow_id="test-wf", agent_id="agent-001") as span:
        with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
            client.chat.completions.create(model="gpt-4o", messages=[])

    event = mock_collector.emit.call_args[0][0]
    assert event["parent_span_id"] == span.span_id
    assert event["agent_id"]       == "agent-001"

def test_user_id_propagated_from_context(client, mock_collector):
    """user_id set on SpanContext reaches the emitted event — SDK previously never sent it."""
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens    = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens     = 15
    mock_response.model                  = "gpt-4o"

    with SpanContext(workflow_id="test-wf", user_id="user-42"):
        with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
            client.chat.completions.create(model="gpt-4o", messages=[])

    event = mock_collector.emit.call_args[0][0]
    assert event["user_id"] == "user-42"

def test_user_id_absent_when_no_context(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens    = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens     = 15
    mock_response.model                  = "gpt-4o"

    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model="gpt-4o", messages=[])

    event = mock_collector.emit.call_args[0][0]
    assert event["user_id"] is None

def test_tokens_cached_defaults_to_zero_when_absent(client, mock_collector):
    mock_response = MagicMock(spec=["usage", "model"])
    mock_response.usage = MagicMock(spec=["prompt_tokens", "completion_tokens", "total_tokens"])
    mock_response.usage.prompt_tokens    = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens     = 15
    mock_response.model                  = "gpt-4o"

    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model="gpt-4o", messages=[])

    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_cached"] == 0

def test_workload_type_included_in_telemetry(mock_collector):
    client = SaviOpenAI(
        api_key="test-key", savi_key="sk_test",
        tenant_id="ten_test", mask_pii=False,
        workload_type="rag", _collector=mock_collector,
    )
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model = "gpt-4o"

    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model="gpt-4o", messages=[])

    event = mock_collector.emit.call_args[0][0]
    assert event["workload_type"] == "rag"


def test_workload_type_absent_when_not_set(mock_collector):
    client = SaviOpenAI(
        api_key="test-key", savi_key="sk_test",
        tenant_id="ten_test", mask_pii=False,
        _collector=mock_collector,
    )
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model = "gpt-4o"

    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model="gpt-4o", messages=[])

    event = mock_collector.emit.call_args[0][0]
    assert event.get("workload_type") is None


def test_emit_failure_does_not_break_caller(client, mock_collector):
    """A bug in _build_payload/emit must not turn a successful provider call into
    a failed one — the real response is already returned before emit runs."""
    mock_collector.emit.side_effect = RuntimeError("boom")
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens    = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens     = 15
    mock_response.model                  = "gpt-4o"

    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        response = client.chat.completions.create(model="gpt-4o", messages=[])

    assert response is mock_response


def test_emits_error_event_when_provider_call_raises(client, mock_collector):
    """A failed provider call must still be visible to SAVI (is_error=True),
    not silently vanish - and the original exception must still propagate to
    the caller unchanged."""
    class _FakeRateLimitError(Exception):
        status_code = 429

    with patch.object(client._inner.chat.completions, "create", side_effect=_FakeRateLimitError("rate limited")):
        with pytest.raises(_FakeRateLimitError):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["is_error"] is True
    assert event["error_code"] == "429"
    assert event["error_message"] == "rate limited"
    assert event["model"] == "gpt-4o"
    assert event["tokens_in"] == 0
    assert event["tokens_out"] == 0


def test_error_event_falls_back_to_exception_class_name(client, mock_collector):
    """Exceptions with no status_code (timeouts, connection errors) still
    produce a usable error_code via the exception's class name."""
    with patch.object(client._inner.chat.completions, "create", side_effect=TimeoutError("took too long")):
        with pytest.raises(TimeoutError):
            client.chat.completions.create(model="gpt-4o", messages=[])

    event = mock_collector.emit.call_args[0][0]
    assert event["error_code"] == "TimeoutError"
    assert event["error_message"] == "took too long"


def test_error_emit_failure_does_not_mask_original_exception(client, mock_collector):
    """A bug in _build_error_payload/emit must not hide the real provider
    error behind a SAVI-side exception."""
    mock_collector.emit.side_effect = RuntimeError("emit boom")

    with patch.object(client._inner.chat.completions, "create", side_effect=ValueError("provider boom")):
        with pytest.raises(ValueError, match="provider boom"):
            client.chat.completions.create(model="gpt-4o", messages=[])


def test_timestamp_utc_present(client, mock_collector):
    """LLMEventIn (backend schema) requires timestamp_utc - confirmed live
    during a production pilot that its absence 422s the ingest endpoint
    silently (collector._flush() never checks the response status)."""
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens    = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens     = 15
    mock_response.model                  = "gpt-4o"

    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model="gpt-4o", messages=[])

    event = mock_collector.emit.call_args[0][0]
    assert event["timestamp_utc"]


# ── SaviOpenAI.embeddings ────────────────────────────────────────────────────
# The SDK previously had no wrapper for the embeddings endpoint at all -
# callers had to hand-roll emission (see an earlier production pilot). This proxy
# mirrors .chat.completions: same masking/fingerprint/fail-open/timestamp
# guarantees, adapted for embeddings' flat `input: str | list[str]` shape
# (no {role, content} messages) and single total_tokens usage figure.

def test_embeddings_emits_event(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.total_tokens = 12
    mock_response.model              = "text-embedding-3-small"

    with patch.object(client._inner.embeddings, "create", return_value=mock_response):
        client.embeddings.create(model="text-embedding-3-small", input="hello world")

    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"]     == "openai"
    assert event["model"]        == "text-embedding-3-small"
    assert event["tokens_in"]    == 12
    assert event["tokens_out"]   == 0
    assert event["total_tokens"] == 12
    assert event["tenant_id"]    == "ten_test"
    assert event["team_id"]      == "engineering"
    assert event["timestamp_utc"]


def test_embeddings_accepts_list_input():
    mock_collector = MagicMock()
    client = SaviOpenAI(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector,
    )
    mock_response = MagicMock()
    mock_response.usage.total_tokens = 30
    mock_response.model              = "text-embedding-3-small"

    with patch.object(client._inner.embeddings, "create", return_value=mock_response) as mock_create:
        client.embeddings.create(model="text-embedding-3-small", input=["a", "b", "c"])

    assert mock_create.call_args.kwargs["input"] == ["a", "b", "c"]
    event = mock_collector.emit.call_args[0][0]
    assert event["tokens_in"] == 30


def test_embeddings_pii_masking_sets_flagged_and_types(mock_collector):
    masker = MagicMock(spec=PiiMasker)
    masker.mask.return_value = ("Hello <REDACTED>", {"PERSON": 1})

    client = SaviOpenAI(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector,
    )
    client.embeddings._masker = masker

    mock_response = MagicMock()
    mock_response.usage.total_tokens = 8
    mock_response.model              = "text-embedding-3-small"

    with patch.object(client._inner.embeddings, "create", return_value=mock_response):
        client.embeddings.create(model="text-embedding-3-small", input="Hello John Smith")

    event = mock_collector.emit.call_args[0][0]
    assert event["pii_flagged"] is True
    assert event["pii_types"] == {"PERSON": 1}


def test_embeddings_original_input_sent_to_provider_not_masked(mock_collector):
    masker = MagicMock(spec=PiiMasker)
    masker.mask.return_value = ("Hello <REDACTED>", {"PERSON": 1})

    client = SaviOpenAI(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector,
    )
    client.embeddings._masker = masker

    mock_response = MagicMock()
    mock_response.usage.total_tokens = 8
    mock_response.model              = "text-embedding-3-small"

    with patch.object(client._inner.embeddings, "create", return_value=mock_response) as mock_create:
        client.embeddings.create(model="text-embedding-3-small", input="Hello John Smith")

    assert mock_create.call_args.kwargs["input"] == "Hello John Smith"


def test_embeddings_workload_type_included_when_set(mock_collector):
    client = SaviOpenAI(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, embedding_workload_type="embedding", _collector=mock_collector,
    )
    mock_response = MagicMock()
    mock_response.usage.total_tokens = 5
    mock_response.model              = "text-embedding-3-small"

    with patch.object(client._inner.embeddings, "create", return_value=mock_response):
        client.embeddings.create(model="text-embedding-3-small", input="hi")

    event = mock_collector.emit.call_args[0][0]
    assert event["workload_type"] == "embedding"


def test_embeddings_and_chat_workload_types_are_independent(mock_collector):
    """A single client instance can tag chat completions and embeddings
    differently - confirmed this doesn't cross-contaminate."""
    client = SaviOpenAI(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, workload_type="sql-generation",
        embedding_workload_type="embedding", _collector=mock_collector,
    )
    mock_chat_response = MagicMock()
    mock_chat_response.usage.prompt_tokens = 1
    mock_chat_response.usage.completion_tokens = 1
    mock_chat_response.usage.total_tokens = 2
    mock_chat_response.model = "gpt-4o"
    mock_embed_response = MagicMock()
    mock_embed_response.usage.total_tokens = 5
    mock_embed_response.model = "text-embedding-3-small"

    with patch.object(client._inner.chat.completions, "create", return_value=mock_chat_response):
        client.chat.completions.create(model="gpt-4o", messages=[])
    with patch.object(client._inner.embeddings, "create", return_value=mock_embed_response):
        client.embeddings.create(model="text-embedding-3-small", input="hi")

    chat_event, embed_event = [c[0][0] for c in mock_collector.emit.call_args_list]
    assert chat_event["workload_type"]  == "sql-generation"
    assert embed_event["workload_type"] == "embedding"


def test_embeddings_emit_failure_does_not_break_caller(client, mock_collector):
    mock_collector.emit.side_effect = RuntimeError("boom")
    mock_response = MagicMock()
    mock_response.usage.total_tokens = 5
    mock_response.model              = "text-embedding-3-small"

    with patch.object(client._inner.embeddings, "create", return_value=mock_response):
        response = client.embeddings.create(model="text-embedding-3-small", input="hi")

    assert response is mock_response


def test_event_id_is_unique(client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens    = 5
    mock_response.usage.completion_tokens = 2
    mock_response.usage.total_tokens     = 7
    mock_response.model                  = "gpt-4o"

    with patch.object(client._inner.chat.completions, "create", return_value=mock_response):
        client.chat.completions.create(model="gpt-4o", messages=[])
        client.chat.completions.create(model="gpt-4o", messages=[])

    calls  = mock_collector.emit.call_args_list
    ids    = [c[0][0]["event_id"] for c in calls]
    assert ids[0] != ids[1]
    assert all(eid.startswith("evt_") for eid in ids)


# ── Response caching (backlog item 26) ──────────────────────────────────────
# enable_cache=False by default (all other tests above use the default) - a
# repeat prompt always calls the provider again unless a test opts in below.

def _mock_completion(content_tokens=(10, 5, 15), model="gpt-4o"):
    r = MagicMock()
    r.usage.prompt_tokens     = content_tokens[0]
    r.usage.completion_tokens = content_tokens[1]
    r.usage.total_tokens      = content_tokens[2]
    r.model                   = model
    return r


def test_cache_disabled_by_default_calls_provider_every_time(mock_collector):
    client = SaviOpenAI(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector,
    )
    with patch.object(client._inner.chat.completions, "create", return_value=_mock_completion()) as mock_create:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same prompt"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same prompt"}])
    assert mock_create.call_count == 2


def test_cache_hit_avoids_second_provider_call(mock_collector):
    client = SaviOpenAI(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    with patch.object(client._inner.chat.completions, "create", return_value=_mock_completion()) as mock_create:
        r1 = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same prompt"}])
        r2 = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same prompt"}])
    assert mock_create.call_count == 1  # second call replayed from cache, provider never called again
    assert r2 is r1


def test_cache_does_not_collide_across_different_models(mock_collector):
    # Same messages, different model - must never serve the other model's
    # cached response just because the fingerprint used to ignore model.
    client = SaviOpenAI(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    with patch.object(client._inner.chat.completions, "create", return_value=_mock_completion()) as mock_create:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same prompt"}])
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "same prompt"}])
    assert mock_create.call_count == 2


def test_cache_does_not_collide_across_different_kwargs(mock_collector):
    # Same messages and model, different generation params (temperature
    # here) - must not collide on the same fingerprint either.
    client = SaviOpenAI(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    with patch.object(client._inner.chat.completions, "create", return_value=_mock_completion()) as mock_create:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same prompt"}], temperature=0.2)
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same prompt"}], temperature=0.9)
    assert mock_create.call_count == 2


def test_cache_hit_emits_is_cache_hit_true(mock_collector):
    client = SaviOpenAI(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    with patch.object(client._inner.chat.completions, "create", return_value=_mock_completion()):
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same prompt"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same prompt"}])
    first_event, second_event = [c[0][0] for c in mock_collector.emit.call_args_list]
    assert first_event["is_cache_hit"] is False
    assert second_event["is_cache_hit"] is True


def test_cache_miss_for_different_prompt(mock_collector):
    client = SaviOpenAI(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    with patch.object(client._inner.chat.completions, "create", return_value=_mock_completion()) as mock_create:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "prompt one"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "a totally different prompt"}])
    assert mock_create.call_count == 2


def test_cache_never_replays_pii_flagged_prompts(mock_collector):
    """The privacy carve-out: two different real prompts that mask to the
    same template must never share a cached response (see ResponseCache
    docstring - fingerprints are computed on masked, not raw, content)."""
    masker = MagicMock(spec=PiiMasker)
    masker.mask_messages.return_value = (
        [{"role": "user", "content": "My name is <REDACTED>, what's my balance?"}],
        True, {"PERSON": 1},
    )
    client = SaviOpenAI(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    client.chat.completions._masker = masker

    with patch.object(client._inner.chat.completions, "create", return_value=_mock_completion()) as mock_create:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "My name is John, what's my balance?"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "My name is Jane, what's my balance?"}])
    assert mock_create.call_count == 2  # never served Jane a cached answer meant for John


def test_cache_bypassed_for_streaming_calls(mock_collector):
    client = SaviOpenAI(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    with patch.object(client._inner.chat.completions, "create", return_value=_mock_completion()) as mock_create:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same"}], stream=True)
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same"}], stream=True)
    assert mock_create.call_count == 2


def test_cache_hit_response_object_identical_to_original(mock_collector):
    """Replaying must hand back the exact same response the provider gave the
    first time, not a re-serialised copy that might drop provider-specific
    fields the customer's code depends on."""
    client = SaviOpenAI(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    original = _mock_completion()
    with patch.object(client._inner.chat.completions, "create", return_value=original):
        r1 = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same"}])
        r2 = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same"}])
    assert r1 is original
    assert r2 is original


def test_cache_respects_custom_ttl_and_max_size(mock_collector):
    """Constructor params reach the underlying ResponseCache, not just the
    enable/disable flag."""
    client = SaviOpenAI(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True,
        cache_ttl_seconds=60.0, cache_max_size=5,
        _collector=mock_collector,
    )
    cache = client.chat.completions._cache
    assert cache._ttl == 60.0
    assert cache._max_size == 5


@pytest.mark.asyncio
async def test_async_cache_hit_avoids_second_provider_call(mock_collector):
    client = SaviAsyncOpenAI(
        api_key="k", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, enable_cache=True, _collector=mock_collector,
    )
    with patch.object(
        client._inner.chat.completions, "create",
        new=AsyncMock(return_value=_mock_completion()),
    ) as mock_create:
        r1 = await client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same"}])
        r2 = await client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "same"}])
    assert mock_create.call_count == 1
    assert r2 is r1
    events = [c[0][0] for c in mock_collector.emit.call_args_list]
    assert events[1]["is_cache_hit"] is True


# ── SaviAsyncOpenAI ──────────────────────────────────────────────────────────
# Async-native equivalent, added because SaviOpenAI wraps the *synchronous*
# openai.OpenAI client and cannot be awaited from async application code.

@pytest.fixture
def async_client(mock_collector):
    return SaviAsyncOpenAI(
        api_key="test-key",
        savi_key="sk_test",
        tenant_id="ten_test",
        team_id="engineering",
        mask_pii=False,
        _collector=mock_collector,
    )


@pytest.mark.asyncio
async def test_async_emits_event_on_chat_completion(async_client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens     = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens      = 150
    mock_response.model                   = "gpt-4o"

    with patch.object(
        async_client._inner.chat.completions, "create",
        new=AsyncMock(return_value=mock_response),
    ):
        response = await async_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hi"}]
        )

    assert response is mock_response
    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"]   == "openai"
    assert event["model"]      == "gpt-4o"
    assert event["tokens_in"]  == 100
    assert event["tokens_out"] == 50
    assert event["tenant_id"]  == "ten_test"
    assert event["team_id"]    == "engineering"


@pytest.mark.asyncio
async def test_async_original_messages_sent_to_provider_not_masked(mock_collector):
    """Same guarantee as the sync wrapper: provider always gets unmasked messages."""
    masker = MagicMock(spec=PiiMasker)
    masked_msgs = [{"role": "user", "content": "Hello <REDACTED>"}]
    masker.mask_messages.return_value = (masked_msgs, True, {"PERSON": 1})

    client = SaviAsyncOpenAI(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector,
    )
    client.chat.completions._masker = masker

    mock_response = MagicMock()
    mock_response.usage.prompt_tokens     = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens      = 15
    mock_response.model                   = "gpt-4o"

    original_messages = [{"role": "user", "content": "Hello John Smith"}]
    with patch.object(
        client._inner.chat.completions, "create",
        new=AsyncMock(return_value=mock_response),
    ) as mock_create:
        await client.chat.completions.create(model="gpt-4o", messages=original_messages)

    assert mock_create.call_args.kwargs["messages"][0]["content"] == "Hello John Smith"
    event = mock_collector.emit.call_args[0][0]
    assert event["pii_flagged"] is True
    assert event["pii_types"]   == {"PERSON": 1}


@pytest.mark.asyncio
async def test_async_span_id_propagated_from_context(async_client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens     = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens      = 15
    mock_response.model                   = "gpt-4o"

    with SpanContext(workflow_id="test-wf", agent_id="agent-001") as span:
        with patch.object(
            async_client._inner.chat.completions, "create",
            new=AsyncMock(return_value=mock_response),
        ):
            await async_client.chat.completions.create(model="gpt-4o", messages=[])

    event = mock_collector.emit.call_args[0][0]
    assert event["parent_span_id"] == span.span_id
    assert event["agent_id"]       == "agent-001"


@pytest.mark.asyncio
async def test_async_emit_failure_does_not_break_caller(async_client, mock_collector):
    mock_collector.emit.side_effect = RuntimeError("boom")
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens     = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens      = 15
    mock_response.model                   = "gpt-4o"

    with patch.object(
        async_client._inner.chat.completions, "create",
        new=AsyncMock(return_value=mock_response),
    ):
        response = await async_client.chat.completions.create(model="gpt-4o", messages=[])

    assert response is mock_response


@pytest.mark.asyncio
async def test_async_emits_error_event_when_provider_call_raises(async_client, mock_collector):
    class _FakeRateLimitError(Exception):
        status_code = 429

    with patch.object(
        async_client._inner.chat.completions, "create",
        new=AsyncMock(side_effect=_FakeRateLimitError("rate limited")),
    ):
        with pytest.raises(_FakeRateLimitError):
            await async_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["is_error"] is True
    assert event["error_code"] == "429"
    assert event["error_message"] == "rate limited"
    assert event["tokens_in"] == 0
    assert event["tokens_out"] == 0


def test_timeout_and_max_retries_forwarded_to_inner_client(mock_collector):
    with patch("savi.openai._OpenAI") as mock_openai_cls:
        SaviOpenAI(
            api_key="test-key", savi_key="sk_test", tenant_id="ten_test",
            timeout=30.0, max_retries=2, mask_pii=False, _collector=mock_collector,
        )
    mock_openai_cls.assert_called_once_with(api_key="test-key", timeout=30.0, max_retries=2)


def test_timeout_and_max_retries_omitted_when_not_set(mock_collector):
    """Passing None explicitly to the inner OpenAI client would disable its
    default timeout rather than use it — omit the kwargs instead."""
    with patch("savi.openai._OpenAI") as mock_openai_cls:
        SaviOpenAI(
            api_key="test-key", savi_key="sk_test", tenant_id="ten_test",
            mask_pii=False, _collector=mock_collector,
        )
    mock_openai_cls.assert_called_once_with(api_key="test-key")


def test_base_url_forwarded_to_inner_client(mock_collector):
    """Lets SaviOpenAI point at any OpenAI-compatible endpoint (e.g. a local
    Ollama instance at http://localhost:11434/v1), not just the real
    OpenAI API — distinct from `endpoint`, which is SAVI's own collector
    endpoint, not the provider's."""
    with patch("savi.openai._OpenAI") as mock_openai_cls:
        SaviOpenAI(
            api_key="test-key", savi_key="sk_test", tenant_id="ten_test",
            base_url="http://localhost:11434/v1", mask_pii=False, _collector=mock_collector,
        )
    mock_openai_cls.assert_called_once_with(api_key="test-key", base_url="http://localhost:11434/v1")


def test_base_url_omitted_when_not_set(mock_collector):
    with patch("savi.openai._OpenAI") as mock_openai_cls:
        SaviOpenAI(
            api_key="test-key", savi_key="sk_test", tenant_id="ten_test",
            mask_pii=False, _collector=mock_collector,
        )
    mock_openai_cls.assert_called_once_with(api_key="test-key")


def test_async_base_url_forwarded_to_inner_client(mock_collector):
    with patch("savi.openai._AsyncOpenAI") as mock_async_openai_cls:
        SaviAsyncOpenAI(
            api_key="test-key", savi_key="sk_test", tenant_id="ten_test",
            base_url="http://localhost:11434/v1", mask_pii=False, _collector=mock_collector,
        )
    mock_async_openai_cls.assert_called_once_with(api_key="test-key", base_url="http://localhost:11434/v1")


def test_async_client_raises_clear_error_when_openai_async_unavailable(mock_collector):
    import savi.openai as openai_mod
    with patch.object(openai_mod, "_AsyncOpenAI", None):
        with pytest.raises(ImportError, match="AsyncOpenAI"):
            SaviAsyncOpenAI(
                api_key="k", savi_key="sk_test", tenant_id="ten_test",
                _collector=mock_collector,
            )


# ── SaviAsyncOpenAI.embeddings ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_async_embeddings_emits_event(async_client, mock_collector):
    mock_response = MagicMock()
    mock_response.usage.total_tokens = 12
    mock_response.model              = "text-embedding-3-small"

    with patch.object(
        async_client._inner.embeddings, "create",
        new=AsyncMock(return_value=mock_response),
    ):
        response = await async_client.embeddings.create(
            model="text-embedding-3-small", input="hello world"
        )

    assert response is mock_response
    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"]  == "openai"
    assert event["tokens_in"] == 12
    assert event["tenant_id"] == "ten_test"
    assert event["timestamp_utc"]


@pytest.mark.asyncio
async def test_async_embeddings_original_input_sent_to_provider_not_masked(mock_collector):
    masker = MagicMock(spec=PiiMasker)
    masker.mask.return_value = ("Hello <REDACTED>", {"PERSON": 1})

    client = SaviAsyncOpenAI(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_test",
        mask_pii=False, _collector=mock_collector,
    )
    client.embeddings._masker = masker

    mock_response = MagicMock()
    mock_response.usage.total_tokens = 8
    mock_response.model              = "text-embedding-3-small"

    with patch.object(
        client._inner.embeddings, "create",
        new=AsyncMock(return_value=mock_response),
    ) as mock_create:
        await client.embeddings.create(model="text-embedding-3-small", input="Hello John Smith")

    assert mock_create.call_args.kwargs["input"] == "Hello John Smith"
    event = mock_collector.emit.call_args[0][0]
    assert event["pii_flagged"] is True
    assert event["pii_types"]   == {"PERSON": 1}


@pytest.mark.asyncio
async def test_async_embeddings_emit_failure_does_not_break_caller(async_client, mock_collector):
    mock_collector.emit.side_effect = RuntimeError("boom")
    mock_response = MagicMock()
    mock_response.usage.total_tokens = 5
    mock_response.model              = "text-embedding-3-small"

    with patch.object(
        async_client._inner.embeddings, "create",
        new=AsyncMock(return_value=mock_response),
    ):
        response = await async_client.embeddings.create(model="text-embedding-3-small", input="hi")

    assert response is mock_response
