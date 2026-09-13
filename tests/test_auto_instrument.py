import pytest
from unittest.mock import MagicMock
from savi import auto_instrument
from savi.pii import PiiMasker

# auto_instrument patches class-level methods shared by every test file in
# this suite (openai.resources.chat.completions.Completions.create etc.) -
# a leaked patch would silently break every other SDK test file that
# constructs a plain or Savi-wrapped client. Snapshotting the real class
# attributes here and force-restoring them after every test (regardless of
# what auto_instrument's own bookkeeping thinks happened, and regardless of
# test failure) is a belt-and-suspenders guarantee on top of
# disable_auto_instrumentation() - correct even if a test directly
# overwrites Completions.create rather than going through enable/disable.
@pytest.fixture(autouse=True)
def _cleanup():
    from openai.resources.chat.completions import Completions, AsyncCompletions
    from anthropic.resources.messages import Messages, AsyncMessages
    snapshot = {
        (Completions, "create"):      Completions.create,
        (AsyncCompletions, "create"): AsyncCompletions.create,
        (Messages, "create"):         Messages.create,
        (AsyncMessages, "create"):    AsyncMessages.create,
    }
    yield
    auto_instrument.disable_auto_instrumentation()
    for (cls, method_name), original in snapshot.items():
        setattr(cls, method_name, original)


@pytest.fixture
def mock_collector():
    return MagicMock()


def _openai_response(model="gpt-4o", prompt_tokens=100, completion_tokens=50):
    resp = MagicMock()
    resp.model = model
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens = prompt_tokens + completion_tokens
    return resp


def _anthropic_response(model="claude-3-5-sonnet", input_tokens=80, output_tokens=40):
    resp = MagicMock()
    resp.model = model
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    return resp


# ── Plain (unwrapped) clients get instrumented ─────────────────────────────

def test_plain_openai_client_gets_instrumented(mock_collector):
    from openai import OpenAI
    from openai.resources.chat.completions import Completions

    mock_response = _openai_response()
    original = MagicMock(return_value=mock_response)
    Completions.create = original
    auto_instrument.enable_auto_instrumentation(
        savi_key="sk_test", tenant_id="ten_test", mask_pii=False, _collector=mock_collector,
    )
    client = OpenAI(api_key="test-key")
    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Hi"}],
    )
    # autouse _cleanup fixture disables and restores the real Completions.create afterward

    assert response is mock_response
    original.assert_called_once()
    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"] == "openai"
    assert event["tenant_id"] == "ten_test"
    assert event["tokens_in"] == 100
    assert event["tokens_out"] == 50


@pytest.mark.asyncio
async def test_plain_async_openai_client_gets_instrumented(mock_collector):
    from openai import AsyncOpenAI
    from openai.resources.chat.completions import AsyncCompletions

    mock_response = _openai_response()

    async def _acreate(self, *args, **kwargs):
        return mock_response

    AsyncCompletions.create = _acreate
    auto_instrument.enable_auto_instrumentation(
        savi_key="sk_test", tenant_id="ten_test", mask_pii=False, _collector=mock_collector,
    )
    client = AsyncOpenAI(api_key="test-key")
    response = await client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Hi"}],
    )

    assert response is mock_response
    mock_collector.emit.assert_called_once()
    assert mock_collector.emit.call_args[0][0]["provider"] == "openai"


def test_plain_anthropic_client_gets_instrumented(mock_collector):
    from anthropic import Anthropic
    from anthropic.resources.messages import Messages

    mock_response = _anthropic_response()
    original = MagicMock(return_value=mock_response)
    Messages.create = original

    auto_instrument.enable_auto_instrumentation(
        savi_key="sk_test", tenant_id="ten_test", mask_pii=False, _collector=mock_collector,
    )
    client = Anthropic(api_key="test-key")
    response = client.messages.create(
        model="claude-3-5-sonnet", max_tokens=100, messages=[{"role": "user", "content": "Hi"}],
    )

    assert response is mock_response
    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"] == "anthropic"
    assert event["tokens_in"] == 80
    assert event["tokens_out"] == 40


# ── finish_reason ────────────────────────────────────────────────────────

def test_openai_finish_reason_extracted_from_choices(mock_collector):
    from openai import OpenAI
    from openai.resources.chat.completions import Completions

    mock_response = _openai_response()
    mock_response.choices = [MagicMock(finish_reason="stop")]
    Completions.create = MagicMock(return_value=mock_response)

    auto_instrument.enable_auto_instrumentation(
        savi_key="sk_test", tenant_id="ten_test", mask_pii=False, _collector=mock_collector,
    )
    OpenAI(api_key="test-key").chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Hi"}],
    )

    assert mock_collector.emit.call_args[0][0]["finish_reason"] == "stop"


def test_anthropic_finish_reason_extracted_from_stop_reason(mock_collector):
    from anthropic import Anthropic
    from anthropic.resources.messages import Messages

    mock_response = _anthropic_response()
    mock_response.stop_reason = "end_turn"
    Messages.create = MagicMock(return_value=mock_response)

    auto_instrument.enable_auto_instrumentation(
        savi_key="sk_test", tenant_id="ten_test", mask_pii=False, _collector=mock_collector,
    )
    Anthropic(api_key="test-key").messages.create(
        model="claude-3-5-sonnet", max_tokens=100, messages=[{"role": "user", "content": "Hi"}],
    )

    assert mock_collector.emit.call_args[0][0]["finish_reason"] == "end_turn"


def test_openai_finish_reason_none_when_choices_empty(mock_collector):
    # isinstance-guarded fallback: an empty choices[] must not raise.
    from openai import OpenAI
    from openai.resources.chat.completions import Completions

    mock_response = _openai_response()
    mock_response.choices = []
    Completions.create = MagicMock(return_value=mock_response)

    auto_instrument.enable_auto_instrumentation(
        savi_key="sk_test", tenant_id="ten_test", mask_pii=False, _collector=mock_collector,
    )
    OpenAI(api_key="test-key").chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Hi"}],
    )

    assert mock_collector.emit.call_args[0][0]["finish_reason"] is None


# ── PII masking shapes SAVI's telemetry, never what the provider receives ──

def test_masked_content_never_reaches_the_provider(mock_collector):
    """Matches SaviOpenAI/SaviAnthropic's contract: masking only shapes
    SAVI's own telemetry signal. The provider always gets the real content
    it needs to actually answer the request - an earlier version of this
    module sent the masked version to the provider instead, silently
    degrading every auto-instrumented call's answer quality whenever PII
    was detected."""
    from openai import OpenAI
    from openai.resources.chat.completions import Completions

    captured_kwargs = {}
    original_messages = [{"role": "user", "content": "My name is John"}]

    def _create(self, *args, **kwargs):
        captured_kwargs.update(kwargs)
        return _openai_response()

    Completions.create = _create

    masker = MagicMock(spec=PiiMasker)
    masker.mask_messages.return_value = (
        [{"role": "user", "content": "Hello <REDACTED>"}], True, {"PERSON": 1},
    )
    # mask_pii=False here to skip constructing a real PiiMasker (needs
    # presidio-analyzer, not part of the [dev] test extras) - inject a
    # controllable mock directly into module state instead, same effect.
    auto_instrument.enable_auto_instrumentation(
        savi_key="sk_test", tenant_id="ten_test", mask_pii=False, _collector=mock_collector,
    )
    auto_instrument._state["masker"] = masker

    client = OpenAI(api_key="test-key")
    client.chat.completions.create(model="gpt-4o", messages=original_messages)

    # The provider call gets the REAL content, unmodified.
    assert captured_kwargs["messages"] == original_messages
    # SAVI's own telemetry still reflects the masked-content PII signal.
    event = mock_collector.emit.call_args[0][0]
    assert event["pii_flagged"] is True
    assert event["pii_types"] == {"PERSON": 1}


# ── Coexistence with SaviOpenAI/SaviAnthropic: no double-emit ──────────────

def test_savi_openai_wrapper_not_double_instrumented(mock_collector):
    """SaviOpenAI already masks+emits for its own client - auto-instrumentation
    must not additionally instrument that same instance (would double-count
    spend), even though enable_auto_instrumentation() is active process-wide."""
    from openai.resources.chat.completions import Completions
    from savi.openai import SaviOpenAI

    mock_response = _openai_response()
    Completions.create = MagicMock(return_value=mock_response)

    auto_instrument.enable_auto_instrumentation(
        savi_key="sk_test", tenant_id="ten_auto", mask_pii=False, _collector=MagicMock(),
    )

    client = SaviOpenAI(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_wrapped",
        mask_pii=False, _collector=mock_collector,
    )
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    # Exactly one emit - from SaviOpenAI's own proxy, not from the process-wide patch too.
    mock_collector.emit.assert_called_once()
    assert mock_collector.emit.call_args[0][0]["tenant_id"] == "ten_wrapped"


@pytest.mark.asyncio
async def test_savi_async_openai_wrapper_not_double_instrumented(mock_collector):
    from openai.resources.chat.completions import AsyncCompletions
    from savi.openai import SaviAsyncOpenAI

    mock_response = _openai_response()

    async def _acreate(self, *args, **kwargs):
        return mock_response

    AsyncCompletions.create = _acreate

    auto_instrument.enable_auto_instrumentation(
        savi_key="sk_test", tenant_id="ten_auto", mask_pii=False, _collector=MagicMock(),
    )

    client = SaviAsyncOpenAI(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_wrapped",
        mask_pii=False, _collector=mock_collector,
    )
    await client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    mock_collector.emit.assert_called_once()
    assert mock_collector.emit.call_args[0][0]["tenant_id"] == "ten_wrapped"


def test_savi_anthropic_wrapper_not_double_instrumented(mock_collector):
    from anthropic.resources.messages import Messages
    from savi.anthropic import SaviAnthropic

    mock_response = _anthropic_response()
    Messages.create = MagicMock(return_value=mock_response)

    auto_instrument.enable_auto_instrumentation(
        savi_key="sk_test", tenant_id="ten_auto", mask_pii=False, _collector=MagicMock(),
    )

    client = SaviAnthropic(
        api_key="test-key", savi_key="sk_test", tenant_id="ten_wrapped",
        mask_pii=False, _collector=mock_collector,
    )
    client.messages.create(model="claude-3-5-sonnet", max_tokens=100, messages=[{"role": "user", "content": "Hi"}])

    mock_collector.emit.assert_called_once()
    assert mock_collector.emit.call_args[0][0]["tenant_id"] == "ten_wrapped"


# ── Lifecycle ────────────────────────────────────────────────────────────

def test_enable_twice_warns_and_does_not_repatch(mock_collector, caplog):
    from openai.resources.chat.completions import Completions
    original = MagicMock(return_value=_openai_response())
    Completions.create = original

    auto_instrument.enable_auto_instrumentation(
        savi_key="sk_test", tenant_id="ten_1", mask_pii=False, _collector=mock_collector,
    )
    first_patched = Completions.create
    auto_instrument.enable_auto_instrumentation(
        savi_key="sk_test", tenant_id="ten_2", mask_pii=False, _collector=mock_collector,
    )
    assert Completions.create is first_patched
    assert auto_instrument._state["tenant_id"] == "ten_1"  # unchanged - second call was a no-op


def test_disable_restores_original_method():
    from openai.resources.chat.completions import Completions
    original = MagicMock(return_value=_openai_response())
    Completions.create = original

    auto_instrument.enable_auto_instrumentation(
        savi_key="sk_test", tenant_id="ten_1", mask_pii=False, _collector=MagicMock(),
    )
    assert Completions.create is not original

    auto_instrument.disable_auto_instrumentation()
    assert Completions.create is original
    assert auto_instrument._state["enabled"] is False


def test_missing_provider_package_is_not_an_error(mock_collector, monkeypatch):
    """If a provider package genuinely isn't importable, enabling
    auto-instrumentation must not raise - just skip that provider (mirrors
    savi/__init__.py's lazy-import contract for the wrapper classes)."""
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "anthropic.resources.messages":
            raise ImportError("simulated: anthropic not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    auto_instrument.enable_auto_instrumentation(
        savi_key="sk_test", tenant_id="ten_1", mask_pii=False, _collector=mock_collector,
    )
    assert auto_instrument._state["enabled"] is True


# ── local_mode (same option as every direct wrapper, see resolve_emitter) ──

def test_local_mode_requires_no_credentials():
    from savi.local import LocalEventEmitter
    auto_instrument.enable_auto_instrumentation(local_mode=True, mask_pii=False)
    assert isinstance(auto_instrument._state["collector"], LocalEventEmitter)


def test_without_credentials_and_without_local_mode_raises():
    with pytest.raises(ValueError, match="local_mode=True"):
        auto_instrument.enable_auto_instrumentation(mask_pii=False)
    assert auto_instrument._state["enabled"] is False


def test_local_mode_never_calls_httpx_post():
    from openai.resources.chat.completions import Completions
    from unittest.mock import patch

    Completions.create = MagicMock(return_value=_openai_response())
    auto_instrument.enable_auto_instrumentation(local_mode=True, mask_pii=False)

    from openai import OpenAI
    client = OpenAI(api_key="test-key")
    with patch("httpx.post") as mock_post:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])
    mock_post.assert_not_called()


def test_positional_savi_key_and_tenant_id_still_work():
    # Backward-compat guard, matching test_local_mode.py's equivalent for
    # SaviOpenAI: savi_key/tenant_id were required positional params before
    # local_mode existed, so existing positional callers must keep working.
    mock_collector = MagicMock()
    auto_instrument.enable_auto_instrumentation("sk_test", "ten_test", mask_pii=False, _collector=mock_collector)
    assert auto_instrument._state["collector"] is mock_collector
    assert auto_instrument._state["tenant_id"] == "ten_test"
