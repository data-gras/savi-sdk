"""Wire-level check of what the Python SDK sends. Nothing is mocked above the network layer: a real emitter builds
and serialises the HTTP request, and the bytes are captured where the HTTP transport would send them. Each test
plants a marker in the prompt, the response and the tool arguments, then looks for it in every byte sent."""
import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import httpx
import pytest

from savi.anthropic import SaviAnthropic
from savi.collector import AsyncEventEmitter
from savi.openai import SaviOpenAI

MARKER = "WIREMARK-91C2"
CONTENT = ("prompt_text", "response_text", "tool_arguments", "tool_results", "system_prompt", "history_text", "retrieved_context")


class Wire:
    def __init__(self, status=200):
        self.requests: list[httpx.Request] = []
        self.status = status

    def handle(self, transport, request):
        self.requests.append(request)
        return httpx.Response(self.status, request=request)

    @property
    def bodies(self) -> list[str]:
        return [r.content.decode() for r in self.requests]


@pytest.fixture
def wire():
    w = Wire()
    with patch.object(httpx.HTTPTransport, "handle_request", lambda self, req: w.handle(self, req)):
        yield w


def _emitter():
    return AsyncEventEmitter("https://savi.invalid", "sk_test", flush_interval_secs=3600)


def _openai(emitter, **kw):
    return SaviOpenAI(api_key="k", savi_key="sk", tenant_id="ten", mask_pii=False, _collector=emitter, **kw)


def _openai_response():
    tool = NS(function=NS(name="lookup", arguments=json.dumps({"q": MARKER})))
    return NS(model="gpt-4o", usage=NS(prompt_tokens=10, completion_tokens=5, total_tokens=15),
              choices=[NS(finish_reason="stop", message=NS(content=f"reply {MARKER}", tool_calls=[tool]))])


def _call(client, response, **kw):
    with patch.object(client._inner.chat.completions, "create", return_value=response):
        return client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": f"hello {MARKER}"}], **kw)


def test_default_call_sends_a_request_and_no_text(wire):
    emitter = _emitter()
    _call(_openai(emitter), _openai_response())
    emitter.flush()
    assert len(wire.requests) == 1 and wire.requests[0].url.path == "/v1/events/ingest"
    assert MARKER not in wire.bodies[0]
    event = json.loads(wire.bodies[0])["events"][0]
    assert not set(CONTENT) & set(event)


def test_a_streamed_call_sends_one_event_with_no_text(wire):
    emitter = _emitter()
    chunks = [NS(model="gpt-4o", usage=None, choices=[NS(delta=NS(content=f"part {MARKER}"), finish_reason=None)]),
              NS(model="gpt-4o", usage=None, choices=[NS(delta=NS(content=None), finish_reason="stop")]),
              NS(model="gpt-4o", choices=[], usage=NS(prompt_tokens=10, completion_tokens=5, total_tokens=15))]
    list(_call(_openai(emitter, capture_content=True), iter(chunks), stream=True))
    emitter.flush()
    assert len(wire.requests) == 1 and MARKER not in wire.bodies[0]
    event = json.loads(wire.bodies[0])["events"][0]
    assert event["tokens_in"] == 10 and event["finish_reason"] == "stop"
    assert not set(CONTENT) & set(event)


def test_capture_on_puts_the_text_only_in_the_content_fields(wire):
    emitter = _emitter()
    _call(_openai(emitter, capture_content=True), _openai_response())
    emitter.flush()
    event = json.loads(wire.bodies[0])["events"][0]
    carrying = {k for k, v in event.items() if MARKER in json.dumps(v)}
    assert carrying <= set(CONTENT) and carrying


def test_a_rejected_batch_that_is_retried_carries_no_text_either():
    w = Wire(status=503)
    emitter = _emitter()
    with patch.object(httpx.HTTPTransport, "handle_request", lambda self, req: w.handle(self, req)):
        _call(_openai(emitter), _openai_response())
        emitter.flush()
        emitter.flush()
    assert len(w.requests) >= 2 and all(MARKER not in b for b in w.bodies)


def test_anthropic_default_and_streamed(wire):
    emitter = _emitter()
    client = SaviAnthropic(api_key="k", savi_key="sk", tenant_id="ten", mask_pii=False, _collector=emitter)
    response = NS(model="claude-sonnet-4-6", stop_reason="end_turn",
                  usage=NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=0),
                  content=[NS(type="text", text=f"reply {MARKER}")])
    with patch.object(client._inner.messages, "create", return_value=response):
        client.messages.create(model="claude-sonnet-4-6", max_tokens=10,
                               messages=[{"role": "user", "content": f"hello {MARKER}"}])
    emitter.flush()
    assert len(wire.requests) == 1 and MARKER not in wire.bodies[0]
    events = [NS(type="message_start", message=NS(model="claude-sonnet-4-6", usage=NS(input_tokens=9, output_tokens=1, cache_read_input_tokens=0))),
              NS(type="content_block_delta", delta=NS(text=f"part {MARKER}")),
              NS(type="message_delta", delta=NS(stop_reason="end_turn"), usage=NS(output_tokens=4))]
    with patch.object(client._inner.messages, "create", return_value=iter(events)):
        list(client.messages.create(model="claude-sonnet-4-6", max_tokens=10, stream=True,
                                    messages=[{"role": "user", "content": f"hello {MARKER}"}]))
    emitter.flush()
    assert len(wire.requests) == 2 and MARKER not in wire.bodies[1]


def test_a_provider_error_message_is_sent_as_is_and_can_echo_the_prompt(wire):
    """Known gap, recorded as a test so it cannot change unnoticed: on a failed call the provider's own error
    message (first 512 characters) is part of the event. A provider that echoes prompt text in an error would put
    that text on the wire even with capture off."""
    emitter = _emitter()
    client = _openai(emitter)
    with patch.object(client._inner.chat.completions, "create", side_effect=RuntimeError(f"rejected: {MARKER}")):
        with pytest.raises(RuntimeError):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    emitter.flush()
    assert MARKER in wire.bodies[0]
