"""Streamed calls send one metadata-only event when the stream ends, fails or is closed (troubleshooting pack G12)."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import pytest

from savi.anthropic import SaviAnthropic, SaviAsyncAnthropic
from savi.openai import SaviAsyncOpenAI, SaviOpenAI

TEXT = "STREAMTEXT-77A"


class Sink:
    def __init__(self):
        self.events = []

    def emit(self, payload):
        self.events.append(payload)


def _oa_chunk(text=None, finish=None, usage=None, model="gpt-4o"):
    choices = [NS(delta=NS(content=text), finish_reason=finish)] if (text is not None or finish) else []
    return NS(model=model, choices=choices, usage=usage)


def _oa_usage():
    return NS(prompt_tokens=12, completion_tokens=7, total_tokens=19, prompt_tokens_details=NS(cached_tokens=4))


def _openai(sink):
    return SaviOpenAI(api_key="k", savi_key="sk", tenant_id="ten", mask_pii=False, _collector=sink)


def _stream_call(client, chunks):
    with patch.object(client._inner.chat.completions, "create", return_value=iter(chunks)):
        return client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)


def test_a_completed_openai_stream_sends_one_event_with_the_reported_usage():
    sink = Sink()
    chunks = [_oa_chunk(TEXT), _oa_chunk(TEXT), _oa_chunk(finish="stop"), _oa_chunk(usage=_oa_usage())]
    got = list(_stream_call(_openai(sink), chunks))
    assert got == chunks and all(a is b for a, b in zip(got, chunks))          # the caller sees the same chunks
    assert len(sink.events) == 1
    e = sink.events[0]
    assert (e["tokens_in"], e["tokens_out"], e["tokens_cached"], e["finish_reason"]) == (12, 7, 4, "stop")
    assert e["provider"] == "openai" and "is_error" not in e and TEXT not in str(e)


def test_without_reported_usage_output_is_estimated_from_text_chunks_and_input_is_zero():
    sink = Sink()
    list(_stream_call(_openai(sink), [_oa_chunk(TEXT), _oa_chunk(TEXT), _oa_chunk(TEXT), _oa_chunk(finish="stop")]))
    e = sink.events[0]
    assert (e["tokens_in"], e["tokens_out"]) == (0, 3) and e["finish_reason"] == "stop"


def test_a_stream_that_fails_midway_sends_an_error_event_and_the_error_still_reaches_the_caller():
    sink = Sink()

    def gen():
        yield _oa_chunk(TEXT)
        raise ConnectionError(f"reset while streaming")
    with pytest.raises(ConnectionError):
        list(_stream_call(_openai(sink), gen()))
    assert len(sink.events) == 1
    e = sink.events[0]
    assert e["is_error"] is True and e["error_code"] == "ConnectionError" and e["tokens_out"] == 1


def test_a_stream_closed_early_is_reported_once_as_abandoned():
    sink = Sink()
    stream = _stream_call(_openai(sink), _Closable([_oa_chunk(TEXT), _oa_chunk(TEXT)]))
    next(stream)
    stream.close()
    stream.close()
    assert len(sink.events) == 1 and sink.events[0]["finish_reason"] == "stream_abandoned"


class _Closable:
    def __init__(self, chunks):
        self._it = iter(chunks)
        self.closed = False
        self.response = "http-response-object"

    def __iter__(self): return self
    def __next__(self): return next(self._it)
    def close(self): self.closed = True
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def test_leaving_a_with_block_early_reports_abandoned_and_other_attributes_pass_through():
    sink = Sink()
    stream = _stream_call(_openai(sink), _Closable([_oa_chunk(TEXT), _oa_chunk(TEXT)]))
    with stream as s:
        next(s)
        assert s.response == "http-response-object"
    assert len(sink.events) == 1 and sink.events[0]["finish_reason"] == "stream_abandoned"


def test_an_emit_failure_never_reaches_the_caller():
    class Broken:
        def emit(self, payload):
            raise RuntimeError("collector down")
    client = SaviOpenAI(api_key="k", savi_key="sk", tenant_id="ten", mask_pii=False, _collector=Broken())
    assert len(list(_stream_call(client, [_oa_chunk(TEXT), _oa_chunk(finish="stop")]))) == 2


# ── Anthropic ─────────────────────────────────────────────────────────────────

def _an_events(text=TEXT):
    return [
        NS(type="message_start", message=NS(model="claude-sonnet-4-6",
                                            usage=NS(input_tokens=20, output_tokens=1, cache_read_input_tokens=5))),
        NS(type="content_block_delta", delta=NS(text=text)),
        NS(type="content_block_delta", delta=NS(text=text)),
        NS(type="message_delta", delta=NS(stop_reason="end_turn"), usage=NS(output_tokens=33)),
        NS(type="message_stop"),
    ]


def test_a_completed_anthropic_stream_reports_input_output_cache_and_stop_reason():
    sink = Sink()
    client = SaviAnthropic(api_key="k", savi_key="sk", tenant_id="ten", mask_pii=False, _collector=sink)
    events = _an_events()
    with patch.object(client._inner.messages, "create", return_value=iter(events)):
        got = list(client.messages.create(model="claude-sonnet-4-6", max_tokens=50, stream=True,
                                          messages=[{"role": "user", "content": "hi"}]))
    assert got == events and len(sink.events) == 1
    e = sink.events[0]
    assert (e["provider"], e["tokens_in"], e["tokens_out"], e["tokens_cached"], e["finish_reason"]) == \
           ("anthropic", 20, 33, 5, "end_turn")
    assert TEXT not in str(e)


# ── Async ─────────────────────────────────────────────────────────────────────

async def _agen(items, fail_after=None):
    for i, x in enumerate(items):
        if fail_after is not None and i == fail_after:
            raise TimeoutError("stalled")
        yield x


def test_async_openai_completed_and_failed_streams():
    sink = Sink()
    client = SaviAsyncOpenAI(api_key="k", savi_key="sk", tenant_id="ten", mask_pii=False, _collector=sink)

    async def run(chunks, fail_after=None):
        with patch.object(client._inner.chat.completions, "create", new=AsyncMock(return_value=_agen(chunks, fail_after))):
            stream = await client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
            return [c async for c in stream]

    got = asyncio.run(run([_oa_chunk(TEXT), _oa_chunk(finish="stop"), _oa_chunk(usage=_oa_usage())]))
    assert len(got) == 3 and len(sink.events) == 1 and sink.events[0]["tokens_in"] == 12
    with pytest.raises(TimeoutError):
        asyncio.run(run([_oa_chunk(TEXT), _oa_chunk(TEXT)], fail_after=1))
    assert len(sink.events) == 2 and sink.events[1]["is_error"] is True and sink.events[1]["error_code"] == "TimeoutError"


def test_async_anthropic_completed_stream():
    sink = Sink()
    client = SaviAsyncAnthropic(api_key="k", savi_key="sk", tenant_id="ten", mask_pii=False, _collector=sink)

    async def run():
        with patch.object(client._inner.messages, "create", new=AsyncMock(return_value=_agen(_an_events()))):
            stream = await client.messages.create(model="claude-sonnet-4-6", max_tokens=50, stream=True,
                                                  messages=[{"role": "user", "content": "hi"}])
            return [e async for e in stream]

    assert len(asyncio.run(run())) == 5 and len(sink.events) == 1 and sink.events[0]["tokens_out"] == 33


# ── The other providers ───────────────────────────────────────────────────────

def test_azure_openai_stream_reports_usage_with_provider_azure():
    from savi.azure_openai import SaviAzureOpenAI
    sink = Sink()
    client = SaviAzureOpenAI(azure_endpoint="https://x.openai.azure.com/", api_key="k", api_version="2024-02-01",
                             savi_key="sk", tenant_id="ten", mask_pii=False, _collector=sink)
    chunks = [_oa_chunk(TEXT), _oa_chunk(finish="stop"), _oa_chunk(usage=_oa_usage())]
    with patch.object(client._inner.chat.completions, "create", return_value=iter(chunks)):
        got = list(client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True))
    assert len(got) == 3 and len(sink.events) == 1
    e = sink.events[0]
    assert (e["provider"], e["tokens_in"], e["tokens_out"], e["finish_reason"]) == ("azure", 12, 7, "stop")
    assert TEXT not in str(e)


def test_mistral_stream_unwraps_events_and_reports_usage():
    from savi.mistral import SaviMistral
    from unittest.mock import MagicMock
    sink = Sink()
    inner = MagicMock()
    events = [NS(data=_oa_chunk(TEXT)), NS(data=_oa_chunk(finish="stop")),
              NS(data=_oa_chunk(usage=NS(prompt_tokens=30, completion_tokens=9, total_tokens=39, prompt_tokens_details=None)))]
    inner.chat.complete.return_value = iter(events)
    client = SaviMistral(api_key="k", savi_key="sk", tenant_id="ten", mask_pii=False, _collector=sink, _client=inner)
    got = list(client.chat.complete(model="mistral-large-latest", messages=[{"role": "user", "content": "hi"}], stream=True))
    assert got == events and len(sink.events) == 1
    e = sink.events[0]
    assert (e["provider"], e["tokens_in"], e["tokens_out"], e["finish_reason"]) == ("mistral", 30, 9, "stop")


def test_cohere_stream_reads_the_message_end_usage():
    from savi.cohere import SaviCohere
    from unittest.mock import MagicMock
    sink = Sink()
    inner = MagicMock()
    events = [NS(type="content-delta", delta=NS(message=NS(content=NS(text=TEXT)))),
              NS(type="content-delta", delta=NS(message=NS(content=NS(text=TEXT)))),
              NS(type="message-end", delta=NS(finish_reason="COMPLETE", usage=NS(tokens=NS(input_tokens=44, output_tokens=12))))]
    inner.chat.return_value = iter(events)
    client = SaviCohere(api_key="k", savi_key="sk", tenant_id="ten", mask_pii=False, _collector=sink, _client=inner)
    got = list(client.chat(model="command-r-plus", messages=[{"role": "user", "content": "hi"}], stream=True))
    assert got == events and len(sink.events) == 1
    e = sink.events[0]
    assert (e["provider"], e["tokens_in"], e["tokens_out"], e["finish_reason"]) == ("cohere", 44, 12, "complete")
    assert TEXT not in str(e)


def test_vertex_stream_reads_usage_metadata_and_finish_reason():
    from savi.google_vertex import _WrappedModel
    sink = Sink()

    def chunk(text=None, usage=None, reason=None):
        parts = [NS(text=text)] if text else []
        cand = NS(content=NS(parts=parts), finish_reason=NS(name=reason) if reason else None)
        return NS(candidates=[cand], usage_metadata=usage)

    chunks = [chunk(TEXT), chunk(TEXT), chunk(reason="STOP", usage=NS(prompt_token_count=21, candidates_token_count=8,
                                                                       cached_content_token_count=0))]
    from unittest.mock import MagicMock
    inner = MagicMock()
    inner.generate_content.return_value = iter(chunks)
    model = _WrappedModel(inner, "gemini-1.5-pro", sink.emit, "ten", "eng", masker=None)
    got = list(model.generate_content("hello", stream=True))
    assert got == chunks and len(sink.events) == 1
    e = sink.events[0]
    assert (e["provider"], e["tokens_in"], e["tokens_out"], e["finish_reason"]) == ("google", 21, 8, "stop")
    assert TEXT not in str(e)


def test_bedrock_converse_stream_wraps_the_event_stream_and_reports_usage():
    from savi.bedrock import SaviBedrockRuntime
    from unittest.mock import MagicMock
    sink = Sink()
    events = [{"contentBlockDelta": {"delta": {"text": TEXT}}}, {"contentBlockDelta": {"delta": {"text": TEXT}}},
              {"messageStop": {"stopReason": "end_turn"}},
              {"metadata": {"usage": {"inputTokens": 50, "outputTokens": 25, "cacheReadInputTokens": 10}}}]
    client = MagicMock()
    client.converse_stream.return_value = {"ResponseMetadata": {"x": 1}, "stream": iter(events)}
    runtime = SaviBedrockRuntime(savi_key="sk", tenant_id="ten", team_id="eng", mask_pii=False, _collector=sink,
                                 _client=client)
    response = runtime.converse_stream(model_id="anthropic.claude-3", messages=[{"role": "user", "content": [{"text": "hi"}]}])
    assert response["ResponseMetadata"] == {"x": 1}
    assert list(response["stream"]) == events and len(sink.events) == 1
    e = sink.events[0]
    assert (e["provider"], e["tokens_in"], e["tokens_out"], e["tokens_cached"], e["finish_reason"]) == \
           ("bedrock", 50, 25, 10, "end_turn")
    assert TEXT not in str(e)
