"""Opt-in content capture: off by default, bounded to the server's limits,
masked when a masker is present, never in local-mode logs,
and a failure never costs the event."""
import json
import logging
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest

import savi
from savi import content
from savi.content import ContentCapture, add_content, conversation, resolve_capture, retrieval, split_prompt, strip_content
from savi.local import LocalEventEmitter
from savi.openai import SaviOpenAI, SaviAsyncOpenAI
from savi.anthropic import SaviAnthropic

MARKER = "CONTENTMARK-7F3A"


def _openai_response(text="the answer", tool_calls=None):
    message = NS(content=text, tool_calls=tool_calls)
    return NS(model="gpt-4o", usage=NS(prompt_tokens=10, completion_tokens=5, total_tokens=15),
              choices=[NS(finish_reason="stop", message=message)])


def _client(capture, masker=False):
    collector = MagicMock()
    client = SaviOpenAI(api_key="k", savi_key="sk", tenant_id="ten", mask_pii=masker,
                        capture_content=capture, _collector=collector)
    return client, collector


def _call(client, messages, response):
    with patch.object(client._inner.chat.completions, "create", return_value=response):
        client.chat.completions.create(model="gpt-4o", messages=messages)


def test_off_by_default_no_content_field_is_sent():
    client, collector = _client(False)
    _call(client, [{"role": "user", "content": f"hello {MARKER}"}], _openai_response(f"reply {MARKER}"))
    event = collector.emit.call_args[0][0]
    assert not any(k in event for k in content.CONTENT_FIELDS)
    assert MARKER not in json.dumps(event, default=str)


def test_on_sends_the_latest_message_the_system_prompt_and_the_response_in_their_own_fields():
    client, collector = _client(True)
    _call(client, [{"role": "system", "content": "be brief"}, {"role": "user", "content": f"hello {MARKER}"}],
          _openai_response(f"reply {MARKER}"))
    event = collector.emit.call_args[0][0]
    assert event["prompt_text"] == f"hello {MARKER}"
    assert event["system_prompt"] == "be brief"
    assert event["response_text"] == f"reply {MARKER}"
    assert "history_text" not in event and "retrieved_context" not in event
    assert "tool_arguments" not in event and "conversation_id" not in event


def test_capture_parts_can_be_chosen():
    client, collector = _client(ContentCapture(prompt=False, response=True, tools=False))
    _call(client, [{"role": "user", "content": "hi"}], _openai_response("out"))
    event = collector.emit.call_args[0][0]
    assert "prompt_text" not in event and event["response_text"] == "out"


def test_an_all_false_capture_is_off():
    assert resolve_capture(ContentCapture(False, False, False, False, False, False)) is None
    assert resolve_capture(ContentCapture(False, False, False)) is not None     # the newer kinds are still on
    assert resolve_capture(None) is None and resolve_capture(False) is None
    with pytest.raises(TypeError):
        resolve_capture("yes")


def test_tool_calls_and_tool_results_are_captured():
    call = NS(function=NS(name="lookup", arguments='{"id": 7}'))
    client, collector = _client(True)
    messages = [
        {"role": "user", "content": "find it"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "tool_call_id": "c1", "content": f"row {MARKER}"},
    ]
    _call(client, messages, _openai_response(None, tool_calls=[call]))
    event = collector.emit.call_args[0][0]
    assert event["tool_arguments"] == [{"name": "lookup", "arguments": '{"id": 7}'}]
    assert event["tool_results"] == [{"tool_call_id": "c1", "content": f"row {MARKER}"}]
    assert "row" not in event["prompt_text"]      # tool output goes only in tool_results
    assert "response_text" not in event            # a tool-call-only reply has no text


def test_conversation_numbers_turns_in_order_and_is_absent_outside():
    client, collector = _client(True)
    with conversation("support-1"):
        for _ in range(3):
            _call(client, [{"role": "user", "content": "hi"}], _openai_response("ok"))
    events = [c[0][0] for c in collector.emit.call_args_list]
    assert [e["conversation_id"] for e in events] == ["support-1"] * 3
    assert [e["turn_index"] for e in events] == [0, 1, 2]
    _call(client, [{"role": "user", "content": "again"}], _openai_response("ok"))
    assert "conversation_id" not in collector.emit.call_args[0][0]


def test_conversation_without_an_id_generates_one_within_the_server_limit():
    with conversation() as conv:
        assert conv.conversation_id.startswith("conv_") and len(conv.conversation_id) <= 128
    assert len(conversation("x" * 500).conversation_id) == 128


def test_a_masker_is_applied_before_the_text_leaves():
    masker = MagicMock()
    masker.mask.side_effect = lambda t: (t.replace("jane@example.com", "<EMAIL>"), {"EMAIL": 1})
    payload = add_content({"event_id": "e"}, ContentCapture(), masker,
                          messages=[{"role": "user", "content": "mail jane@example.com"}],
                          response_text="sent to jane@example.com",
                          tool_calls=[{"name": "n", "arguments": {"to": "jane@example.com"}}])
    blob = json.dumps(payload)
    assert "jane@example.com" not in blob and "<EMAIL>" in blob


def test_oversize_content_is_bounded_so_the_metadata_event_is_never_rejected():
    big = "x" * 50_000
    payload = add_content({"event_id": "e"}, ContentCapture(), None,
                          messages=[{"role": "user", "content": big + "END"}],
                          response_text=big,
                          tool_calls=[{"name": "n", "arguments": big}] * 5,
                          tool_results=[{"content": big}])
    assert len(payload["prompt_text"]) <= 10_000 and payload["prompt_text"].endswith("END")   # the latest part is kept
    assert len(payload["response_text"]) <= 10_000
    assert len(json.dumps(payload["tool_arguments"])) <= 10_000
    assert len(json.dumps(payload["tool_results"])) <= 10_000


def test_a_failure_while_attaching_returns_the_event_unchanged():
    masker = MagicMock()
    masker.mask.side_effect = RuntimeError("detector down")
    payload = {"event_id": "e", "tokens_in": 1}
    out = add_content(payload, ContentCapture(), masker, messages=[{"role": "user", "content": "hi"}], response_text="r")
    assert out == payload


def test_local_mode_log_never_carries_content(caplog):
    emitter = LocalEventEmitter()
    with caplog.at_level(logging.INFO, logger="savi.local"):
        emitter.emit({"event_id": "e", "model": "m", "tokens_in": 1, "prompt_text": MARKER,
                      "response_text": MARKER, "tool_arguments": [MARKER], "tool_results": [MARKER],
                      "system_prompt": MARKER, "history_text": MARKER, "retrieved_context": [MARKER],
                      "conversation_id": "c", "turn_index": 0})
    assert MARKER not in caplog.text and '"event_id": "e"' in caplog.text
    assert strip_content({"prompt_text": "p", "system_prompt": "s", "history_text": "h", "retrieved_context": ["r"], "a": 1}) == {"a": 1}


def test_local_mode_client_with_capture_on_logs_no_content(caplog):
    client = SaviOpenAI(api_key="k", local_mode=True, mask_pii=False, capture_content=True)
    with caplog.at_level(logging.INFO, logger="savi.local"):
        _call(client, [{"role": "user", "content": f"secret {MARKER}"}], _openai_response(f"reply {MARKER}"))
    assert "[savi:local]" in caplog.text and MARKER not in caplog.text


def test_async_client_captures_too():
    import asyncio
    collector = MagicMock()
    client = SaviAsyncOpenAI(api_key="k", savi_key="sk", tenant_id="ten", mask_pii=False,
                             capture_content=True, _collector=collector)

    async def run():
        async def fake(**_):
            return _openai_response("async reply")
        with patch.object(client._inner.chat.completions, "create", side_effect=fake):
            await client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "q"}])
    asyncio.run(run())
    event = collector.emit.call_args[0][0]
    assert event["response_text"] == "async reply" and event["prompt_text"] == "q"


def test_anthropic_system_prompt_text_and_tool_use_are_captured():
    collector = MagicMock()
    client = SaviAnthropic(api_key="k", savi_key="sk", tenant_id="ten", mask_pii=False,
                           capture_content=True, _collector=collector)
    response = NS(model="claude", usage=NS(input_tokens=5, output_tokens=3, cache_read_input_tokens=0),
                  stop_reason="tool_use",
                  content=[NS(type="text", text="thinking"), NS(type="tool_use", name="search", input={"q": "x"})])
    messages = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "found"}]}]
    with patch.object(client._inner.messages, "create", return_value=response):
        client.messages.create(model="claude", messages=messages, max_tokens=10, system="be brief")
    event = collector.emit.call_args[0][0]
    assert event["system_prompt"] == "be brief"
    assert "prompt_text" not in event          # the only message is a tool result
    assert event["response_text"] == "thinking"
    assert event["tool_arguments"] == [{"name": "search", "arguments": {"q": "x"}}]
    assert event["tool_results"] == [{"tool_call_id": "t1", "content": "found"}]


def test_bedrock_shapes():
    response = {"output": {"message": {"content": [{"text": "hi"}, {"toolUse": {"name": "f", "input": {"a": 1}}}]}}}
    messages = [{"role": "user", "content": [{"toolResult": {"toolUseId": "u", "content": [{"text": "res"}]}}]}]
    assert content.bedrock_response_text(response) == "hi"
    assert content.bedrock_tool_calls(response) == [{"name": "f", "arguments": {"a": 1}}]
    assert content.bedrock_tool_results(messages) == [{"tool_call_id": "u", "content": "res"}]
    assert content.render_prompt([{"role": "user", "content": [{"text": "q"}]}], [{"text": "sys"}]) == "system: sys\nuser: q"


def test_cohere_and_vertex_shapes():
    response = NS(message=NS(content=[NS(type="text", text="c")],
                             tool_calls=[NS(function=NS(name="n", arguments="{}"))]))
    assert content.cohere_response_text(response) == "c"
    assert content.cohere_tool_calls(response) == [{"name": "n", "arguments": "{}"}]
    sent = MagicMock()
    holder = NS(_emit=sent, _masker=None, _content=ContentCapture())
    content.VertexContentMixin._emit_with_content(holder, "the prompt", NS(text="the reply"), {"event_id": "e"})
    event = sent.call_args[0][0]
    assert event["prompt_text"] == "the prompt" and event["response_text"] == "the reply"
    content.VertexContentMixin._emit_with_content(holder, [object()], NS(text="r"), {"event_id": "e"})
    assert "prompt_text" not in sent.call_args[0][0]


def test_auto_instrumentation_captures_when_asked_and_not_otherwise():
    from savi import auto_instrument
    from openai.resources.chat.completions import Completions
    original = Completions.create
    collector = MagicMock()
    try:
        auto_instrument.enable_auto_instrumentation(savi_key="sk", tenant_id="ten", mask_pii=False,
                                                    capture_content=True, _collector=collector)
        response = _openai_response("auto reply")
        auto_instrument._emit("openai", response, 5, "fp", False, None, [{"role": "user", "content": "auto q"}], None)
        event = collector.emit.call_args[0][0]
        assert event["prompt_text"] == "auto q" and event["response_text"] == "auto reply"
        auto_instrument.disable_auto_instrumentation()
        auto_instrument.enable_auto_instrumentation(savi_key="sk", tenant_id="ten", mask_pii=False, _collector=collector)
        auto_instrument._emit("openai", response, 5, "fp", False, None, [{"role": "user", "content": "auto q"}], None)
        assert "prompt_text" not in collector.emit.call_args[0][0]
    finally:
        auto_instrument.disable_auto_instrumentation()
        Completions.create = original


def test_the_package_exports_conversation_and_retrieval():
    assert savi.conversation is conversation and savi.ContentCapture is ContentCapture and savi.retrieval is retrieval


# --- system prompt, earlier messages, retrieved documents ----------------------

def test_split_prompt_separates_system_earlier_messages_and_the_latest_user_message():
    system, history, prompt = split_prompt([
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "tool", "tool_call_id": "t", "content": "tool output"},
        {"role": "user", "content": "second question"},
    ], system="outer rules")
    assert system == "outer rules\nrules"
    assert history == "user: first question\nassistant: first answer"
    assert prompt == "second question"
    assert "tool output" not in system + history + prompt


def test_split_prompt_copes_with_no_messages_and_with_no_user_message():
    assert split_prompt(None) == ("", "", "")
    assert split_prompt([{"role": "assistant", "content": "only me"}]) == ("", "assistant: only me", "")
    assert split_prompt([{"role": "user", "content": [{"text": "block"}]}], [{"text": "sys"}]) == ("sys", "", "block")


def test_history_is_sent_only_with_earlier_messages_and_only_when_asked_for():
    messages = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"}]
    on = add_content({"event_id": "e"}, ContentCapture(), None, messages=messages)
    assert on["prompt_text"] == "three" and on["history_text"] == "user: one\nassistant: two"
    off = add_content({"event_id": "e"}, ContentCapture(history=False), None, messages=messages)
    assert "history_text" not in off and off["prompt_text"] == "three"


def test_each_kind_can_be_left_out_on_its_own():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    payload = add_content({"event_id": "e"}, ContentCapture(prompt=False, system=True, history=False, response=False,
                                                         tools=False, retrieval=False), None, messages=messages)
    assert payload["system_prompt"] == "s" and "prompt_text" not in payload


def test_a_long_system_prompt_keeps_its_start_and_a_long_history_keeps_its_end():
    big = "x" * 50_000
    payload = add_content({"event_id": "e"}, ContentCapture(), None, system="START" + big,
                          messages=[{"role": "user", "content": big + "OLDEND"}, {"role": "assistant", "content": "a"},
                                    {"role": "user", "content": "latest"}])
    assert payload["system_prompt"].startswith("START") and len(payload["system_prompt"]) <= 10_000
    assert payload["history_text"].endswith("assistant: a") and len(payload["history_text"]) <= 10_000


def test_retrieved_documents_are_sent_only_inside_a_retrieval_block():
    docs = [{"source": "kb/refunds", "text": f"Refunds take 5 days {MARKER}"}]
    messages = [{"role": "user", "content": "q"}]
    outside = add_content({"event_id": "e"}, ContentCapture(), None, messages=messages)
    assert "retrieved_context" not in outside
    with retrieval(docs):
        inside = add_content({"event_id": "e"}, ContentCapture(), None, messages=messages)
    assert inside["retrieved_context"] == docs
    after = add_content({"event_id": "e"}, ContentCapture(), None, messages=messages)
    assert "retrieved_context" not in after


def test_the_retrieval_block_ends_even_when_the_code_inside_it_fails():
    with pytest.raises(RuntimeError):
        with retrieval(["a"]):
            raise RuntimeError("search failed")
    assert content.retrieved_documents() == []


def test_nested_retrieval_blocks_use_the_inner_documents_then_return_to_the_outer():
    with retrieval(["outer"]):
        with retrieval(["inner"]):
            assert content.retrieved_documents() == ["inner"]
        assert content.retrieved_documents() == ["outer"]


def test_a_single_document_is_accepted_without_a_list():
    with retrieval("just one"):
        assert content.retrieved_documents() == ["just one"]


def test_retrieved_documents_are_masked_and_bounded_before_they_leave():
    masker = MagicMock()
    masker.mask.side_effect = lambda t: (t.replace("jane@example.com", "<EMAIL>"), {})
    with retrieval([{"text": "write to jane@example.com"}]):
        masked = add_content({"event_id": "e"}, ContentCapture(), masker, messages=[{"role": "user", "content": "q"}])
    assert masked["retrieved_context"] == [{"text": "write to <EMAIL>"}]
    with retrieval([{"text": "x" * 6_000}, {"text": "y" * 6_000}]):
        bounded = add_content({"event_id": "e"}, ContentCapture(), None, messages=[{"role": "user", "content": "q"}])
    assert len(json.dumps(bounded["retrieved_context"])) <= 10_000


def test_retrieval_can_be_switched_off_in_the_sdk():
    with retrieval(["doc"]):
        payload = add_content({"event_id": "e"}, ContentCapture(retrieval=False), None, messages=[{"role": "user", "content": "q"}])
    assert "retrieved_context" not in payload


def test_a_client_sends_the_retrieved_documents_declared_around_the_call():
    client, collector = _client(True)
    with retrieval([{"source": "kb/a", "text": f"doc {MARKER}"}]):
        _call(client, [{"role": "user", "content": "q"}], _openai_response("ok"))
    assert collector.emit.call_args[0][0]["retrieved_context"] == [{"source": "kb/a", "text": f"doc {MARKER}"}]


def test_capture_off_sends_none_of_the_new_fields_even_inside_a_retrieval_block():
    client, collector = _client(False)
    with retrieval([MARKER]):
        _call(client, [{"role": "system", "content": MARKER}, {"role": "user", "content": MARKER}, {"role": "assistant", "content": MARKER},
                       {"role": "user", "content": MARKER}], _openai_response(MARKER))
    event = collector.emit.call_args[0][0]
    assert not any(k in event for k in content.CONTENT_FIELDS) and MARKER not in json.dumps(event, default=str)


GOLDEN = __import__("pathlib").Path(__file__).resolve().parents[3] / "backend" / "savi_api" / "tests" / "content" / "fixtures" / "sdk_content_event.json"


def _golden_event() -> dict:
    """The event the SDK really sends for a fixed call, with the volatile fields pinned."""
    call = NS(function=NS(name="send_email", arguments='{"to": "jane@example.com"}'))
    client, collector = _client(True)
    messages = [{"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi, how can I help?"},
                {"role": "user", "content": "email jane@example.com"},
                {"role": "tool", "tool_call_id": "c1", "content": "sent to jane@example.com"}]
    with conversation("golden-conv"), retrieval([{"source": "kb/contacts", "text": "Jane is at jane@example.com"}]):
        _call(client, messages, _openai_response("done, emailed jane@example.com", tool_calls=[call]))
    event = dict(collector.emit.call_args[0][0])
    event.update(event_id="evt_golden", timestamp_utc="2000-01-01T00:00:00+00:00", tenant_id="ten_golden")
    return event


def test_the_event_matches_the_golden_file_the_backend_test_ingests():
    if not GOLDEN.parent.exists():
        pytest.skip("backend tree not present (SDK checked out alone)")
    event = _golden_event()
    if not GOLDEN.exists() or __import__("os").environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert json.loads(GOLDEN.read_text(encoding="utf-8")) == json.loads(json.dumps(event, default=str))
