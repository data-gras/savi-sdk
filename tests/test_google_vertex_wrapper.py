"""Unit tests for savi.google_vertex's _WrappedModel (backlog item 26 —
response-caching extension to the Google Vertex provider).

No test file existed for this module before: SaviVertexAI.__init__()
unconditionally does `import vertexai; vertexai.init(...)` with no `_client=`
bypass the way the other providers' wrappers have, and `vertexai` isn't in
this SDK's [dev] extras — so it's exercised here by constructing
_WrappedModel directly (it takes `inner` as a plain constructor arg), never
importing the real vertexai package at all.
"""
import pytest
from unittest.mock import MagicMock
from savi.google_vertex import _WrappedModel

_MODEL = "gemini-1.5-pro-002"


def _make_response(prompt=100, candidates=50):
    resp = MagicMock()
    resp.usage_metadata.prompt_token_count         = prompt
    resp.usage_metadata.candidates_token_count     = candidates
    resp.usage_metadata.cached_content_token_count = 0
    return resp


def _wrapped(inner, emit_fn, enable_cache=False, workload_type=None):
    from savi.cache import ResponseCache
    cache = ResponseCache() if enable_cache else None
    return _WrappedModel(inner, _MODEL, emit_fn, "ten_test", "engineering", masker=None,
                          workload_type=workload_type, cache=cache)


def test_emits_event_on_generate_content():
    inner = MagicMock()
    inner.generate_content.return_value = _make_response()
    emit = MagicMock()
    model = _wrapped(inner, emit)

    model.generate_content("Summarise this contract.")

    emit.assert_called_once()
    event = emit.call_args[0][0]
    assert event["provider"]   == "google"
    assert event["model"]      == _MODEL
    assert event["tokens_in"]  == 100
    assert event["tokens_out"] == 50


def test_timestamp_utc_present():
    # Required (non-nullable) by LLMEventIn - was missing entirely, so
    # every event this wrapper emitted 422'd silently against the ingest
    # endpoint.
    inner = MagicMock()
    inner.generate_content.return_value = _make_response()
    emit = MagicMock()
    model = _wrapped(inner, emit)

    model.generate_content("Summarise this contract.")

    event = emit.call_args[0][0]
    assert "timestamp_utc" in event
    assert event["timestamp_utc"] is not None


def test_workload_type_included_when_set():
    inner = MagicMock()
    inner.generate_content.return_value = _make_response()
    emit = MagicMock()
    model = _wrapped(inner, emit, workload_type="summarization")

    model.generate_content("Summarise this contract.")

    assert emit.call_args[0][0]["workload_type"] == "summarization"


def test_workload_type_omitted_when_not_set():
    inner = MagicMock()
    inner.generate_content.return_value = _make_response()
    emit = MagicMock()
    model = _wrapped(inner, emit)

    model.generate_content("Summarise this contract.")

    assert "workload_type" not in emit.call_args[0][0]


def test_emit_failure_does_not_break_the_call():
    inner = MagicMock()
    inner.generate_content.return_value = _make_response()
    emit = MagicMock(side_effect=RuntimeError("collector exploded"))
    model = _wrapped(inner, emit)

    response = model.generate_content("Summarise this contract.")
    assert response is not None  # did not raise


def test_cache_disabled_by_default():
    inner = MagicMock()
    inner.generate_content.return_value = _make_response()
    emit = MagicMock()
    model = _wrapped(inner, emit)

    model.generate_content("same prompt")
    model.generate_content("same prompt")
    assert inner.generate_content.call_count == 2


def test_cache_hit_avoids_second_provider_call():
    inner = MagicMock()
    inner.generate_content.return_value = _make_response()
    emit = MagicMock()
    model = _wrapped(inner, emit, enable_cache=True)

    r1 = model.generate_content("same prompt")
    r2 = model.generate_content("same prompt")
    assert inner.generate_content.call_count == 1
    assert r2 is r1
    events = [c[0][0] for c in emit.call_args_list]
    assert events[0]["is_cache_hit"] is False
    assert events[1]["is_cache_hit"] is True


def test_cache_miss_for_different_prompt():
    inner = MagicMock()
    inner.generate_content.return_value = _make_response()
    emit = MagicMock()
    model = _wrapped(inner, emit, enable_cache=True)

    model.generate_content("prompt one")
    model.generate_content("a totally different prompt")
    assert inner.generate_content.call_count == 2


def test_cache_bypassed_for_streaming_calls():
    inner = MagicMock()
    inner.generate_content.return_value = _make_response()
    emit = MagicMock()
    model = _wrapped(inner, emit, enable_cache=True)

    model.generate_content("same", stream=True)
    model.generate_content("same", stream=True)
    assert inner.generate_content.call_count == 2


@pytest.mark.asyncio
async def test_generate_content_async_inherits_caching():
    """generate_content_async() delegates to generate_content() via a thread
    executor - it must inherit the same caching behaviour without any
    separate wiring."""
    inner = MagicMock()
    inner.generate_content.return_value = _make_response()
    emit = MagicMock()
    model = _wrapped(inner, emit, enable_cache=True)

    r1 = await model.generate_content_async("same prompt")
    r2 = await model.generate_content_async("same prompt")
    assert inner.generate_content.call_count == 1
    assert r2 is r1


def test_finish_reason_captured_from_candidate_enum():
    # Vertex reports finish_reason as an enum-like object with a .name
    # attribute (e.g. FinishReason.MAX_TOKENS) — lowercased to match every
    # other provider's vocabulary.
    inner = MagicMock()
    response = _make_response()
    finish_reason_enum = MagicMock()
    finish_reason_enum.name = "MAX_TOKENS"
    response.candidates = [MagicMock(finish_reason=finish_reason_enum)]
    inner.generate_content.return_value = response
    emit = MagicMock()
    model = _wrapped(inner, emit)

    model.generate_content("Summarise this contract.")

    event = emit.call_args[0][0]
    assert event["finish_reason"] == "max_tokens"


def test_total_tokens_prefers_vertex_reported_total_over_the_sum():
    # Vertex's own total_token_count can legitimately differ from
    # prompt+candidates (e.g. tool-call/thinking token overhead) - matches
    # every other provider wrapper's convention of trusting the API's own
    # total field (e.g. bedrock.py's usage["totalTokens"]) rather than
    # recomputing it.
    inner = MagicMock()
    response = _make_response(prompt=100, candidates=50)
    response.usage_metadata.total_token_count = 200
    inner.generate_content.return_value = response
    emit = MagicMock()
    model = _wrapped(inner, emit)

    model.generate_content("Summarise this contract.")

    event = emit.call_args[0][0]
    assert event["total_tokens"] == 200


def test_total_tokens_falls_back_to_sum_when_total_token_count_absent():
    inner = MagicMock()
    response = MagicMock()
    response.usage_metadata = MagicMock(
        spec=["prompt_token_count", "candidates_token_count", "cached_content_token_count"]
    )
    response.usage_metadata.prompt_token_count = 100
    response.usage_metadata.candidates_token_count = 50
    response.usage_metadata.cached_content_token_count = 0
    inner.generate_content.return_value = response
    emit = MagicMock()
    model = _wrapped(inner, emit)

    model.generate_content("Summarise this contract.")

    event = emit.call_args[0][0]
    assert event["total_tokens"] == 150


def test_finish_reason_none_when_no_candidates():
    inner = MagicMock()
    response = _make_response()
    response.candidates = []
    inner.generate_content.return_value = response
    emit = MagicMock()
    model = _wrapped(inner, emit)

    model.generate_content("Summarise this contract.")

    event = emit.call_args[0][0]
    assert event["finish_reason"] is None
