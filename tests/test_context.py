import pytest
from savi.context import SpanContext, get_current_span_id, get_current_context

def test_span_context_enters_and_exits():
    assert get_current_span_id() is None
    with SpanContext(workflow_id="test-wf") as span:
        assert get_current_span_id() == span.span_id
    assert get_current_span_id() is None

def test_nested_spans_inherit_parent():
    with SpanContext(workflow_id="parent-wf") as parent:
        with SpanContext(workflow_id="child-wf") as child:
            assert child.parent_id == parent.span_id
        assert get_current_span_id() == parent.span_id  # restored after child exits

def test_span_id_format():
    with SpanContext(workflow_id="test-wf") as span:
        assert span.span_id.startswith("span_")
        assert len(span.span_id) > 10

def test_agent_id_stored_and_retrievable():
    with SpanContext(workflow_id="wf", agent_id="agent-001") as span:
        ctx = get_current_context()
        assert ctx is not None
        assert ctx.agent_id == "agent-001"
    assert get_current_context() is None


def test_run_id_auto_generated_at_root():
    """Root span with no explicit run_id auto-generates one."""
    with SpanContext(workflow_id="wf") as span:
        assert span.run_id is not None
        assert span.run_id.startswith("run_")

def test_run_id_inherited_by_child():
    """Child span inherits parent's run_id."""
    with SpanContext(workflow_id="wf") as parent:
        with SpanContext(workflow_id="wf-child") as child:
            assert child.run_id == parent.run_id

def test_run_id_explicit_overrides():
    """Explicit run_id is preserved, not overridden."""
    with SpanContext(workflow_id="wf", run_id="run_custom_123") as span:
        assert span.run_id == "run_custom_123"

def test_attribution_fields_returns_dict():
    """attribution_fields() returns all five keys with correct values.

    parent_span_id is the active span's own span_id — it is the span under
    which any LLM call is made, so the event is a child of that span.
    """
    with SpanContext(workflow_id="my-wf", agent_id="agent-1", user_id="user-42") as span:
        from savi.context import attribution_fields
        fields = attribution_fields()
        assert fields["workflow_id"] == "my-wf"
        assert fields["agent_id"] == "agent-1"
        assert fields["user_id"] == "user-42"
        assert fields["parent_span_id"] == span.span_id
        assert fields["agent_run_id"] == span.run_id

def test_user_id_defaults_to_none():
    """user_id is optional — omitting it leaves attribution unset, same as agent_id."""
    with SpanContext(workflow_id="wf") as span:
        assert span.user_id is None

def test_attribution_fields_outside_context_returns_nones():
    """Module-level attribution_fields() returns all-None when no active span."""
    from savi.context import attribution_fields
    fields = attribution_fields()
    assert fields == {
        "workflow_id": None, "agent_id": None, "user_id": None,
        "parent_span_id": None, "agent_run_id": None,
    }

def test_start_run_creates_named_run():
    """start_run() convenience function creates a SpanContext with auto-generated run_id."""
    from savi.context import start_run
    with start_run("my-workflow", agent_id="agent-1") as span:
        assert span.run_id.startswith("run_")
        assert span.workflow_id == "my-workflow"
        assert span.agent_id == "agent-1"

def test_start_run_explicit_run_id():
    """start_run() accepts an explicit run_id."""
    from savi.context import start_run
    with start_run("wf", run_id="run_explicit_abc") as span:
        assert span.run_id == "run_explicit_abc"

def test_start_run_accepts_user_id():
    """start_run() passes user_id through to the created SpanContext."""
    from savi.context import start_run
    with start_run("my-workflow", user_id="user-42") as span:
        assert span.user_id == "user-42"


# ── Cross-process propagation (to_headers/from_headers, to_env/from_env) ────

def test_to_headers_and_from_headers_round_trip_the_same_run():
    """The generic hand-off case (a raw HTTP call between a customer's own
    services, not going through MCP) — mcp_instrument.py already closes this
    for the MCP-tool-call channel specifically; this is the SDK-level
    primitive for everything else."""
    with SpanContext(workflow_id="kyc-review", agent_id="intake", user_id="user-1") as sender:
        headers = sender.to_headers()

    resumed = SpanContext.from_headers(headers)
    assert resumed is not None
    assert resumed.run_id == sender.run_id, "the whole point: one run across the hop, not two"
    assert resumed.workflow_id == "kyc-review"
    assert resumed.agent_id == "intake"
    assert resumed.user_id == "user-1"
    assert resumed.parent_id == sender.span_id, "the sender's span is this hop's parent"


def test_to_headers_lookup_is_case_insensitive():
    """Real HTTP frameworks normalize header casing inconsistently with each
    other — a receiver using a plain, non-normalizing dict must still find
    a lowercased header."""
    with SpanContext(workflow_id="wf") as sender:
        headers = sender.to_headers()
    lowercased = {k.lower(): v for k, v in headers.items()}
    resumed = SpanContext.from_headers(lowercased)
    assert resumed is not None
    assert resumed.run_id == sender.run_id


def test_from_headers_finds_an_arbitrarily_cased_header_via_full_scan():
    """Neither the canonical name nor the all-lowercase fallback matches a
    caller-built plain dict using some other casing (e.g. copied verbatim
    from a raw socket read) - falls back to a full case-insensitive scan
    rather than missing the header entirely."""
    with SpanContext(workflow_id="wf") as sender:
        headers = sender.to_headers()
    oddly_cased = {"x-SAVI-trace-CONTEXT": headers["X-Savi-Trace-Context"]}
    resumed = SpanContext.from_headers(oddly_cased)
    assert resumed is not None
    assert resumed.run_id == sender.run_id


def test_from_headers_with_no_trace_header_returns_none():
    """Same shape as attribution_fields() with no active context — a missing
    header is 'no context to resume', not an error."""
    assert SpanContext.from_headers({}) is None


def test_from_headers_with_malformed_json_returns_none_not_raise():
    """A corrupted or truncated header must never crash the receiving
    process's own real work — same never-raise contract as every other
    best-effort path in this SDK (mcp_instrument.py, auto_instrument.py)."""
    from savi.context import TRACE_HEADER
    assert SpanContext.from_headers({TRACE_HEADER: "{not valid json"}) is None


def test_to_env_and_from_env_round_trip_the_same_run():
    """The subprocess hand-off case — merge to_env()'s dict into what you
    pass to subprocess.Popen/run; the child resumes via from_env()."""
    with SpanContext(workflow_id="batch-job", agent_id="worker") as sender:
        env = sender.to_env()

    resumed = SpanContext.from_env(env)
    assert resumed is not None
    assert resumed.run_id == sender.run_id
    assert resumed.agent_id == "worker"
    assert resumed.parent_id == sender.span_id


def test_from_env_defaults_to_os_environ(monkeypatch):
    """No explicit env= means read the process's real environment — the
    actual subprocess use case, not just the explicit-dict test convenience."""
    from savi.context import TRACE_ENV

    with SpanContext(workflow_id="wf") as sender:
        payload = sender.to_env()[TRACE_ENV]
    monkeypatch.setenv(TRACE_ENV, payload)

    resumed = SpanContext.from_env()
    assert resumed is not None
    assert resumed.run_id == sender.run_id


def test_from_env_with_nothing_set_returns_none():
    assert SpanContext.from_env(env={}) is None


def test_resumed_context_correlates_a_third_hop():
    """Three-hop chain (A -> B -> C), matching a real multi-agent hand-off
    depth, not just a single A -> B pair — every hop shares one run_id."""
    with SpanContext(workflow_id="wf-a", agent_id="a") as span_a:
        headers_a = span_a.to_headers()

    span_b = SpanContext.from_headers(headers_a)
    with span_b:
        headers_b = span_b.to_headers()

    span_c = SpanContext.from_headers(headers_b)
    assert span_c.run_id == span_a.run_id
    assert span_c.parent_id == span_b.span_id
    assert span_b.parent_id == span_a.span_id
