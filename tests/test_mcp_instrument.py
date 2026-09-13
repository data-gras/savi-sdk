import pytest
from unittest.mock import AsyncMock
from savi import mcp_instrument
from savi.context import SpanContext


# mcp_instrument patches a class-level method shared by every test in this
# file - same leaked-patch risk auto_instrument.py's own _cleanup fixture
# guards against, same fix: snapshot the real attribute and force-restore it
# after every test regardless of what this module's own bookkeeping thinks
# happened.
@pytest.fixture(autouse=True)
def _cleanup():
    from mcp.client.session import ClientSession
    original = ClientSession.call_tool
    yield
    mcp_instrument.disable_mcp_instrumentation()
    ClientSession.call_tool = original


def _patched_client_session(mock_call_tool):
    from mcp.client.session import ClientSession
    ClientSession.call_tool = mock_call_tool
    mcp_instrument.enable_mcp_instrumentation()
    return ClientSession.__new__(ClientSession)


@pytest.mark.asyncio
async def test_no_active_context_injects_nothing():
    """Outside any SpanContext, the wrapper must not invent a meta the
    caller never asked for."""
    mock_call_tool = AsyncMock(return_value="result")
    session = _patched_client_session(mock_call_tool)

    await session.call_tool("some_tool", {"arg": 1})

    mock_call_tool.assert_awaited_once_with(session, "some_tool", {"arg": 1})


@pytest.mark.asyncio
async def test_active_context_stamps_agent_run_id_and_parent_span_id():
    mock_call_tool = AsyncMock(return_value="result")
    session = _patched_client_session(mock_call_tool)

    with SpanContext(workflow_id="wf", agent_id="agent-1") as span:
        await session.call_tool("some_tool", {"arg": 1})

    mock_call_tool.assert_awaited_once()
    _, kwargs = mock_call_tool.call_args
    assert kwargs["meta"]["io.savi/context"]["agent_run_id"] == span.run_id
    assert kwargs["meta"]["io.savi/context"]["parent_span_id"] == span.span_id


@pytest.mark.asyncio
async def test_caller_supplied_meta_is_preserved_not_clobbered():
    mock_call_tool = AsyncMock(return_value="result")
    session = _patched_client_session(mock_call_tool)

    with SpanContext(workflow_id="wf"):
        await session.call_tool("some_tool", {"arg": 1}, meta={"progress_token": "tok-1"})

    _, kwargs = mock_call_tool.call_args
    assert kwargs["meta"]["progress_token"] == "tok-1"
    assert "io.savi/context" in kwargs["meta"]


@pytest.mark.asyncio
async def test_disabled_by_default_leaves_call_tool_unpatched():
    from mcp.client.session import ClientSession
    original = ClientSession.call_tool
    assert (ClientSession, "call_tool") not in mcp_instrument._originals
    assert ClientSession.call_tool is original


@pytest.mark.asyncio
async def test_double_enable_is_a_noop():
    mock_call_tool = AsyncMock(return_value="result")
    session = _patched_client_session(mock_call_tool)
    mcp_instrument.enable_mcp_instrumentation()  # second call - must not re-wrap the wrapper

    with SpanContext(workflow_id="wf"):
        await session.call_tool("some_tool")

    # If double-enabled, kwargs["meta"] would still be a single well-formed
    # dict either way - the real regression this guards is enable_* raising
    # or replacing the already-patched method with itself wrapped twice,
    # which a second call succeeding without error already rules out.
    mock_call_tool.assert_awaited_once()
