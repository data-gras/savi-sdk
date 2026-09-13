"""
Auto-instrumentation compatibility probe against a real agent framework
(LangChain), not just savi's own clients.

enable_auto_instrumentation() patches openai.resources.chat.completions
.Completions.create at the class level - the same method langchain_openai's
ChatOpenAI calls internally, several layers of LangChain's own code above
it. Because the patch targets the class rather than a specific instance, it
should apply to a client LangChain constructs too, regardless of whether
LangChain built that client before or after enable_auto_instrumentation()
ran - only the order relative to the actual *call* matters.

That's a plausible argument, not a verified one: LangChain isn't part of
this SDK's own dependency chain, and nothing here has actually run it before
today. This file exists to settle it either way, going forward, rather than
leave it as an open question in the README indefinitely. If a future
langchain-openai release changes how it reaches the openai client, this is
what would catch that regression.

Skipped entirely unless langchain-openai is installed - `pip install
langchain-openai`. Not a runtime dependency of the SDK itself, so it isn't
part of any extras_require group; install it only to run this file.
"""
import pytest

langchain_openai = pytest.importorskip("langchain_openai")

from unittest.mock import MagicMock, patch
from savi import auto_instrument


@pytest.fixture(autouse=True)
def _cleanup():
    from openai.resources.chat.completions import Completions
    original = Completions.create
    yield
    auto_instrument.disable_auto_instrumentation()
    Completions.create = original


def _openai_response(content="OK", model="gpt-4o-mini"):
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=0,
        model=model,
        choices=[Choice(
            index=0, finish_reason="stop",
            message=ChatCompletionMessage(role="assistant", content=content),
        )],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def test_langchain_chatopenai_gets_instrumented():
    """The concrete claim this settles: does a call made through
    langchain_openai.ChatOpenAI (never touching SaviOpenAI at all) still get
    reported, purely as a side effect of where auto-instrumentation patches?
    """
    from openai.resources.chat.completions import Completions

    mock_collector = MagicMock()
    with patch.object(Completions, "create", return_value=_openai_response()):
        auto_instrument.enable_auto_instrumentation(
            savi_key="sk_test", tenant_id="ten_test", mask_pii=False, _collector=mock_collector,
        )
        llm = langchain_openai.ChatOpenAI(model="gpt-4o-mini", api_key="test-key")
        llm.invoke("Say OK.")

    mock_collector.emit.assert_called_once()
    event = mock_collector.emit.call_args[0][0]
    assert event["provider"] == "openai"
    assert event["tenant_id"] == "ten_test"
    assert event["tokens_in"] == 10
    assert event["tokens_out"] == 5
