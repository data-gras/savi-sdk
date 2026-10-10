"""
savi/content.py, opt-in capture of prompt, response, tool, system prompt, earlier-message and retrieved-document content.

Off by default. Nothing here runs unless a wrapper is built with
capture_content=True (or a ContentCapture). Even then the server stores the
text only when the workspace has itself opted in (consent, capture settings);
otherwise the server discards it on arrival. Content never goes to local-mode
logs, only to SAVI's ingest endpoint.

Masking: when the wrapper has a PII masker (mask_pii on and Presidio installed)
the text is masked here before it leaves the process. Whether or not it was,
the server masks again with its own redactor, which is the authoritative one.

A failure here never affects the caller's LLM call and never costs the
metadata event: every field is bounded to the server's limits, and any error
returns the event without content.
"""
import contextvars
import json
import logging
import threading
import uuid
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# The server rejects a whole event (and so loses its metadata) if a text field
# is longer than this or a tool field serializes to more than this.
MAX_TEXT_CHARS = 10_000
MAX_ID_CHARS = 128

CONTENT_FIELDS = ("prompt_text", "response_text", "tool_arguments", "tool_results",
                  "system_prompt", "history_text", "retrieved_context")


@dataclass(frozen=True)
class ContentCapture:
    """Which parts of a call to send. Each part is still subject to the
    workspace's own capture settings on the server."""
    prompt: bool = True       # the latest user message
    response: bool = True
    tools: bool = True
    system: bool = True       # the system prompt
    history: bool = True      # the earlier messages sent with the call
    retrieval: bool = True    # documents passed in with savi.retrieval(...)


def resolve_capture(value) -> "ContentCapture | None":
    """capture_content=False/None -> off; True -> everything; a
    ContentCapture -> as given."""
    if value is None or value is False:
        return None
    if value is True:
        return ContentCapture()
    if isinstance(value, ContentCapture):
        if not (value.prompt or value.response or value.tools or value.system or value.history or value.retrieval):
            return None
        return value
    raise TypeError("capture_content must be a bool or a ContentCapture")


# --- conversation context ---------------------------------------------------

_CONVERSATION: contextvars.ContextVar["conversation | None"] = contextvars.ContextVar(
    "savi_conversation", default=None)


class conversation:
    """Groups the calls made inside the block into one conversation and numbers
    them 0, 1, 2 in call order, so stored content can be shown as a replay.

        with savi.conversation("support-4711"):
            client.chat.completions.create(...)   # turn 0
            client.chat.completions.create(...)   # turn 1

    With no id one is generated. Outside a block no conversation fields are sent.
    """

    def __init__(self, conversation_id: "str | None" = None):
        self.conversation_id = (conversation_id or f"conv_{uuid.uuid4().hex[:24]}")[:MAX_ID_CHARS]
        self._turn = 0
        self._lock = threading.Lock()
        self._token = None

    def __enter__(self):
        self._token = _CONVERSATION.set(self)
        return self

    def __exit__(self, *_):
        _CONVERSATION.reset(self._token)

    def next_turn(self) -> int:
        with self._lock:
            turn = self._turn
            self._turn += 1
        return turn


def conversation_fields() -> dict:
    """conversation_id and turn_index for the next call in the active
    conversation, or {} outside one."""
    active = _CONVERSATION.get()
    if active is None:
        return {}
    return {"conversation_id": active.conversation_id, "turn_index": active.next_turn()}


# --- retrieved documents ----------------------------------------------------

_RETRIEVAL: contextvars.ContextVar["list | None"] = contextvars.ContextVar("savi_retrieval", default=None)


class retrieval:
    """Tells SAVI which documents your code retrieved for the calls made inside the block.

    SAVI cannot see inside your search or vector store, so retrieval is declared, not detected:

        with savi.retrieval([{"source": "kb/refunds", "text": chunk.text} for chunk in chunks]):
            client.chat.completions.create(...)

    Each item can be a string or a dict. The documents are masked here and sent only when the workspace has
    switched on "retrieved documents" for this agent and environment. Nested blocks: the inner one wins.
    """

    def __init__(self, documents):
        self.documents = list(documents) if isinstance(documents, (list, tuple)) else [documents]
        self._token = None

    def __enter__(self):
        self._token = _RETRIEVAL.set(self.documents)
        return self

    def __exit__(self, *_):
        _RETRIEVAL.reset(self._token)


def retrieved_documents() -> list:
    """The documents declared by the active savi.retrieval block, or [] outside one."""
    return _RETRIEVAL.get() or []


# --- extraction --------------------------------------------------------------

def _text_of(content) -> str:
    """Plain text of a message body: a string, or a list of blocks
    ({"type": "text", "text": ...} or bare strings)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "\n".join(p for p in parts if p)
    return ""


def render_prompt(messages, system=None) -> str:
    """The conversation sent to the model as 'role: text' lines. Tool-result
    messages are left out here; they are sent as tool_results. Longer than the
    limit keeps the end, because the latest turn matters most."""
    lines = []
    if system:
        lines.append(f"system: {_text_of(system)}")
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool":
            continue
        text = _text_of(message.get("content"))
        if text:
            lines.append(f"{role or 'user'}: {text}")
    return "\n".join(lines)


def split_prompt(messages, system=None) -> tuple:
    """(system prompt, earlier messages, latest user message) as three texts, each possibly empty.

    The system prompt is the `system` argument plus any system or developer message. The latest user message
    is the last message with the user role. Everything else with text, in order, is the earlier messages, as
    "role: text" lines. Tool messages are left out here; they are sent as tool results."""
    system_parts = []
    if system:
        text = _text_of(system)
        if text:
            system_parts.append(text)
    turns = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or "user"
        if role == "tool":
            continue
        text = _text_of(message.get("content"))
        if not text:
            continue
        if role in ("system", "developer"):
            system_parts.append(text)
        else:
            turns.append((role, text))
    latest = max((i for i, (role, _) in enumerate(turns) if role == "user"), default=None)
    prompt = turns.pop(latest)[1] if latest is not None else ""
    history = "\n".join(f"{role}: {text}" for role, text in turns)
    return "\n".join(system_parts), history, prompt


def _mask_value(masker, value):
    """Mask every string inside a JSON-like value."""
    if isinstance(value, str):
        return masker.mask(value)[0]
    if isinstance(value, list):
        return [_mask_value(masker, v) for v in value]
    if isinstance(value, dict):
        return {k: _mask_value(masker, v) for k, v in value.items()}
    return value


def _bound_text(text: str, keep: str = "end") -> "str | None":
    """At most MAX_TEXT_CHARS. Conversations keep their end (the latest turn matters most);
    a system prompt keeps its start, where the instructions are."""
    if not text:
        return None
    return text[:MAX_TEXT_CHARS] if keep == "start" else text[-MAX_TEXT_CHARS:]


def _bound_json(value):
    """A tool value the server will accept: at most MAX_TEXT_CHARS when
    serialized. Lists lose their oldest items first."""
    if value in (None, [], {}):
        return None

    def size(v):
        return len(json.dumps(v, default=str))

    if size(value) <= MAX_TEXT_CHARS:
        return value
    if isinstance(value, list):
        items = list(value)
        while items and size(items) > MAX_TEXT_CHARS:
            items.pop(0)
        if items:
            return items
    return {"truncated": True}


def openai_tool_calls(response) -> list:
    """Tool calls the model asked for, from an OpenAI-shaped response
    (choices[0].message.tool_calls). Arguments are kept as the model sent them."""
    try:
        calls = response.choices[0].message.tool_calls or []
        return [{"name": c.function.name, "arguments": c.function.arguments} for c in calls]
    except Exception:
        return []


def openai_tool_results(messages) -> list:
    """Tool outputs the caller passed back in this call's messages: the
    role='tool' messages after the last assistant message."""
    results = []
    for message in reversed(messages or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            results.append({"tool_call_id": message.get("tool_call_id"),
                            "content": _text_of(message.get("content"))})
        elif message.get("role") == "assistant":
            break
    return list(reversed(results))


def openai_response_text(response) -> str:
    try:
        return _text_of(response.choices[0].message.content)
    except Exception:
        return ""


def anthropic_response_text(response) -> str:
    try:
        return "\n".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    except Exception:
        return ""


def anthropic_tool_calls(response) -> list:
    try:
        return [{"name": b.name, "arguments": b.input}
                for b in response.content if getattr(b, "type", None) == "tool_use"]
    except Exception:
        return []


def anthropic_tool_results(messages) -> list:
    """tool_result blocks in the latest user message."""
    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), list):
            blocks = [b for b in message["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
            if blocks:
                return [{"tool_call_id": b.get("tool_use_id"), "content": _text_of(b.get("content"))}
                        for b in blocks]
            return []
    return []


# --- attaching to an event ----------------------------------------------------

def add_content(payload: dict, capture: "ContentCapture | None", masker, *,
                messages=None, system=None, response_text: str = "",
                tool_calls=None, tool_results=None) -> dict:
    """Add the chosen content fields, and the conversation fields, to an event
    payload. Returns the payload unchanged when capture is off or anything
    goes wrong."""
    if capture is None:
        return payload
    try:
        extra: dict = {}
        system_text, history_text, prompt_text = split_prompt(messages, system)
        for wanted, key, text, keep in (
            (capture.prompt, "prompt_text", prompt_text, "end"),
            (capture.system, "system_prompt", system_text, "start"),
            (capture.history, "history_text", history_text, "end"),
        ):
            if not wanted:
                continue
            if masker is not None and text:
                text = masker.mask(text)[0]
            bounded = _bound_text(text, keep)
            if bounded:
                extra[key] = bounded
        if capture.retrieval:
            documents = retrieved_documents()
            if masker is not None and documents:
                documents = _mask_value(masker, documents)
            bounded_docs = _bound_json(documents)
            if bounded_docs is not None:
                extra["retrieved_context"] = bounded_docs
        if capture.response:
            text = response_text or ""
            if masker is not None and text:
                text = masker.mask(text)[0]
            bounded = _bound_text(text)
            if bounded:
                extra["response_text"] = bounded
        if capture.tools:
            for key, value in (("tool_arguments", tool_calls), ("tool_results", tool_results)):
                if masker is not None and value:
                    value = _mask_value(masker, value)
                bounded = _bound_json(value)
                if bounded is not None:
                    extra[key] = bounded
        if not extra:
            return payload
        extra.update(conversation_fields())
        return {**payload, **extra}
    except Exception:
        _log.debug("savi.content: could not attach content; sending the event without it", exc_info=True)
        return payload


def strip_content(event: dict) -> dict:
    """The event without any content field, for outputs that must never carry
    it (local-mode logs)."""
    return {k: v for k, v in event.items() if k not in CONTENT_FIELDS}


# --- provider mixins ------------------------------------------------------------

class OpenAIContentMixin:
    """For proxies whose responses are OpenAI-shaped (OpenAI, Azure OpenAI,
    Mistral). The wrapper sets `_content`; None means capture is off."""
    _content = None

    def _emit_with_content(self, messages, response, payload):
        self._emit(add_content(
            payload, self._content, self._masker, messages=messages,
            response_text=openai_response_text(response),
            tool_calls=openai_tool_calls(response),
            tool_results=openai_tool_results(messages),
        ))


class AnthropicContentMixin:
    _content = None

    def _emit_with_content(self, messages, system, response, payload):
        self._emit(add_content(
            payload, self._content, self._masker, messages=messages, system=system,
            response_text=anthropic_response_text(response),
            tool_calls=anthropic_tool_calls(response),
            tool_results=anthropic_tool_results(messages),
        ))


class _CollectorContentMixin:
    """For wrappers that hold the collector themselves (Bedrock, Cohere)."""
    _content = None


def _bedrock_blocks(response):
    try:
        return response["output"]["message"]["content"] or []
    except Exception:
        return []


def bedrock_response_text(response) -> str:
    return "\n".join(b["text"] for b in _bedrock_blocks(response) if isinstance(b, dict) and isinstance(b.get("text"), str))


def bedrock_tool_calls(response) -> list:
    return [{"name": b["toolUse"].get("name"), "arguments": b["toolUse"].get("input")}
            for b in _bedrock_blocks(response) if isinstance(b, dict) and isinstance(b.get("toolUse"), dict)]


def bedrock_tool_results(messages) -> list:
    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), list):
            found = [b["toolResult"] for b in message["content"]
                     if isinstance(b, dict) and isinstance(b.get("toolResult"), dict)]
            return [{"tool_call_id": r.get("toolUseId"), "content": _text_of(r.get("content"))} for r in found]
    return []


class BedrockContentMixin(_CollectorContentMixin):
    def _emit_with_content(self, messages, system, response, payload):
        self._collector.emit(add_content(
            payload, self._content, self._masker, messages=messages, system=system,
            response_text=bedrock_response_text(response),
            tool_calls=bedrock_tool_calls(response),
            tool_results=bedrock_tool_results(messages),
        ))


def cohere_response_text(response) -> str:
    try:
        return "\n".join(getattr(b, "text", "") or "" for b in response.message.content
                         if getattr(b, "type", "text") == "text")
    except Exception:
        return ""


def cohere_tool_calls(response) -> list:
    try:
        return [{"name": c.function.name, "arguments": c.function.arguments}
                for c in (response.message.tool_calls or [])]
    except Exception:
        return []


class CohereContentMixin(_CollectorContentMixin):
    def _emit_with_content(self, messages, response, payload):
        self._collector.emit(add_content(
            payload, self._content, self._masker, messages=messages,
            response_text=cohere_response_text(response),
            tool_calls=cohere_tool_calls(response),
            tool_results=openai_tool_results(messages),
        ))


class VertexContentMixin:
    """Vertex prompts are 'contents': a string or a list of strings is captured;
    richer Content objects are not (there is no plain text to send)."""
    _content = None

    def _emit_with_content(self, contents, response, payload):
        if isinstance(contents, str):
            messages = [{"role": "user", "content": contents}]
        elif isinstance(contents, list) and all(isinstance(c, str) for c in contents):
            messages = [{"role": "user", "content": c} for c in contents]
        else:
            messages = []
        try:
            text = response.text
        except Exception:
            text = ""
        self._emit(add_content(payload, self._content, self._masker, messages=messages,
                               response_text=text if isinstance(text, str) else ""))
