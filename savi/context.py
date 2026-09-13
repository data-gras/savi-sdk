# sdks/python-savi/savi/context.py
import contextvars
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Mapping

logger = logging.getLogger(__name__)

_SPAN_VAR: contextvars.ContextVar[str | None] = contextvars.ContextVar("savi_span", default=None)
_CONTEXT_STORE: dict[str, "SpanContext"] = {}

# Cross-process propagation (HTTP header / env var), for a subprocess or
# HTTP hand-off that doesn't go through MCP (see mcp_instrument.py for
# that channel). A custom header rather than W3C traceparent, since that
# format's fields can't carry workflow_id/agent_id/user_id.
TRACE_HEADER = "X-Savi-Trace-Context"
TRACE_ENV = "SAVI_TRACE_CONTEXT"


@dataclass
class SpanContext:
    workflow_id: str
    agent_id:    str | None = None
    user_id:     str | None = None
    run_id:      str | None = None
    span_id:     str        = field(default_factory=lambda: f"span_{uuid.uuid4().hex[:16]}")
    parent_id:   str | None = field(default=None, init=False)
    _token:      object     = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.parent_id = _SPAN_VAR.get()
        # Resolution: explicit → inherit parent's run_id → auto-generate at root
        if self.run_id is None:
            parent_ctx = _CONTEXT_STORE.get(self.parent_id) if self.parent_id else None
            self.run_id = (parent_ctx.run_id if parent_ctx else None) or f"run_{uuid.uuid4().hex}"

    def __enter__(self):
        self._token = _SPAN_VAR.set(self.span_id)
        _CONTEXT_STORE[self.span_id] = self
        return self

    def __exit__(self, *_):
        _CONTEXT_STORE.pop(self.span_id, None)
        _SPAN_VAR.reset(self._token)

    def attribution_fields(self) -> dict:
        return {
            "workflow_id":    self.workflow_id,
            "agent_id":       self.agent_id,
            "user_id":        self.user_id,
            "parent_span_id": self.span_id,
            "agent_run_id":   self.run_id,
        }

    def _wire_payload(self) -> dict:
        return {
            "workflow_id": self.workflow_id, "agent_id": self.agent_id,
            "user_id": self.user_id, "run_id": self.run_id, "span_id": self.span_id,
        }

    def to_headers(self) -> dict[str, str]:
        """Call inside the active `with` block before an outbound call to
        another process. The receiving side resumes the same run_id via
        SpanContext.from_headers()."""
        return {TRACE_HEADER: json.dumps(self._wire_payload())}

    def to_env(self) -> dict[str, str]:
        """Same as to_headers(), for a subprocess hand-off: merge into the
        env passed to subprocess.Popen/run, and the child resumes via
        SpanContext.from_env()."""
        return {TRACE_ENV: json.dumps(self._wire_payload())}

    @classmethod
    def _from_wire(cls, raw: str | None) -> "SpanContext | None":
        if not raw:
            return None
        try:
            data = json.loads(raw)
            ctx = cls(
                workflow_id=data["workflow_id"], agent_id=data.get("agent_id"),
                user_id=data.get("user_id"), run_id=data.get("run_id"),
            )
        except Exception:
            # A malformed trace context must never break the receiving
            # process; it just starts a fresh, uncorrelated run.
            logger.debug("savi.context: failed to parse incoming trace context", exc_info=True)
            return None
        # __post_init__ can't set this itself - it only reads the current
        # process's contextvar, which is empty in a fresh process.
        ctx.parent_id = data.get("span_id")
        return ctx

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> "SpanContext | None":
        """Resume a SpanContext a sender attached via to_headers(). Header
        lookup is case-insensitive: tries the canonical name, a lowercased
        fallback, then a full case-insensitive scan for a plain dict built
        with different casing. Returns None if absent or unparseable."""
        raw = headers.get(TRACE_HEADER)
        if raw is None:
            raw = headers.get(TRACE_HEADER.lower())
        if raw is None:
            target = TRACE_HEADER.lower()
            raw = next((v for k, v in headers.items() if k.lower() == target), None)
        return cls._from_wire(raw)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SpanContext | None":
        """Resume a SpanContext a parent process attached via to_env().
        Reads os.environ by default, pass env explicitly only in tests or
        if you're not propagating through the process's real environment."""
        return cls._from_wire((env if env is not None else os.environ).get(TRACE_ENV))


def start_run(
    workflow_id: str,
    agent_id: str | None = None,
    user_id: str | None = None,
    run_id: str | None = None,
) -> SpanContext:
    """Convenience constructor for a top-level agent run.

    Always generates a new run_id (or uses the one provided). Does NOT inherit
    run_id from any active parent span, call SpanContext() directly if you need
    inheritance from the current context.
    """
    return SpanContext(
        workflow_id=workflow_id,
        agent_id=agent_id,
        user_id=user_id,
        run_id=run_id or f"run_{uuid.uuid4().hex}",
    )


def get_current_span_id() -> str | None:
    return _SPAN_VAR.get()


def get_current_context() -> SpanContext | None:
    sid = _SPAN_VAR.get()
    return _CONTEXT_STORE.get(sid) if sid else None


def attribution_fields() -> dict:
    """Module-level helper, returns attribution dict for the active context or all-None."""
    ctx = get_current_context()
    if ctx:
        return ctx.attribution_fields()
    return {
        "workflow_id": None, "agent_id": None, "user_id": None,
        "parent_span_id": None, "agent_run_id": None,
    }
