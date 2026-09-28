"""
Correlate an agent hand-off across two processes using SpanContext.

Needs only `pip install savi-sdk` (no provider extra, no API key, no SAVI
account). This simulates Agent A calling Agent B over HTTP without actually
making a network call: to_headers()/from_headers() are pure serialization,
so you can see the correlation work end to end without a real server.

Run: python examples/02_agent_tracing_handoff.py
"""
from savi import SpanContext

if __name__ == "__main__":
    # --- Agent A: starts the run, is about to hand off to Agent B ---
    with SpanContext(workflow_id="kyc-review", agent_id="intake") as span_a:
        print("Agent A run_id: ", span_a.run_id)
        headers = span_a.to_headers()
        print("Headers Agent A would attach to its outbound HTTP call:")
        print(" ", headers)

    # --- Agent B: a separate process, receives Agent A's headers ---
    resumed = SpanContext.from_headers(headers)
    with resumed:
        print("Agent B run_id: ", resumed.run_id)
        print("Same run?       ", resumed.run_id == span_a.run_id)

    # A hand-off with no trace context (e.g. a caller that isn't SAVI-aware)
    # never raises; it just starts a fresh, uncorrelated run.
    orphan = SpanContext.from_headers({}) or SpanContext(workflow_id="unknown")
    print("\nOrphaned call falls back to a fresh run_id:", orphan.run_id)
