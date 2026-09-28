"""
Show SAVI's opt-in response cache avoiding a second real API call.

Needs:
  pip install "savi-sdk[openai]"
  export OPENAI_API_KEY=sk-...

The cache is entirely local (no SAVI account involved). The same question
asked twice in a row should hit OpenAI once, not twice - the second call
returns instantly from the in-memory cache.

Run: python examples/04_response_caching.py
"""
import os
import sys
import time

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(
        "Set OPENAI_API_KEY first, e.g.:\n"
        "  export OPENAI_API_KEY=sk-...\n"
        "This example makes a real OpenAI call for the cache miss."
    )

try:
    from savi import SaviOpenAI
except ImportError:
    sys.exit('Missing the openai extra. Run: pip install "savi-sdk[openai]"')

if __name__ == "__main__":
    client = SaviOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        local_mode=True,
        enable_cache=True,
        cache_ttl_seconds=300,
    )

    question = [{"role": "user", "content": "What are your business hours?"}]

    t0 = time.monotonic()
    r1 = client.chat.completions.create(model="gpt-4o-mini", messages=question)
    t1 = time.monotonic()
    print(f"First call  (cache miss): {t1 - t0:.2f}s - real OpenAI call, billed as normal")

    t0 = time.monotonic()
    r2 = client.chat.completions.create(model="gpt-4o-mini", messages=question)
    t1 = time.monotonic()
    print(f"Second call (cache hit):  {t1 - t0:.4f}s - OpenAI never called")

    print("\nSame response object:", r1 is r2)
