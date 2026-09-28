"""
Wrap a real OpenAI call with SaviOpenAI in local_mode, zero SAVI account.

Needs:
  pip install "savi-sdk[openai]"
  export OPENAI_API_KEY=sk-...

local_mode=True only changes where SAVI's own telemetry goes (your terminal
instead of SAVI's backend). The OpenAI call itself is real and still uses
your OpenAI quota, same as it would with the plain openai client.

Run: python examples/03_quickstart_local_mode.py
"""
import os
import sys

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(
        "Set OPENAI_API_KEY first, e.g.:\n"
        "  export OPENAI_API_KEY=sk-...\n"
        "This example makes a real OpenAI call; SAVI's own telemetry stays "
        "local either way."
    )

try:
    from savi import SaviOpenAI
except ImportError:
    sys.exit('Missing the openai extra. Run: pip install "savi-sdk[openai]"')

if __name__ == "__main__":
    client = SaviOpenAI(api_key=os.environ["OPENAI_API_KEY"], local_mode=True)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in five words or fewer."}],
    )

    print("\nModel replied:", response.choices[0].message.content)
    print("\n(the [savi:local] line above is SAVI's own telemetry - tokens, "
          "latency, PII flags, fingerprint - printed locally, never sent anywhere)")
