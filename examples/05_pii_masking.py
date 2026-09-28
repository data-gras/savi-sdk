"""
Show SAVI flagging PII in a prompt, without sending that PII to SAVI itself.

Needs:
  pip install "savi-sdk[openai,pii]"
  python -m spacy download en_core_web_lg
  export OPENAI_API_KEY=sk-...

Masking only protects what SAVI's own telemetry receives (entity counts, not
raw text). Your actual prompt still goes to OpenAI completely unmodified,
same as it would without this SDK - see the README's "PII masking" section
for why. Watch the [savi:local] line below: it carries pii_flagged/pii_types,
never the name or email itself.

Run: python examples/05_pii_masking.py
"""
import os
import sys

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(
        "Set OPENAI_API_KEY first, e.g.:\n"
        "  export OPENAI_API_KEY=sk-...\n"
        "This example makes a real OpenAI call; only SAVI's local telemetry "
        "is what gets masked, not the call itself."
    )

try:
    from savi import SaviOpenAI
except ImportError:
    sys.exit('Missing the openai extra. Run: pip install "savi-sdk[openai]"')

if __name__ == "__main__":
    # mask_pii defaults to on-if-Presidio's-installed. Passing True explicitly
    # makes a missing Presidio/spaCy model a hard error instead of a silent
    # unmasked fallback - useful here so a missing setup step is obvious.
    client = SaviOpenAI(api_key=os.environ["OPENAI_API_KEY"], local_mode=True, mask_pii=True)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": "Draft a follow-up email to John Smith at john.smith@example.com "
                       "about his renewal.",
        }],
    )

    print("\nModel replied:", response.choices[0].message.content)
    print("\n(the [savi:local] line above should show pii_flagged=true and "
          "pii_types listing PERSON/EMAIL_ADDRESS - counts only, never the "
          "name or address itself)")
