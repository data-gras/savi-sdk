# savi-sdk

[![Tests](https://github.com/data-gras/savi-sdk/actions/workflows/test.yml/badge.svg)](https://github.com/data-gras/savi-sdk/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/data-gras/savi-sdk/branch/main/graph/badge.svg)](https://codecov.io/gh/data-gras/savi-sdk)
[![PyPI version](https://img.shields.io/pypi/v/savi-sdk.svg)](https://pypi.org/project/savi-sdk/)
[![Python versions](https://img.shields.io/pypi/pyversions/savi-sdk.svg)](https://pypi.org/project/savi-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Lightweight observability SDK for LLM costs, carbon, and compliance.
Drop one wrapper around your LLM calls. SAVI captures spend, tokens, latency,
PII flags, and carbon automatically, with zero changes to your prompts or model logic.

> **The SAVI service is hosted and closed-source. This SDK is the open-source client.**
> You do not need an account to start: [Local Mode](#start-here-try-it-from-zero) runs on your own computer
> and sends nothing out. To use SAVI across your company, [book a demo](https://datagras.com/savi/contact?utm_source=github&utm_medium=readme&utm_campaign=sdk-readme) and we will set it up with you.

Runnable examples, from "needs nothing" to "needs a provider key", live in
[`examples/`](examples/).

---

## Start here: try it from zero

This part assumes you know nothing. You do not need a SAVI account or a SAVI server. Local mode runs inside your own program on your own computer, and nothing leaves it. You need Python and one AI provider key (this guide uses OpenAI). A test costs a few cents.

This guide needs savi-sdk 0.16 or newer. After step 4, check with `pip show savi-sdk`.

No key yet? Jump to [Try it without a key](#try-it-without-a-key).

### 1. Install Python

You need Python 3.11 or newer. Check what you have:

```
python --version
```

On Mac or Linux you may need `python3 --version`. If it shows 3.11 or higher, go to step 2. If not, download Python from [python.org/downloads](https://www.python.org/downloads/). On Windows, tick **Add Python to PATH** on the first screen of the installer.

### 2. Make a folder and open a terminal in it

Make a new folder, for example `savi-try`.

- **Windows:** open the folder, click the address bar, type `cmd`, press Enter.
- **Mac:** open Terminal, type `cd ` (with a space after it), drag the folder into the window, press Enter.

### 3. Make a clean Python space

This keeps SAVI separate from everything else on your computer.

```
python -m venv venv
```

Turn it on:

- **Windows:** `venv\Scripts\activate`
- **Mac or Linux:** `source venv/bin/activate`

You should now see `(venv)` at the start of the line. Do this again each time you open a new terminal.

### 4. Install SAVI

```
pip install "savi-sdk[openai]"
```

### 5. Give it your OpenAI key

Get a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys). Treat it like a password: never put it in a file you share. Set it for this terminal only:

- **Windows, Command Prompt:** `set OPENAI_API_KEY=sk-your-key`
- **Windows, PowerShell:** `$env:OPENAI_API_KEY="sk-your-key"`
- **Mac or Linux:** `export OPENAI_API_KEY="sk-your-key"`

### 6. Make your first calls

Save this as `first_call.py` in your folder:

```python
import os
from savi import SaviOpenAI, local_report

local_report.start()   # turns on the readable output and the report

client = SaviOpenAI(api_key=os.environ["OPENAI_API_KEY"], local_mode=True)

# The first two questions are the same on purpose, so you can see SAVI notice a repeat.
for question in ["What is a token?", "What is a token?", "Name three colours."]:
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
```

Run it:

```
python first_call.py
```

### 7. What you will see

Each call prints one line. When your script ends, a summary prints and SAVI writes `savi-report.html` in your folder. This is the output of the sample run (`python -m savi.local_report --demo`, made-up calls), so you know what to expect:

```text
  [ OK  ] openai     gpt-4o-mini-2024-07-18       in   214  out    96    812 ms  cost ~$0.000090  workflow=support-inbox agent=triage
  [ OK  ] openai     gpt-4o-2024-08-06            in  1840  out   402     2.3 s  cost ~$0.0086  workflow=support-inbox agent=drafter
  [ OK  ] anthropic  claude-haiku-4-5-20251001    in   620  out   140    940 ms  cost ~$0.0013  workflow=kyc-review agent=extractor
  [CACHE] openai     gpt-4o-mini-2024-07-18       in   214  out    96     14 ms  cost $0.00 (saved ~$0.000090)  answered from cache  workflow=support-inbox agent=triage
  [ OK  ] openai     gpt-4o-mini-2024-07-18       in   330  out    60    701 ms  cost ~$0.000085  personal data: EMAIL_ADDRESS x1, PERSON x1  workflow=support-inbox user=priya
  [ OK  ] openai     gpt-4o-mini-2024-07-18       in   214  out    96    790 ms  cost ~$0.000090  workflow=support-inbox agent=triage
  [ OK  ] mistral    mistral-small-latest         in   410  out    88     1.1 s  cost no price
  [FAIL ] openai     gpt-4o                       in     0  out     0    30.2 s  cost -  failed timeout  workflow=support-inbox agent=drafter

  Summary  (this computer only; nothing was sent anywhere)
    8 calls, 1 failed   tokens in 3,842 / out 978   average response 1.1 s
    cost about $0.01. Cost covers 6 of 7 calls. No price for: mistral-small-latest.
    1 answered from the cache, saving about $0.000090
    personal data found in 1 call: EMAIL_ADDRESS x1, PERSON x1
    Costs marked ~ use the prices you entered. They are estimates, not a bill.
```

You may also see a line that starts with "PII masking is on by default but Presidio isn't installed". It is a notice, not an error. It means the personal-data add-on is not installed, so SAVI will not flag personal data yet. Your calls work either way. To add it, see [If something goes wrong](#if-something-goes-wrong).

How to read a line: the status (`OK`, `CACHE` when the answer came from your own cache, `FAIL`), the provider and model, tokens in and out, how long the call took, and the cost. A `~` before a cost means it is an estimate from the prices you entered. Any labels (workflow, agent, user) come last.

Now open `savi-report.html` in your browser (double-click it). It is one file with everything inside. The top of the sample report looks like this ([open the full sample](https://github.com/data-gras/savi-sdk/blob/main/docs/sample-report.html)):

![The top of a SAVI local report: a headline, six totals, cost by model, and what SAVI noticed](https://raw.githubusercontent.com/data-gras/savi-sdk/main/docs/sample-report.png)

The full report also lists every call, the prices it used, and a box that explains what local mode cannot show.

### 8. Add your own prices

SAVI does not know what you pay, so it shows no cost until you tell it. Make a file called `my-prices.json`:

```json
{
  "_note": "USD per 1 million tokens: [input, output]. Check your provider's pricing page.",
  "gpt-4o-mini": [0.15, 0.60],
  "gpt-4o": [2.50, 10.00]
}
```

These numbers are examples. Look up your provider's current prices and use those. Then change one line in `first_call.py`:

```python
local_report.start(pricing="my-prices.json")
```

A price matches a model name that starts with it, so `gpt-4o-mini` also covers `gpt-4o-mini-2024-07-18`, the name OpenAI sends back. If two prices match, the longest name wins. A model with no price is left out of the total, and the report says which one. You can also set the file once for every script with the `SAVI_LOCAL_PRICING` environment variable. If the file has a mistake, SAVI tells you what and where.

### 9. Share the report

The report is one file, so you can email it or drop it in a chat. It holds counts, model names, tokens, response times, costs from your prices, and the types of personal data found. It never holds what was asked or answered. Workflow, agent and user names do appear. To swap them for neutral labels:

```python
local_report.start(pricing="my-prices.json", hide_names=True)
```

Open the file yourself before you send it.

### Try it without a key

This writes a sample report from made-up calls. The report says it is a sample, so nobody mistakes it for real use.

```
python -m savi.local_report --demo
```

### If something goes wrong

Run the setup check. It shows your Python version, what is installed, which keys are set (never the keys themselves), settings that send your calls somewhere else, and whether your computer can reach the provider:

```
python -m savi.doctor
```

- **"Connection error" on every call.** Often a setting such as `OPENAI_BASE_URL` points your calls at another address. The setup check lists it. Remove it for this terminal to go back to OpenAI.
- **No personal data flags.** Flagging is an add-on: `pip install "savi-sdk[pii]"` and then `python -m spacy download en_core_web_lg`. It is a large download, and on some computers (for example Windows on ARM) it may not install.
- **No cost shown.** Add prices (step 8).

### What local mode does not do

It shows one run on one computer. It has no history across runs, no view of other teams, no spend limits, no rules, and no alerts. Those come when SAVI is connected to your company. [Book a demo](https://datagras.com/savi/contact?utm_source=github&utm_medium=readme&utm_campaign=sdk-readme) and tell us which providers and tools your teams use.

### Do I need a SAVI server?

No. There is nothing to start and nothing to sign up for. You will not see `localhost` anywhere in this guide. The `api.datagras.com` addresses further down are for connected mode, which we set up with you.

---

## Install

```bash
# Core (no provider extras, bring your own client)
pip install savi-sdk

# With your provider
pip install "savi-sdk[openai]"      # OpenAI / Azure OpenAI
pip install "savi-sdk[anthropic]"   # Anthropic Claude
pip install "savi-sdk[google]"      # Google Vertex AI
pip install "savi-sdk[bedrock]"     # AWS Bedrock (boto3)
pip install "savi-sdk[cohere]"      # Cohere
pip install "savi-sdk[mistral]"     # Mistral AI

# PII masking (requires spaCy model, see below)
pip install "savi-sdk[pii]"

# Everything
pip install "savi-sdk[all]"
```

---

## Quick start

Replace `OpenAI(...)` with `SaviOpenAI(...)`. No SAVI account needed to try
it: pass `local_mode=True` and everything the wrapper computes client-side
(tokens, latency, PII flags, LSH fingerprint) prints straight to your
terminal instead of being sent anywhere.

```python
from savi import SaviOpenAI

client = SaviOpenAI(api_key="sk-...", local_mode=True)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarise this contract."}],
)
# prints: [savi:local] {"event_id": "...", "provider": "openai", "model": "gpt-4o",
#          "tokens_in": 42, "tokens_out": 118, "latency_ms": 810, ...}
```

See [Local Mode](#local-mode) below for the full picture (what's included,
optional cost estimates). Every provider wrapper supports `local_mode=True`
the same way.

Once your company has a SAVI account ([ask us](https://datagras.com/savi/contact?utm_source=github&utm_medium=readme&utm_campaign=sdk-readme)),
swap in your real `savi_key`/`tenant_id` instead and telemetry goes to SAVI
itself rather than your terminal — same call, same code shape either way:

```python
from savi import SaviOpenAI

client = SaviOpenAI(
    api_key="sk-...",          # your OpenAI key
    savi_key="sk_savi_...",    # from your SAVI account
    tenant_id="ten_acme",
    team_id="engineering",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarise this contract."}],
)
```

SAVI captures: tokens in/out, cost (USD + AUD), latency, model, carbon, LSH fingerprint.
Nothing changes in the API call; your app code is identical either way.

**Streaming (`stream=True`) is not instrumented by any wrapper today.** The
call itself still works exactly as it would with the plain provider client,
your code gets the real stream object back, nothing raises, but SAVI's
telemetry (cost, tokens, PII flags, fingerprint) and the opt-in response
cache both silently skip that call, since neither can be computed from a
stream object the way they can from a full response. If your workload
streams, you currently won't see it in SAVI at all.

LangChain, LlamaIndex, CrewAI, and gateway/router setups (LiteLLM, an
internal proxy) are not officially tested or supported. For LangChain
specifically, there's a real, documented reason it may work anyway:
`enable_auto_instrumentation()` patches the underlying `openai`/`anthropic`
packages directly, the same libraries `langchain_openai`'s `ChatOpenAI` calls
under its own layers - so the patch sits beneath the framework rather than
requiring framework-specific integration. That's a plausible mechanism, not
a guarantee (`tests/test_langchain_compat.py`, run with `langchain-openai`
installed, is a real automated check of it, not just a claim). CrewAI and
LiteLLM don't necessarily share that same call path, so the same reasoning
doesn't automatically extend to them. Check with your SAVI contact for the
current provider/framework support matrix before integrating into a
codebase that streams or uses an agent framework.

---

## How this compares

Most LLM observability tools take one of two shapes: a proxy you route
calls through by swapping your base URL (Helicone, Portkey), or a server
you deploy and point your app at (Langfuse runs on Postgres, ClickHouse,
Redis, and S3; Phoenix ships as a Docker container). savi-sdk is neither.
It's a Python import that wraps your existing client in-process; your call
still goes straight to the provider, nothing sits in the request path, and
nothing gets deployed, not even for [Local Mode](#local-mode)'s zero-account
path. A real SAVI account changes where the telemetry goes (SAVI's backend
instead of your terminal), not how the SDK runs.

---

## Supported providers

| Provider | Class | Method |
|---|---|---|
| OpenAI | `SaviOpenAI` | `client.chat.completions.create()` |
| Anthropic | `SaviAnthropic` | `client.messages.create()` |
| Azure OpenAI | `SaviAzureOpenAI` | `client.chat.completions.create()` |
| Google Vertex AI | `SaviVertexAI` | `model.generate_content()` |
| AWS Bedrock | `SaviBedrockRuntime` | `client.converse()` |
| Cohere | `SaviCohere` | `client.chat()` |
| Mistral | `SaviMistral` | `client.chat.complete()` |

### Anthropic
```python
from savi import SaviAnthropic

client = SaviAnthropic(api_key="...", savi_key="sk_savi_...", tenant_id="ten_acme")
response = client.messages.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=1024,
)
```

### AWS Bedrock
```python
from savi import SaviBedrockRuntime

client = SaviBedrockRuntime(savi_key="sk_savi_...", tenant_id="ten_acme",
                             region_name="ap-southeast-2")
response = client.converse(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
)
```

### Cohere
```python
from savi import SaviCohere

client = SaviCohere(api_key="...", savi_key="sk_savi_...", tenant_id="ten_acme")
response = client.chat(model="command-r-plus-08-2024",
                        messages=[{"role": "user", "content": "Hello"}])
```

### Mistral
```python
from savi import SaviMistral

client = SaviMistral(api_key="...", savi_key="sk_savi_...", tenant_id="ten_acme")
response = client.chat.complete(model="mistral-large-latest",
                                 messages=[{"role": "user", "content": "Hello"}])
```

---

## PII masking

All wrappers enable PII masking **by default**. This needs Presidio
(`pip install "savi-sdk[pii]"` plus the spaCy model below) — if it isn't
installed and you never touched `mask_pii` yourself, the SDK logs a warning
and continues unmasked rather than crashing, since the call itself shouldn't
fail over an optional dependency you didn't explicitly ask for. Pass
`mask_pii=True` explicitly if you want a missing dependency to be a hard
error instead (e.g. in an environment where masking is a compliance
requirement, not a nice-to-have). Content is scanned for personal data
before the LSH fingerprint is computed.
**The fingerprint and PII entity counts (not the raw prompt) are the only
things sent to SAVI.** To be precise about what this does and doesn't cover:
this masking protects what *SAVI's telemetry* receives, not what your LLM
provider receives. **Your actual prompt is still sent to Anthropic/OpenAI/etc.
completely unmodified**, exactly as it would be without this SDK — masking a
customer's own prompt before the model sees it would break the call. Nothing
here is a redaction/DLP layer for outbound provider traffic; it exists purely
to keep PII out of SAVI's own systems.

```bash
# First-time setup: download the spaCy model
python -m spacy download en_core_web_lg
```

```python
# Disable masking (e.g. for non-sensitive internal tools)
client = SaviOpenAI(..., mask_pii=False)

# Custom entity list: an exact replacement for the default list above
client = SaviOpenAI(..., pii_entities=["PERSON", "EMAIL_ADDRESS", "AU_TFN"])

# Exclude specific entities from the default list instead of replacing it
# outright, for a workload where a generic recognizer is a known
# false-positive source rather than a real signal (e.g. LOCATION/DATE_TIME
# routinely fire on ordinary query content in a text-to-SQL or analytics
# integration, not personal data). Stays in sync automatically if the
# default entity list ever changes, unlike pii_entities above.
client = SaviOpenAI(..., pii_exclude_entities=["LOCATION", "DATE_TIME"])
```

Supported Australian entities: `AU_ABN`, `AU_ACN`, `AU_TFN`, `AU_MEDICARE`
(APRA CPG 234 compliance).

**Using this outside Australia.** The 4 entities above are Australia-specific;
everything else in the default list (`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`,
`CREDIT_CARD`, `IBAN_CODE`, `IP_ADDRESS`, `LOCATION`, `DATE_TIME`) is generic
and works identically regardless of where you or your users are. Having the AU
entities enabled doesn't break or misfire for non-Australian data, they just
never match anything, so nothing needs to be turned off. But it also means a
US/UK/India/etc. deployment gets no *country-specific* identifier detection by
default the way Australia does (no automatic SSN, NINO, or Aadhaar detection,
for example) unless you opt in.

The underlying detection library, [Microsoft
Presidio](https://microsoft.github.io/presidio/supported_entities/), ships
built-in recognizers for many other countries already, so this is a matter of
using `pii_entities` to add the ones relevant to you, not a limitation of this
SDK. Check Presidio's own supported-entities list for the exact,
version-current identifiers for your country before using them; the list
above is only what SAVI's own default enables. Example shape (verify exact
names against the link above; do not copy these as a promise of what's
currently supported without checking):

```python
# e.g. adding US-specific identifiers to the default list
client = SaviOpenAI(..., pii_entities=[
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
    "IBAN_CODE", "IP_ADDRESS", "LOCATION", "DATE_TIME",
    "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE",
])
```

---

## Response caching

**This is entirely local. No SAVI account, backend, or network access is
required** — it works identically in [Local Mode](#local-mode) or with a real
`savi_key`. Opt in with `enable_cache=True`; if the same call (same messages,
model, and other options) comes in again within `cache_ttl_seconds`, the
provider is never called again and the prior response is returned instantly.

```python
client = SaviOpenAI(api_key="sk-...", local_mode=True, enable_cache=True)

r1 = client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "What are your business hours?"}],
)
# cache miss: real call to OpenAI, billed as normal

r2 = client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "What are your business hours?"}],
)
# cache hit: same fingerprint, within the 300s default TTL -> OpenAI is
# never called; r2 is r1, returned instantly at no additional cost
```

Useful for FAQ-style workloads or an agent loop that keeps re-asking the same
question. Two things worth knowing before you rely on it:

- **Per-process, in-memory only.** The cache lives inside one running Python
  process (`savi/cache.py`'s `ResponseCache`, a plain dict with TTL + LRU
  eviction). It doesn't survive a restart, and if your app runs as multiple
  replicas/pods, each one has its own separate cache — there's no shared or
  distributed cache across instances.
- **Never caches anything flagged as PII**, on both the read and the write
  side, even if `mask_pii=True`. This is a deliberate safety gate on top of
  masking itself: two different real prompts can mask down to an identical
  redacted template (e.g. two different names), and without this gate one
  person's cached answer could otherwise be served to someone else's lookup.

Not to be confused with the LSH fingerprint's *other* use (previous section):
that fingerprint is also sent to SAVI on every call, cached or not, for
prompt-dedup analytics. That's a separate, SAVI-backend feature — it needs a
real account to do anything useful with the fingerprint, whereas the cache
above needs nothing beyond this SDK.

Config: `cache_ttl_seconds` (default `300`), `cache_max_size` (default `1000`,
oldest entries evicted first once full). Not available on the
[auto-instrumentation](#auto-instrumentation-for-multi-agent-hand-offs) path;
use `SaviOpenAI`/`SaviAnthropic`/etc. directly to use it. Also skipped for any
call with `stream=True` (see the streaming note in Quick Start above).

---

## Auto-instrumentation for multi-agent hand-offs

`SaviOpenAI`/`SaviAnthropic` only mask and report calls made through their
own client instance. If your agent hands work off to another agent that
constructs its own plain `openai.OpenAI()` or `anthropic.Anthropic()`
(its own framework's default, a sub-agent library, or just a client you
forgot to wrap), that call is invisible to SAVI by default: no PII
masking, no telemetry.

`enable_auto_instrumentation()` closes that gap by patching the
`openai`/`anthropic` client classes for the whole process, so **any**
client constructed afterward is covered, not just ones you build with
`SaviOpenAI`. Call it once, before any client (yours or a framework's)
gets constructed:

```python
from savi import enable_auto_instrumentation

enable_auto_instrumentation(
    savi_key="sk_...",
    tenant_id="your-tenant-id",
    team_id="engineering",   # optional, default "default"
    mask_pii=True,           # optional; default is "on if Presidio's installed" (see PII masking above) - explicit True here makes a missing Presidio a hard error instead
)

# Any client from here on is covered, even ones SAVI's own code never sees:
import openai
client = openai.OpenAI()  # not SaviOpenAI, still masked + reported
response = client.chat.completions.create(model="gpt-4o", messages=[...])
```

Safe to use alongside `SaviOpenAI`/`SaviAnthropic` in the same process;
a wrapped client's own masking/reporting isn't duplicated by the
process-wide patch.

Scope: chat/message completion calls only, sync and async, for OpenAI and
Anthropic. No embeddings and no response caching on this path (both still
require `SaviOpenAI`/`SaviAnthropic` directly). Only the providers whose
package is actually installed are patched.

Same `savi_key`/`tenant_id`-or-`local_mode` choice as every direct wrapper
above ([Local Mode](#local-mode) applies here too):

```python
enable_auto_instrumentation(local_mode=True)
# same zero-account, zero-network-call behavior as SaviOpenAI(..., local_mode=True),
# applied process-wide instead of to one client
```

---

## Local Mode

No SAVI account, no `SAVI_KEY`, no network call. Everything the wrapper
already computes client-side (tokens, latency, PII flags, LSH fingerprint)
is logged to your own terminal instead of sent anywhere.

```python
from savi import SaviOpenAI

client = SaviOpenAI(api_key="sk-...", local_mode=True)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarise this contract."}],
)
# prints: [savi:local] {"event_id": "...", "provider": "openai", "model": "gpt-4o",
#          "tokens_in": 42, "tokens_out": 118, "latency_ms": 810, ...}
```

**What's included:** tokens in/out, latency, PII flags, LSH fingerprint:
everything the real product captures client-side today.

**What's not included by default:** a dollar `cost_usd` figure. The real
product computes cost server-side against a pricing table SAVI actively
keeps in sync and alerts on staleness for; shipping an equivalent bundled
table here would mean it goes stale silently, with nothing to catch it. If
you want a local `$` estimate anyway, supply your own rates:

```python
client = SaviOpenAI(
    api_key="sk-...", local_mode=True,
    local_pricing={"gpt-4o": (2.50, 10.00)},  # (USD per 1M input, per 1M output tokens)
)
# now includes "cost_usd" in the logged event, computed from the rate you gave it
```

Every provider wrapper supports `local_mode=True` the same way. Local mode
and a real `savi_key`/`tenant_id` are mutually exclusive per client instance:
pass one or the other, never both, and never neither (the constructor
raises a clear error if you pass neither).

---

## Agent tracing with SpanContext

Wrap multi-step agent workflows to capture call chains and correlate calls
into one logical run:

```python
from savi import SaviOpenAI, SpanContext

client = SaviOpenAI(...)

with SpanContext(workflow_id="kyc-review", agent_id="doc-extractor") as span:
    response = client.chat.completions.create(...)
    # event is linked to this span - this part works the same in local_mode
```

`SpanContext` itself is pure client-side correlation (no network call), so it
works identically with or without a real account. One thing worth being
precise about: **loop-detection *enforcement* (the "fires if agent_id exceeds
the velocity threshold" behavior) is a SAVI backend feature — the circuit
breaker — and requires a real account.** `SpanContext` only tags events with
the `agent_id`/`parent_span_id` the backend needs to evaluate that; it
doesn't detect loops itself, and nothing fires in `local_mode=True` since
there's no backend watching. If you want loop detection with zero account
(fully local, in-process), that's exactly what
[`savi-loop-guard`](https://pypi.org/project/savi-loop-guard/) is for — a
separate, standalone package (`pip install savi-loop-guard`), unrelated to
this SDK's account/local-mode split.

`SpanContext` also accepts `user_id`, the end user on whose behalf the call was
made. This is unrelated to `team_id`/`tenant_id` on the client (which are fixed
for the lifetime of the client instance) and is meant for services that make
calls on behalf of many different people, e.g. a backend handling requests for
your whole user base through one shared client:

```python
with SpanContext(workflow_id="support-chat", user_id=current_request.user.email):
    response = client.chat.completions.create(...)
    # per-user attribution now available on this event, e.g. for governance
    # policies scoped to a specific person rather than just team/tenant
```

### Correlating a hand-off across processes

`SpanContext` is per-process. If Agent A hands work to Agent B over HTTP, or by
spawning it as a subprocess, Agent B's calls are invisible to Agent A's run by
default. `to_headers()`/`from_headers()` (HTTP) and `to_env()`/`from_env()`
(subprocess) carry the run across that hop, so both sides' calls attribute to one
logical agent run instead of two disconnected ones.

```python
# Agent A (the caller)
with SpanContext(workflow_id="kyc-review", agent_id="intake") as span:
    response = requests.post(
        "http://agent-b.internal/handle",
        json={...},
        headers=span.to_headers(),   # attach alongside your own headers
    )
```

```python
# Agent B (the process that receives the hand-off)
resumed = SpanContext.from_headers(request.headers) or SpanContext(workflow_id="unknown")
with resumed:
    response = client.chat.completions.create(...)
    # this call's agent_run_id matches Agent A's: one run, not two
```

For a subprocess instead of an HTTP call, use `to_env()`/`from_env()` the same way:

```python
# Parent process
with SpanContext(workflow_id="batch-job", agent_id="worker") as span:
    subprocess.run(["python", "worker.py"], env={**os.environ, **span.to_env()})
```

```python
# worker.py (the child process)
with SpanContext.from_env() or SpanContext(workflow_id="unknown"):
    ...
```

`from_headers()`/`from_env()` return `None` if there's nothing to resume (no
header/env var, or it's malformed); they never raise. If your hand-off goes
through MCP (`mcp.ClientSession.call_tool`) instead, call
`savi.enable_mcp_instrumentation()` once at startup and this correlation
happens automatically, stamped onto the tool call's own `_meta` field. You
only need `to_headers()`/`to_env()` yourself for a hand-off SAVI can't see
into on its own (a plain HTTP call or subprocess spawn between two of your
own services).

---

## Reporting real outcomes (cost-per-outcome attribution)

**Requires a real SAVI account** (a `SAVI_KEY`, not `local_mode=True`) — this
posts to SAVI's backend, which is what computes acceptance-rate and
cost-per-outcome numbers from it. There's no local-mode equivalent, since
there's nothing running locally to compute those numbers against.

SAVI's acceptance-rate and cost-per-outcome numbers are only as good as what feeds
them. `POST /v1/outcomes` is built to be called by **whatever system already knows
whether an agent's work was actually accepted** (a ticket-reopen, a case override, a
document resubmission), not a rating widget shown to anyone. There's no SDK helper
for this yet; it's a plain HTTP call, usually made from the *other* system (your
support desk, your case-management tool), not from wherever the agent itself ran.

```bash
curl -X POST https://api.datagras.com/v1/outcomes \
  -H "Authorization: Bearer $SAVI_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_run_id": "run_8f2c1a",
    "outcome_status": "accepted_first_pass",
    "outcome_source": "downstream_system",
    "outcome_timestamp": "2026-09-07T03:14:00Z",
    "workflow_id": "kyc-review"
  }'
```

`agent_run_id` is whatever value the SDK attached to the original call (from
`SpanContext`, or the `agent_run_id` field on a Direct-ingest event): the thing you're
reporting an outcome *for*. `outcome_status` is one of:
`accepted_first_pass`, `accepted_after_correction`, `correctly_escalated`, `rejected`,
`abandoned`, `ineligible`, `reopened`, `incident`. `outcome_source` is
`"human"`, `"automated_test"`, or `"downstream_system"`; use `"downstream_system"`
for all three examples below, since none of them ask a person to rate anything.

A status change always resets the acceptance window (a run reported `accepted_first_pass`
today and `reopened` next week correctly loses its earlier "accepted" credit), so call
this again with the new status rather than trying to patch or delete the old report.

**Three real integration patterns**, each firing from an event that was already going
to happen in your own system:

**1. A support ticket gets reopened (Zendesk, or any helpdesk with reopen webhooks)**

```python
# In your own Zendesk webhook handler, not the SAVI SDK
import requests, os

def on_zendesk_ticket_reopened(ticket):
    requests.post(
        "https://api.datagras.com/v1/outcomes",
        headers={"Authorization": f"Bearer {os.environ['SAVI_KEY']}"},
        json={
            "agent_run_id": ticket.custom_fields["savi_agent_run_id"],
            "outcome_status": "reopened",
            "outcome_source": "downstream_system",
            "outcome_timestamp": ticket.reopened_at.isoformat(),
            "workflow_id": "support-chat",
        },
    )
```

The one piece of setup this needs on your side: store the originating `agent_run_id`
somewhere on the ticket when it's first created (a custom field, as above), so the
reopen handler has something to report an outcome against.

**2. A case-management system flags a case for review (ServiceNow, or similar)**

```python
def on_servicenow_case_needs_review(case):
    requests.post(
        "https://api.datagras.com/v1/outcomes",
        headers={"Authorization": f"Bearer {os.environ['SAVI_KEY']}"},
        json={
            "agent_run_id": case.savi_agent_run_id,
            "outcome_status": "correctly_escalated" if case.escalation_was_warranted
                               else "rejected",
            "outcome_source": "downstream_system",
            "outcome_timestamp": case.flagged_at.isoformat(),
        },
    )
```

**3. A KYC decision gets overridden in your case-management system**

```python
def on_kyc_decision_overridden(decision):
    requests.post(
        "https://api.datagras.com/v1/outcomes",
        headers={"Authorization": f"Bearer {os.environ['SAVI_KEY']}"},
        json={
            "agent_run_id": decision.savi_agent_run_id,
            "outcome_status": "accepted_after_correction",
            "outcome_source": "downstream_system",
            "outcome_timestamp": decision.overridden_at.isoformat(),
            "workflow_id": "kyc-review",
            "rework_minutes": decision.reviewer_minutes_spent,
        },
    )
```

`rework_minutes` and `value_score` (0-5) are both optional; include `rework_minutes`
wherever your system already tracks how long a human spent fixing the agent's output;
it's what turns "accepted after correction" from a label into a real cost-of-rework
number on the Finance dashboard.

If you don't currently have any system reporting outcomes and aren't sure where to
start, the honest default is: whichever human review step already exists downstream of
the agent (a QA queue, an approval step, a support escalation) is almost always the
right place to add one `POST /v1/outcomes` call. You're not building new review
infrastructure, just reporting a decision that review step was already making.

---

## Outcome Explainer (opt-in judge: explains *why* an outcome happened)

Reported outcomes (above) say *what* happened: accepted, rejected, reopened.
`OutcomeExplainer` optionally adds *why*, by running a judge model of your choice
over the actual prompt/response text for that run. **This reads content, so it's
off by default in two places**: nothing runs unless you call it, and even if you
do, SAVI's backend silently drops the result unless the `outcome_explainer`
feature flag is turned on for your tenant in Settings.

**This text is never sent to SAVI. Only the resulting explanation and a 0-5
score are, and that's all SAVI stores or displays.** What happens to the text
itself depends on which backend you choose:

- **`backend="local"`**: the text never leaves your machine at all. Points at
  any Ollama-compatible server you run (default `http://localhost:11434`); no
  extra install needed, `OutcomeExplainer` talks to it over plain HTTP.
- **`backend="byo_cloud"`**: the text goes only to the provider you name
  (`provider="openai"`/`"anthropic"`, using *your own* API key), the same
  provider already serving your app, never SAVI. Check that provider's own
  current data-retention terms if that distinction matters for your compliance
  posture; SAVI doesn't control or guarantee it. Uses this package's existing
  `savi-sdk[openai]`/`savi-sdk[anthropic]` extras, nothing new to install.

```python
from savi import OutcomeExplainer

judge = OutcomeExplainer()

explanation, score = judge.explain(
    prompt="Summarise this contract.",
    response="The contract renews automatically every 90 days.",
    outcome_status="rejected",
    backend="local",
    model="qwen2.5-coder:14b",   # any model you've pulled, SAVI ships no default
)
# explanation, score = "The renewal term is wrong: the contract says 12 months, not 90 days.", 1

requests.post(
    "https://api.datagras.com/v1/outcomes",
    headers={"Authorization": f"Bearer {os.environ['SAVI_KEY']}"},
    json={
        "agent_run_id": "run_8f2c1a",
        "outcome_status": "rejected",
        "outcome_source": "downstream_system",
        "outcome_timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_score": score,
        "judge_explanation": explanation,
    },
)
```

**Choosing a judge model.** SAVI doesn't maintain or recommend a specific model;
this is guidance, not a shipped default. Frontier models generally track human
judgment better than small local ones, but nothing here is SAVI-verified. Using
the *same* model family to judge that judges itself tends to inflate its own
scores, so a reasonable default is picking a different family for judging than
for the agent's own calls. Smaller local models trade some of that accuracy for
the `local` backend's stronger privacy guarantee. That tradeoff is yours to make,
per call, via `model=`.

**Before trusting these scores in a decision.** A `judge_score` is only as good
as the model and prompt behind it. Before wiring scores into an automated
decision (e.g. "auto-flag anything under 3 for review"), hand-grade a small
sample of your own runs, compare against the judge's scores, and only automate
once that comparison looks reasonable. This calibration step is your
responsibility, same as it would be for any eval tool. `explanation` carries the
actual reasoning; `score` (0 = completely wrong/unhelpful, 5 = completely
correct/helpful) is a coarse summary of it, not a substitute for reading it.

`explain()` never raises. On any internal failure (backend unreachable, bad
response, anything) it returns `(None, None)`, matching PII masking's fail-safe
behavior. Omit `judge_score`/`judge_explanation` from your `POST /v1/outcomes`
body entirely if you get `(None, None)` back, same as any other optional field.

---

## Alerting to your own systems (generic webhook)

**Requires a real SAVI account.** Budget breaches/spend spikes are computed
by SAVI's backend watching your accumulated spend over time, something a
`local_mode=True` client has no concept of (it has no memory of past calls at
all, let alone a budget to compare against) — so there's no local equivalent
to this feature, and the request below will fail without a valid `SAVI_KEY`.

SAVI ships three fixed alert channels (Slack, email, PagerDuty). If your
team runs its own incident tooling (an internal ops bot, ServiceNow, a
custom automation), configure a webhook instead, so your own systems get a
machine-readable event rather than requiring someone to read a Slack message
and re-key it.

```bash
curl -X PUT https://api.datagras.com/v1/settings/alert-destinations \
  -H "Authorization: Bearer $SAVI_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_url": "https://your-systems.example.com/savi-alerts",
    "webhook_secret": "a-secret-you-generate-yourself-16-chars-min"
  }'
```

`webhook_url` must be `https://` and resolve to a public address. SAVI
rejects a URL pointing at a private/internal network at both save time and
every dispatch. `webhook_secret` is optional; when set, every request is
HMAC-SHA256-signed.

Once configured, budget breaches, warnings, forecasts, and spend spikes
(more alert types are on the roadmap) POST this shape to your endpoint:

```json
{
  "event_type": "budget_breach",
  "severity": "needs_action",
  "tenant_id": "ten_yourcompany",
  "summary": "Budget breach: engineering — Spent $512.00 of $500.00 (102%) — monthly",
  "occurred_at": "2026-09-07T03:14:00+00:00",
  "incident_url": null,
  "metadata": {"pct_used": 1.024, "cap_usd": 500.0, "current_spend": 512.0,
               "period": "monthly", "team_id": "engineering"}
}
```

`severity` is machine-readable on purpose: `"needs_action"` means a human
should look at it, `"auto_resolved"` (used by future alert types where SAVI
itself resolves the situation, e.g. a stopped loop) means it's informational.
Route on this field rather than parsing `summary`.

Verifying the signature (Python example, adapt to your own stack):

```python
import hashlib, hmac

def verify_savi_webhook(raw_body: bytes, signature_header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)
```

---

## Exporting to your existing Datadog / Grafana / Prometheus

**Requires a real SAVI account.** This endpoint reads aggregated metrics back
*out of* SAVI's backend — in `local_mode=True` there's no backend holding
your data to export, so there's nothing for this to expose.

`GET /v1/metrics/prometheus` exposes cost, token, latency, anomaly/quality-
flag, budget-pool, and governance-policy signals in standard Prometheus
text format, so they show up in whatever you already use to watch metrics
instead of requiring a separate SAVI dashboard tab. Check with your SAVI
contact for the full metric list and example scrape configs for Prometheus,
Grafana Agent, and Datadog's OpenMetrics check.

---

## What happens if SAVI is unreachable

Telemetry delivery **never blocks or raises in your application code**,
regardless of whether the SAVI backend is up. `emit()` only appends to an
in-memory ring buffer (10,000 events, oldest dropped once full); a background
thread batches and POSTs it every `flush_interval_secs` (default 5s).

- A permanent rejection (401/403/422, e.g. a dead key) is logged
  (`savi-sdk: ingest rejected ...`) and the batch is dropped, since retrying
  an invalid key won't fix itself.
- A transient failure (connection error, timeout, or 408/429/500/502/503/504)
  is logged and the batch is requeued to try again on the next flush tick.
- A prolonged outage can still lose events once the 10,000-event ring buffer
  fills, since it's bounded, not unbounded queueing. This is a real limit,
  not just a theoretical one, on a very long outage under sustained volume.
- On process exit, whatever's still buffered is flushed once, best-effort
  (`atexit`), so a normal shutdown doesn't lose the last few seconds of data.

None of this applies in [Local Mode](#local-mode): there's no network call to
fail in the first place. And none of it applies to the [response
cache](#response-caching) either — that's a separate, always-local mechanism
that doesn't touch the network regardless of `savi_key`/`local_mode`.

---

## Configuration

| Parameter | Env var | Default |
|---|---|---|
| `savi_key` | `SAVI_KEY` | required, unless `local_mode=True` |
| `endpoint` | `SAVI_ENDPOINT` | `https://api.datagras.com` (ignored when `local_mode=True` - never contacted) |
| `tenant_id` | — | required, unless `local_mode=True` |
| `team_id` | — | `"default"` |
| `mask_pii` | — | on if Presidio's installed, else unmasked with a warning (see [PII masking](#pii-masking)); pass `True` explicitly to make a missing Presidio a hard error, `False` to disable outright |
| `local_mode` | — | `False` (see [Local Mode](#local-mode)) |
| `local_pricing` | — | `None` (only used when `local_mode=True`) |
| `batch_size` | — | `100` (events per telemetry POST; ignored when `local_mode=True`; see [above](#what-happens-if-savi-is-unreachable)) |
| `flush_interval_secs` | — | `5.0` (how often the background thread flushes; ignored when `local_mode=True`) |
| `enable_cache` | — | `False` (see [Response caching](#response-caching), 100% local) |
| `cache_ttl_seconds` | — | `300.0` (only used when `enable_cache=True`) |
| `cache_max_size` | — | `1000` (oldest entries evicted first; only used when `enable_cache=True`) |

---

## License

MIT. See [LICENSE](LICENSE).
