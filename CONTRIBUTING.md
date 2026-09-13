# Contributing to savi-sdk

Thanks for taking a look. This SDK wraps several LLM provider clients
(OpenAI, Anthropic, Azure OpenAI, Bedrock, Cohere, Mistral, Vertex AI) with
the same contract each time: capture cost/latency/PII signal client-side,
emit it (to SAVI, or nowhere, see Local Mode in the README), and never get
in the way of the underlying provider call.

## Dev setup

```bash
git clone https://github.com/data-gras/savi-sdk.git
cd savi-sdk
pip install -e ".[dev]"
pytest -q
```

`[dev]` installs the provider SDKs needed to run the full suite
(`openai`, `anthropic`, etc.) plus `pytest`/`pytest-asyncio`. The test suite
never makes a real network call to a provider or to SAVI. Provider clients
are mocked or injected (`_client=...`/`_collector=...` constructor params
exist specifically for this).

If you're only touching one provider wrapper, `pip install -e ".[openai]"`
(or whichever extra) is enough to import it, but you'll want `[dev]` to run
the shared tests (`context`, `pii`, `local_mode`, etc.).

## Before opening a PR

- **Never commit a real API key**, not even a test/sandbox one, not even
  in a test fixture. Every test in this repo uses an obviously-fake string
  (`"test-key"`, `"sk_test_..."`).
- Add or update a test. `tests/` mirrors the module layout closely, e.g. a
  change to `savi/local.py` belongs with `tests/test_local_mode.py`.
- If you're adding a new provider wrapper: match the existing shape
  (constructor accepts `local_mode`/`local_pricing` like every other
  wrapper does, see `savi/local.py`'s `resolve_emitter()`, the single
  validation point every wrapper calls into) rather than inventing a
  parallel path.
- If you're changing what gets sent in an emitted event, check
  `savi/local.py`'s docstring and `PiiMasker` in `savi/pii.py`: this SDK's
  privacy contract (mask before it ever leaves the caller's machine) is a
  hard line, not a preference.
- Run `pytest -q` locally before pushing. CI runs the same command.

## Reporting a bug

Which provider wrapper, the SDK version, and (with real secrets/PII
redacted) the smallest code snippet that reproduces it. A stack trace alone
is rarely enough; provider client versions vary a lot.

## Security issues

Don't open a public issue for a security or privacy concern, see
[SECURITY.md](SECURITY.md).

## License

By contributing, you agree your contribution is licensed under this
project's [MIT License](LICENSE).
