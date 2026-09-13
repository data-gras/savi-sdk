# Security Policy

## Scope

This SDK sits directly in the path of API keys and, often, sensitive
end-user data (the whole point of `PiiMasker` and local mode is handling
that data safely). Security and privacy reports here are taken seriously
and get priority over a normal bug report.

Particularly in scope:

- Anything that could cause a real credential (a provider API key, a
  `savi_key`) to be logged, cached, or sent somewhere it shouldn't be.
- A `PiiMasker` false negative on one of its documented entity types (PII
  that should have been masked but wasn't): this is a privacy defect, not
  just a correctness one.
- Local mode (`local_mode=True`) making a network call under any
  condition: the entire point of that mode is that it never does.
- A vulnerability in how the SDK handles data from an LLM provider's
  response (e.g. an untrusted-content injection path).

## Supported versions

Only the latest published version on PyPI is supported.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a security or privacy
concern.

Instead, email **contact@datagras.com** with a description of the issue,
steps to reproduce, and, if it involves real data, confirmation that
you've redacted or synthesized it rather than pasting anything real. You
should get an acknowledgment within a few business days. We'll coordinate
on disclosure timing with you once a fix is confirmed.

## Out of scope

- Vulnerabilities in the underlying provider SDKs this package wraps
  (`openai`, `anthropic`, `boto3`, etc.): please report those upstream.
- Issues that require a compromised API key to already be leaked by some
  other means (not by this SDK).
