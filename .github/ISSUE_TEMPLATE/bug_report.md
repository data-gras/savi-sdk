---
name: Bug report
about: Something in the SDK isn't behaving as documented
title: ""
labels: bug
---

**Which wrapper**
`SaviOpenAI` / `SaviAsyncOpenAI` / `SaviAnthropic` / `SaviAzureOpenAI` /
`SaviBedrockRuntime` / `SaviCohere` / `SaviMistral` / `SaviVertexAI` / other

**What happened**
A clear description of the incorrect behavior.

**Minimal repro**
The smallest code snippet that reproduces it, with any real API key or
PII replaced by an obviously-fake placeholder:

```python
from savi import SaviOpenAI
# ...
```

**Expected vs. actual**


**Environment**
- `savi-sdk` version:
- Provider SDK + version (e.g. `openai==1.x`):
- Python version:
- Local mode or a real `savi_key`/`tenant_id`?

⚠️ Please do not paste real API keys, tokens, or PII into an issue,
redact or use a placeholder.
