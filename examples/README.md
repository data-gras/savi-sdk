# Examples

Runnable scripts, ordered from "needs nothing" to "needs a real provider key".
None of them need a SAVI account.

| Script | Needs |
|---|---|
| [`01_verify_webhook_signature.py`](01_verify_webhook_signature.py) | nothing — stdlib only |
| [`02_agent_tracing_handoff.py`](02_agent_tracing_handoff.py) | `pip install savi-sdk` |
| [`03_quickstart_local_mode.py`](03_quickstart_local_mode.py) | `pip install "savi-sdk[openai]"` + `OPENAI_API_KEY` |
| [`04_response_caching.py`](04_response_caching.py) | `pip install "savi-sdk[openai]"` + `OPENAI_API_KEY` |
| [`05_pii_masking.py`](05_pii_masking.py) | `pip install "savi-sdk[openai,pii]"` + `OPENAI_API_KEY` + `python -m spacy download en_core_web_lg` |

Each script prints what it's doing as it runs, and exits with a clear message
instead of a stack trace if a prerequisite is missing. Run any of them with:

```bash
python examples/03_quickstart_local_mode.py
```
