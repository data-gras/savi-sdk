"""OutcomeExplainer, opt-in, customer-side "why was this outcome X" judge.

Runs entirely on the customer's own side, prompt/response content never
reaches SAVI. Two backends, no default for either:

  - "local": any Ollama-compatible server (default http://localhost:11434).
  - "byo_cloud": the customer's own OpenAI/Anthropic/etc. key, via that
    provider's official SDK (savi-sdk[openai], savi-sdk[anthropic], ...).

The tenant's `outcome_explainer` feature flag must also be enabled
server-side for judge_score/judge_explanation to persist once posted to
POST /v1/outcomes; this module computes them regardless, but the backend
drops both fields if the flag is off.

See the README's "Choosing a judge model" section before wiring a score
into automated decisions; this module doesn't verify judging quality.
"""
import json
import logging
from typing import Literal, Optional

import httpx

_log = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"

_JUDGE_PROMPT_TEMPLATE = """You are grading whether an AI assistant's response was helpful and correct. A downstream system recorded this outcome for it: {outcome_status}.

Prompt given to the assistant:
{prompt}

Assistant's response:
{response}

Respond with ONLY a JSON object of the exact shape {{"score": <int 0-5>, "explanation": "<one or two sentences>"}}. 0 means completely wrong or unhelpful, 5 means completely correct and helpful. No other text, no code fence."""


def _parse_judge_reply(text: str) -> "tuple[str | None, int | None]":
    """Best-effort JSON extraction, models sometimes wrap the JSON in prose
    or a code fence despite instructions. Returns (None, None) on anything
    unparseable rather than raising, matching PiiMasker.mask()'s fail-safe
    contract."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        obj = json.loads(text[start:end])
        score = int(obj["score"])
        if not 0 <= score <= 5:
            return None, None
        explanation = str(obj.get("explanation", "")).strip()[:2000] or None
        return explanation, score
    except Exception:
        return None, None


class OutcomeExplainer:
    """Explains a reported outcome using a judge model the customer
    chooses and runs. See module docstring for the privacy model."""

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout

    def explain(
        self,
        prompt: str,
        response: str,
        outcome_status: str,
        backend: Literal["local", "byo_cloud"],
        model: str,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        ollama_url: str = _DEFAULT_OLLAMA_URL,
        **provider_kwargs,
    ) -> "tuple[str | None, int | None]":
        """
        Returns (explanation_text, judge_score), (None, None) on any
        internal failure, never raises. `model` is always customer-supplied;
        SAVI ships no default and doesn't verify judging quality for any
        model, see the README before trusting a score in an automated
        decision.
        """
        try:
            judge_prompt = _JUDGE_PROMPT_TEMPLATE.format(
                outcome_status=outcome_status, prompt=prompt, response=response,
            )
            if backend == "local":
                reply = self._call_local(judge_prompt, model, ollama_url)
            elif backend == "byo_cloud":
                reply = self._call_byo_cloud(
                    judge_prompt, model, provider, api_key, **provider_kwargs
                )
            else:
                return None, None
            return _parse_judge_reply(reply)
        except Exception as exc:
            # DEBUG, not WARNING: opt-in feature, but still logged so
            # "judging is off" is distinguishable from "misconfigured"
            # when the customer enables debug logging.
            _log.debug("savi.judge: explain() failed - %s: %s", type(exc).__name__, exc)
            return None, None

    def _call_local(self, judge_prompt: str, model: str, ollama_url: str) -> str:
        """Ollama's own REST API (POST /api/generate), any model the
        customer has already pulled locally, named by them, never assumed
        by SAVI. No network call beyond this localhost/self-hosted address."""
        resp = httpx.post(
            f"{ollama_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": judge_prompt, "stream": False},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def _call_byo_cloud(
        self,
        judge_prompt: str,
        model: str,
        provider: Optional[str],
        api_key: Optional[str],
        **provider_kwargs,
    ) -> str:
        """The customer's own key, sent to the provider they name, never to
        SAVI. Reuses this package's existing per-provider optional extras
        (savi-sdk[openai]/[anthropic]) rather than a new one."""
        if not provider or not api_key:
            raise ValueError("byo_cloud backend requires provider= and api_key=")
        if provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ValueError(
                    "byo_cloud provider='openai' requires the openai package - "
                    "install with: pip install 'savi-sdk[openai]'"
                )

            client = OpenAI(api_key=api_key, **provider_kwargs)
            completion = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": judge_prompt}],
            )
            return completion.choices[0].message.content or ""
        if provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ValueError(
                    "byo_cloud provider='anthropic' requires the anthropic package - "
                    "install with: pip install 'savi-sdk[anthropic]'"
                )

            client = Anthropic(api_key=api_key, **provider_kwargs)
            message = client.messages.create(
                model=model, max_tokens=500,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            return message.content[0].text if message.content else ""
        raise ValueError(
            f"Unsupported byo_cloud provider {provider!r}, supported today: "
            "'openai', 'anthropic'. Open an issue if you need another provider."
        )
