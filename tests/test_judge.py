import logging
import pytest
from unittest.mock import MagicMock, patch
from savi.judge import OutcomeExplainer, _parse_judge_reply


# ---------------------------------------------------------------------------
# _parse_judge_reply — no network needed
# ---------------------------------------------------------------------------

def test_parse_judge_reply_valid_json():
    explanation, score = _parse_judge_reply(
        '{"score": 4, "explanation": "Mostly correct, minor wording issue."}'
    )
    assert score == 4
    assert explanation == "Mostly correct, minor wording issue."


def test_parse_judge_reply_json_wrapped_in_prose():
    # Models sometimes ignore "no other text" — extraction should still work.
    text = 'Sure, here is my grading:\n{"score": 1, "explanation": "Wrong answer."}\nHope that helps!'
    explanation, score = _parse_judge_reply(text)
    assert score == 1
    assert explanation == "Wrong answer."


def test_parse_judge_reply_out_of_range_score_returns_none():
    explanation, score = _parse_judge_reply('{"score": 9, "explanation": "x"}')
    assert (explanation, score) == (None, None)


def test_parse_judge_reply_malformed_json_returns_none():
    explanation, score = _parse_judge_reply("not json at all")
    assert (explanation, score) == (None, None)


def test_parse_judge_reply_missing_score_key_returns_none():
    explanation, score = _parse_judge_reply('{"explanation": "no score field"}')
    assert (explanation, score) == (None, None)


def test_parse_judge_reply_explanation_truncated_at_2000_chars():
    long_explanation = "x" * 3000
    explanation, score = _parse_judge_reply(
        '{"score": 3, "explanation": "%s"}' % long_explanation
    )
    assert score == 3
    assert len(explanation) == 2000


# ---------------------------------------------------------------------------
# OutcomeExplainer.explain() — fail-safe contract, same as PiiMasker.mask()
# ---------------------------------------------------------------------------

def test_explain_unknown_backend_returns_none_none():
    judge = OutcomeExplainer()
    explanation, score = judge.explain(
        prompt="p", response="r", outcome_status="rejected",
        backend="not_a_real_backend", model="whatever",
    )
    assert (explanation, score) == (None, None)


def test_explain_byo_cloud_missing_credentials_returns_none_none():
    judge = OutcomeExplainer()
    explanation, score = judge.explain(
        prompt="p", response="r", outcome_status="rejected",
        backend="byo_cloud", model="gpt-4o-mini",
        # provider/api_key deliberately omitted
    )
    assert (explanation, score) == (None, None)


def test_explain_byo_cloud_unsupported_provider_returns_none_none():
    judge = OutcomeExplainer()
    explanation, score = judge.explain(
        prompt="p", response="r", outcome_status="rejected",
        backend="byo_cloud", model="whatever",
        provider="not_a_real_provider", api_key="sk-fake",
    )
    assert (explanation, score) == (None, None)


def test_explain_local_backend_network_error_returns_none_none():
    judge = OutcomeExplainer()
    with patch("savi.judge.httpx.post", side_effect=ConnectionError("no ollama running")):
        explanation, score = judge.explain(
            prompt="p", response="r", outcome_status="rejected",
            backend="local", model="qwen2.5-coder:14b",
        )
    assert (explanation, score) == (None, None)


def test_explain_local_backend_success():
    judge = OutcomeExplainer()
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "response": '{"score": 2, "explanation": "Cited a policy that does not exist."}'
    }
    fake_response.raise_for_status.return_value = None
    with patch("savi.judge.httpx.post", return_value=fake_response) as mock_post:
        explanation, score = judge.explain(
            prompt="What's our refund policy?",
            response="You get a full refund within 90 days per Policy 7.2.",
            outcome_status="rejected",
            backend="local",
            model="qwen2.5-coder:14b",
            ollama_url="http://localhost:11434",
        )
    assert score == 2
    assert explanation == "Cited a policy that does not exist."
    called_url = mock_post.call_args.args[0]
    assert called_url == "http://localhost:11434/api/generate"
    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload["model"] == "qwen2.5-coder:14b"
    assert called_payload["stream"] is False


def test_explain_byo_cloud_openai_success():
    judge = OutcomeExplainer()
    fake_message = MagicMock()
    fake_message.content = '{"score": 5, "explanation": "Correct and complete."}'
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=fake_message)]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_completion

    with patch("openai.OpenAI", return_value=fake_client):
        explanation, score = judge.explain(
            prompt="p", response="r", outcome_status="accepted_first_pass",
            backend="byo_cloud", model="gpt-4o-mini",
            provider="openai", api_key="sk-fake",
        )
    assert score == 5
    assert explanation == "Correct and complete."


# ---------------------------------------------------------------------------
# byo_cloud with the provider SDK not installed — a customer who forgot
# `pip install 'savi-sdk[openai]'` (or [anthropic]) would otherwise see a bare
# ImportError swallowed into a silent (None, None), indistinguishable from
# "judging is off" or a transient network failure.
# ---------------------------------------------------------------------------

def test_call_byo_cloud_openai_missing_package_raises_clear_value_error():
    judge = OutcomeExplainer()
    with patch.dict("sys.modules", {"openai": None}):
        with pytest.raises(ValueError, match=r"savi-sdk\[openai\]"):
            judge._call_byo_cloud("prompt", "gpt-4o-mini", "openai", "sk-fake")


def test_call_byo_cloud_anthropic_missing_package_raises_clear_value_error():
    judge = OutcomeExplainer()
    with patch.dict("sys.modules", {"anthropic": None}):
        with pytest.raises(ValueError, match=r"savi-sdk\[anthropic\]"):
            judge._call_byo_cloud("prompt", "claude-sonnet-4-6", "anthropic", "sk-fake")


def test_explain_logs_failure_reason_at_debug_level(caplog):
    """The outer except in explain() is fail-safe by design (never raises),
    but must not be silent past the point of self-diagnosis - a customer
    with debug logging on should be able to tell 'judging is misconfigured'
    apart from 'judging is off'."""
    judge = OutcomeExplainer()
    with caplog.at_level(logging.DEBUG, logger="savi.judge"):
        with patch("savi.judge.httpx.post", side_effect=ConnectionError("no ollama running")):
            judge.explain(
                prompt="p", response="r", outcome_status="rejected",
                backend="local", model="qwen2.5-coder:14b",
            )
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG
    assert "ConnectionError" in caplog.records[0].message


def test_explain_byo_cloud_missing_openai_package_logged_with_clear_message(caplog):
    judge = OutcomeExplainer()
    with caplog.at_level(logging.DEBUG, logger="savi.judge"):
        with patch.dict("sys.modules", {"openai": None}):
            explanation, score = judge.explain(
                prompt="p", response="r", outcome_status="rejected",
                backend="byo_cloud", model="gpt-4o-mini",
                provider="openai", api_key="sk-fake",
            )
    assert (explanation, score) == (None, None)
    assert any("savi-sdk[openai]" in r.message for r in caplog.records)
