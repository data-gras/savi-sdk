import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from savi import local, local_report
from savi.local import LocalEventEmitter
from savi.openai import SaviOpenAI


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SAVI_LOCAL_PRICING", raising=False)
    local_report.stop()
    local.set_default_pricing(None)
    yield
    local_report.stop()
    local.set_default_pricing(None)


def _event(**kw):
    base = {"provider": "openai", "model": "gpt-4o-mini-2024-07-18", "tokens_in": 1_000_000, "tokens_out": 0, "latency_ms": 500,
            "timestamp_utc": "2026-10-05T07:10:00+00:00", "pii_flagged": False, "is_cache_hit": False}
    base.update(kw)
    return base


def _run(events, **start_kwargs):
    local_report.start(save=None, **start_kwargs)
    emitter = LocalEventEmitter()
    for e in events:
        emitter.emit(e)
    return emitter


def test_start_prints_readable_lines_and_not_raw_json(capsys):
    _run([_event()], pricing={"gpt-4o-mini": (0.15, 0.60)})
    out = capsys.readouterr().out
    assert "[ OK  ]" in out and "gpt-4o-mini-2024-07-18" in out and "~$0.15" in out
    assert "event_id" not in out and "{" not in out


def test_stop_brings_back_the_sdk_console_line(caplog):
    local_report.start(save=None)
    local_report.stop()
    with caplog.at_level(logging.INFO, logger="savi.local"):
        LocalEventEmitter().emit(_event())
    assert caplog.records and caplog.records[0].getMessage().startswith("[savi:local]")


def test_quiet_prints_no_call_lines(capsys):
    _run([_event()], quiet=True)
    assert capsys.readouterr().out == ""


def test_prices_from_a_file_ignore_comment_keys(tmp_path):
    f = tmp_path / "prices.json"
    f.write_text(json.dumps({"_note": "my prices", "gpt-4o-mini": [0.15, 0.60]}), encoding="utf-8")
    _run([_event()], pricing=str(f))
    e = local_report._run.events[0]
    assert e["cost_usd"] == pytest.approx(0.15) and e["cost_estimated"] is True
    assert e["cost_rate_key"] == "gpt-4o-mini" and e["cost_rates"] == [0.15, 0.60]


def test_a_broken_prices_file_explains_what_is_wrong(tmp_path):
    f = tmp_path / "prices.json"
    f.write_text('{"gpt-4o": [1, 2],}', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        local_report.start(pricing=str(f), save=None)


@pytest.mark.parametrize("body, message", [
    ('{"gpt-4o": 5}', "two numbers"),
    ('{"gpt-4o": [1, "x"]}', "two numbers"),
    ('{"gpt-4o": [-1, 2]}', "negative"),
    ('[1, 2]', "one JSON object"),
])
def test_wrong_price_shapes_get_clear_errors(tmp_path, body, message):
    f = tmp_path / "prices.json"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        local_report.start(pricing=str(f), save=None)


def test_a_missing_prices_file_is_named_in_the_error():
    with pytest.raises(ValueError, match="not found: nope.json"):
        local_report.start(pricing="nope.json", save=None)


def test_prices_can_come_from_an_environment_variable(tmp_path, monkeypatch):
    f = tmp_path / "prices.json"
    f.write_text(json.dumps({"gpt-4o-mini": [0.15, 0.60]}), encoding="utf-8")
    monkeypatch.setenv("SAVI_LOCAL_PRICING", str(f))
    _run([_event()])
    assert local_report._run.events[0]["cost_usd"] == pytest.approx(0.15)


def test_a_clients_own_price_beats_the_shared_default():
    local.set_default_pricing({"gpt-4o-mini": (0.15, 0.60)})
    seen = []
    local.add_listener(seen.append)
    try:
        LocalEventEmitter(local_pricing={"gpt-4o-mini": (1.0, 1.0)}).emit(_event())
    finally:
        local.remove_listener(seen.append)
    assert seen[0]["cost_usd"] == pytest.approx(1.0)


def test_summary_says_how_many_calls_have_a_price():
    _run([_event(), _event(model="mistral-small-latest"), _event(is_error=True, tokens_in=0)], pricing={"gpt-4o-mini": (0.15, 0.60)})
    s = local_report.summarize()
    assert (s["calls"], s["errors"], s["billable_calls"], s["priced_calls"]) == (3, 1, 2, 1)
    assert s["unpriced_models"] == ["mistral-small-latest"]
    assert "1 of 2 calls" in local_report._coverage(s) and "mistral-small-latest" in local_report._coverage(s)


def test_summary_with_no_prices_says_so():
    _run([_event()])
    assert local_report.summarize()["cost_usd"] is None
    assert local_report._coverage(local_report.summarize()) == "No prices added yet."


def test_cache_hits_cost_nothing_and_failures_have_no_cost():
    _run([_event(), _event(is_cache_hit=True), _event(is_error=True, error_code="timeout")], pricing={"gpt-4o-mini": (0.15, 0.60)})
    s = local_report.summarize()
    assert s["cost_usd"] == pytest.approx(0.15) and s["saved_usd"] == pytest.approx(0.15) and s["cache_hits"] == 1
    assert "cost_usd" not in local_report._run.events[2]


def test_the_same_prompt_sent_twice_is_noticed():
    _run([_event(lsh_fingerprint="aa"), _event(lsh_fingerprint="aa"), _event(lsh_fingerprint="bb")])
    repeats = local_report.summarize()["repeats"]
    assert len(repeats) == 1 and len(repeats[0]) == 2


def test_personal_data_is_counted_by_type():
    _run([_event(pii_flagged=True, pii_types={"EMAIL_ADDRESS": 1}), _event(pii_flagged=True, pii_types={"EMAIL_ADDRESS": 2, "PERSON": 1})])
    s = local_report.summarize()
    assert s["pii_calls"] == 2 and s["pii_types"] == {"EMAIL_ADDRESS": 3, "PERSON": 1}


def _html(events, **kw):
    _run(events, pricing={"gpt-4o-mini": (0.15, 0.60)})
    return local_report.build_html(list(local_report._run.events), **kw)


def test_html_has_a_headline_branding_prices_and_the_demo_link():
    page = _html([_event(), _event(pii_flagged=True, pii_types={"EMAIL_ADDRESS": 1})])
    assert "2 AI calls, about $0.30, 1 with personal data" in page
    assert "data:image/png;base64," in page and "Local report" in page
    assert "Prices used" in page and "$0.15" in page and "$0.60" in page
    assert "https://datagras.com/savi/contact?utm_source=sdk&amp;utm_medium=report&amp;utm_campaign=local-report" in page


def test_html_never_points_at_the_signup_address_that_does_not_exist():
    page = _html([_event()])
    assert "app.datagras.com" not in page and "api.datagras.com" not in page


def test_html_escapes_names_so_a_label_cannot_inject_markup():
    page = _html([_event(workflow_id="<script>alert(1)</script>")])
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page


def test_hide_names_swaps_workflow_agent_and_user_for_labels():
    page = _html([_event(workflow_id="payroll-secret", agent_id="cfo-bot", user_id="priya@example.com"),
                  _event(workflow_id="payroll-secret", agent_id="other", user_id="priya@example.com")], hide_names=True)
    assert "payroll-secret" not in page and "cfo-bot" not in page and "priya@example.com" not in page
    assert "workflow=workflow-1" in page and "agent=agent-1" in page and "agent=agent-2" in page and "user=user-1" in page


def test_the_demo_banner_only_shows_on_sample_reports():
    assert "Sample report" not in _html([_event()])
    assert "Sample report" in _html([_event()], demo=True)


def test_prompts_and_answers_never_reach_the_report():
    page = _html([_event(prompt_text="TOP SECRET PROMPT", response_text="TOP SECRET ANSWER", tool_arguments="x", tool_results="y")])
    assert "TOP SECRET" not in page


def test_a_real_client_call_flows_into_the_report(capsys):
    local_report.start(save=None, pricing={"gpt-4o-mini": (0.15, 0.60)})
    client = SaviOpenAI(api_key="test-key", local_mode=True, mask_pii=False)
    response = MagicMock()
    response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens = 1_000, 500, 1_500
    response.model = "gpt-4o-mini-2024-07-18"
    with patch("httpx.post") as post, patch.object(client._inner.chat.completions, "create", return_value=response):
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}])
    post.assert_not_called()
    assert local_report.summarize()["calls"] == 1
    assert "[ OK  ]" in capsys.readouterr().out


def test_finish_prints_the_summary_and_saves_the_file(tmp_path, capsys):
    target = tmp_path / "out.html"
    local_report.start(save=str(target), pricing={"gpt-4o-mini": (0.15, 0.60)})
    LocalEventEmitter().emit(_event())
    local_report._finish()
    out = capsys.readouterr().out
    assert "Summary" in out and "Report saved:" in out and target.exists()
    assert "SAVI" in target.read_text(encoding="utf-8")


def test_finish_writes_nothing_when_there_were_no_calls(tmp_path):
    target = tmp_path / "out.html"
    local_report.start(save=str(target))
    local_report._finish()
    assert not target.exists()


def test_the_demo_command_writes_a_clearly_labelled_sample(tmp_path, capsys):
    target = tmp_path / "sample.html"
    assert local_report.main(["--demo", "--out", str(target)]) == 0
    page = target.read_text(encoding="utf-8")
    assert "Sample report" in page and "Cost covers 6 of 7 calls" in page
    assert local._default_pricing == {} and local._console is True


def test_the_demo_command_without_a_flag_shows_help_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert local_report.main([]) == 0
    assert not list(tmp_path.iterdir()) and "--demo" in capsys.readouterr().out
