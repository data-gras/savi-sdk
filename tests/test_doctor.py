import pytest

from savi import doctor


@pytest.mark.parametrize("raw, shown", [
    ("http://user:s3cretpass@localhost:8889/openai?token=abc123", "http://***@localhost:8889/openai"),
    ("https://proxy.example.com:3128", "https://proxy.example.com:3128"),
    ("http://localhost:8889/openai", "http://localhost:8889/openai"),
])
def test_redact_hides_passwords_and_query_tokens(raw, shown):
    assert doctor.redact(raw) == shown


def test_doctor_never_prints_a_key_or_a_url_secret(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-this-must-never-print-0000")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://user:s3cretpass@localhost:8889/openai?token=abc123")
    monkeypatch.setenv("HTTPS_PROXY", "http://bob:hunter2@proxy.local:3128")
    assert doctor.main(["--offline"]) == 0
    out = capsys.readouterr().out
    for secret in ("sk-this-must-never-print", "s3cretpass", "abc123", "hunter2"):
        assert secret not in out
    assert "OPENAI_API_KEY" in out and "set" in out


def test_doctor_flags_a_base_url_that_redirects_calls(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8889/openai")
    doctor.main(["--offline"])
    out = capsys.readouterr().out
    assert "[FIX ] OPENAI_BASE_URL" in out and "not the provider" in out


def test_doctor_says_local_mode_needs_no_server(capsys):
    doctor.main(["--offline"])
    assert "needs no SAVI server" in capsys.readouterr().out


def test_doctor_reports_an_unreachable_host_in_plain_words(monkeypatch):
    def boom(*a, **k):
        raise OSError("name not known")
    monkeypatch.setattr(doctor.socket, "getaddrinfo", boom)
    ok, detail = doctor.check_provider("api.example.invalid")
    assert ok is False and "cannot find api.example.invalid" in detail and "internet connection" in detail
