import sys
import pytest
from unittest.mock import MagicMock, patch
from savi.pii import fingerprint, PiiMasker, _DEFAULT_ALLOW_LIST


# ---------------------------------------------------------------------------
# fingerprint() — no Presidio needed
# ---------------------------------------------------------------------------

def test_fingerprint_is_deterministic():
    msgs = [{"role": "user", "content": "Hello world"}]
    assert fingerprint(msgs) == fingerprint(msgs)


def test_fingerprint_differs_on_content_change():
    assert fingerprint([{"role": "user", "content": "Hello world"}]) != \
           fingerprint([{"role": "user", "content": "Hello earth"}])


def test_fingerprint_string_input_returns_256_char_hex():
    # 256 = 32 min-hash values x 8 hex chars each (matches lsh-fingerprint.ts).
    result = fingerprint("plain text")
    assert isinstance(result, str)
    assert len(result) == 256


def test_fingerprint_near_duplicate_prompts_share_most_of_signature():
    # MinHash's whole point: near-duplicate text shouldn't fingerprint as
    # unrelated the way a SHA-256 hash of the raw bytes would.
    fp_a = fingerprint("What is our total AWS spend this month?")
    fp_b = fingerprint("What is our total AWS spend this month? ")  # trailing space
    assert fp_a == fp_b  # normalised whitespace - literally identical after normalisation


def test_fingerprint_empty_input_returns_empty_string():
    assert fingerprint("") == ""
    assert fingerprint([]) == ""


def test_fingerprint_after_masking_matches_same_template():
    masked1 = [{"role": "user", "content": "Invoice for <REDACTED>"}]
    masked2 = [{"role": "user", "content": "Invoice for <REDACTED>"}]
    assert fingerprint(masked1) == fingerprint(masked2)


def test_fingerprint_accepts_arbitrary_object():
    result = fingerprint({"key": "value", "nested": [1, 2, 3]})
    assert isinstance(result, str) and len(result) == 256


# ---------------------------------------------------------------------------
# PiiMasker — Presidio internals mocked
# ---------------------------------------------------------------------------

@pytest.fixture
def masker():
    """PiiMasker with Presidio load stubbed out; analyzer/anonymizer are MagicMocks."""
    with patch.object(PiiMasker, "_load"):
        m = object.__new__(PiiMasker)
        m._entities   = ["PERSON", "EMAIL_ADDRESS"]
        m._allow_list = ["allowed_term"]
        m._language   = "en"
        m._analyzer   = MagicMock()
        m._anonymizer = MagicMock()
        m._redact_op  = {"DEFAULT": MagicMock()}
        return m


def test_mask_returns_original_when_no_pii(masker):
    masker._analyzer.analyze.return_value = []
    text, counts = masker.mask("Tell me about cloud costs")
    assert text == "Tell me about cloud costs"
    assert counts == {}


def test_mask_replaces_detected_pii(masker):
    hit = MagicMock()
    hit.entity_type = "EMAIL_ADDRESS"
    masker._analyzer.analyze.return_value = [hit]

    anon = MagicMock()
    anon.text = "Send invoice to <REDACTED>"
    masker._anonymizer.anonymize.return_value = anon

    text, counts = masker.mask("Send invoice to user@example.com")
    assert text == "Send invoice to <REDACTED>"
    assert counts == {"EMAIL_ADDRESS": 1}


def test_mask_counts_multiple_entity_types(masker):
    hits = [MagicMock(entity_type="PERSON"), MagicMock(entity_type="EMAIL_ADDRESS"),
            MagicMock(entity_type="PERSON")]
    masker._analyzer.analyze.return_value = hits

    anon = MagicMock()
    anon.text = "<REDACTED> at <REDACTED> and <REDACTED>"
    masker._anonymizer.anonymize.return_value = anon

    _, counts = masker.mask("John Smith at john@example.com and Jane Doe")
    assert counts == {"PERSON": 2, "EMAIL_ADDRESS": 1}


def test_mask_returns_original_on_analyzer_exception(masker):
    masker._analyzer.analyze.side_effect = RuntimeError("NLP failure")
    text, counts = masker.mask("some prompt with PII")
    assert text == "some prompt with PII"
    assert counts == {}


def test_mask_passes_allow_list_to_analyzer(masker):
    """The allow_list defense against LOCATION/DATE_TIME false positives
    only works if it's actually forwarded to Presidio on every call - a
    silent drop here would make the default list decorative."""
    masker._analyzer.analyze.return_value = []
    masker.mask("some text")
    _, kwargs = masker._analyzer.analyze.call_args
    assert kwargs["allow_list"] == masker._allow_list
    assert kwargs["allow_list_match"] == "exact"


# ---------------------------------------------------------------------------
# allow_list construction — backlog item 3 (LOCATION/DATE_TIME false
# positives). allow_list_match must stay "exact" (whole-span match), not
# regex/substring - a word-boundary regex for "west" was empirically found
# to also match inside "West Melbourne" and suppress that entire genuine
# LOCATION detection. Exact matching can't have that failure mode, since
# "West Melbourne" as a string is never equal to "west"/"West"/"WEST".
# ---------------------------------------------------------------------------

def test_default_allow_list_has_all_case_variants():
    for base in ("west", "quarter", "march"):
        assert base in _DEFAULT_ALLOW_LIST
        assert base.capitalize() in _DEFAULT_ALLOW_LIST
        assert base.upper() in _DEFAULT_ALLOW_LIST


def test_default_allow_list_covers_multiword_relative_time_phrases():
    # Presidio detects "last quarter" as one two-word span - an entry for
    # "quarter" alone can never exact-match that longer span.
    assert "last quarter" in _DEFAULT_ALLOW_LIST
    assert "Last quarter" in _DEFAULT_ALLOW_LIST


def test_piimasker_uses_default_allow_list_when_not_specified():
    with patch.object(PiiMasker, "_load"):
        m = PiiMasker()
    assert m._allow_list == _DEFAULT_ALLOW_LIST


def test_piimasker_custom_allow_list_overrides_default():
    with patch.object(PiiMasker, "_load"):
        m = PiiMasker(allow_list=["custom_term"])
    assert m._allow_list == ["custom_term"]


def test_piimasker_empty_allow_list_disables_default():
    """An integrator who wants every LOCATION/DATE_TIME hit flagged, no
    exceptions, passes allow_list=[] rather than being stuck with SAVI's
    business-vocabulary default."""
    with patch.object(PiiMasker, "_load"):
        m = PiiMasker(allow_list=[])
    assert m._allow_list == []


def test_mask_messages_masks_string_content(masker):
    hit = MagicMock(entity_type="PERSON")
    masker._analyzer.analyze.return_value = [hit]
    anon = MagicMock()
    anon.text = "Hello <REDACTED>"
    masker._anonymizer.anonymize.return_value = anon

    messages = [{"role": "user", "content": "Hello John Smith"}]
    masked, flagged, types = masker.mask_messages(messages)

    assert masked[0]["content"] == "Hello <REDACTED>"
    assert masked[0]["role"] == "user"
    assert flagged is True
    assert types == {"PERSON": 1}


def test_mask_messages_preserves_non_content_keys(masker):
    masker._analyzer.analyze.return_value = []
    messages = [{"role": "system", "content": "You are helpful", "name": "system-v2"}]
    masked, _, _ = masker.mask_messages(messages)
    assert masked[0]["name"] == "system-v2"
    assert masked[0]["role"] == "system"


def test_mask_messages_does_not_mutate_original(masker):
    masker._analyzer.analyze.return_value = []
    original = [{"role": "user", "content": "clean prompt"}]
    import copy
    snapshot = copy.deepcopy(original)
    masker.mask_messages(original)
    assert original == snapshot


def test_mask_messages_skips_non_string_content(masker):
    messages = [{"role": "user", "content": None}]
    masked, flagged, types = masker.mask_messages(messages)
    assert masked == messages
    assert flagged is False
    assert types is None


def test_mask_messages_no_pii_returns_none_types(masker):
    masker._analyzer.analyze.return_value = []
    _, flagged, types = masker.mask_messages([{"role": "user", "content": "safe text"}])
    assert flagged is False
    assert types is None


def test_mask_messages_masks_text_parts_in_list_content(masker):
    """OpenAI 'content parts' format (tool results, multimodal messages) sends
    content as a list of {"type": "text", "text": ...} dicts rather than a
    plain string. Previously this shape bypassed masking entirely."""
    hit = MagicMock(entity_type="EMAIL_ADDRESS")
    masker._analyzer.analyze.return_value = [hit]
    anon = MagicMock()
    anon.text = "Reported by <REDACTED>"
    masker._anonymizer.anonymize.return_value = anon

    messages = [{
        "role": "tool",
        "content": [{"type": "text", "text": "Reported by jane.doe@example.com"}],
    }]
    masked, flagged, types = masker.mask_messages(messages)

    assert masked[0]["content"][0]["text"] == "Reported by <REDACTED>"
    assert flagged is True
    assert types == {"EMAIL_ADDRESS": 1}


def test_mask_messages_list_content_preserves_non_text_parts(masker):
    masker._analyzer.analyze.return_value = []
    messages = [{
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}],
    }]
    masked, flagged, types = masker.mask_messages(messages)

    assert masked[0]["content"] == messages[0]["content"]
    assert flagged is False
    assert types is None


def test_pii_masker_raises_import_error_when_presidio_missing():
    with patch.dict(sys.modules, {"presidio_analyzer": None, "presidio_anonymizer": None}):
        with pytest.raises(ImportError, match="Presidio not installed"):
            PiiMasker()
