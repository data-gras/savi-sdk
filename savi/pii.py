import json
import sys
from typing import Optional

# The first 8 are generic and work the same regardless of country. The last
# 4 (AU_*) are Australia-specific, for APRA CPG 234 compliance; they simply
# never match on non-Australian text, so they're harmless elsewhere, but they
# also mean no country-specific identifier is auto-detected outside Australia
# by default (no US SSN, UK NINO, India Aadhaar, etc.). Presidio ships
# built-in recognizers for many other countries already - pass those entity
# names via the entities/pii_entities constructor arg to add them; see
# https://microsoft.github.io/presidio/supported_entities/ for the current,
# version-accurate list rather than assuming a name here is still correct.
_DEFAULT_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
    "IBAN_CODE", "IP_ADDRESS", "LOCATION", "DATE_TIME",
    "AU_ABN", "AU_ACN", "AU_TFN", "AU_MEDICARE",
]

# Common business vocabulary that Presidio's LOCATION/DATE_TIME recognizers
# correctly but unhelpfully flag as PII here ("west region", "last quarter"
# aren't personal data in a usage-analytics product). Multi-word phrases
# need to be listed whole, since allow_list_match="exact" matches the full
# detected span. Case variants are generated below, not hand-duplicated here.
_DEFAULT_ALLOW_LIST_TERMS = [
    "north", "south", "east", "west", "region", "zone",
    "quarter", "yesterday", "today", "tomorrow",
    "last quarter", "this quarter", "next quarter",
    "last week", "this week", "next week",
    "last month", "this month", "next month",
    "last year", "this year", "next year",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]


def _expand_case_variants(terms: list) -> list:
    """lower / Capitalised / UPPER for each term. Used with Presidio's
    allow_list_match="exact", not regex/substring matching: a word-boundary
    regex for "west" would also match inside "West Melbourne" and suppress
    that whole LOCATION detection. Exact whole-span matching doesn't have
    that failure mode."""
    out = []
    for t in terms:
        out.extend([t, t.capitalize(), t.upper()])
    return out


_DEFAULT_ALLOW_LIST = _expand_case_variants(_DEFAULT_ALLOW_LIST_TERMS)

# MinHash LSH parameters. Must match lsh-fingerprint.ts exactly, so the
# same text produces the same fingerprint regardless of which SAVI SDK
# sent it. Fixed constants: changing them invalidates every fingerprint
# already stored.
_NUM_HASHES   = 32
_SHINGLE_SIZE = 3
_PRIME        = 2147483647  # Mersenne prime 2^31 - 1
_HASH_A = [(i * 1234567 + 987654) % _PRIME for i in range(_NUM_HASHES)]
_HASH_B = [(i * 7654321 + 123456) % _PRIME for i in range(_NUM_HASHES)]


def _to_shingles(text: str) -> set:
    normalized = " ".join(text.lower().split())
    shingles = set()
    for i in range(len(normalized) - _SHINGLE_SIZE + 1):
        s = normalized[i : i + _SHINGLE_SIZE]
        h = 0
        for ch in s:
            h = ((h * 31) + ord(ch)) % _PRIME
        shingles.add(h)
    return shingles


def _compute_minhash(shingles: set) -> list:
    sig = [_PRIME] * _NUM_HASHES
    for sh in shingles:
        for i in range(_NUM_HASHES):
            hv = (_HASH_A[i] * sh + _HASH_B[i]) % _PRIME
            if hv < sig[i]:
                sig[i] = hv
    return sig


def fingerprint(content) -> str:
    """MinHash LSH fingerprint of JSON-normalised content (256 lowercase
    hex chars: 32 min-hash values x 8 hex chars). Near-duplicate text
    produces the same or a close signature, unlike a plain hash of the
    raw content. Used for prompt dedup/clustering and as the cache key in
    savi.cache.ResponseCache.

    Returns '' for empty/too-short input, so callers can treat a falsy
    fingerprint as "never cache this".
    """
    if isinstance(content, str):
        raw = content
    else:
        raw = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
    shingles = _to_shingles(raw)
    if not shingles:
        return ""
    sig = _compute_minhash(shingles)
    return "".join(f"{v:08x}" for v in sig)


class PiiMasker:
    """
    Detects and masks PII in prompt messages using Microsoft Presidio.
    Runs entirely on the customer's machine, only masked metadata reaches SAVI servers.

    Requires: pip install 'savi-sdk[pii]'
    """

    def __init__(
        self,
        entities: Optional[list] = None,
        exclude_entities: Optional[list] = None,
        language: str = "en",
        allow_list: Optional[list] = None,
    ):
        # exclude_entities drops a few from the default list without having
        # to hand-copy and maintain the rest. entities (an exact list)
        # takes precedence when both are given.
        if entities is not None:
            self._entities = entities
        elif exclude_entities:
            self._entities = [e for e in _DEFAULT_ENTITIES if e not in set(exclude_entities)]
        else:
            self._entities = _DEFAULT_ENTITIES
        # allow_list: exact strings that never count as PII regardless of
        # what a recognizer scores them. None uses SAVI's default business-
        # vocabulary list (see _DEFAULT_ALLOW_LIST above); pass [] to disable
        # entirely; pass a custom list to replace it outright.
        self._allow_list = _DEFAULT_ALLOW_LIST if allow_list is None else allow_list
        self._language = language
        self._analyzer = None
        self._anonymizer = None
        self._load()

    def _load(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig
        except ImportError:
            raise ImportError(
                "Presidio not installed. "
                "Install with: pip install 'savi-sdk[pii]'"
            )
        self._analyzer  = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._redact_op  = {"DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"})}

    def mask(self, text: str) -> tuple[str, dict]:
        """
        Returns (masked_text, entity_type_counts).
        On any internal error returns the original text, never breaks calling code.
        """
        if not text:
            return text, {}
        try:
            results = self._analyzer.analyze(
                text=text, entities=self._entities, language=self._language,
                allow_list=self._allow_list, allow_list_match="exact",
            )
            if not results:
                return text, {}
            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=self._redact_op,
            )
            counts: dict = {}
            for r in results:
                counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
            return anonymized.text, counts
        except Exception:
            return text, {}

    def mask_messages(self, messages: list) -> tuple[list, bool, "dict | None"]:
        """
        Returns (masked_messages, pii_flagged, pii_types_or_None).
        Masks string 'content' fields, and text parts of the OpenAI "content
        parts" list format (e.g. tool results / multimodal messages:
        content=[{"type": "text", "text": "..."}, {"type": "image_url", ...}]).
        Non-text parts (image_url etc.) pass through unmasked. Does not
        mutate the original list.
        """
        masked = []
        all_counts: dict = {}

        def _accumulate(counts: dict) -> None:
            for k, v in counts.items():
                all_counts[k] = all_counts.get(k, 0) + v

        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str):
                new_content, counts = self.mask(content)
                _accumulate(counts)
                masked.append({**msg, "content": new_content})
            elif isinstance(content, list):
                new_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        new_text, counts = self.mask(part["text"])
                        _accumulate(counts)
                        new_parts.append({**part, "text": new_text})
                    else:
                        new_parts.append(part)
                masked.append({**msg, "content": new_parts})
            else:
                masked.append(msg)
        pii_types = all_counts if all_counts else None
        return masked, bool(all_counts), pii_types
