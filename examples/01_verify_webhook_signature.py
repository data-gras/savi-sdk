"""
Verify the HMAC-SHA256 signature on a SAVI alert webhook.

Needs nothing: no SAVI account, no provider key, no extra pip install.
This is the exact check from the README's "Alerting to your own systems"
section, with a fake request built inline so you can see it pass and fail.

Run: python examples/01_verify_webhook_signature.py
"""
import hashlib
import hmac
import json


def verify_savi_webhook(raw_body: bytes, signature_header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


if __name__ == "__main__":
    secret = "a-secret-you-generate-yourself-16-chars-min"

    # What SAVI would actually POST to your webhook_url.
    body = json.dumps({
        "event_type": "budget_breach",
        "severity": "needs_action",
        "tenant_id": "ten_yourcompany",
        "summary": "Budget breach: engineering — Spent $512.00 of $500.00 (102%) — monthly",
    }).encode()

    # What SAVI would send in the request's signature header.
    real_signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    print("Valid signature: ", verify_savi_webhook(body, real_signature, secret))
    print("Wrong secret:    ", verify_savi_webhook(body, real_signature, "not-the-real-secret"))
    print("Tampered body:   ", verify_savi_webhook(body + b"tampered", real_signature, secret))
