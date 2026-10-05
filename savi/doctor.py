"""Check your setup for the SAVI SDK:  python -m savi.doctor

Prints what is installed, which keys are set (never their values), settings that send your calls somewhere else,
and whether your computer can reach an AI provider. Safe to paste into a message: secrets are hidden.
"""
import argparse
import importlib.util
import os
import re
import socket
import ssl
import sys
from importlib.metadata import PackageNotFoundError, version

KEY_NAMES = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "COHERE_API_KEY", "MISTRAL_API_KEY"]
REDIRECTS = ["OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_BASE_URL", "AZURE_OPENAI_BASE_URL",
             "COHERE_BASE_URL", "MISTRAL_BASE_URL", "OLLAMA_BASE_URL", "SAVI_ENDPOINT"]
PROXIES = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"]
OPTIONAL = [("openai", "openai", 'pip install "savi-sdk[openai]"'), ("anthropic", "anthropic", 'pip install "savi-sdk[anthropic]"'),
            ("presidio_analyzer", "personal-data flagging", 'pip install "savi-sdk[pii]"')]


def redact(value):
    """Hide passwords and tokens inside a URL: user:pass@host and anything after ?."""
    value = re.sub(r"//[^/@\s]*@", "//***@", value)
    return value.split("?", 1)[0]


def _line(ok, label, detail=""):
    mark = {True: "[ ok ]", False: "[FIX ]", None: "[ -- ]"}[ok]
    print(f"{mark} {label}" + (f"  {detail}" if detail else ""))


def check_provider(host):
    try:
        addresses = sorted({info[4][0] for info in socket.getaddrinfo(host, 443)})
    except OSError as exc:
        return False, f"cannot find {host} ({exc}). Check your internet connection or DNS."
    try:
        with socket.create_connection((host, 443), timeout=8) as raw:
            with ssl.create_default_context().wrap_socket(raw, server_hostname=host):
                return True, f"reached {host} securely"
    except OSError as exc:
        return False, f"found {host} ({addresses[0]}) but could not connect: {exc}. A firewall or proxy may be in the way."


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m savi.doctor", description="Check your setup for the SAVI SDK.")
    p.add_argument("--host", default="api.openai.com", help="provider address to test (default api.openai.com)")
    p.add_argument("--offline", action="store_true", help="skip the internet check")
    args = p.parse_args(argv)

    print("SAVI setup check\n")
    py = ".".join(map(str, sys.version_info[:3]))
    _line(sys.version_info >= (3, 11), f"Python {py}", "" if sys.version_info >= (3, 11) else "SAVI needs Python 3.11 or newer. Install it from python.org/downloads")
    try:
        _line(True, f"savi-sdk {version('savi-sdk')}")
    except PackageNotFoundError:
        _line(False, "savi-sdk is not installed", 'run: pip install "savi-sdk[openai]"')
    _line(None, "Environment", "inside a virtual environment" if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else "not in a virtual environment (fine, but a venv keeps things tidy)")

    print("\nOptional parts")
    for module, label, how in OPTIONAL:
        found = importlib.util.find_spec(module) is not None
        _line(found if found else None, label, "installed" if found else f"not installed. To add it: {how}")

    print("\nKeys (only whether each is set, never the value)")
    for name in KEY_NAMES:
        _line(True if os.environ.get(name) else None, name, "set" if os.environ.get(name) else "not set")

    redirected = [(n, os.environ[n]) for n in REDIRECTS if os.environ.get(n)]
    print("\nSettings that send calls somewhere else")
    if not redirected:
        _line(True, "none set")
    for name, value in redirected:
        _line(False, f"{name} = {redact(value)}", "your calls go to this address, not the provider. If that is not what you want, remove it.")
    for name in PROXIES + [n.lower() for n in PROXIES]:
        if os.environ.get(name):
            _line(None, f"{name} = {redact(os.environ[name])}", "a proxy is set")

    if not args.offline:
        print("\nInternet")
        ok, detail = check_provider(args.host)
        _line(ok, args.host, detail)

    print("\nLocal mode needs no SAVI server and no SAVI account. Nothing here is required for it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
