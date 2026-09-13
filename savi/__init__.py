# Provider wrappers are imported lazily (PEP 562) so that installing only the
# provider you use is enough, e.g. SaviAnthropic must not require the openai,
# boto3, cohere, or mistralai packages to be installed.
from savi.context import SpanContext
from savi.pii     import PiiMasker, fingerprint
from savi.judge   import OutcomeExplainer
from savi.auto_instrument import enable_auto_instrumentation, disable_auto_instrumentation
from savi.mcp_instrument import enable_mcp_instrumentation, disable_mcp_instrumentation

_PROVIDER_EXPORTS = {
    "SaviOpenAI":         ("savi.openai",        "SaviOpenAI"),
    "SaviAsyncOpenAI":    ("savi.openai",        "SaviAsyncOpenAI"),
    "SaviAnthropic":      ("savi.anthropic",     "SaviAnthropic"),
    "SaviAsyncAnthropic": ("savi.anthropic",     "SaviAsyncAnthropic"),
    "SaviVertexAI":       ("savi.google_vertex", "SaviVertexAI"),
    "SaviAzureOpenAI":    ("savi.azure_openai",  "SaviAzureOpenAI"),
    "SaviAsyncAzureOpenAI": ("savi.azure_openai", "SaviAsyncAzureOpenAI"),
    # SaviBedrockRuntime and SaviMistral are async-capable via methods on the
    # same class (converse_async / complete_async) rather than separate
    # classes - boto3 has no native async client, and mistralai's SDK exposes
    # async as `_async`-suffixed methods on one client, not a second class.
    "SaviBedrockRuntime": ("savi.bedrock",       "SaviBedrockRuntime"),
    "SaviCohere":         ("savi.cohere",        "SaviCohere"),
    "SaviAsyncCohere":    ("savi.cohere",        "SaviAsyncCohere"),
    "SaviMistral":        ("savi.mistral",       "SaviMistral"),
}

__all__ = [
    *_PROVIDER_EXPORTS,
    "SpanContext", "PiiMasker", "fingerprint", "OutcomeExplainer",
    "enable_auto_instrumentation", "disable_auto_instrumentation",
    "enable_mcp_instrumentation", "disable_mcp_instrumentation",
]


def __getattr__(name: str):
    if name in _PROVIDER_EXPORTS:
        import importlib
        module_name, attr = _PROVIDER_EXPORTS[name]
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"{name} requires the provider package {exc.name!r}. "
                f"Install it (e.g. pip install {exc.name}) to use this wrapper."
            ) from exc
        return getattr(module, attr)
    raise AttributeError(f"module 'savi' has no attribute {name!r}")
