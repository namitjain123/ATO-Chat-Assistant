import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings

# Azure OpenAI primary, Groq automatic fallback (in that order — see notes.md
# for why: fallback-only means the system never actually touches Azure in
# normal operation, which undercuts "runs on Azure OpenAI" as a true claim).
#
# Fallback is implemented at the APPLICATION level (see FALLBACK_TARGETS,
# create_completion_with_fallback, get_structured_llm_with_fallback below),
# not via Portkey's own server-side fallback strategy. That was the original
# plan — GATEWAY_CONFIG below still documents the intended target chain and
# is still used for its cache/retry settings — but Portkey's server-side
# fallback has a confirmed bug/limitation for Azure OpenAI targets
# specifically: every direct-addressed call to the Azure target succeeded
# (verified repeatedly, including through Portkey's own "Run Test Request"
# tool), while every identical call routed through the saved fallback config
# failed with "azure-openai error: Resource not found" — despite the Azure
# resource, the Portkey integration, the deployment/alias/api-version
# mapping, and the config's target JSON all being verified correct. Most
# likely cause: Azure's REST API requires the deployment name in the URL
# path (not just a body field, unlike Groq/OpenAI-style providers), and
# Portkey's fallback-iteration code path appears not to construct that URL
# correctly, while its simpler direct-request path does.
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503]
    },
    "targets": [
        {"override_params": {"model": f"@{settings.AZURE_SLUG}/{settings.AZURE_OPENAI_DEPLOYMENT}"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/openai/gpt-oss-20b"}},
    ]
}

PORTKEY_CONFIG = settings.PORTKEY_CONFIG_SLUG or GATEWAY_CONFIG

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=PORTKEY_CONFIG
)

# A `config` attached at the CLIENT level (like portkey_client above) keeps
# applying Portkey's server-side fallback-strategy routing on every call made
# through it — even ones with an explicit `model=` override — which is
# exactly the broken path being avoided here. Verified directly: the same
# call through portkey_client (config attached) landed on Groq every time;
# the identical call through this config-free client landed on Azure
# (model in the response: "gpt-5-mini-2025-08-07"). Used specifically for the
# application-level fallback functions below. Trade-off: loses Portkey's
# gateway-level response caching (tied to the config's "cache" setting) for
# calls made this way — acceptable given a working primary/fallback chain is
# the actual requirement here.
portkey_client_direct = Portkey(api_key=settings.PORTKEY_API_KEY)

# Ordered primary -> fallback targets, addressed directly by @slug/model — the
# path verified to work reliably, bypassing Portkey's broken server-side
# fallback strategy for Azure specifically (see comment above GATEWAY_CONFIG).
FALLBACK_TARGETS = [
    f"@{settings.AZURE_SLUG}/{settings.AZURE_OPENAI_DEPLOYMENT}",
    f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b",
    f"@{settings.GROQ_SLUG_2}/openai/gpt-oss-20b",
]

# No `temperature` override anywhere below: gpt-5-mini (Azure primary) is a
# reasoning model and rejects any value other than the default 1 — verified
# directly ("Unsupported value: 'temperature' does not support 0.1 with this
# model"). The Groq fallback targets (openai/gpt-oss-120b, openai/gpt-oss-20b
# — also reasoning models) tolerate the default fine, so omitting it works
# for every target in the fallback chain.
#
# Groq fallback models: llama-3.3-70b-versatile and llama-3.1-8b-instant were
# both removed from Groq's catalog entirely at some point after this was
# first built (confirmed via GET /v1/models 404ing on both) — this broke
# every single query in production, since guardrails (app/guardrails/
# rails.py, a separate direct-Groq call, not through this fallback chain)
# also depended on the first one and runs on every request. Replaced with
# openai/gpt-oss-120b / openai/gpt-oss-20b, both confirmed present in
# Groq's current model list at the time of this fix.
#
# max_completion_tokens (not max_tokens) everywhere below: Azure rejects
# `max_tokens` outright for gpt-5-mini ("Unsupported parameter... Use
# 'max_completion_tokens' instead") — verified both params directly against
# both Azure and Groq targets; max_completion_tokens is accepted by both.
# Set well past typical response length: reasoning models spend tokens on
# internal chain-of-thought before the visible answer — verified a one-word
# reply alone used 128 reasoning tokens; too small a limit silently returns
# empty content instead of erroring.


def create_completion_with_fallback(messages: list, max_completion_tokens: int = 4096, **kwargs):
    """
    Calls each target in FALLBACK_TARGETS in order (Azure primary, then Groq
    fallbacks), moving to the next on any failure. Raw-Portkey-client
    equivalent of get_structured_llm_with_fallback below — used by
    responder.py, which needs the native client (not LangChain) to read the
    x-portkey-cache-status header.
    """
    last_error = None
    for i, target in enumerate(FALLBACK_TARGETS):
        try:
            response = portkey_client_direct.chat.completions.create(
                model=target,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                **kwargs,
            )
            if i > 0:
                logfire.warning(f"⚠️ Primary target failed — succeeded on fallback #{i}: {target}")
            return response
        except Exception as e:
            last_error = e
            logfire.warning(f"Target {target} failed ({type(e).__name__}): {e}")
    raise last_error


def _make_chat_model(model_str: str, feature: str) -> ChatOpenAI:
    # No `config=` in createHeaders here — same reason as portkey_client_direct
    # above: a config attached (even just via headers) keeps triggering
    # Portkey's broken server-side fallback routing for the Azure target, even
    # when this specific request already names its model explicitly.
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=model_str,
        model_kwargs={"max_completion_tokens": 2048},
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.
    No fallback applied here (single target, Azure primary) — for a node that
    needs the primary/fallback chain, use get_structured_llm_with_fallback instead.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + config). The @slug/model-name format is Portkey-specific — the underlying
      provider's own client does not understand it.
    """
    return _make_chat_model(FALLBACK_TARGETS[0], feature)


def get_structured_llm_with_fallback(schema, feature: str = "rag", method: str = "function_calling"):
    """
    Structured-output chain (Azure primary, Groq fallbacks) with the same
    application-level fallback as create_completion_with_fallback, built via
    LangChain's native .with_fallbacks() — each candidate is wrapped with
    with_structured_output first, then chained, so the whole thing still
    behaves like a single Runnable returning `schema` instances on .invoke().
    """
    candidates = [_make_chat_model(t, feature).with_structured_output(schema, method=method) for t in FALLBACK_TARGETS]
    primary, *fallbacks = candidates
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"
