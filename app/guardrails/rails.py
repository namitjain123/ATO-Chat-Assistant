import re

import logfire
from langchain_openai import ChatOpenAI
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.

    Model history: llama-3.3-70b-versatile (Groq) was originally used, then
    replaced with openai/gpt-oss-120b (also Groq) after Groq deprecated
    every Llama model in its catalog — that broke every single query in
    production, since guardrails runs on every one, before Groq's model
    catalog even entered the picture as a design concern here.

    gpt-oss-120b introduced a NEW problem, though: it's a raw open-weight
    reasoning model that writes its chain-of-thought inline as
    <think>...</think> in the response text itself — NeMo's response
    parsing doesn't know to strip that, so the model's raw internal
    reasoning was leaking straight into what users saw. It also started
    incorrectly refusing legitimate ATO questions, reasoning (visible in
    the leaked <think> block) that IT personally needed to know the answer
    rather than just classifying intent and passing through to the real
    RAG pipeline.

    Switched to Azure OpenAI's gpt-5-mini (via Portkey, the same primary LLM
    used elsewhere) as a first fix: Azure/OpenAI-hosted reasoning models
    keep their reasoning trace server-side, out of the visible response
    text, so the <think>-leak class of bug isn't structurally possible the
    way it is with a raw open-weight model's exposed completion text. That
    fixed 2 of 3 test cases, but "tell me a joke" still silently passed
    through unblocked — NeMo's action dispatcher was swallowing an
    exception and returning its own hardcoded internal-error string, which
    didn't match any of our detection patterns.

    Now switched to going direct to OpenAI.com (plain ChatOpenAI, no
    Portkey base_url/headers) once OPENAI_API_KEY was fixed: one fewer
    layer between this gate and the model (no Portkey gateway, no Azure
    deployment-routing quirks) for what should be the simplest, cheapest,
    highest-uptime piece of the whole pipeline — a narrow classification
    gate, not the main answer-generation path. gpt-4o-mini: a standard
    chat-completions model (not a raw reasoning model), so no <think>
    leakage risk here either.
    """
    global _rails

    guard_llm = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0,
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("NeMo Guardrails initialised (OpenAI gpt-4o-mini, direct).")
    
    


# Generic refusal phrasing NeMo's underlying LLM uses when it declines a request
# via its own wording rather than one of our canned Colang responses (e.g. for
# jailbreak/harmful-content attempts the general-purpose flow sometimes answers
# in its own words instead of hitting our exact "bot refuse jailbreak" text).
# Checked case-insensitively as a fallback alongside the exact RAIL_INDICATORS.
GENERIC_REFUSAL_PATTERNS = [
    "i cannot provide",
    "i can't provide",
    "i can't respond to that",
    "i cannot respond to that",
    "i must refuse",
    "i'm not able to help with that",
    "i'm unable to assist with that",
    "i can't assist with that",
    "i cannot assist with that",
]

# Defensive strip for a reasoning model's inline chain-of-thought (e.g.
# Groq's raw open-weight gpt-oss models write <think>...</think> directly in
# the completion text — real incident: this leaked straight into a user-
# facing response before the guardrails model was switched to Azure, whose
# reasoning trace stays server-side). Azure shouldn't produce this at all,
# but stripping it unconditionally is free insurance against any future
# model choice doing the same thing.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        content = _THINK_BLOCK_RE.sub("", content).strip()
        content_lower = content.lower()

        fired = any(indicator in content for indicator in RAIL_INDICATORS) or any(
            pattern in content_lower for pattern in GENERIC_REFUSAL_PATTERNS
        )

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
