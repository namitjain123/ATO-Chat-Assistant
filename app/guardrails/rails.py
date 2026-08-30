import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.

    Model history: llama-3.3-70b-versatile was originally chosen over the
    smaller llama-3.1-8b-instant, which wasn't reliably completing NeMo's
    internal few-shot prompt template for the general-purpose dialog flow —
    it would sometimes echo an unrelated early example line from the
    template instead of generating a real answer. Both Llama models were
    subsequently removed from Groq's catalog entirely (confirmed via
    GET /v1/models returning a 404 for both at request time — this broke
    every single query in production, since guardrails runs on every one).
    openai/gpt-oss-120b replaces it: verified elsewhere in this project
    (golden_synthetic.py) for reliable instruction-following without the
    template-echoing failure mode the 8b model had.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="openai/gpt-oss-120b",
        temperature=0,
        max_retries=6,  # this call bypasses Portkey's retry/fallback entirely — needs its own resilience against Groq's free-tier TPM limit
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("NeMo Guardrails initialised (openai/gpt-oss-120b).")
    
    


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
        content_lower = content.lower()

        fired = any(indicator in content for indicator in RAIL_INDICATORS) or any(
            pattern in content_lower for pattern in GENERIC_REFUSAL_PATTERNS
        )

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
