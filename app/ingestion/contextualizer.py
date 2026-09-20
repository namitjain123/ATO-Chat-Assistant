"""
Contextual retrieval: before embedding, each child chunk gets a short
LLM-written note situating it within its whole document. A chunk like
"The threshold is $97,000" says nothing about *which* threshold; the note
("...Medicare levy surcharge, singles, 2024-25") gets embedded with it, so
ambiguous chunks become findable by the question they actually answer.

The same call also tags the chunk's topics (from metadata.TOPICS) for
query-time filtering — one LLM call per chunk, not two.
"""
from concurrent.futures import ThreadPoolExecutor

import logfire
from pydantic import BaseModel, Field

from app.gateway.client import get_structured_llm_with_fallback
from app.ingestion.metadata import TOPICS, normalize_topics

MAX_WORKERS = 8


class ChunkAnnotation(BaseModel):
    context: str = Field(
        description="1-2 sentences situating the chunk within the overall document, to improve "
        "search retrieval. If the chunk leaves its subject implicit, name the specific topic, "
        "tax year, threshold, or entity it relates to."
    )
    topics: list[str] = Field(
        description=f"Every topic the chunk is substantively about (usually 1-2), chosen only from: "
        f"{', '.join(TOPICS)}. Empty list if none fit."
    )


# Document first, chunk last: keeps the long shared prefix identical across
# every chunk of a document, so provider-side prompt caching can reuse it.
CONTEXT_PROMPT = """<document>
{document}
</document>

Here is a chunk from the document above:
<chunk>
{chunk}
</chunk>

Situate this chunk within the overall document to improve search retrieval of the chunk, and tag its topics."""

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_structured_llm_with_fallback(
            ChunkAnnotation, feature="contextualizer", method="function_calling",
            max_completion_tokens=4096,  # long whole-document prompt; headroom for reasoning
        )
    return _llm


def _annotate_one(document: str, chunk: str) -> dict:
    try:
        result = _get_llm().invoke(CONTEXT_PROMPT.format(document=document, chunk=chunk))
        return {"context": (result.context or "").strip(), "topics": normalize_topics(result.topics)}
    except Exception as e:
        # One failed chunk shouldn't sink the whole file — index it without annotation.
        logfire.warning(f"Chunk annotation failed, indexing without context/topics: {e}")
        return {"context": "", "topics": []}


def contextualize_chunks(document: str, chunks: list[str]) -> list[dict]:
    """One {"context", "topics"} per chunk, same order as `chunks` (empty where generation failed)."""
    with logfire.span("Contextual Retrieval", chunks=len(chunks)):
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            annotations = list(pool.map(lambda c: _annotate_one(document, c), chunks))
        missing = sum(1 for a in annotations if not a["context"])
        if missing:
            logfire.warning(f"{missing}/{len(chunks)} chunks indexed without context.")
        return annotations


def embedding_prefix(title: str, section: str, context: str) -> str:
    """'Title › Section' heading path plus the context note — the text
    prepended to a chunk before embedding. The heading is deterministic, so it
    still helps when the context note is empty (disabled or failed)."""
    heading = f"{title} › {section}" if section else title
    return "\n".join(part for part in (heading, context) if part)


def build_embedding_inputs(prefixes: list[str], chunks: list[str]) -> list[str]:
    """Text actually embedded (dense and sparse): prefix (heading path + context) prepended to its chunk."""
    return [f"{prefix}\n\n{chunk}" if prefix else chunk for prefix, chunk in zip(prefixes, chunks)]
