import uuid
from typing import List, TypedDict
import logfire

PARENT_CHUNK_SIZE = 2500
CHILD_CHUNK_SIZE = 400


class ParentChildChunk(TypedDict):
    parent_id: str
    parent_text: str
    child_text: str


def _hard_split(paragraph: str, chunk_size: int) -> List[str]:
    """Split a single paragraph that's already >= chunk_size on its own, breaking
    on whitespace (not mid-word) so no piece exceeds chunk_size."""
    words = paragraph.split(" ")
    pieces = []
    current = ""
    for w in words:
        if current and len(current) + 1 + len(w) > chunk_size:
            pieces.append(current)
            current = w
        else:
            current = f"{current} {w}" if current else w
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """
    Simple semantic-ish chunker that splits by paragraphs.
    Ensures chunks do not exceed the specified size — including a single
    paragraph that's already larger than chunk_size on its own, which
    previously passed through unsplit (real chunks up to 4442 chars were
    observed against a 1500 target).
    """
    with logfire.span("Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                if len(p) >= chunk_size:
                    chunks.extend(_hard_split(p, chunk_size))
                else:
                    current_chunk = p + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [c for c in chunks if c.strip()]
        oversized = [c for c in valid_chunks if len(c) > chunk_size]
        if oversized:
            logfire.warning(f"⚠️ {len(oversized)} chunk(s) still exceed {chunk_size} chars after hard-split")
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks


def chunk_parent_child(
    text: str,
    parent_size: int = PARENT_CHUNK_SIZE,
    child_size: int = CHILD_CHUNK_SIZE,
) -> List[ParentChildChunk]:
    """
    Parent-child ("small-to-big") chunking, replacing flat ~1500-char chunks
    for ingestion. A single chunk size forces a tradeoff that's wrong both
    ways: small enough to embed precisely, it's too small to answer from
    (cuts off mid-explanation); big enough to answer from, dense/sparse
    similarity gets diluted across unrelated sentences sharing one vector.

    Splitting the two apart removes the tradeoff: small child chunks
    (~400 chars) are what gets embedded and searched, so a query's vector
    lands close to the exact sentence/paragraph it's asking about. Each
    child's larger parent chunk (~2500 chars) is what actually gets
    returned to the LLM, so the answer isn't built from an isolated
    fragment — it has the surrounding context that fragment came from.

    Reuses chunk_text at two granularities rather than a second splitting
    algorithm: parent- and child-level splitting are the same paragraph-
    combining + hard-split logic, just run at different target sizes. A
    parent already smaller than child_size (e.g. a short trailing section)
    comes back as its own single child — chunk_text already handles that
    case (a chunk under the target size returned whole), so it needs no
    special-casing here.

    Returns one entry per CHILD, each carrying its parent_id/parent_text —
    multiple entries share the same parent when a parent split into several
    children. Ingestion (processor.py) embeds child_text and stores
    parent_text as the point's returned content; retrieval (qdrant_service.py)
    deduplicates by parent_id so one parent can't occupy multiple result
    slots via its several children all matching.
    """
    parents = chunk_text(text, chunk_size=parent_size)
    pairs: List[ParentChildChunk] = []
    for parent_text in parents:
        parent_id = str(uuid.uuid4())
        children = chunk_text(parent_text, chunk_size=child_size)
        for child_text in children:
            pairs.append({
                "parent_id": parent_id,
                "parent_text": parent_text,
                "child_text": child_text,
            })

    logfire.info(f"✅ Generated {len(pairs)} child chunks across {len(parents)} parent chunks")
    return pairs