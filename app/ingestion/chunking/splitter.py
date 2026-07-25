from typing import List
import logfire


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