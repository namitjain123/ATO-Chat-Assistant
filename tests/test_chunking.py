"""
Tests for app/ingestion/chunking/splitter.py.

chunk_text used to silently let a single oversized paragraph pass through
whole (real chunks up to 4442 chars were observed against a 1500 target,
before _hard_split was added) — these tests guard against that regressing.
"""
from app.ingestion.chunking.splitter import chunk_text, _hard_split


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_returns_single_chunk():
    text = "This is a short paragraph well under the chunk size."
    chunks = chunk_text(text, chunk_size=1500)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_multiple_paragraphs_combine_until_chunk_size():
    # Three short paragraphs that together stay under chunk_size should
    # combine into a single chunk, not one chunk each.
    paragraphs = ["Paragraph one.", "Paragraph two.", "Paragraph three."]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, chunk_size=1500)
    assert len(chunks) == 1
    for p in paragraphs:
        assert p in chunks[0]


def test_no_chunk_exceeds_chunk_size_even_for_oversized_paragraph():
    # A single paragraph larger than chunk_size on its own — this is the
    # exact bug that was found and fixed: previously passed through unsplit.
    chunk_size = 100
    oversized_paragraph = "word " * 50  # ~250 chars, well over 100
    chunks = chunk_text(oversized_paragraph, chunk_size=chunk_size)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= chunk_size


def test_hard_split_never_exceeds_chunk_size():
    chunk_size = 50
    paragraph = "word " * 30  # ~150 chars
    pieces = _hard_split(paragraph, chunk_size)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= chunk_size


def test_hard_split_does_not_split_mid_word():
    chunk_size = 20
    paragraph = "supercalifragilisticexpialidocious is a long word standalone"
    pieces = _hard_split(paragraph, chunk_size)
    # Every word from the original paragraph should appear whole in exactly
    # one piece — hard_split breaks on whitespace only, never mid-word.
    original_words = paragraph.split(" ")
    rejoined_words = " ".join(pieces).split(" ")
    assert rejoined_words == original_words


def test_hard_split_word_longer_than_chunk_size_kept_whole():
    # A single word longer than chunk_size can't be split further (splitter
    # only breaks on whitespace) — it should still come out as its own piece
    # rather than being silently dropped or crashing.
    chunk_size = 10
    pieces = _hard_split("supercalifragilisticexpialidocious", chunk_size)
    assert pieces == ["supercalifragilisticexpialidocious"]


def test_realistic_oversized_paragraph_from_a_real_bug():
    # Mirrors the actual bug: a single ~4400 char paragraph against a 1500
    # char target should be split into multiple chunks, none exceeding it.
    chunk_size = 1500
    oversized_paragraph = ("This is a sentence in a very long paragraph. " * 90).strip()
    assert len(oversized_paragraph) > chunk_size  # sanity check on the fixture itself

    chunks = chunk_text(oversized_paragraph, chunk_size=chunk_size)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= chunk_size
