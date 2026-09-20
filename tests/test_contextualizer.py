"""
Tests for app/ingestion/contextualizer.py — the structured LLM is always mocked.
"""
import pytest

from app.ingestion import contextualizer as cx


class _FakeAnnotation:
    def __init__(self, context, topics=None):
        self.context = context
        self.topics = topics or []


class _FakeLLM:
    def __init__(self, fn):
        self.fn = fn
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.fn(prompt)


@pytest.fixture
def fake_llm(mocker):
    def install(fn):
        llm = _FakeLLM(fn)
        mocker.patch.object(cx, "_get_llm", return_value=llm)
        return llm
    return install


def _chunk_of(prompt):
    return prompt.split("<chunk>\n", 1)[1].split("\n</chunk>", 1)[0]


def test_returns_one_annotation_per_chunk_in_order(fake_llm):
    # Echo the chunk back so ordering is checkable even though calls run
    # concurrently on a thread pool.
    fake_llm(lambda p: _FakeAnnotation(f"context for {_chunk_of(p)}", ["deductions"]))

    out = cx.contextualize_chunks("the whole document", ["chunk A", "chunk B", "chunk C"])

    assert [a["context"] for a in out] == ["context for chunk A", "context for chunk B", "context for chunk C"]
    assert all(a["topics"] == ["deductions"] for a in out)


def test_prompt_puts_document_before_chunk(fake_llm):
    llm = fake_llm(lambda p: _FakeAnnotation("ctx"))

    cx.contextualize_chunks("FULL DOCUMENT TEXT", ["one chunk"])

    prompt = llm.prompts[0]
    assert prompt.index("FULL DOCUMENT TEXT") < prompt.index("one chunk")  # cacheable shared prefix


def test_unknown_topics_are_dropped_and_known_ones_normalised(fake_llm):
    fake_llm(lambda p: _FakeAnnotation("ctx", ["Deductions", "record-keeping", "astrology", "deductions"]))

    [annotation] = cx.contextualize_chunks("doc", ["chunk"])

    assert annotation["topics"] == ["deductions", "record_keeping"]


def test_failed_generation_yields_empty_annotation_not_an_exception(fake_llm):
    def boom(_):
        raise RuntimeError("all targets failed")
    fake_llm(boom)

    assert cx.contextualize_chunks("doc", ["a", "b"]) == [
        {"context": "", "topics": []},
        {"context": "", "topics": []},
    ]


def test_empty_model_output_yields_empty_context(fake_llm):
    fake_llm(lambda p: _FakeAnnotation(None))

    assert cx.contextualize_chunks("doc", ["chunk"]) == [{"context": "", "topics": []}]


def test_context_is_stripped(fake_llm):
    fake_llm(lambda p: _FakeAnnotation("  ctx with padding \n"))

    assert cx.contextualize_chunks("doc", ["chunk"])[0]["context"] == "ctx with padding"


def test_build_embedding_inputs_prepends_prefix():
    assert cx.build_embedding_inputs(["Title › Section\nctx"], ["chunk"]) == ["Title › Section\nctx\n\nchunk"]


def test_build_embedding_inputs_leaves_chunk_alone_without_prefix():
    assert cx.build_embedding_inputs([""], ["chunk"]) == ["chunk"]
