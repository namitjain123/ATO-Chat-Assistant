"""
Tests for app/services/retrieval/sparse_embedding.py — the BM25 sparse-vector
half of hybrid retrieval. Mocks FastEmbed's SparseTextEmbedding entirely, so
these never trigger a real model download or run real inference; what's being
pinned down is the *wiring* — that document embedding calls embed() (BM25's
indexing-side weighting) and query embedding calls query_embed() (BM25's
query-side weighting), and that both convert cleanly into
qdrant_client.http.models.SparseVector.
"""
import numpy as np
import pytest
from qdrant_client.http import models

from app.services.retrieval import sparse_embedding as se


class _FakeSparseEmbedding:
    """Stand-in for fastembed.sparse.sparse_embedding_base.SparseEmbedding."""
    def __init__(self, indices, values):
        self.indices = np.array(indices)
        self.values = np.array(values, dtype=float)


class _FakeModel:
    def __init__(self):
        self.embed_calls = []
        self.query_embed_calls = []

    def embed(self, texts):
        self.embed_calls.append(list(texts))
        return [_FakeSparseEmbedding([i, i + 1], [1.0, 0.5]) for i in range(len(texts))]

    def query_embed(self, query):
        self.query_embed_calls.append(query)
        return [_FakeSparseEmbedding([7, 9], [2.0, 1.0])]


@pytest.fixture(autouse=True)
def _reset_model_singleton():
    """_model is a lazily-initialised module-level singleton — reset it around
    every test so one test's fake model can't leak into the next."""
    se._model = None
    yield
    se._model = None


def test_embed_documents_sparse_calls_embed_not_query_embed(mocker):
    fake = _FakeModel()
    mocker.patch.object(se, "_get_model", return_value=fake)

    result = se.embed_documents_sparse(["deductions text", "medicare levy text"])

    assert fake.embed_calls == [["deductions text", "medicare levy text"]]
    assert fake.query_embed_calls == []


def test_embed_documents_sparse_returns_one_sparsevector_per_input(mocker):
    fake = _FakeModel()
    mocker.patch.object(se, "_get_model", return_value=fake)

    result = se.embed_documents_sparse(["a", "b", "c"])

    assert len(result) == 3
    assert all(isinstance(v, models.SparseVector) for v in result)
    assert result[0].indices == [0, 1]
    assert result[0].values == [1.0, 0.5]
    assert result[1].indices == [1, 2]


def test_embed_query_sparse_calls_query_embed_not_embed(mocker):
    fake = _FakeModel()
    mocker.patch.object(se, "_get_model", return_value=fake)

    result = se.embed_query_sparse("what is the medicare levy surcharge")

    assert fake.query_embed_calls == ["what is the medicare levy surcharge"]
    assert fake.embed_calls == []
    assert isinstance(result, models.SparseVector)
    assert result.indices == [7, 9]
    assert result.values == [2.0, 1.0]


def test_get_model_is_a_lazy_singleton(mocker):
    mock_ctor = mocker.patch.object(se, "SparseTextEmbedding", return_value=_FakeModel())

    first = se._get_model()
    second = se._get_model()

    mock_ctor.assert_called_once_with(model_name="Qdrant/bm25")
    assert first is second
