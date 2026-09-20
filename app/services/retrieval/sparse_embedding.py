
import logfire
from fastembed import SparseTextEmbedding
from qdrant_client.http import models

_SPARSE_MODEL_NAME = "Qdrant/bm25"
_model: SparseTextEmbedding | None = None


def _get_model() -> SparseTextEmbedding:
    global _model
    if _model is None:
        logfire.info(f"Loading FastEmbed sparse model ({_SPARSE_MODEL_NAME}) for hybrid retrieval.")
        _model = SparseTextEmbedding(model_name=_SPARSE_MODEL_NAME)
    return _model


def embed_documents_sparse(texts: list[str]) -> list[models.SparseVector]:
    """Sparse vectors for ingestion — one per chunk, same order as `texts`."""
    embeddings = _get_model().embed(texts)
    return [
        models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in embeddings
    ]


def embed_query_sparse(query: str) -> models.SparseVector:
    """Sparse vector for a single query — uses BM25's query-side weighting."""
    embedding = next(iter(_get_model().query_embed(query)))
    return models.SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )
