"""
Tests for app/services/retrieval/qdrant_service.py — no real Qdrant/embedding
calls (client.get_collection and client.query_points are always mocked).

Two independent things are pinned down here:

1. ENABLE_HYBRID_SEARCH controls whether sparse+RRF fusion is attempted, but
   only when the collection's actual schema supports it (see
   _has_named_vectors' docstring for why this can't just be the flag alone —
   a real bug was caught here: a collection migrated to named vectors but
   deployed before the flag was flipped would otherwise send a query shape
   the collection rejects outright, breaking every single search).

2. Both the dense-only and hybrid paths must over-fetch and deduplicate by
   parent_id — see app/ingestion/chunking/splitter.py's chunk_parent_child
   for why several results can share one parent.
"""
import pytest
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval import qdrant_service as qs


@pytest.fixture(autouse=True)
def _reset_module_state():
    original_flag = settings.ENABLE_HYBRID_SEARCH
    original_cache = qs._collection_has_named_vectors
    yield
    settings.ENABLE_HYBRID_SEARCH = original_flag
    qs._collection_has_named_vectors = original_cache


class _FakePoint:
    def __init__(self, text, source, score, id="pt-id", parent_id=None):
        self.payload = {"text": text, "source": source}
        if parent_id is not None:
            self.payload["parent_id"] = parent_id
        self.score = score
        self.id = id


class _FakeResponse:
    def __init__(self, points):
        self.points = points


class _FakeCollectionInfo:
    """Stand-in for QdrantClient.get_collection()'s return shape, just the
    one nested field _has_named_vectors actually reads."""
    def __init__(self, vectors):
        class _Params:
            pass
        class _Config:
            pass
        self.config = _Config()
        self.config.params = _Params()
        self.config.params.vectors = vectors


# -- _has_named_vectors -------------------------------------------------------

def test_has_named_vectors_true_for_dict_schema(mocker):
    qs._collection_has_named_vectors = None
    mocker.patch.object(
        qs.client, "get_collection",
        return_value=_FakeCollectionInfo({"dense": object(), "sparse": object()}),
    )
    assert qs._has_named_vectors() is True


def test_has_named_vectors_false_for_legacy_unnamed_schema(mocker):
    qs._collection_has_named_vectors = None
    mocker.patch.object(
        qs.client, "get_collection",
        return_value=_FakeCollectionInfo(object()),  # a single VectorParams, not a dict
    )
    assert qs._has_named_vectors() is False


def test_has_named_vectors_is_cached_after_first_call(mocker):
    qs._collection_has_named_vectors = None
    mock_get = mocker.patch.object(
        qs.client, "get_collection",
        return_value=_FakeCollectionInfo({"dense": object()}),
    )
    qs._has_named_vectors()
    qs._has_named_vectors()
    mock_get.assert_called_once()


def test_has_named_vectors_defaults_false_and_does_not_cache_on_error(mocker):
    qs._collection_has_named_vectors = None
    mocker.patch.object(qs.client, "get_collection", side_effect=RuntimeError("network blip"))

    assert qs._has_named_vectors() is False
    assert qs._collection_has_named_vectors is None  # not pinned - safe to retry next call


# -- search_enterprise_knowledge: schema x flag combinations -----------------

def test_legacy_schema_flag_off_issues_plain_unnamed_query(mocker):
    settings.ENABLE_HYBRID_SEARCH = False
    mocker.patch.object(qs, "_has_named_vectors", return_value=False)
    mocker.patch.object(qs, "embed_query", return_value=[0.1, 0.2, 0.3])
    mock_sparse = mocker.patch.object(qs, "embed_query_sparse")
    mock_query_points = mocker.patch.object(qs.client, "query_points", return_value=_FakeResponse([]))

    qs.search_enterprise_knowledge("what deductions can I claim", limit=8)

    kwargs = mock_query_points.call_args.kwargs
    assert kwargs["query"] == [0.1, 0.2, 0.3]
    assert "using" not in kwargs
    assert "prefetch" not in kwargs
    assert kwargs["limit"] == 24  # over-fetched (limit * 3) so parent-id dedup still yields ~8
    mock_sparse.assert_not_called()


def test_legacy_schema_flag_on_still_issues_plain_unnamed_query(mocker):
    # Flipping the flag before migrating must be a no-op, not a crash - the
    # schema, not the flag, has final say over whether fusion is attempted.
    settings.ENABLE_HYBRID_SEARCH = True
    mocker.patch.object(qs, "_has_named_vectors", return_value=False)
    mocker.patch.object(qs, "embed_query", return_value=[0.1, 0.2, 0.3])
    mock_sparse = mocker.patch.object(qs, "embed_query_sparse")
    mock_query_points = mocker.patch.object(qs.client, "query_points", return_value=_FakeResponse([]))

    qs.search_enterprise_knowledge("query", limit=8)

    kwargs = mock_query_points.call_args.kwargs
    assert "using" not in kwargs
    assert "prefetch" not in kwargs
    mock_sparse.assert_not_called()


def test_migrated_schema_flag_off_names_the_dense_vector(mocker):
    # The gap this test exists for: collection already migrated to named
    # vectors, but ENABLE_HYBRID_SEARCH hasn't been flipped on yet. The
    # dense-only path must now explicitly name "dense" - an unnamed query
    # against this schema fails outright (verified directly against a real
    # in-memory Qdrant collection).
    settings.ENABLE_HYBRID_SEARCH = False
    mocker.patch.object(qs, "_has_named_vectors", return_value=True)
    mocker.patch.object(qs, "embed_query", return_value=[0.1, 0.2, 0.3])
    mock_sparse = mocker.patch.object(qs, "embed_query_sparse")
    mock_query_points = mocker.patch.object(qs.client, "query_points", return_value=_FakeResponse([]))

    qs.search_enterprise_knowledge("query", limit=8)

    kwargs = mock_query_points.call_args.kwargs
    assert kwargs["using"] == settings.DENSE_VECTOR_NAME
    assert "prefetch" not in kwargs
    mock_sparse.assert_not_called()  # still dense-only - fusion wasn't requested


def test_migrated_schema_flag_on_issues_prefetch_fusion_query(mocker):
    settings.ENABLE_HYBRID_SEARCH = True
    mocker.patch.object(qs, "_has_named_vectors", return_value=True)
    mocker.patch.object(qs, "embed_query", return_value=[0.1, 0.2, 0.3])
    mocker.patch.object(
        qs, "embed_query_sparse",
        return_value=models.SparseVector(indices=[1, 2], values=[0.9, 0.4]),
    )
    mock_query_points = mocker.patch.object(qs.client, "query_points", return_value=_FakeResponse([]))

    qs.search_enterprise_knowledge("medicare levy surcharge threshold", limit=8)

    kwargs = mock_query_points.call_args.kwargs
    assert isinstance(kwargs["query"], models.FusionQuery)
    assert kwargs["query"].fusion == models.Fusion.RRF

    prefetch = kwargs["prefetch"]
    assert len(prefetch) == 2
    using_names = {p.using for p in prefetch}
    assert using_names == {settings.DENSE_VECTOR_NAME, settings.SPARSE_VECTOR_NAME}
    assert all(p.limit == 24 for p in prefetch)  # over-fetched (limit * 3), same as the dense-only path
    assert kwargs["limit"] == 24


def test_hybrid_results_are_normalised_the_same_way(mocker):
    settings.ENABLE_HYBRID_SEARCH = True
    mocker.patch.object(qs, "_has_named_vectors", return_value=True)
    mocker.patch.object(qs, "embed_query", return_value=[0.1, 0.2, 0.3])
    mocker.patch.object(
        qs, "embed_query_sparse",
        return_value=models.SparseVector(indices=[1], values=[0.9]),
    )
    fake_points = [_FakePoint("chunk text", "page.md", 0.87, id="pt-1", parent_id="parent-1")]
    mocker.patch.object(qs.client, "query_points", return_value=_FakeResponse(fake_points))

    results = qs.search_enterprise_knowledge("query", limit=8)

    assert results == [{"content": "chunk text", "source": "page.md", "title": "", "section": "", "score": 0.87}]


def test_qdrant_failure_returns_empty_list_not_an_exception(mocker):
    mocker.patch.object(qs, "_has_named_vectors", return_value=False)
    mocker.patch.object(qs, "embed_query", side_effect=RuntimeError("boom"))

    results = qs.search_enterprise_knowledge("query", limit=8)

    assert results == []


# -- parent-id deduplication (schema-independent) ----------------------------

def test_dedupes_by_parent_id_keeping_highest_scoring_child_only(mocker):
    # Two children of "parent-A" (A is the higher-scoring, listed first -
    # Qdrant returns points in score order) plus one child of "parent-B".
    # Deduping must keep parent-A's first (best) occurrence and parent-B's,
    # dropping parent-A's second child rather than returning it twice.
    mocker.patch.object(qs, "_has_named_vectors", return_value=False)
    mocker.patch.object(qs, "embed_query", return_value=[0.1, 0.2, 0.3])
    fake_points = [
        _FakePoint("parent A, child 1 matched here", "page-a.md", 0.95, id="c1", parent_id="parent-A"),
        _FakePoint("parent A, but a different child matched", "page-a.md", 0.91, id="c2", parent_id="parent-A"),
        _FakePoint("parent B content", "page-b.md", 0.80, id="c3", parent_id="parent-B"),
    ]
    mocker.patch.object(qs.client, "query_points", return_value=_FakeResponse(fake_points))

    results = qs.search_enterprise_knowledge("query", limit=8)

    assert len(results) == 2
    assert results[0]["content"] == "parent A, child 1 matched here"
    assert results[0]["score"] == 0.95
    assert results[1]["content"] == "parent B content"


def test_dedupe_falls_back_to_point_id_for_pre_migration_points(mocker):
    # Points indexed before the parent-child migration carry no parent_id at
    # all - each must still count as its own distinct result rather than
    # colliding under a shared "missing key" value.
    mocker.patch.object(qs, "_has_named_vectors", return_value=False)
    mocker.patch.object(qs, "embed_query", return_value=[0.1, 0.2, 0.3])
    fake_points = [
        _FakePoint("old-schema chunk one", "legacy.md", 0.9, id="old-1", parent_id=None),
        _FakePoint("old-schema chunk two", "legacy.md", 0.85, id="old-2", parent_id=None),
    ]
    mocker.patch.object(qs.client, "query_points", return_value=_FakeResponse(fake_points))

    results = qs.search_enterprise_knowledge("query", limit=8)

    assert len(results) == 2


def test_results_respect_limit_after_dedup(mocker):
    mocker.patch.object(qs, "_has_named_vectors", return_value=False)
    mocker.patch.object(qs, "embed_query", return_value=[0.1, 0.2, 0.3])
    fake_points = [
        _FakePoint(f"chunk {i}", "page.md", 1.0 - i * 0.01, id=f"pt-{i}", parent_id=f"parent-{i}")
        for i in range(24)  # a full over-fetched batch, all distinct parents
    ]
    mocker.patch.object(qs.client, "query_points", return_value=_FakeResponse(fake_points))

    results = qs.search_enterprise_knowledge("query", limit=8)

    assert len(results) == 8


# -- metadata filtering -------------------------------------------------------

def _points(n, prefix="p"):
    return [_FakePoint(f"{prefix} {i}", "page.md", 1.0 - i * 0.01, id=f"{prefix}-{i}", parent_id=f"{prefix}-parent-{i}")
            for i in range(n)]


def test_build_filter_none_when_nothing_usable():
    assert qs._build_filter(None) is None
    assert qs._build_filter({"topics": [], "income_year": ""}) is None
    assert qs._build_filter({"topics": ["made_up"], "income_year": "2025"}) is None  # invalid values dropped


def test_build_filter_topics_and_year_each_allow_untagged_chunks():
    f = qs._build_filter({"topics": ["deductions"], "income_year": "2025\u201326"})
    assert len(f.must) == 2
    topic_clause, year_clause = f.must
    assert topic_clause.should[0].key == "topics"
    assert topic_clause.should[0].match.any == ["deductions"]
    assert year_clause.should[0].match.any == ["2025-26"]  # normalised from the en-dash form
    assert all(isinstance(c.should[1], models.IsEmptyCondition) for c in f.must)


def test_filters_applied_on_migrated_collection(mocker):
    settings.ENABLE_HYBRID_SEARCH = False
    mocker.patch.object(qs, "_has_named_vectors", return_value=True)
    mocker.patch.object(qs, "embed_query", return_value=[0.1])
    mock_qp = mocker.patch.object(qs.client, "query_points", return_value=_FakeResponse(_points(10)))

    results = qs.search_enterprise_knowledge("q", limit=8, filters={"topics": ["deductions"], "income_year": ""})

    assert len(results) == 8
    mock_qp.assert_called_once()  # enough filtered hits — no unfiltered backfill
    assert mock_qp.call_args.kwargs["query_filter"] is not None


def test_filters_ignored_on_legacy_collection(mocker):
    mocker.patch.object(qs, "_has_named_vectors", return_value=False)
    mocker.patch.object(qs, "embed_query", return_value=[0.1])
    mock_qp = mocker.patch.object(qs.client, "query_points", return_value=_FakeResponse([]))

    qs.search_enterprise_knowledge("q", limit=8, filters={"topics": ["deductions"], "income_year": ""})

    assert "query_filter" not in mock_qp.call_args.kwargs


def test_thin_filtered_results_are_backfilled_filtered_first(mocker):
    settings.ENABLE_HYBRID_SEARCH = False
    mocker.patch.object(qs, "_has_named_vectors", return_value=True)
    mocker.patch.object(qs, "embed_query", return_value=[0.1])
    filtered = _points(2, "filtered")
    unfiltered = [filtered[0]] + _points(10, "other")  # overlaps one filtered parent
    mock_qp = mocker.patch.object(
        qs.client, "query_points", side_effect=[_FakeResponse(filtered), _FakeResponse(unfiltered)]
    )

    results = qs.search_enterprise_knowledge("q", limit=8, filters={"topics": ["deductions"], "income_year": ""})

    assert mock_qp.call_count == 2
    assert "query_filter" not in mock_qp.call_args_list[1].kwargs
    assert [r["content"] for r in results[:2]] == ["filtered 0", "filtered 1"]  # precision hits stay first
    assert len(results) == 8
    assert len({r["content"] for r in results}) == 8  # the overlapping parent isn't duplicated


def test_failed_filtered_query_falls_back_to_unfiltered(mocker):
    settings.ENABLE_HYBRID_SEARCH = False
    mocker.patch.object(qs, "_has_named_vectors", return_value=True)
    mocker.patch.object(qs, "embed_query", return_value=[0.1])
    mocker.patch.object(
        qs.client, "query_points", side_effect=[RuntimeError("index missing"), _FakeResponse(_points(8))]
    )

    results = qs.search_enterprise_knowledge("q", limit=8, filters={"topics": ["deductions"], "income_year": ""})

    assert len(results) == 8


def test_hybrid_prefetches_carry_the_filter(mocker):
    settings.ENABLE_HYBRID_SEARCH = True
    mocker.patch.object(qs, "_has_named_vectors", return_value=True)
    mocker.patch.object(qs, "embed_query", return_value=[0.1])
    mocker.patch.object(qs, "embed_query_sparse", return_value=models.SparseVector(indices=[1], values=[1.0]))
    mock_qp = mocker.patch.object(qs.client, "query_points", return_value=_FakeResponse(_points(10)))

    qs.search_enterprise_knowledge("q", limit=8, filters={"topics": ["tax_rates"], "income_year": ""})

    assert all(p.filter is not None for p in mock_qp.call_args.kwargs["prefetch"])


def test_embedding_computed_once_even_with_backfill(mocker):
    settings.ENABLE_HYBRID_SEARCH = False
    mocker.patch.object(qs, "_has_named_vectors", return_value=True)
    mock_embed = mocker.patch.object(qs, "embed_query", return_value=[0.1])
    mocker.patch.object(qs.client, "query_points", side_effect=[_FakeResponse([]), _FakeResponse(_points(8))])

    qs.search_enterprise_knowledge("q", limit=8, filters={"topics": ["deductions"], "income_year": ""})

    mock_embed.assert_called_once()  # a Gemini network call — not repeated for the backfill query


# -- fetch_parents (graph evidence) --------------------------------------------

def test_fetch_parents_dedupes_children_and_keeps_requested_order(mocker):
    mocker.patch.object(qs, "_has_named_vectors", return_value=True)
    children = [
        _FakePoint("parent B text", "b.md", None, id="c1", parent_id="B"),
        _FakePoint("parent A text", "a.md", None, id="c2", parent_id="A"),
        _FakePoint("parent B text", "b.md", None, id="c3", parent_id="B"),
    ]
    mocker.patch.object(qs.client, "scroll", return_value=(children, None))

    out = qs.fetch_parents(["A", "B", "A"])

    assert [d["content"] for d in out] == ["parent A text", "parent B text"]


def test_fetch_parents_empty_on_legacy_collection(mocker):
    mocker.patch.object(qs, "_has_named_vectors", return_value=False)
    scroll = mocker.patch.object(qs.client, "scroll")
    assert qs.fetch_parents(["A"]) == []
    scroll.assert_not_called()


def test_fetch_parents_failure_returns_empty(mocker):
    mocker.patch.object(qs, "_has_named_vectors", return_value=True)
    mocker.patch.object(qs.client, "scroll", side_effect=RuntimeError("no index"))
    assert qs.fetch_parents(["A"]) == []
