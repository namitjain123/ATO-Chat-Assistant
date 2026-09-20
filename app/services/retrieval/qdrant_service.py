import time
import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embedding import embed_query
from app.services.retrieval.sparse_embedding import embed_query_sparse
from app.ingestion.metadata import normalize_topics, income_years

# Below this many filtered results, top up with unfiltered ones.
MIN_FILTERED_RESULTS = 5


# Initialize Qdrant Client. Explicit timeout + one retry (see _query_with_retry):
# the library default timed out on a live query ("The read operation timed
# out") against the free-tier cluster, which can be slow to respond after idle.
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    timeout=20,
)

_collection_has_named_vectors: bool | None = None


def _has_named_vectors() -> bool:
    """
    Whether QDRANT_COLLECTION was built with named dense/sparse vectors
    (see app/ingestion/processor.py) — checked once per process against the
    live collection, NOT trusted from settings.ENABLE_HYBRID_SEARCH.

    This split matters: a collection built with named vectors REJECTS a
    query with no `using=` ("Dense vector "" is not found in the
    collection" — verified directly), while the old single-unnamed-vector
    schema REJECTS a query that names one ("Dense vector dense is not
    found in the collection" — also verified directly). The two schemas
    take mutually exclusive request shapes.

    If ENABLE_HYBRID_SEARCH alone decided which shape to send, the gap
    between running the --wipe migration and remembering to flip that flag
    would break every single query — exactly the "code and live data
    silently disagreeing" failure this project already hit once with the
    Groq model deprecation, self-inflicted this time. Detecting the actual
    schema at runtime means the request shape always matches the data,
    regardless of flag/migration ordering; ENABLE_HYBRID_SEARCH then only
    controls whether to spend the extra sparse-fusion work when the schema
    supports it, a pure preference rather than a correctness switch.
    """
    global _collection_has_named_vectors
    if _collection_has_named_vectors is None:
        try:
            info = client.get_collection(settings.QDRANT_COLLECTION)
            _collection_has_named_vectors = isinstance(info.config.params.vectors, dict)
        except Exception as e:
            logfire.warning(f"Could not inspect collection schema ({e}) — assuming legacy unnamed-vector schema.")
            return False  # not cached: retry next call rather than pin a guess for the process lifetime
    return _collection_has_named_vectors


def _match_or_untagged(key: str, values: list[str]) -> models.Filter:
    """Chunk is tagged with one of `values`, OR carries no tag for `key` at
    all. A filter should only exclude chunks positively tagged with something
    else — most chunks mention no income year, and those must stay eligible
    for a year-specific question."""
    return models.Filter(should=[
        models.FieldCondition(key=key, match=models.MatchAny(any=values)),
        models.IsEmptyCondition(is_empty=models.PayloadField(key=key)),
    ])


def _build_filter(filters: dict | None) -> models.Filter | None:
    """Planner-extracted filters -> Qdrant filter; None when nothing usable."""
    if not filters:
        return None
    conditions = []
    topics = normalize_topics(filters.get("topics"))
    if topics:
        conditions.append(_match_or_untagged("topics", topics))
    years = income_years(filters.get("income_year") or "")  # validates + normalizes '2024–25' -> '2024-25'
    if years:
        conditions.append(_match_or_untagged("income_years", years))
    return models.Filter(must=conditions) if conditions else None


def _query_points(dense_vector, sparse_vector, named: bool, limit: int, qfilter: models.Filter | None):
    """
    One Qdrant query. With sparse_vector: dense + sparse hybrid search —
    prefetch top candidates from each vector space independently, then fuse
    the two rankings with Reciprocal Rank Fusion. Dense (Gemini) carries
    semantic recall; sparse (BM25) carries exact lexical recall — the numbers,
    form names, and ATO-specific terms dense embeddings tend to blur together.
    RRF combines both rankings by position rather than raw score, which
    sidesteps dense/sparse scores living on incomparable scales.
    """
    if sparse_vector is not None:
        return client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vector, using=settings.DENSE_VECTOR_NAME, limit=limit, filter=qfilter),
                models.Prefetch(query=sparse_vector, using=settings.SPARSE_VECTOR_NAME, limit=limit, filter=qfilter),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

    kwargs = dict(
        collection_name=settings.QDRANT_COLLECTION,
        query=dense_vector,
        limit=limit,
        with_payload=True,
    )
    # Migrated collections have no unnamed vector at all — must name it
    # even on the dense-only path (see _has_named_vectors' docstring).
    if named:
        kwargs["using"] = settings.DENSE_VECTOR_NAME
    if qfilter is not None:
        kwargs["query_filter"] = qfilter
    return client.query_points(**kwargs)


def _query_with_retry(*args):
    """One retry on failure. Matters beyond latency: search failures are
    swallowed into an empty result, which the grader would read as "nothing
    relevant" — two transient timeouts in a row would otherwise produce a
    false "not covered by the knowledge base" answer."""
    try:
        return _query_points(*args)
    except Exception as e:
        logfire.warning(f"Qdrant query failed ({e}) — retrying once.")
        time.sleep(1)
        return _query_points(*args)


def _collect(points, results: list, seen_parents: set, limit: int) -> None:
    """Append points to results, one per parent, until `limit`."""
    for res in points:
        if len(results) >= limit:
            return
        # Points indexed before the parent-child migration carry no parent_id
        # at all — fall back to the point's own id so old-schema data still
        # dedupes safely (each point is its own "parent").
        dedupe_key = res.payload.get("parent_id") or res.id
        if dedupe_key in seen_parents:
            continue
        seen_parents.add(dedupe_key)
        results.append({
            "content": res.payload.get("text", ""),
            "source": res.payload.get("source", "Unknown"),
            "title": res.payload.get("title", ""),
            "section": res.payload.get("section", ""),
            "score": res.score,
        })


def fetch_parents(parent_ids: list[str], limit: int = 8) -> list[dict]:
    """Parent passages by parent_id, in the given order — how knowledge-graph
    facts (which store the parent_ids they were extracted from) get their
    evidence text back. [] on a legacy collection, which has no parent_ids."""
    wanted = list(dict.fromkeys(parent_ids))[:limit]
    if not wanted or not _has_named_vectors():
        return []
    try:
        points, _ = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="parent_id", match=models.MatchAny(any=wanted)),
            ]),
            limit=len(wanted) * 12,  # every child point carries its parent's text; enough to cover each parent
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logfire.warning(f"Fetching graph evidence passages failed: {e}")
        return []

    by_parent = {}
    for p in points:
        by_parent.setdefault(p.payload.get("parent_id"), {
            "content": p.payload.get("text", ""),
            "source": p.payload.get("source", "Unknown"),
            "title": p.payload.get("title", ""),
            "section": p.payload.get("section", ""),
            "score": None,
        })
    return [by_parent[pid] for pid in wanted if pid in by_parent]


def search_enterprise_knowledge(query: str, limit: int = 8, filters: dict | None = None):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface — hybrid (dense + sparse, fused
    via RRF) when settings.ENABLE_HYBRID_SEARCH is on, plain dense otherwise.

    Matching happens against small child chunks (see app/ingestion/chunking/
    splitter.py's chunk_parent_child), but each result's `content` is its
    larger parent chunk. Because several children of the same parent can each
    score highly, candidates are over-fetched (fetch_limit) and deduplicated
    by parent_id, keeping only the highest-scoring child per parent.

    `filters` ({"topics": [...], "income_year": "2024-25"}, from the planner)
    narrow the search to matching or untagged chunks. A wrong filter must not
    starve the answer, so when the filtered search returns fewer than
    MIN_FILTERED_RESULTS parents, it's topped up with unfiltered results —
    filtered hits stay first, the precision gain survives, recall isn't lost.
    Filters only apply to migrated collections: legacy points carry no
    metadata to filter on (and no payload indexes).
    """
    try:
        fetch_limit = max(limit * 3, limit)
        named = _has_named_vectors()
        dense_vector = embed_query(query)
        sparse_vector = embed_query_sparse(query) if settings.ENABLE_HYBRID_SEARCH and named else None
        qfilter = _build_filter(filters) if named else None

        results, seen_parents = [], set()
        if qfilter is not None:
            try:
                _collect(_query_with_retry(dense_vector, sparse_vector, named, fetch_limit, qfilter).points,
                         results, seen_parents, limit)
            except Exception as e:
                logfire.warning(f"Filtered search failed ({e}) — falling back to unfiltered.")
            if len(results) >= min(limit, MIN_FILTERED_RESULTS):
                return results
            logfire.info(f"Filtered search returned {len(results)} — backfilling with unfiltered results.")

        _collect(_query_with_retry(dense_vector, sparse_vector, named, fetch_limit, None).points,
                 results, seen_parents, limit)
        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []