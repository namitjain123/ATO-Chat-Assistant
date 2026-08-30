"""
Tests for app/services/cache.py — the two-layer (in-process + Redis) cache.

Redis may or may not be running wherever these tests execute (it's optional
infra — see app/services/cache.py's own docstring). These tests deliberately
don't require it: the module is designed to degrade to L1-only automatically,
and that degraded behavior is exactly what's under test here.

Every test uses a fresh uuid4-suffixed key, not just a "descriptive enough"
static string — L2 (Redis) is genuinely persistent by design, so a static
string that passed on a fresh Redis will fail its own "assert miss" on the
*second* run against the same instance, since the previous run's value is
still sitting there. Learned this the hard way: it's exactly the kind of bug
this cache is supposed to guard the app against elsewhere.
"""
import uuid

from app.services import cache


def _unique(label: str) -> str:
    return f"{label}-{uuid.uuid4()}"


def test_embedding_cache_miss_then_hit():
    text = _unique("embedding cache miss then hit")
    assert cache.cache_get_embedding(text) is None

    vector = [0.1, 0.2, 0.3]
    cache.cache_set_embedding(text, vector)

    assert cache.cache_get_embedding(text) == vector


def test_retrieval_cache_miss_then_hit():
    query = _unique("retrieval cache miss then hit")
    limit = 8
    assert cache.cache_get_retrieval(query, limit) is None

    results = [{"content": "foo", "source": "bar.md", "score": 0.9}]
    cache.cache_set_retrieval(query, limit, results)

    assert cache.cache_get_retrieval(query, limit) == results


def test_retrieval_cache_is_keyed_by_limit_too():
    # Same query text, different `limit` — must be treated as different
    # cache entries, since a limit=5 result set isn't a valid limit=8 result.
    query = _unique("retrieval cache limit keying")
    cache.cache_set_retrieval(query, 5, [{"content": "five"}])

    assert cache.cache_get_retrieval(query, 8) is None
    assert cache.cache_get_retrieval(query, 5) == [{"content": "five"}]


def test_key_is_deterministic():
    key1 = cache._key("emb", "some text")
    key2 = cache._key("emb", "some text")
    assert key1 == key2


def test_key_differs_by_namespace():
    # Same literal text, different namespace ("emb" vs "retr") should not
    # collide, since they represent different kinds of cached data.
    key_emb = cache._key("emb", "same text")
    key_retr = cache._key("retr", "same text")
    assert key_emb != key_retr


def test_key_differs_for_different_text():
    key1 = cache._key("emb", "text one")
    key2 = cache._key("emb", "text two")
    assert key1 != key2


def test_get_redis_does_not_raise_when_unavailable():
    # Whether or not Redis is actually running here, this must never raise —
    # the whole point of the two-layer design is graceful degradation.
    result = cache._get_redis()
    assert result is None or hasattr(result, "ping")
