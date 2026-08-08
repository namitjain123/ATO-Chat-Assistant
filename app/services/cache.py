"""
Two-layer cache for the retrieval pipeline.

L1 (in-process, cachetools TTLCache): near-zero latency, but scoped to this
one worker process — lost on restart, not shared across replicas.
L2 (Redis): shared across every app instance/replica and survives restarts —
the layer that actually matters once this scales beyond a single container.

Why this exists specifically for embedding + retrieval (not LLM responses):
Portkey's gateway config already caches full LLM completions
(app/gateway/client.py: {"cache": {"mode": "simple"}}) — duplicating that at
this layer would be redundant. But embedding calls (Gemini, direct — bypasses
Portkey entirely) and Qdrant vector search are not covered by that cache at
all, and both are fully deterministic for a given input, so they're safe and
valuable to cache here.

Redis is optional infrastructure: if it's unreachable (not running locally,
not provisioned yet, etc.) the app degrades to L1-only rather than crashing —
logged loudly once, not silently swallowed (see notes.md's running list of
"silent failure" bugs this project has hit and fixed).
"""
import hashlib
import json

import logfire
from cachetools import TTLCache

from app.config import settings

REDIS_URL = settings.REDIS_URL

L1_MAXSIZE = 2_000
EMBEDDING_L1_TTL = 60 * 60        # 1h — in-process cache, small footprint is fine to keep short
RETRIEVAL_L1_TTL = 5 * 60         # 5m

EMBEDDING_L2_TTL = 24 * 60 * 60   # 24h — embeddings are deterministic for the same text, safe to keep long
RETRIEVAL_L2_TTL = 15 * 60        # 15m — shorter: underlying Qdrant content can change after a re-ingestion

_l1_embedding: TTLCache = TTLCache(maxsize=L1_MAXSIZE, ttl=EMBEDDING_L1_TTL)
_l1_retrieval: TTLCache = TTLCache(maxsize=L1_MAXSIZE, ttl=RETRIEVAL_L1_TTL)

_redis_client = None
_redis_checked = False


def _get_redis():
    """Lazy-connect to Redis once per process. Returns None (not an exception)
    if unreachable, so callers can degrade to L1-only without special-casing."""
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True
    try:
        import redis
        client = redis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
        client.ping()
        logfire.info(f"✅ Redis L2 cache connected ({REDIS_URL}).")
        _redis_client = client
    except Exception as e:
        logfire.warning(
            f"⚠️ Redis L2 cache unavailable ({e}) — running with L1 (in-process) cache only. "
            f"Start Redis locally (`docker compose up redis`) to enable the shared layer."
        )
        _redis_client = None
    return _redis_client


def _key(namespace: str, *parts: str) -> str:
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def cache_get_embedding(text: str):
    key = _key("emb", text)
    if key in _l1_embedding:
        return _l1_embedding[key]

    r = _get_redis()
    if r is not None:
        try:
            raw = r.get(key)
            if raw is not None:
                value = json.loads(raw)
                _l1_embedding[key] = value  # backfill L1
                return value
        except Exception as e:
            logfire.warning(f"Redis L2 read failed for embedding cache: {e}")
    return None


def cache_set_embedding(text: str, vector: list[float]) -> None:
    key = _key("emb", text)
    _l1_embedding[key] = vector
    r = _get_redis()
    if r is not None:
        try:
            r.setex(key, EMBEDDING_L2_TTL, json.dumps(vector))
        except Exception as e:
            logfire.warning(f"Redis L2 write failed for embedding cache: {e}")


def cache_get_retrieval(query: str, limit: int):
    key = _key("retr", query, str(limit))
    if key in _l1_retrieval:
        return _l1_retrieval[key]

    r = _get_redis()
    if r is not None:
        try:
            raw = r.get(key)
            if raw is not None:
                value = json.loads(raw)
                _l1_retrieval[key] = value  # backfill L1
                return value
        except Exception as e:
            logfire.warning(f"Redis L2 read failed for retrieval cache: {e}")
    return None


def cache_set_retrieval(query: str, limit: int, results: list[dict]) -> None:
    key = _key("retr", query, str(limit))
    _l1_retrieval[key] = results
    r = _get_redis()
    if r is not None:
        try:
            r.setex(key, RETRIEVAL_L2_TTL, json.dumps(results))
        except Exception as e:
            logfire.warning(f"Redis L2 write failed for retrieval cache: {e}")
