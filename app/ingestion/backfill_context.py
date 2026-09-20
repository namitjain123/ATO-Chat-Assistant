"""
Backfill contextual-retrieval notes for chunks that were indexed without one.

Annotation failures degrade per chunk (contextualizer._annotate_one) rather
than failing ingestion — first real run: 24 of 318 chunks, all from Azure
429s while the deployment was capped at 10K tokens/minute, with the Groq
fallback also out of daily quota. This re-annotates ONLY those chunks,
re-embeds them (dense + sparse), and overwrites the same points in place —
same ids and parent_ids, so the graph's evidence pointers stay valid.
Idempotent: safe to re-run until nothing is missing.

Usage:  python -m app.ingestion.backfill_context DATA/ato_deductions
        (targets settings.QDRANT_COLLECTION — set QDRANT_COLLECTION to pick one)
"""
import os
import sys
import time
from collections import defaultdict

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.ingestion.contextualizer import contextualize_chunks, build_embedding_inputs, embedding_prefix
from app.ingestion.metadata import income_years
from app.services.retrieval.embedding import embed_texts
from app.services.retrieval.sparse_embedding import embed_documents_sparse

logfire.configure(service_name="enterprise-ingestion-service", token=os.getenv("LOGFIRE_TOKEN"), send_to_logfire="if-token-present")


def _retry(fn, attempts: int = 4):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts - 1:
                raise
            logfire.warning(f"Qdrant call failed ({e}) — retrying.")
            time.sleep(2 ** attempt)


def _all_points(client: QdrantClient) -> list:
    points, offset = [], None
    while True:
        batch, offset = _retry(lambda: client.scroll(
            settings.QDRANT_COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=False,
        ))
        points += batch
        if offset is None:
            return points


def backfill(data_dir: str) -> tuple[int, int]:
    """Returns (fixed, still_missing)."""
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60)
    missing = [p for p in _all_points(client) if not p.payload.get("context")]
    logfire.info(f"{len(missing)} chunks in '{settings.QDRANT_COLLECTION}' have no context note.")

    by_source = defaultdict(list)
    for p in missing:
        by_source[p.payload["source"]].append(p)

    fixed = 0
    for source, points in by_source.items():
        path = os.path.join(data_dir, source)
        if not os.path.exists(path):
            logfire.warning(f"Source document not found, skipping {len(points)} chunks: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            document = f.read()

        with logfire.span("Backfill Context", source=source, chunks=len(points)):
            annotations = contextualize_chunks(document, [p.payload["child_text"] for p in points])
            done = [(p, a) for p, a in zip(points, annotations) if a["context"]]
            if not done:
                continue
            prefixes = [embedding_prefix(p.payload.get("title", ""), p.payload.get("section", ""), a["context"]) for p, a in done]
            inputs = build_embedding_inputs(prefixes, [p.payload["child_text"] for p, _ in done])
            dense, sparse = embed_texts(inputs), embed_documents_sparse(inputs)

            updated = [
                models.PointStruct(
                    id=p.id,
                    vector={settings.DENSE_VECTOR_NAME: d, settings.SPARSE_VECTOR_NAME: s},
                    payload={
                        **p.payload,
                        "context": a["context"],
                        "topics": a["topics"],
                        "income_years": income_years(f"{p.payload['child_text']} {a['context']}"),
                    },
                )
                for (p, a), d, s in zip(done, dense, sparse)
            ]
            _retry(lambda: client.upsert(settings.QDRANT_COLLECTION, points=updated))
            fixed += len(updated)
            logfire.info(f"Backfilled {len(updated)}/{len(points)} chunks from {source}.")

    return fixed, len(missing) - fixed


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "DATA/ato_deductions"
    fixed, still_missing = backfill(data_dir)
    print(f"Backfilled {fixed} chunks in '{settings.QDRANT_COLLECTION}'; {still_missing} still missing context.")
