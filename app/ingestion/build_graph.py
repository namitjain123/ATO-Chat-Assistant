"""
Build (or rebuild) the Neo4j knowledge graph from an already-ingested Qdrant
collection — no re-parsing, re-chunking, re-annotation or re-embedding.

Reads each parent chunk (parent_id + text) back out of Qdrant, so every
relationship's evidence pointer is a parent_id that exists in that exact
collection. Use it when Neo4j is added after ingestion, or to re-extract
after changing the graph schema.

Usage:  python -m app.ingestion.build_graph            (adds to the graph)
        python -m app.ingestion.build_graph --wipe     (clears this project's graph first)
        (reads settings.QDRANT_COLLECTION — set QDRANT_COLLECTION to pick one)
"""
import os
import sys
from collections import defaultdict

import logfire

from app.config import settings
from app.ingestion.backfill_context import _all_points
from app.ingestion.graph_extractor import index_document_graph
from app.services.graph.neo4j_client import get_driver, ensure_schema, wipe_graph
from qdrant_client import QdrantClient

logfire.configure(service_name="enterprise-ingestion-service", token=os.getenv("LOGFIRE_TOKEN"), send_to_logfire="if-token-present")


def build(wipe: bool = False) -> int:
    driver = get_driver()
    if driver is None:
        raise SystemExit("Neo4j unavailable — check NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD and that the instance is running.")
    if wipe:
        wipe_graph(driver)
    ensure_schema(driver)

    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60)
    # {source: (title, {parent_id: parent_text})} — every child point carries its parent's text.
    docs = defaultdict(lambda: ["", {}])
    for p in _all_points(client):
        pid = p.payload.get("parent_id")
        if not pid:
            raise SystemExit(f"'{settings.QDRANT_COLLECTION}' has no parent_ids — it predates parent-child ingestion.")
        doc = docs[p.payload["source"]]
        doc[0] = p.payload.get("title", "")
        doc[1][pid] = p.payload["text"]

    total = 0
    for source, (title, parents) in sorted(docs.items()):
        try:
            total += index_document_graph(driver, title, source, parents)
        except Exception as e:
            logfire.error(f"Graph build failed for {source}: {e}")
    return total


if __name__ == "__main__":
    written = build(wipe="--wipe" in sys.argv)
    print(f"Graph built from '{settings.QDRANT_COLLECTION}': {written} relationships written.")
