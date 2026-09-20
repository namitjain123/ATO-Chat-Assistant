"""
Graph retrieval for relationship-heavy questions ("what do I need to claim
X?", "which deductions require written evidence?").

1. Find the question's starting entities via the full-text index.
2. Traverse with fixed, parameterised Cypher — never LLM-written Cypher,
   which is both fragile and an injection surface.
3. Return readable facts plus the parent_ids of the passages they came from,
   so the retriever can pull those passages back out of Qdrant as evidence.

Traversal is deliberately narrow: every relationship one hop from the
starting entities, plus what they inherit from a broader category via
PART_OF. Open-ended 2-hop traversal would route through hub nodes like
"written evidence" and pull in half the graph.
"""
import re
from collections import Counter

import logfire
from neo4j import RoutingControl

from app.config import settings
from app.services.graph.neo4j_client import get_driver

# Several per term, not one: extraction names the same concept slightly
# differently across passages ("deduction for gifts and donations" / "...or
# donations" / "gifts to DGRs" — observed on the live corpus), so the top few
# full-text matches gather those fragments together.
ENTITIES_PER_TERM = 4
MAX_FACTS = 30

_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/&|])')

FIND_ENTITIES = """
CALL db.index.fulltext.queryNodes('entity_names', $q) YIELD node, score
RETURN node.key AS key, node.name AS name, score
ORDER BY score DESC LIMIT $k
"""

NEIGHBOURHOOD = """
MATCH (e:Entity) WHERE e.key IN $keys
MATCH (e)-[r]-(:Entity)
RETURN startNode(r).name AS subject, type(r) AS relation, endNode(r).name AS object,
       r.detail AS detail, r.parent_ids AS evidence, 1 AS hops
UNION
MATCH (e:Entity) WHERE e.key IN $keys
MATCH (e)-[:PART_OF]->(:Entity)-[r]-(other:Entity) WHERE other <> e
RETURN startNode(r).name AS subject, type(r) AS relation, endNode(r).name AS object,
       r.detail AS detail, r.parent_ids AS evidence, 2 AS hops
"""


def lucene_escape(text: str) -> str:
    """User text goes into a Lucene query string — escape its operators so
    "$1,500 (singles)" is searched as words, not parsed as syntax."""
    return _LUCENE_SPECIAL.sub(r"\\\1", text or "").strip()


def format_fact(subject: str, relation: str, obj: str, detail: str | None) -> str:
    fact = f"{subject} —{relation}→ {obj}"
    return f"{fact} ({detail})" if detail else fact


def graph_search(entity_names: list[str], fallback_query: str) -> dict | None:
    """{"entities", "facts", "parent_ids"} — None when the graph is unavailable
    (unconfigured, unreachable, or the query failed), so callers fall back to
    vector search. An empty result just means nothing matched."""
    driver = get_driver()
    if driver is None:
        return None

    terms = [t for t in (entity_names or []) if t.strip()] or [fallback_query]
    try:
        with logfire.span("🕸️ Graph Retrieval", terms=terms):
            matched = {}
            for term in terms:
                q = lucene_escape(term)
                if not q:
                    continue
                records, _, _ = driver.execute_query(
                    FIND_ENTITIES, q=q, k=ENTITIES_PER_TERM,
                    database_=settings.NEO4J_DATABASE, routing_=RoutingControl.READ,
                )
                for rec in records:
                    matched.setdefault(rec["key"], rec["name"])
            if not matched:
                return {"entities": [], "facts": [], "parent_ids": []}

            records, _, _ = driver.execute_query(
                NEIGHBOURHOOD, keys=list(matched),
                database_=settings.NEO4J_DATABASE, routing_=RoutingControl.READ,
            )
    except Exception as e:
        logfire.warning(f"Graph query failed ({e}) — falling back to vector search.")
        return None

    rows = sorted(records, key=lambda r: r["hops"])[:MAX_FACTS]  # direct relationships first
    facts, seen = [], set()
    backs, nearest_hop = Counter(), {}
    for r in rows:
        fact = format_fact(r["subject"], r["relation"], r["object"], r["detail"])
        if fact in seen:
            continue
        seen.add(fact)
        facts.append(fact)
        for pid in r["evidence"] or []:
            backs[pid] += 1
            nearest_hop[pid] = min(nearest_hop.get(pid, r["hops"]), r["hops"])

    logfire.info(f"Graph: {len(matched)} entities, {len(facts)} facts")
    return {
        "entities": list(matched.values()),
        "facts": facts,
        # Evidence behind direct (1-hop) facts first, then passages backing
        # more facts — so the passage that answers the question outranks one
        # that only backs several inherited, indirect facts.
        "parent_ids": sorted(backs, key=lambda pid: (nearest_hop[pid], -backs[pid])),
    }
