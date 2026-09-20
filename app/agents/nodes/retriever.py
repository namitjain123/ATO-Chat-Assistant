import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge, fetch_parents
from app.services.retrieval.ranking_service import rerank_with_scores
from app.services.graph.graph_retrieval import graph_search

GRAPH_EVIDENCE_PASSAGES = 8
# A hop searches for one missing piece: a tight top-5, no filters (the piece
# may sit under a different topic than the original question), added to the
# context rather than replacing it. Total context capped so hops can't bloat
# the responder's prompt.
HOP_STRATEGY = {"candidates": 20, "keep": 5}
MAX_CONTEXT_DOCS = 14

# Per-route retrieval strategy (see router.py for what each route means).
#   candidates: Qdrant results fetched (per sub-query for multi_part)
#   keep:       passages kept after reranking
# lookup keeps fewer, tighter passages — a figure or threshold lives in one
# place, and extra loosely-related passages only give the model more to
# confuse it with. explanatory keeps more for fuller coverage (20 -> 8 was
# widened from 15 -> 5 earlier to improve Context Recall).
STRATEGIES = {
    "lookup": {"candidates": 20, "keep": 5},
    "explanatory": {"candidates": 20, "keep": 8},
    "multi_part": {"candidates": 10, "keep": 10},
    "relational": {"candidates": 20, "keep": 8},  # plus the graph's evidence passages
}


def _with_header(doc: dict) -> str:
    """'[Page › Section]' header: a locator for the reranker and responder."""
    title, section = doc.get("title", ""), doc.get("section", "")
    header = f"{title} › {section}" if title and section else title or section
    return f"[{header}]\n{doc['content']}" if header else doc["content"]


def _search_and_rerank(query: str, candidates: int, keep: int, filters: dict | None) -> list[tuple[str, float]]:
    docs = search_enterprise_knowledge(query, limit=candidates, filters=filters)
    return rerank_with_scores(query, [_with_header(d) for d in docs], top_n=keep)


def _interleave(per_part: list[list[tuple[str, float]]], keep: int) -> list[tuple[str, float]]:
    """Round-robin across sub-queries so every part of the question is
    represented, instead of the strongest part crowding out the rest."""
    merged, seen = [], set()
    for rank in range(max((len(p) for p in per_part), default=0)):
        for part in per_part:
            if rank < len(part) and part[rank][0] not in seen:
                seen.add(part[rank][0])
                merged.append(part[rank])
    return merged[:keep]


def _relational(query: str, entities: list[str], strategy: dict, filters: dict | None):
    """Knowledge graph + vector search together. The graph contributes (a)
    relationship facts for the responder and (b) the parent passages those
    facts were extracted from, pulled back out of Qdrant. Vector search runs
    alongside — the graph only knows what extraction captured, and misses
    anything phrased outside its schema. Both candidate sets are reranked
    together, so graph evidence competes on relevance like everything else.
    Returns (ranked, facts, plan note)."""
    graph = graph_search(entities, query)
    evidence = fetch_parents(graph["parent_ids"], limit=GRAPH_EVIDENCE_PASSAGES) if graph else []
    vector = search_enterprise_knowledge(query, limit=strategy["candidates"], filters=filters)

    candidates, seen = [], set()
    for doc in evidence + vector:
        if doc["content"] not in seen:
            seen.add(doc["content"])
            candidates.append(doc)
    ranked = rerank_with_scores(query, [_with_header(d) for d in candidates], top_n=strategy["keep"])

    if graph is None:
        note = "Graph: unavailable — vector search only"
    else:
        note = (f"Graph: {len(graph['entities'])} entities matched, {len(graph['facts'])} relationships, "
                f"{len(evidence)} evidence passages")
    return ranked, (graph["facts"] if graph else []), note


def _hop(state: AgentState):
    """Follow-up search for what the grader found missing — accumulates."""
    hop, query = state.get("hops") or 1, state["current_query"]
    with logfire.span("🔁 Retrieval Hop", hop=hop, query=query):
        docs = search_enterprise_knowledge(query, limit=HOP_STRATEGY["candidates"])
        ranked = rerank_with_scores(query, [_with_header(d) for d in docs], top_n=HOP_STRATEGY["keep"])

    existing = state.get("documents") or []
    new = [f"CONTENT: {text}" for text, _ in ranked if f"CONTENT: {text}" not in existing]
    scores = [s for _, s in ranked if s is not None]
    top = max(scores) if scores else (None if ranked else 0.0)
    shown = f"{top:.2f}" if top is not None else "unscored"
    return {
        "documents": (existing + new)[:MAX_CONTEXT_DOCS],
        "top_relevance": top,  # of THIS hop — the grader judges whether the hop found anything
        "hop_new_docs": len(new),  # 0 = only re-found existing passages; the grader stops hopping
        "status": f"Hop {hop}: searching for missing information.",
        "plan": state["plan"] + [f"Hop {hop} retrieved {len(new)} new passages (top relevance {shown})"],
    }


def retrieve_node(state: AgentState):
    """Search with the strategy the router chose, then rerank and score."""
    if state.get("retrieval_mode") == "hop":
        return _hop(state)

    route = state.get("route") or "explanatory"
    filters = state.get("search_filters")
    sub_queries = state.get("sub_queries") or []
    # multi_part's numbers are per sub-query; once the rewriter collapses it to
    # one broadened search, it's searched like an explanatory question.
    strategy = STRATEGIES["explanatory"] if route == "multi_part" and not sub_queries else STRATEGIES.get(route, STRATEGIES["explanatory"])
    attempt = (state.get("retrieval_attempts") or 0) + 1

    graph_facts, extra_plan = [], []
    with logfire.span("🔍 Knowledge Retrieval", route=route, attempt=attempt):
        if route == "relational":
            ranked, graph_facts, note = _relational(
                state["current_query"], state.get("entities") or [], strategy, filters,
            )
            extra_plan.append(note)
        elif route == "multi_part" and sub_queries:
            per_part_keep = max(3, strategy["keep"] // len(sub_queries))
            per_part = []
            for sub_query in sub_queries:
                logfire.info(f"Searching Qdrant for sub-query: {sub_query}")
                per_part.append(_search_and_rerank(sub_query, strategy["candidates"], per_part_keep, filters))
            ranked = _interleave(per_part, strategy["keep"])
        else:
            query = state["current_query"]
            logfire.info(f"Searching Qdrant for: {query}")
            ranked = _search_and_rerank(query, strategy["candidates"], strategy["keep"], filters)

        scores = [s for _, s in ranked if s is not None]
        # None = reranker unavailable: relevance unknown, not zero (see grader).
        top_relevance = max(scores) if scores else (None if ranked else 0.0)
        logfire.info(f"Kept {len(ranked)} passages, top relevance {top_relevance}")

    relevance = f"{top_relevance:.2f}" if top_relevance is not None else "unscored"
    return {
        "documents": [f"CONTENT: {text}" for text, _ in ranked],
        "graph_facts": graph_facts,
        "top_relevance": top_relevance,
        "retrieval_attempts": attempt,
        "status": "Found technical context.",
        # "Context Retrieved" is what the eval suite's detect_tool keys on.
        "plan": state["plan"] + extra_plan + [f"Context Retrieved (attempt {attempt}, {len(ranked)} passages, top relevance {relevance})"],
    }
