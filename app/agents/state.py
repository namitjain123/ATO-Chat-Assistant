from typing import TypedDict, List, Annotated
import operator


class AgentState(TypedDict):
    # Using Annotated with operator.add ensures that messages
    # are appended to the history rather than replaced.
    messages: Annotated[List[dict], operator.add]
    current_query: str
    # Bounded history (app/agents/history.py): older turns live in the summary,
    # recent ones stay verbatim in `messages`. Persist across turns per thread.
    history_summary: str
    summarized_msgs: int      # how many of `messages` the summary already covers
    search_filters: dict  # {"topics": [...], "income_year": "YYYY-YY" or ""} — set by the router
    documents: List[str]
    plan: List[str]
    status: str
    final_answer: str

    # Retrieval routing — all reset by the router every turn (state is
    # checkpointed per thread, so stale values would otherwise carry over).
    route: str                # conversational | lookup | explanatory | multi_part | relational
    sub_queries: List[str]    # multi_part only: one search per part
    entities: List[str]       # relational only: starting points in the knowledge graph
    graph_facts: List[str]    # relational only: relationships found in Neo4j
    retrieval_attempts: int   # passes so far this turn (first search + rewrites)
    top_relevance: float      # best reranker score from the latest pass
    retrieval_grade: str      # "" | relevant | sufficient | retry | hop | partial | insufficient
    question: str             # standalone form of the user's question (from the router) — what sufficiency is judged against
    retrieval_mode: str       # "initial" (replace documents) | "hop" (add to them)
    hops: int                 # follow-up searches done this turn
    hop_queries: List[str]    # follow-ups already tried — never repeated
    hop_new_docs: int         # passages the latest hop added (0 = no progress, stop)
    missing_info: str         # what the knowledge base couldn't supply (for an honest partial answer)
