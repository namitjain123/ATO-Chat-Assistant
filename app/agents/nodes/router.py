"""
Router: the first decision in the graph. Replaces a planner that only chose
search-or-don't — the router picks a retrieval *strategy* per question, and
the graph can loop back (grader -> rewriter -> retriever) when that strategy
comes up empty. See README's "Retrieval Routing" section.
"""
from pydantic import BaseModel, Field
from app.agents import history
from app.agents.state import AgentState
from app.gateway.client import get_structured_llm_with_fallback
from app.ingestion.metadata import TOPICS, normalize_topics, income_years
import logfire

ROUTES = ("conversational", "lookup", "explanatory", "multi_part", "relational")
MAX_SUB_QUERIES = 4


# All fields required (not defaulted): Groq's tool-call validation has treated
# defaulted fields as required before and rejected the call.
class RouteDecision(BaseModel):
    route: str = Field(
        description="One of: "
        "'conversational' — greetings, small talk, or questions answerable purely from the "
        "conversation history; never for substantive questions, even ones you think you know. "
        "'lookup' — a specific fact: a figure, threshold, rate, limit, date, or yes/no eligibility. "
        "'explanatory' — how/why/what-counts questions needing a fuller explanation. "
        "'multi_part' — several distinct questions, or a comparison between things. "
        "'relational' — how things CONNECT: what something requires or depends on, which things "
        "share a requirement or limit, what applies to an occupation/organisation, what a category "
        "includes (e.g. 'which deductions need written evidence?', 'what do I need before I can "
        "claim a donation?')."
    )
    search_query: str = Field(
        description="A standalone search query for the whole question (resolve references to earlier "
        "turns, use official ATO terminology). Empty string for 'conversational'."
    )
    sub_queries: list[str] = Field(
        description=f"'multi_part' only: 2-{MAX_SUB_QUERIES} standalone search queries, one per part. "
        "Empty list for every other route."
    )
    entities: list[str] = Field(
        description="'relational' only: the specific concepts the question is about, as short "
        "names (e.g. ['donations', 'deductible gift recipient']). Empty list for every other route."
    )
    topics: list[str] = Field(
        description=f"Search filter. Topics the question is clearly about, chosen only from: "
        f"{', '.join(TOPICS)}. Leave EMPTY when unsure or when the question spans many "
        "topics — an empty list searches everything, a wrong topic hides relevant answers."
    )
    income_year: str = Field(
        description="Search filter. Australian financial year as 'YYYY-YY' (e.g. '2024-25'), ONLY "
        "when the user explicitly names a financial/income year ('2024-25', '2025-26 income "
        "year', 'FY2025' = '2024-25'). Empty string for a bare calendar year like '2025' or "
        "no year at all."
    )


# function_calling rather than json_schema mode — works across both Azure and
# the Groq fallback targets (see app/gateway/client.py).
structured_llm = get_structured_llm_with_fallback(RouteDecision, feature="router", method="function_calling")


def _normalize_route(route: str) -> str:
    route = (route or "").strip().lower().replace("-", "_").replace(" ", "_")
    # An unrecognised route still searches — defaulting to conversational would
    # answer a real question from memory alone.
    return route if route in ROUTES else "explanatory"


def router_node(state: AgentState):
    """Classify the latest message into a retrieval route, with filters."""
    # Compaction runs here, not in the responder: the router is the first node
    # every turn, so the summary it writes is already in state by the time the
    # responder builds its own prompt — one compaction per turn, not two.
    summary = state.get("history_summary") or ""
    summarized = state.get("summarized_msgs") or 0
    to_compact, verbatim = history.plan(state["messages"], summarized)
    if to_compact:
        summary = history.compact(summary, to_compact)
        summarized += len(to_compact)
    history_text = history.format_for_prompt(summary, verbatim)

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You route questions for an Australian Taxation Office (ATO) knowledge-base assistant,
    choosing how to search the document store.

    EXAMPLES:
    - "hello" -> route=conversational
    - "what did I just ask you?" -> route=conversational (answerable from history alone)
    - "What is the capital of France?" -> route=lookup, search_query="capital of France" (never answer from your own knowledge — search)
    - "How much can I claim for gifts to political parties?" -> route=lookup, search_query="deduction limit gifts political parties", topics=["deductions"]
    - "How do I work out what I can claim for working from home?" -> route=explanatory, search_query="working from home expenses deduction method", topics=["deductions"]
    - "What records do I need for work expenses in 2025-26?" -> route=explanatory, search_query="record keeping work-related expenses", topics=["record_keeping", "deductions"], income_year="2025-26"
    - "Can I claim union fees, and what records do I need for donations?" -> route=multi_part, sub_queries=["union fees deduction", "records for gifts and donations"], topics=["deductions", "record_keeping"]
    - "Which deductions require written evidence?" -> route=relational, search_query="deductions requiring written evidence", entities=["written evidence"], topics=["deductions", "record_keeping"]
    - "What do I need before I can claim a donation?" -> route=relational, search_query="requirements to claim gift donation deduction", entities=["donations", "deductible gift recipient"], topics=["deductions"]
    - "tax slab for 2025" -> route=lookup, search_query="individual income tax rates", topics=["tax_rates"], income_year="" (bare calendar year — ambiguous, don't filter)

    CONVERSATION HISTORY:
    {history_text}

    LATEST MESSAGE:
    "{user_message}"
    """

    with logfire.span("Retrieval Routing"):
        decision = structured_llm.invoke(prompt)
        route = _normalize_route(decision.route)
        logfire.info(f"Route: {route}, query={decision.search_query!r}, sub_queries={decision.sub_queries!r}")

    # Per-turn fields — state is checkpointed per thread, so anything not reset
    # here would leak from the previous turn (e.g. last turn's sources showing
    # up under a conversational reply).
    reset = {
        "retrieval_attempts": 0, "top_relevance": 0.0, "retrieval_grade": "", "documents": [], "graph_facts": [],
        "retrieval_mode": "initial", "hops": 0, "hop_queries": [], "missing_info": "",
        # Carried on every return so a compaction done this turn is persisted
        # even when the turn is conversational.
        "history_summary": summary, "summarized_msgs": summarized,
    }

    if route == "conversational":
        return {
            **reset,
            "route": route,
            "current_query": "CONVERSATIONAL",
            "sub_queries": [],
            "entities": [],
            "search_filters": {},
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Route: conversational", "Retrieval: Skipped"],
        }

    search_query = decision.search_query.strip() or user_message
    sub_queries = [q.strip() for q in decision.sub_queries if q.strip()][:MAX_SUB_QUERIES]
    if route == "multi_part" and len(sub_queries) < 2:
        route, sub_queries = "explanatory", []  # "multi-part" with one part is just one search
    elif route != "multi_part":
        sub_queries = []
    entities = [e.strip() for e in decision.entities if e.strip()][:MAX_SUB_QUERIES] if route == "relational" else []

    # Validate before they reach Qdrant — an invented topic or a malformed
    # year is dropped rather than silently filtering everything out.
    search_filters = {
        "topics": normalize_topics(decision.topics),
        "income_year": (income_years(decision.income_year) or [""])[0],
    }

    # "Intent: Technical" / "Search Term:" are what the eval suite's
    # detect_tool keys on (evals/pipeline.py) — keep them.
    plan = ["Intent: Technical", f"Route: {route}", f"Search Term: {search_query}"]
    if sub_queries:
        plan.append(f"Sub-queries: {' | '.join(sub_queries)}")
    if entities:
        plan.append(f"Graph entities: {' | '.join(entities)}")
    if search_filters["topics"] or search_filters["income_year"]:
        plan.append(f"Filters: topics={search_filters['topics']}, income_year={search_filters['income_year'] or 'any'}")

    return {
        **reset,
        "route": route,
        "current_query": search_query,
        "question": search_query,
        "sub_queries": sub_queries,
        "entities": entities,
        "search_filters": search_filters,
        "status": f"Route: {route}. Searching for: {search_query}",
        "plan": plan,
    }
