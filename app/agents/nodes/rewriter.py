"""
Rewriter: when the grader finds nothing relevant, reformulate the search
instead of giving up. Only runs on that unhappy path — the common case
(relevant first try) costs no extra LLM call.
"""
from pydantic import BaseModel, Field
import logfire
from app.agents.state import AgentState
from app.gateway.client import get_structured_llm_with_fallback


class RewrittenQuery(BaseModel):
    search_query: str = Field(description="A different standalone search query for the same question.")


structured_llm = get_structured_llm_with_fallback(RewrittenQuery, feature="rewriter", method="function_calling")


def rewrite_node(state: AgentState):
    question = state["messages"][-1]["content"] if state["messages"] else ""
    tried = state.get("sub_queries") or [state["current_query"]]

    prompt = f"""
    A search of an Australian Taxation Office (ATO) knowledge base found nothing relevant.

    USER QUESTION: "{question}"
    SEARCHES THAT FOUND NOTHING: {tried}

    Write ONE different search query for the same question: use the official ATO
    terminology for the concept, or broaden it to the general topic the answer
    would sit under. Don't repeat a search that already failed.
    """
    with logfire.span("Query Rewrite"):
        new_query = structured_llm.invoke(prompt).search_query.strip() or question
        logfire.info(f"Rewrote {tried!r} -> {new_query!r}")

    return {
        "current_query": new_query,
        # One broadened query from here on — a multi-part split that found
        # nothing is retried as a single search.
        "sub_queries": [],
        # Graph lookup restarts from the rewritten query's words too.
        "entities": [],
        # Filters may be why nothing matched; the retry searches unfiltered.
        "search_filters": {},
        "plan": state["plan"] + [f"Rewritten Search Term: {new_query}"],
    }
