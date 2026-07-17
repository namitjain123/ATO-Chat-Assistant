from pydantic import BaseModel, Field
from app.agents.state import AgentState
from app.gateway.client import get_langchain_llm
import logfire

# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatGroq
llm = get_langchain_llm(feature="planner")


class PlannerDecision(BaseModel):
    needs_search: bool = Field(
        description="True if this requires searching the document store to answer well; "
        "false ONLY for greetings/farewells/small-talk, or questions answerable purely "
        "from the conversation history (e.g. 'what did I just ask you'). Never set this "
        "to false just because you think you already know the answer — always defer to "
        "the document store for substantive questions."
    )
    search_query: str = Field(
        description="A refined search query for the topic, if needs_search is true; "
        "empty string otherwise."
    )


# function_calling (tool-calling) rather than the default json_schema mode — Groq's
# llama models don't support OpenAI's native structured-output response format.
structured_llm = llm.with_structured_output(PlannerDecision, method="function_calling")


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner deciding whether to search a document store.

    EXAMPLES:
    - "hello" -> needs_search=false (greeting)
    - "what did I just ask you?" -> needs_search=false (answerable from history alone)
    - "How do I reset my password?" -> needs_search=true, query="password reset process"
    - "What is the capital of France?" -> needs_search=true, query="capital of France" (even general-knowledge-sounding questions must search — never answer from your own knowledge)
    - "What deductions can I claim?" -> needs_search=true, query="tax deductions eligibility"

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Decide whether answering this well requires searching the document store.
    """

    with logfire.span("Planner Decision"):
        decision = structured_llm.invoke(prompt)
        logfire.info(f"Intent identified: needs_search={decision.needs_search}, query={decision.search_query!r}")

    if not decision.needs_search:
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    return {
        "current_query": decision.search_query,
        "status": f"Technical research needed. Searching for: {decision.search_query}",
        "plan": ["Intent: Technical", f"Search Term: {decision.search_query}"]
    }
