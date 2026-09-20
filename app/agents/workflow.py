"""
The agent graph's wiring, kept free of side effects (no checkpointer, no DB
connection) so tests can compile and run it — graph.py compiles it with the
real checkpointer.

    router ──conversational────────────────────────────────────► responder
      │                                                              ▲
      └─ lookup / explanatory / multi_part / relational              │ sufficient, partial,
            ▼                                                        │ or insufficient
        retriever ◄──────── hop (add missing piece) ──┐              │
            │                                          │              │
            ▼                                          │              │
         grader ──────────────────────────────────────┴──────────────┘
            │ retry (nothing relevant)
            ▼
         rewriter ──► retriever
"""
from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.nodes.router import router_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.grader import grade_node, route_after_grade
from app.agents.nodes.rewriter import rewrite_node
from app.agents.nodes.responder import generate_node


def route_after_router(state: AgentState) -> str:
    return "responder" if state.get("route") == "conversational" else "retriever"


def build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("grader", grade_node)
    workflow.add_node("rewriter", rewrite_node)
    workflow.add_node("responder", generate_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", route_after_router, {"retriever": "retriever", "responder": "responder"})
    workflow.add_edge("retriever", "grader")
    # Two loops out of the grader, both bounded there: retry (rewrite, then
    # search again — MAX_RETRIEVAL_ATTEMPTS) and hop (search for the missing
    # piece, adding to context — MAX_HOPS).
    workflow.add_conditional_edges(
        "grader", route_after_grade,
        {"rewriter": "rewriter", "retriever": "retriever", "responder": "responder"},
    )
    workflow.add_edge("rewriter", "retriever")
    workflow.add_edge("responder", END)
    return workflow
