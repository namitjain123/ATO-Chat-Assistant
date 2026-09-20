import logfire
import os
from dotenv import load_dotenv

load_dotenv()
# APP_ENV tags every trace, so local test runs (APP_ENV=local) can be filtered
# out of the same Logfire project production reports to.
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), send_to_logfire="if-token-present",
                  environment=os.getenv("APP_ENV", "production"))

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# Shared, never shut down: each /query runs the graph here so the request can
# abandon a pathological call at its deadline. Sized above one so a timed-out
# request's orphaned worker doesn't block the next caller.
_GRAPH_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="graph")

from fastapi import FastAPI, Response
from app.agents.graph import rag_agent
from app.config import settings
from app.guardrails import initialize_rails, guard
from app.services.retrieval.embedding import get_embedding_dim
from app.services.retrieval.ranking_service import warm_up as warm_up_ranker

from pydantic import BaseModel
from typing import Optional


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")


@app.on_event("startup")
def startup_event():
    initialize_rails()
    get_embedding_dim()
    warm_up_ranker()
    logfire.info("✅ Embedding model and reranker warmed up.")

class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"
    
    
@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }
    
    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q) ## return as (true, response) if guardrails fired, else (False, None)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": []
            }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables.
        # Deadline: per-call timeouts bound each LLM call, but not their sum —
        # one question can make ~10 (guardrails, router, a grade per retrieval
        # pass, rewriter, responder). Run it on a worker so a pathological
        # request returns a usable answer instead of hanging the client.
        # NOT a `with` block: ThreadPoolExecutor.__exit__ joins its workers, so
        # exiting one would wait for the very call the deadline is meant to
        # escape. The shared pool is never shut down here.
        future = _GRAPH_POOL.submit(rag_agent.invoke, initial_state, config=config)
        try:
            final_output = future.result(timeout=settings.REQUEST_DEADLINE_SECONDS)
        except FuturesTimeout:
            logfire.error(
                f"⏱️ Request exceeded {settings.REQUEST_DEADLINE_SECONDS}s deadline | thread={thread_id}"
            )
            # The worker keeps running to completion (its checkpoint still
            # lands), but the caller isn't left waiting on it.
            return {
                "question": q,
                "answer": "That took longer than expected to answer. Please try again, or ask a narrower question.",
                "thought_process": ["Timed out before an answer was ready."],
                "status": "timeout",
                "sources": []
            }
        
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }