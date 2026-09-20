import os

from langgraph.checkpoint.memory import MemorySaver
from app.agents.workflow import build_workflow


# Wiring (router -> retriever -> grader -> rewriter loop -> responder) lives in
# app/agents/workflow.py, side-effect free so tests can run it.
workflow = build_workflow()


# --- MEMORY UPGRADE ---
# MemorySaver keeps thread state in this process's RAM only — fine for a single
# local instance, but silently loses conversations across restarts or when
# scaled to >1 replica. POSTGRES_URL switches to a shared, durable checkpointer
# so conversation memory survives deploys/restarts and works behind autoscaling.


def _build_checkpointer():
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        return MemorySaver()

    from psycopg_pool import ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver

    pool = ConnectionPool(
        conninfo=postgres_url,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    return checkpointer


checkpointer = _build_checkpointer()


# 4. Compile the Graph with Memory
rag_agent = workflow.compile(checkpointer=checkpointer)