import time
import copy
import json
import os
import uuid
import requests
import logfire

# Defaults to local; set EVAL_TARGET_URL to point Phase 1 at a deployed
# backend instead (e.g. the live Azure Container App) — same golden dataset,
# same scoring, just testing the real deployed instance rather than a
# laptop. No code change needed to switch back: unset the env var.
API_URL = os.getenv("EVAL_TARGET_URL", "http://localhost:8000/query")
# Safety cap only (e.g. a runaway/looping completion) — NOT a display truncation.
# Real answers were previously hard-cut at 300 chars, which fed Phase 2's Faithfulness
# and Answer Correctness metrics a broken, incomplete sentence and unfairly tanked
# their scores. Both eval-UI tables already do their own short re-truncation for
# display (evals/app.py), so storing the full response here doesn't affect the UI.
RESPONSE_TRUNCATE = 2000
DELAY_BETWEEN_CALLS = 20   # seconds — each /query triggers ~3-5 internal Groq calls (guardrails + planner + responder); 10s was hitting the free-tier TPM ceiling
REQUEST_TIMEOUT = 120      # seconds — guardrails + LangGraph + Groq can take >60s




def detect_tool(thought_process: list) -> str:
    """
    Maps the thought_process list from /query response to a tool name.
    Router sets:   'Intent: Technical' + 'Search Term: ...' → retrieve_documents
                   'Intent: Conversational/Memory'           → direct_answer
    main.py sets:  'Intent: Guardrails Fired'                → guardrails
    """
    joined = " ".join(thought_process).lower()
    if "guardrails fired" in joined:
        return "guardrails"
    if "intent: technical" in joined or "search term:" in joined or "context retrieved" in joined:
        return "retrieve_documents"
    if "conversational" in joined or "memory" in joined:
        return "direct_answer"
    return "unknown"


def run_pipeline(golden_dataset: dict, progress_callback=None) -> dict:
    """
    Enriches each rag_sample in golden_dataset with live API results.
    Returns a deep copy with actual_response, actual_contexts, actual_tools_called filled.
    progress_callback(i, total, question, stage, response="") is called per step.
    """
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    n = len(samples)
    # thread_id used to be a fixed f"eval_run_{i}" — identical across every
    # Phase 1 invocation, so the backend's conversational-memory checkpointer
    # accumulated history across DIFFERENT eval runs (not just within one run).
    # Once a similar-sounding question had been asked under the same thread_id
    # in an earlier run, the planner treated new questions as continuations
    # and skipped retrieval entirely — silently zeroing out actual_contexts
    # and tanking Context Recall/Precision/Faithfulness for reasons that had
    # nothing to do with real retrieval quality. Each run now gets its own
    # unique thread namespace.
    run_id = uuid.uuid4().hex[:8]

    with logfire.span("🚀 Eval Phase 1 — Live Pipeline", total_samples=n):
        for i, sample in enumerate(samples):
            question = sample["question"]

            if progress_callback:
                progress_callback(i, n, question, "calling")

            with logfire.span(
                f"📤 Live Query {i + 1}/{n}",
                question=question[:80],
                domain=sample.get("domain", ""),
            ):
                try:
                    resp = requests.post(
                        API_URL,
                        json={"q": question, "thread_id": f"eval_{run_id}_{i}"},
                        timeout=REQUEST_TIMEOUT,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    raw_answer = data.get("answer") or ""
                    thought_process = data.get("thought_process") or []
                    sources = data.get("sources") or []

                    sample["actual_response"] = raw_answer[:RESPONSE_TRUNCATE]
                    sample["actual_contexts"] = sources[:8]  # matches retriever.py's top_n=8
                    sample["actual_tools_called"] = [detect_tool(thought_process)]

                    logfire.info(
                        "✅ Response captured",
                        tool=sample["actual_tools_called"][0],
                        response_chars=len(raw_answer),
                        context_chunks=len(sources),
                    )

                except requests.exceptions.ConnectionError:
                    logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_tools_called"] = ["unknown"]

                except Exception as e:
                    logfire.error(f"❌ Query failed: {e}")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_tools_called"] = ["unknown"]

            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_response"])

            if i < n - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    return dataset


def save_results(dataset: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "eval_config.json")


def get_sample_limit() -> int | None:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, encoding="utf-8") as f:
            n = json.load(f).get("sample_limit")
            if isinstance(n, int) and n > 0:
                return n
    return None


def set_sample_limit(n: int | None) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"sample_limit": n}, f)


def load_golden_dataset() -> dict:
    golden_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(golden_path, encoding="utf-8") as f:
        data = json.load(f)

    # Optional cap for a quick demo run instead of the full golden set. Stored
    # in eval_config.json (not just an EVAL_SAMPLE_LIMIT env var) because a
    # Streamlit process restart wipes shell env vars set via a one-off
    # `$env:X = ...; Start-Process ...` — this file survives that.
    n = get_sample_limit()
    if n:
        data["rag_samples"] = data["rag_samples"][:n]

    return data