import os
import sys
import types
import asyncio
import logfire
import pandas as pd
from openai import AsyncOpenAI

# ragas 0.4.3 still imports the legacy langchain_community.chat_models.vertexai
# path, which langchain-community has since removed in favor of the standalone
# langchain-google-vertexai package (already a project dependency — see
# requirements.txt's NeMo Guardrails shim comment for the same class of issue).
# Register a shim so the old import path resolves to the real installed class.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
    _vertexai_shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertexai_shim.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_shim

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas import SingleTurnSample
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MODEL = "openai/gpt-oss-20b"  # llama-3.1-8b-instant AND llama-3.3-70b-versatile both hit their daily (TPD) quota today
# Cooldowns pace requests under Groq's per-minute (TPM) limit. Defaults are tuned
# for large runs; for a small sample count they're overkill, so they're overridable
# via env vars (EVAL_COOLDOWN_STANDARD / EVAL_COOLDOWN_MINI). The per-sample resilience
# in _batched_score still catches any 429 if the cooldown is set too aggressively.
COOLDOWN_STANDARD = int(os.getenv("EVAL_COOLDOWN_STANDARD", "62"))
COOLDOWN_MINI = int(os.getenv("EVAL_COOLDOWN_MINI", "40"))  # between samples — lets the sliding TPM window recover (~2,800 tok/sample)
GENERAL_BATCH_SIZE = 1  # one sample at a time: abatch_score fires calls concurrently per sample,
                         # so batch>1 stacks multiple samples' async calls inside the same second
# Raised from 300/2 (600 chars total) — that was so small vs. the now-untruncated
# ~2000-char response that Faithfulness couldn't find support for genuinely-grounded
# claims and scored ~0. Still capped (not the full ~1500 chars/chunk x5) to stay
# under Groq's TPM ceiling. Override via env if a specific judge model needs tuning.
CONTEXT_TRUNCATE = int(os.getenv("EVAL_CONTEXT_TRUNCATE", "800"))  # chars per context chunk
CONTEXT_LIMIT = int(os.getenv("EVAL_CONTEXT_LIMIT", "8"))          # number of context chunks passed to RAGAS per sample — matches retriever.py's top_n=8


def _build_judge():
    api_key = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_API_KEY")
    client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    # openai/gpt-oss-20b is a reasoning model — it spends tokens on internal reasoning
    # before writing the final JSON. 4096 was already raised once (from a 1024 default
    # that caused truncated/invalid JSON) but still isn't enough headroom for some
    # calls — confirmed via Logfire: "InstructorRetryException: Failed to validate
    # JSON" on Faithfulness's statement-generation step for a longer response.
    llm = llm_factory(JUDGE_MODEL, provider="openai", client=client, max_tokens=8192)
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
    )
    return llm, embeddings

async def _cooldown(seconds: int, label: str, status_cb=None):
    msg = f"⏳ {seconds}s cooldown after {label} (Groq TPM buffer)..."
    if status_cb:
        status_cb(msg)
    for _ in range(seconds // 10):
        await asyncio.sleep(10)
    if status_cb:
        status_cb(f"✅ Ready — starting next experiment.")
        
        
def _prep_samples(golden_dataset: dict) -> list:
    """
    Returns only samples with actual_response populated.
    Truncates contexts to CONTEXT_TRUNCATE chars and limits to CONTEXT_LIMIT chunks
    so a single RAGAS LLM call stays well under the 6,000 TPM ceiling.
    (Live contexts from Qdrant are ~1,500 chars each — without truncation a single
    Faithfulness request exceeds 7,000 tokens which hard-fails on the on_demand tier.)
    """
    valid = []
    for s in golden_dataset["rag_samples"]:
        response = s.get("actual_response", "").strip()
        if not response:
            continue
        raw_contexts = s.get("actual_contexts") or s.get("relevant_contexts") or []
        contexts = [c[:CONTEXT_TRUNCATE] for c in raw_contexts[:CONTEXT_LIMIT]]
        valid.append({**s, "actual_contexts": contexts})
    return valid


def _score_df(metric_key: str, samples: list, scores) -> pd.DataFrame:
    rows = []
    for s, r in zip(samples, scores):
        value = round(float(r.value), 3) if r is not None else None
        rows.append({"question": s["question"][:65], metric_key: value})
    return pd.DataFrame(rows)


async def _batched_score(metric, inputs: list, samples: list, status_cb=None, label: str = "") -> list:
    """
    Runs abatch_score in chunks of GENERAL_BATCH_SIZE with cooldowns between chunks.
    Keeps each burst under 6,000 TPM on Groq's on_demand tier.

    A single sample's failure (bad JSON from the judge, a transient API error, etc.)
    is caught and recorded as None rather than aborting the whole experiment —
    one flaky judge response shouldn't cost the other ~9 samples' results.
    """
    all_scores = []
    batches = [inputs[i : i + GENERAL_BATCH_SIZE] for i in range(0, len(inputs), GENERAL_BATCH_SIZE)]
    for b_idx, batch in enumerate(batches):
        if b_idx > 0:
            await _cooldown(COOLDOWN_MINI, f"{label} batch {b_idx}", status_cb)
        try:
            scores = await metric.abatch_score(batch)
        except Exception as e:
            # Previously only sent to status_cb (the Streamlit page text, which
            # gets overwritten by the next status update and is never
            # persisted) — meaning a failed sample's actual reason was
            # unrecoverable after the fact, even in Logfire. Log it properly
            # so "why did sample N come back None" is traceable later.
            batch_samples = samples[len(all_scores): len(all_scores) + len(batch)]
            questions = [s["question"][:80] for s in batch_samples]
            logfire.error(
                f"{label} batch {b_idx} failed: {type(e).__name__}: {e}",
                metric=label,
                batch_index=b_idx,
                questions=questions,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            if status_cb:
                status_cb(f"⚠️ {label} batch {b_idx} failed ({type(e).__name__}) — recording as skipped.")
            scores = [None] * len(batch)
        all_scores.extend(scores)
    return all_scores

# Ordered so numbering in status messages ("Exp N/M") stays meaningful
# regardless of which subset is selected. Tool Correctness has no LLM call —
# it's a pure Jaccard similarity, so it's handled separately (never fails,
# never needs re-running for quota reasons).
METRIC_ORDER = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]
METRIC_TITLES = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "answer_correctness": "Answer Correctness",
    "tool_correctness": "Tool Correctness",
}


def _build_inputs(metric_key: str, samples: list) -> list:
    if metric_key == "faithfulness":
        return [
            {"user_input": s["question"], "response": s["actual_response"], "retrieved_contexts": s["actual_contexts"]}
            for s in samples
        ]
    if metric_key == "answer_relevancy":
        return [{"user_input": s["question"], "response": s["actual_response"]} for s in samples]
    if metric_key in ("context_precision", "context_recall"):
        return [
            {"user_input": s["question"], "reference": s["reference"], "retrieved_contexts": s["actual_contexts"]}
            for s in samples
        ]
    if metric_key == "answer_correctness":
        return [
            {"user_input": s["question"], "response": s["actual_response"], "reference": s["reference"]}
            for s in samples
        ]
    raise ValueError(f"Unknown metric key: {metric_key}")


def _build_metric(metric_key: str, judge_llm, ragas_embeddings):
    if metric_key == "faithfulness":
        return Faithfulness(llm=judge_llm)
    if metric_key == "answer_relevancy":
        return AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings)
    if metric_key == "context_precision":
        return ContextPrecision(llm=judge_llm)
    if metric_key == "context_recall":
        return ContextRecall(llm=judge_llm)
    if metric_key == "answer_correctness":
        return AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings)
    raise ValueError(f"Unknown metric key: {metric_key}")


async def run_selected_metrics(golden_dataset: dict, metric_keys: list[str], status_cb=None) -> dict:
    """
    Runs only the given LLM-judged metrics (any subset of METRIC_ORDER), plus
    Tool Correctness if requested — lets a partial re-run (e.g. just the ones
    that came back None from judge-quota exhaustion) skip the experiments that
    already succeeded, instead of re-spending quota on all 6 every time.
    Returns dict keyed by metric name → DataFrame (only for the requested keys).
    """
    samples = _prep_samples(golden_dataset)
    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")

    llm_keys = [k for k in metric_keys if k in METRIC_ORDER]
    want_tool_correctness = "tool_correctness" in metric_keys

    results = {}
    judge_llm, ragas_embeddings = _build_judge() if llm_keys else (None, None)
    n_total = len(llm_keys) + (1 if want_tool_correctness else 0)

    with logfire.span("🧪 Eval Phase 2 — Selected Metrics", total_samples=len(samples), metrics=metric_keys):
        for i, key in enumerate(llm_keys, start=1):
            title = METRIC_TITLES[key]
            if status_cb:
                status_cb(f"🧪 {i}/{n_total} — {title} ({len(samples)} samples)...")
            try:
                with logfire.span(f"🧪 {title}"):
                    inputs = _build_inputs(key, samples)
                    metric = _build_metric(key, judge_llm, ragas_embeddings)
                    scores = await _batched_score(metric, inputs, samples, status_cb, title)
                    df = _score_df(key, samples, scores)
                    results[key] = df
                    logfire.info(f"🧪 {title} done", avg=round(df[key].mean(), 3))
            except Exception as e:
                if status_cb:
                    status_cb(f"❌ {i}/{n_total} — {title} failed entirely ({type(e).__name__}): {e}")
                logfire.error(f"{title} experiment failed: {e}")

            if i < n_total:
                await _cooldown(COOLDOWN_STANDARD, title, status_cb)

        if want_tool_correctness:
            if status_cb:
                status_cb(f"⚡ {n_total}/{n_total} — Tool Correctness (zero LLM calls)...")
            with logfire.span("🧪 Tool Correctness"):
                tool_rows = []
                for s in samples:
                    called = set(s.get("actual_tools_called") or [])
                    expected = set(s.get("expected_tools") or [])
                    union = len(called | expected)
                    score = len(called & expected) / union if union > 0 else 0.0
                    tool_rows.append({"question": s["question"][:65], "tool_correctness": round(score, 3)})
                df = pd.DataFrame(tool_rows)
                results["tool_correctness"] = df
                logfire.info("🧪 Tool Correctness done", avg=round(df["tool_correctness"].mean(), 3))

        if status_cb:
            status_cb(f"✅ {len(results)}/{len(metric_keys)} requested experiments complete!")

    return results


async def run_all_metrics(golden_dataset: dict, status_cb=None) -> dict:
    """Runs all 6 experiments. Returns dict keyed by metric name → DataFrame."""
    return await run_selected_metrics(
        golden_dataset, METRIC_ORDER + ["tool_correctness"], status_cb=status_cb
    )