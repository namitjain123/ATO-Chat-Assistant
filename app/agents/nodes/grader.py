"""
Grader: decides, after each retrieval pass, whether to answer, retry, or hop.

Stage 1 — relevance gate (free: thresholds the reranker score already
computed). Catches "found nothing": measured on the live corpus,
out-of-corpus questions reranked at 0.00-0.03 against >= 0.98 for covered
ones. Weak -> rewrite and retry, then give up honestly.

Stage 2 — sufficiency (one LLM call). Catches "found half": relevant
passages that still don't *answer* the question. Asking "what records do I
need for the donation type with a $1,500 cap?" retrieves the $1,500
political-party passage — highly relevant, says nothing about records. The
LLM names what's missing and writes a follow-up search for it; that search
ADDS to the context (a hop) instead of replacing it, then grading repeats.

  relevance weak, attempts left       -> rewriter -> retriever  (retry)
  relevance weak, out of attempts     -> responder, "not covered"
  hop found nothing new               -> responder, answers what it has
  sufficient                          -> responder
  partial, hops left                  -> retriever (hop, accumulating)
  partial, out of hops                -> responder, states what's missing
"""
import logfire
from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.config import settings
from app.gateway.client import get_structured_llm_with_fallback

MAX_CONTEXT_CHARS = 16000  # what the sufficiency judge reads


class Sufficiency(BaseModel):
    verdict: str = Field(description="'sufficient' or 'partial' — see the rules in the prompt.")
    missing: str = Field(description="If partial: the one specific thing the question asks that the CONTEXT lacks, as a short phrase. Empty string if sufficient.")
    follow_up_query: str = Field(
        description="If partial: a short standalone search query for exactly that missing thing, using ATO "
        "terminology and any names the CONTEXT revealed. Empty string if sufficient."
    )


# First live version asked "is anything needed for a complete answer
# missing?" — and the judge demanded examples, exceptions and apportionment
# rules nobody asked for: "Can I claim union fees?" was fully answered by the
# first retrieval ("yes, with written evidence") yet burned both hops and 43s.
# The question is now "is what was ASKED answered?", defaulting to sufficient.
SUFFICIENCY_PROMPT = """You check whether retrieved passages from an Australian Taxation Office
(ATO) knowledge base answer a user's question. Do NOT answer it yourself; judge only the CONTEXT.

QUESTION: "{question}"
(Standalone form: "{standalone}")

CONTEXT:
{context}

Rules:
- "sufficient": the CONTEXT answers what the question actually asks. Further detail, examples,
  exceptions, edge cases or related rules the question did NOT ask for never make it partial.
- "partial" ONLY if either:
  (a) the question explicitly asks for a specific thing the CONTEXT does not contain, or
  (b) the question refers to something indirectly (e.g. "the donation type with a $1,500 cap"),
      the CONTEXT identifies what it is, but lacks the thing the question asks about it —
      then follow_up_query searches for that thing using the name the CONTEXT revealed.
- When unsure, choose "sufficient"."""

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_structured_llm_with_fallback(
            Sufficiency, feature="sufficiency-grader", method="function_calling",
            max_completion_tokens=4096,  # judging long context; reasoning headroom (see gateway)
        )
    return _llm


def check_sufficiency(question: str, standalone: str, documents: list[str]) -> Sufficiency | None:
    """None on failure — the caller then answers from what it has, rather
    than blocking the answer on a failed quality check."""
    context = "\n\n".join(documents)[:MAX_CONTEXT_CHARS]
    try:
        result = _get_llm().invoke(SUFFICIENCY_PROMPT.format(question=question, standalone=standalone, context=context))
        result.verdict = (result.verdict or "").strip().lower()
        return result
    except Exception as e:
        logfire.warning(f"Sufficiency check failed ({e}) — answering from retrieved context.")
        return None


def _relevance_gate(state: AgentState) -> dict | None:
    """Stage 1. Returns a grade update, or None when relevance passes."""
    top = state.get("top_relevance")
    attempts = state.get("retrieval_attempts") or 0

    if state.get("retrieval_mode") == "hop" and state.get("hop_new_docs") == 0:
        # Checked before relevance: live, a hop re-found only passages already
        # in context — at top relevance 1.00 — and was judged again for
        # nothing. Nothing new means the verdict can't change: stop, no LLM call.
        return {"retrieval_grade": "partial", "note": "Grade: hop added no new passages — answering from current context"}

    if top is None or top >= settings.RELEVANCE_THRESHOLD:
        return None  # None = reranker unavailable: relevance unknown, not zero — don't refuse over it

    if state.get("retrieval_mode") == "hop":
        # The earlier passes WERE relevant; only the follow-up came up empty.
        # Answer from what we have and say what's missing — don't retry or refuse.
        return {"retrieval_grade": "partial",
                "note": f"Grade: hop found nothing relevant (top {top:.2f}) — answering from earlier context"}
    if attempts < settings.MAX_RETRIEVAL_ATTEMPTS:
        return {"retrieval_grade": "retry", "note": f"Grade: weak (top {top:.2f}) — rewriting the search"}
    return {"retrieval_grade": "insufficient",
            "note": f"Grade: insufficient after {attempts} attempts — not covered by the knowledge base",
            "documents": []}  # irrelevant passages shouldn't be shown as sources


def grade_node(state: AgentState):
    update = _relevance_gate(state)

    if update is None and not settings.ENABLE_SUFFICIENCY_CHECK:
        update = {"retrieval_grade": "relevant", "note": "Grade: relevant"}

    elif update is None:
        messages = state.get("messages") or []
        question = messages[-1]["content"] if messages else ""
        standalone = state.get("question") or state.get("current_query", "")
        verdict = check_sufficiency(question, standalone, state.get("documents") or [])
        hops = state.get("hops") or 0
        tried = state.get("hop_queries") or []
        follow_up = (verdict.follow_up_query or "").strip() if verdict else ""

        if verdict is None or verdict.verdict != "partial":
            update = {"retrieval_grade": "sufficient", "note": "Grade: sufficient — context answers the question"}
        elif follow_up and hops < settings.MAX_HOPS and follow_up.lower() not in {q.lower() for q in tried}:
            update = {
                "retrieval_grade": "hop",
                "note": f"Grade: partial — missing: {verdict.missing} → hop {hops + 1}: {follow_up}",
                "current_query": follow_up,
                "retrieval_mode": "hop",
                "hops": hops + 1,
                "hop_queries": tried + [follow_up],
                "missing_info": verdict.missing,
            }
        else:
            reason = "out of hops" if hops >= settings.MAX_HOPS else "no new search to try"
            update = {"retrieval_grade": "partial",
                      "note": f"Grade: partial ({reason}) — missing: {verdict.missing}",
                      "missing_info": verdict.missing}

    note = update.pop("note")
    logfire.info(note)
    return {**update, "plan": state["plan"] + [note]}


def route_after_grade(state: AgentState) -> str:
    grade = state.get("retrieval_grade")
    if grade == "retry":
        return "rewriter"
    if grade == "hop":
        return "retriever"
    return "responder"
