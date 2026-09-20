"""
Conversation history, bounded.

state["messages"] accumulates with operator.add and the checkpointer keeps a
thread alive across sessions, so an unbounded history means the prompt grows
every turn — rising cost per turn, then context-limit errors on a thread that
used to work fine.

Truncation alone would silently lose facts the user established earlier ("I'm
a nurse", "for the 2024-25 year") — exactly what they expect the assistant to
remember. So older turns are COMPACTED into a running summary instead, and
only the recent ones are kept word-for-word.

Compaction happens in batches (see settings.HISTORY_COMPACT_BATCH_TURNS): if
it ran every turn past the window it would add an LLM call to every turn
forever. Between batches the verbatim window simply floats a few turns wider —
nothing is ever dropped waiting to be summarized.

Storage is untouched: the checkpointer still holds every message. This only
bounds what goes into a prompt.
"""
import logfire

from app.config import settings
from app.gateway.client import create_completion_with_fallback

MSGS_PER_TURN = 2  # one user message + one assistant reply

COMPACT_PROMPT = """Summarise this earlier part of a conversation between a user and an
Australian Taxation Office (ATO) assistant, so the assistant can keep answering follow-up
questions without the full transcript.

Keep: what the user said about their own situation (occupation, income year, circumstances),
what they asked about, and any figures or rules the assistant already gave them.
Drop: pleasantries, restated questions, and anything the assistant said it couldn't answer.

Write it as compact notes, not prose. If an earlier summary is included, fold everything into
one combined set of notes rather than appending a second summary.

{existing}
EARLIER CONVERSATION:
{transcript}"""


def _transcript(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def plan(messages: list[dict], summarized_count: int) -> tuple[list[dict], list[dict]]:
    """
    Decide what to keep verbatim and what (if anything) to compact now.

    Returns (to_compact, verbatim) — both excluding the latest user message,
    which every caller adds separately. to_compact is empty unless enough turns
    have aged out to be worth an LLM call.
    """
    history = messages[:-1] if messages else []
    keep = settings.HISTORY_KEEP_TURNS * MSGS_PER_TURN

    if len(history) <= keep:
        return [], history[summarized_count:]

    overflow = history[:-keep]
    unsummarized = overflow[summarized_count:]

    if len(unsummarized) >= settings.HISTORY_COMPACT_BATCH_TURNS * MSGS_PER_TURN:
        return unsummarized, history[-keep:]

    # Not enough aged out yet to justify a call — carry them verbatim for now.
    return [], history[summarized_count:]


def compact(existing_summary: str, messages: list[dict]) -> str:
    """Fold `messages` into `existing_summary`. Returns the old summary
    unchanged on failure — a degraded summary beats losing the thread."""
    existing = f"EARLIER SUMMARY (fold this in):\n{existing_summary}\n\n" if existing_summary else ""
    prompt = COMPACT_PROMPT.format(existing=existing, transcript=_transcript(messages))
    try:
        with logfire.span("🗜️ History Compaction", messages=len(messages)):
            response = create_completion_with_fallback(
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2048,
            )
            summary = (response.choices[0].message.content or "").strip()
            logfire.info(f"Compacted {len(messages)} messages into {len(summary)} chars.")
            return summary or existing_summary
    except Exception as e:
        logfire.warning(f"History compaction failed ({e}) — keeping the previous summary.")
        return existing_summary


def format_for_prompt(summary: str, verbatim: list[dict]) -> str:
    """The CONVERSATION HISTORY block: summarized older turns, then recent ones."""
    parts = []
    if summary:
        parts.append(f"[Earlier in this conversation]\n{summary}")
    if verbatim:
        parts.append(_transcript(verbatim))
    return "\n\n".join(parts)
