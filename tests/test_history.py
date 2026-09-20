"""
Conversation history must stay bounded without silently losing what the user
told the assistant earlier.

messages accumulates with operator.add and the checkpointer keeps a thread
alive across sessions, so an uncapped history grows the prompt every turn.
Plain truncation would drop facts the user established ("I'm a nurse", "for
2024-25"), so older turns are compacted into a running summary instead.
"""
import pytest

from app.agents import history
from app.agents.nodes import responder, router
from app.config import settings


def turns(n: int, start: int = 1) -> list[dict]:
    """n complete turns: user message + assistant reply each."""
    msgs = []
    for i in range(start, start + n):
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    return msgs


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(settings, "HISTORY_KEEP_TURNS", 8)
    monkeypatch.setattr(settings, "HISTORY_COMPACT_BATCH_TURNS", 4)


# -- plan() ---------------------------------------------------------------

def test_short_history_is_kept_whole_and_never_compacted():
    messages = turns(3) + [{"role": "user", "content": "current"}]
    to_compact, verbatim = history.plan(messages, summarized_count=0)
    assert to_compact == []
    assert len(verbatim) == 6  # 3 turns, untouched


def test_the_current_question_is_never_part_of_history():
    messages = turns(2) + [{"role": "user", "content": "current"}]
    _, verbatim = history.plan(messages, 0)
    assert {"role": "user", "content": "current"} not in verbatim


def test_nothing_compacts_until_a_whole_batch_has_aged_out():
    # 10 turns: 8 kept verbatim, only 2 aged out — below the batch of 4.
    messages = turns(10) + [{"role": "user", "content": "current"}]
    to_compact, verbatim = history.plan(messages, 0)
    assert to_compact == []
    assert len(verbatim) == 20  # carried verbatim rather than dropped while waiting


def test_a_full_batch_triggers_compaction_and_keeps_the_window():
    # 12 turns: 8 kept, 4 aged out = exactly one batch.
    messages = turns(12) + [{"role": "user", "content": "current"}]
    to_compact, verbatim = history.plan(messages, 0)
    assert len(to_compact) == 8          # 4 turns
    assert len(verbatim) == 16           # 8 turns
    assert to_compact[0]["content"] == "question 1"
    assert verbatim[0]["content"] == "question 5"


def test_already_summarized_turns_are_not_compacted_twice():
    messages = turns(12) + [{"role": "user", "content": "current"}]
    to_compact, _ = history.plan(messages, summarized_count=8)
    assert to_compact == []  # those 8 are already in the summary


def test_nothing_is_ever_dropped_between_batches():
    # Every message must appear either in the summary's coverage or verbatim.
    messages = turns(11) + [{"role": "user", "content": "current"}]
    summarized = 8
    to_compact, verbatim = history.plan(messages, summarized)
    covered = summarized + len(to_compact) + len(verbatim)
    assert covered == len(messages) - 1  # all history accounted for


def test_the_window_grows_not_shrinks_while_waiting_for_a_batch():
    # The point of batching: between compactions the verbatim window floats
    # wider rather than dropping anything.
    at_10 = history.plan(turns(10) + [{"role": "user", "content": "c"}], 0)[1]
    at_12 = history.plan(turns(12) + [{"role": "user", "content": "c"}], 0)[1]
    assert len(at_10) == 20   # 10 turns, wider than the 8-turn window
    assert len(at_12) == 16   # compaction ran, back to the window


# -- compact() ------------------------------------------------------------

def test_compaction_sends_the_transcript_and_returns_the_summary(mocker):
    captured = {}

    class _Resp:
        def __init__(self):
            class _M: content = "  notes: user is a nurse, asked about uniforms  "
            class _C: message = _M()
            self.choices = [_C()]

    def fake(messages, **kw):
        captured["prompt"] = messages[0]["content"]
        return _Resp()

    mocker.patch.object(history, "create_completion_with_fallback", side_effect=fake)

    out = history.compact("", [{"role": "user", "content": "am I a nurse deduction"}])

    assert out == "notes: user is a nurse, asked about uniforms"
    assert "am I a nurse deduction" in captured["prompt"]


def test_an_existing_summary_is_folded_in_not_appended(mocker):
    captured = {}
    mocker.patch.object(
        history, "create_completion_with_fallback",
        side_effect=lambda messages, **kw: captured.update(prompt=messages[0]["content"]) or _fake_response("merged"),
    )
    history.compact("older notes here", turns(1))
    assert "older notes here" in captured["prompt"]
    assert "fold" in captured["prompt"].lower()


def _fake_response(text):
    class _M: content = text
    class _C: message = _M()
    class _R: choices = [_C()]
    return _R()


def test_failed_compaction_keeps_the_previous_summary(mocker):
    mocker.patch.object(history, "create_completion_with_fallback", side_effect=RuntimeError("all targets failed"))
    assert history.compact("previous notes", turns(2)) == "previous notes"


def test_empty_model_output_keeps_the_previous_summary(mocker):
    mocker.patch.object(history, "create_completion_with_fallback", return_value=_fake_response(""))
    assert history.compact("previous notes", turns(2)) == "previous notes"


# -- format_for_prompt() --------------------------------------------------

def test_summary_is_labelled_and_precedes_recent_turns():
    text = history.format_for_prompt("user is a nurse", turns(1))
    assert text.index("user is a nurse") < text.index("question 1")
    assert "Earlier in this conversation" in text


def test_no_summary_yet_gives_just_the_transcript():
    assert history.format_for_prompt("", turns(1)) == "User: question 1\nAssistant: answer 1"


def test_empty_history_formats_to_nothing():
    assert history.format_for_prompt("", []) == ""


# -- wiring ---------------------------------------------------------------

class _Decision:
    route = "conversational"
    search_query = ""
    sub_queries: list = []
    entities: list = []
    topics: list = []
    income_year = ""


def test_router_compacts_once_and_persists_the_result(mocker):
    class _LLM:
        def invoke(self, prompt):
            return _Decision()
    mocker.patch.object(router, "structured_llm", _LLM())
    compact = mocker.patch.object(router.history, "compact", return_value="compacted notes")

    state = {"messages": turns(12) + [{"role": "user", "content": "current"}], "plan": []}
    out = router.router_node(state)

    compact.assert_called_once()
    assert out["history_summary"] == "compacted notes"
    assert out["summarized_msgs"] == 8


def test_responder_reuses_the_summary_without_compacting_again(mocker):
    compact = mocker.patch.object(responder.history, "compact")
    prompts = []
    mocker.patch.object(
        responder, "create_completion_with_fallback",
        side_effect=lambda messages, **kw: prompts.append(messages[0]["content"]) or _fake_response("answer"),
    )

    responder.generate_node({
        "messages": turns(12) + [{"role": "user", "content": "current"}],
        "current_query": "CONVERSATIONAL",
        "plan": [],
        "history_summary": "compacted notes",
        "summarized_msgs": 8,
    })

    compact.assert_not_called()          # the router already did it this turn
    assert "compacted notes" in prompts[0]
    # Turns 1-4 were summarized away; 5-12 stay verbatim. Match whole lines —
    # "question 1" is a substring of "question 10".
    lines = prompts[0].splitlines()
    assert "User: question 1" not in lines
    assert "User: question 4" not in lines
    assert "User: question 5" in lines
    assert "User: question 12" in lines
