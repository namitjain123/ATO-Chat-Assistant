"""Tests for evals/pipeline.py's detect_tool — maps a thought_process list
from a /query response back to the tool name Tool Correctness scores against."""
from evals.pipeline import detect_tool


def test_detects_guardrails():
    assert detect_tool(["Intent: Guardrails Fired", "Retrieval: Skipped"]) == "guardrails"


def test_detects_retrieve_documents_via_intent_technical():
    assert detect_tool(["Intent: Technical", "Search Term: work related expenses"]) == "retrieve_documents"


def test_detects_retrieve_documents_via_context_retrieved():
    assert detect_tool(["Context Retrieved", "Response generated."]) == "retrieve_documents"


def test_detects_direct_answer_via_conversational():
    assert detect_tool(["Intent: Conversational/Memory", "Retrieval: Skipped"]) == "direct_answer"


def test_detects_direct_answer_via_memory():
    assert detect_tool(["Handling from memory."]) == "direct_answer"


def test_unknown_when_nothing_matches():
    assert detect_tool(["Something unrelated entirely"]) == "unknown"


def test_case_insensitive():
    assert detect_tool(["INTENT: TECHNICAL", "SEARCH TERM: gst"]) == "retrieve_documents"


def test_empty_list_is_unknown():
    assert detect_tool([]) == "unknown"


def test_guardrails_takes_priority_over_other_signals():
    # A plan list that happens to mention both should still resolve to
    # guardrails first, matching the function's actual check order.
    assert detect_tool(["Intent: Guardrails Fired", "Intent: Technical"]) == "guardrails"
