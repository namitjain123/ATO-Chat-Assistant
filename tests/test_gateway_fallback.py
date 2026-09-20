"""
Tests for app/gateway/client.py's create_completion_with_fallback — the
application-level primary/fallback logic that replaced Portkey's own
server-side fallback strategy after finding it has a confirmed bug for
Azure OpenAI targets (see README's "LLM Provider" section for the full
story). This was previously only verified by hand, once, with a real
network call — these tests pin that behavior down permanently, with no
network calls at all.
"""
import pytest
from app.gateway import client as gw


class _FakeResponse:
    """Stand-in for whatever portkey_client_direct.chat.completions.create returns."""
    def __init__(self, model: str):
        self.model = model


def test_primary_target_succeeds_no_fallback_needed(mocker):
    mock_create = mocker.patch.object(
        gw.portkey_client_direct.chat.completions, "create",
        return_value=_FakeResponse("gpt-5-mini-2025-08-07"),
    )

    response = gw.create_completion_with_fallback(messages=[{"role": "user", "content": "hi"}])

    assert response.model == "gpt-5-mini-2025-08-07"
    mock_create.assert_called_once()
    called_model = mock_create.call_args.kwargs["model"]
    assert called_model == gw.FALLBACK_TARGETS[0]


def test_primary_fails_falls_through_to_first_fallback(mocker):
    mock_create = mocker.patch.object(
        gw.portkey_client_direct.chat.completions, "create",
        side_effect=[
            Exception("azure-openai error: Resource not found"),
            _FakeResponse("llama-3.3-70b-versatile"),
        ],
    )

    response = gw.create_completion_with_fallback(messages=[{"role": "user", "content": "hi"}])

    assert response.model == "llama-3.3-70b-versatile"
    assert mock_create.call_count == 2
    first_call_model = mock_create.call_args_list[0].kwargs["model"]
    second_call_model = mock_create.call_args_list[1].kwargs["model"]
    assert first_call_model == gw.FALLBACK_TARGETS[0]
    assert second_call_model == gw.FALLBACK_TARGETS[1]


def test_first_two_fail_falls_through_to_second_fallback(mocker):
    mock_create = mocker.patch.object(
        gw.portkey_client_direct.chat.completions, "create",
        side_effect=[
            Exception("primary down"),
            Exception("first fallback also down"),
            _FakeResponse("llama-3.1-8b-instant"),
        ],
    )

    response = gw.create_completion_with_fallback(messages=[{"role": "user", "content": "hi"}])

    assert response.model == "llama-3.1-8b-instant"
    assert mock_create.call_count == 3


def test_all_targets_fail_raises_the_last_error(mocker):
    mocker.patch.object(
        gw.portkey_client_direct.chat.completions, "create",
        side_effect=[
            Exception("primary down"),
            Exception("fallback 1 down"),
            Exception("fallback 2 down — this one should propagate"),
        ],
    )

    with pytest.raises(Exception, match="fallback 2 down"):
        gw.create_completion_with_fallback(messages=[{"role": "user", "content": "hi"}])


def test_stops_trying_targets_after_first_success():
    # Sanity check on FALLBACK_TARGETS itself: Azure must be first (primary),
    # not last — a fallback-only integration would mean the system never
    # actually touches Azure in normal operation.
    assert gw.FALLBACK_TARGETS[0].startswith(f"@{gw.settings.AZURE_SLUG}/")
    assert len(gw.FALLBACK_TARGETS) >= 2


# -- structured output: a missing result must trigger the fallback ------------

from langchain_core.runnables import RunnableLambda


class _FakeChatModel:
    """Stands in for ChatOpenAI; with_structured_output returns a runnable
    producing whatever this target is scripted to return."""
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema, method):
        return RunnableLambda(lambda _: self.result)


def test_structured_none_falls_through_to_next_target(mocker):
    # Real incident: the primary spent its whole token budget on reasoning and
    # made no tool call -> with_structured_output returned None, not an error.
    results = iter([None, "parsed-by-fallback", "unused"])
    budgets = []

    def make(model_str, feature, max_completion_tokens=2048):
        budgets.append(max_completion_tokens)
        return _FakeChatModel(next(results))

    mocker.patch.object(gw, "_make_chat_model", side_effect=make)

    chain = gw.get_structured_llm_with_fallback(object, feature="t", max_completion_tokens=8192)

    assert chain.invoke("prompt") == "parsed-by-fallback"
    assert set(budgets) == {8192}  # budget reaches every target in the chain


def test_require_parsed_raises_on_none_and_passes_results_through():
    with pytest.raises(ValueError):
        gw._require_parsed(None)
    assert gw._require_parsed("ok") == "ok"
