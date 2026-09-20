"""
Every LLM call carries a timeout, and the whole graph carries a deadline.

Without them a hung provider call hangs the request until the SDK gives up on
its own — and one question can make ~10 calls (guardrails, router, a grade per
retrieval pass, rewriter, responder), so per-call timeouts alone don't bound
the total either.
"""
import time

import pytest

from app.config import settings
from app.gateway import client as gw


class _FakeResponse:
    def __init__(self, model="gpt-5-mini"):
        self.model = model


# -- per-call timeouts ---------------------------------------------------------

def test_raw_completion_call_carries_a_timeout(mocker):
    mock_create = mocker.patch.object(
        gw.portkey_client_direct.chat.completions, "create", return_value=_FakeResponse()
    )

    gw.create_completion_with_fallback(messages=[{"role": "user", "content": "hi"}])

    assert mock_create.call_args.kwargs["timeout"] == settings.LLM_TIMEOUT_SECONDS


def test_structured_chat_models_carry_a_timeout():
    model = gw._make_chat_model(gw.FALLBACK_TARGETS[0], "test")
    assert model.request_timeout == settings.LLM_TIMEOUT_SECONDS


def test_every_fallback_target_gets_the_timeout(mocker):
    # A hung PRIMARY has to fail over; without a timeout it would stall instead.
    timeouts = []

    def capture(**kwargs):
        timeouts.append(kwargs.get("timeout"))
        if len(timeouts) < 3:
            raise RuntimeError("target unavailable")
        return _FakeResponse("gpt-oss-20b")

    mocker.patch.object(gw.portkey_client_direct.chat.completions, "create", side_effect=capture)

    gw.create_completion_with_fallback(messages=[{"role": "user", "content": "hi"}])

    assert timeouts == [settings.LLM_TIMEOUT_SECONDS] * 3


def test_guardrails_llm_carries_a_timeout(mocker):
    # Guardrails run before everything else on every request, and bypass the
    # gateway entirely (direct OpenAI) — so they need their own timeout.
    from app.guardrails import rails

    captured = {}

    class _FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    mocker.patch.object(rails, "ChatOpenAI", _FakeChat)
    mocker.patch.object(rails, "RailsConfig", mocker.MagicMock())
    mocker.patch.object(rails, "LLMRails", mocker.MagicMock())

    rails._build_rails()

    assert captured["timeout"] == settings.LLM_TIMEOUT_SECONDS


# -- whole-request deadline ----------------------------------------------------

@pytest.fixture
def api(mocker):
    from fastapi.testclient import TestClient
    import app.main as main

    # TestClient without a context manager doesn't fire startup events, so the
    # real guardrails/model warm-up never runs here.
    mocker.patch.object(main, "guard", return_value=(False, None))
    return TestClient(main.app), main


def test_slow_graph_returns_a_timeout_answer_not_a_hang(api, monkeypatch, mocker):
    client, main = api
    monkeypatch.setattr(settings, "REQUEST_DEADLINE_SECONDS", 0.2)
    mocker.patch.object(main.rag_agent, "invoke", side_effect=lambda *a, **k: time.sleep(5))

    started = time.time()
    body = client.post("/query", json={"q": "slow one", "thread_id": "t"}).json()
    elapsed = time.time() - started

    assert body["status"] == "timeout"
    assert body["sources"] == []
    assert elapsed < 3  # returned on the deadline, not after the 5s call


def test_normal_graph_answers_through_the_pool(api, monkeypatch, mocker):
    client, main = api
    monkeypatch.setattr(settings, "REQUEST_DEADLINE_SECONDS", 10)
    mocker.patch.object(main.rag_agent, "invoke", return_value={
        "final_answer": "an answer", "plan": ["Intent: Technical"], "status": "Response generated.", "documents": ["CONTENT: x"],
    })

    body = client.post("/query", json={"q": "normal", "thread_id": "t"}).json()

    assert body["answer"] == "an answer"
    assert body["status"] == "Response generated."


def test_a_timed_out_request_does_not_block_the_next_one(api, monkeypatch, mocker):
    # The orphaned worker keeps running; the pool has room so the next caller
    # isn't stuck behind it.
    client, main = api
    monkeypatch.setattr(settings, "REQUEST_DEADLINE_SECONDS", 0.2)
    calls = {"n": 0}

    def slow_then_fast(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(3)
            return {}
        return {"final_answer": "second answer", "plan": [], "status": "ok", "documents": []}

    mocker.patch.object(main.rag_agent, "invoke", side_effect=slow_then_fast)

    first = client.post("/query", json={"q": "slow", "thread_id": "a"}).json()
    monkeypatch.setattr(settings, "REQUEST_DEADLINE_SECONDS", 10)
    second = client.post("/query", json={"q": "fast", "thread_id": "b"}).json()

    assert first["status"] == "timeout"
    assert second["answer"] == "second answer"


# -- the removed dead config ---------------------------------------------------

def test_no_config_attached_client_exists():
    # A client with a gateway config re-applies Portkey's broken Azure fallback
    # routing; the previous one also advertised a retry policy nothing used.
    assert not hasattr(gw, "portkey_client")
    assert not hasattr(gw, "GATEWAY_CONFIG")
    assert not hasattr(gw, "PORTKEY_CONFIG")
