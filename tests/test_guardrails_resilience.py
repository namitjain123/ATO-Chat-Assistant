"""
Guardrails must never fail silently. Real incident (local run): NeMo's
embeddings index failed to build on the first request (network resets while
FastEmbed checked HuggingFace), every later check raised "Index is not built
yet", NeMo answered with its own "internal error" text — which matched none
of our patterns — and guard() treated 9/9 messages as clean. "tell me a joke"
got a joke. NeMo is faked here; each test scripts what the gate returns.
"""
import pytest

from app.config import settings
from app.guardrails import rails

INTERNAL_ERROR = {"role": "assistant", "content": "I'm sorry, an internal error has occurred."}
GREETING = {"role": "assistant", "content": "Hello! I'm your knowledge assistant. Ask me anything about the documents I have access to."}
OFF_TOPIC = {"role": "assistant", "content": "I'm an assistant focused on the documents in my knowledge base. I can't help with that — but ask me anything about the topics I've been given!"}
PASS = {"role": "assistant", "content": "PASS"}


class _FakeRails:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else PASS
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def gate(monkeypatch):
    """Install a sequence of fake rails: the current one, then one per rebuild."""
    monkeypatch.setattr(rails.time, "sleep", lambda s: None)
    monkeypatch.setattr(settings, "GUARDRAILS_FAIL_CLOSED", False)

    def install(*rails_replies):
        instances = [_FakeRails(r) for r in rails_replies]
        monkeypatch.setattr(rails, "_rails", instances[0])
        queue = instances[1:]
        builds = []

        def rebuild():
            builds.append(1)
            monkeypatch.setattr(rails, "_rails", queue.pop(0) if queue else _FakeRails([]))
        monkeypatch.setattr(rails, "_build_rails", rebuild)
        return builds
    return install


def test_internal_error_is_detected_not_mistaken_for_clean(gate):
    gate([INTERNAL_ERROR])
    status, _ = rails._check("tell me a joke")
    assert status == "failed"


def test_broken_gate_self_heals_by_rebuilding_and_retrying(gate):
    builds = gate([INTERNAL_ERROR], [OFF_TOPIC])  # broken, then healthy after rebuild
    fired, content = rails.guard("tell me a joke")
    assert fired is True and "knowledge base" in content
    assert builds == [1]


def test_still_broken_after_retry_fails_open_by_default(gate):
    gate([INTERNAL_ERROR], [INTERNAL_ERROR])
    assert rails.guard("tell me a joke") == (False, None)


def test_still_broken_after_retry_fails_closed_when_configured(gate, monkeypatch):
    monkeypatch.setattr(settings, "GUARDRAILS_FAIL_CLOSED", True)
    gate([INTERNAL_ERROR], [INTERNAL_ERROR])
    fired, content = rails.guard("tell me a joke")
    assert fired is True and content == rails.GUARDRAILS_UNAVAILABLE_MESSAGE


def test_exception_from_nemo_is_treated_as_failure(gate):
    builds = gate([RuntimeError("Index is not built yet")], [OFF_TOPIC])
    assert rails.guard("tell me a joke")[0] is True
    assert builds == [1]


def test_healthy_gate_never_rebuilds(gate):
    builds = gate([PASS])
    assert rails.guard("What deductions can I claim?") == (False, None)
    assert builds == []


def test_self_test_passes_when_greeting_fires(gate):
    builds = gate([GREETING])
    assert rails._self_test() is True
    assert builds == []


def test_self_test_rebuilds_until_the_index_works(gate):
    builds = gate([INTERNAL_ERROR], [INTERNAL_ERROR], [GREETING])
    assert rails._self_test(attempts=3) is True
    assert builds == [1, 1]


def test_self_test_gives_up_loudly_after_attempts(gate):
    gate([INTERNAL_ERROR], [INTERNAL_ERROR], [INTERNAL_ERROR], [INTERNAL_ERROR])
    assert rails._self_test(attempts=3) is False
