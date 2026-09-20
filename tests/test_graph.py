"""
Tests for the Neo4j knowledge graph — extraction/validation, writes, graph
retrieval, and the connection's graceful degradation. The Neo4j driver and
the LLM are always faked; no database or network needed.
"""
import pytest

from app.config import settings
from app.ingestion import graph_extractor as gx
from app.services.graph import graph_retrieval as gr
from app.services.graph import neo4j_client as nc


class _Rel:
    def __init__(self, source, source_type, relation, target, target_type, detail=""):
        self.source, self.source_type, self.relation = source, source_type, relation
        self.target, self.target_type, self.detail = target, target_type, detail


class _FakeDriver:
    """Records every execute_query; replies from a queue (or empty)."""
    def __init__(self, replies=None, fail=False):
        self.calls, self.replies, self.fail = [], list(replies or []), fail

    def execute_query(self, query, **params):
        self.calls.append((query, params))
        if self.fail:
            raise RuntimeError("database down")
        return (self.replies.pop(0) if self.replies else []), None, None


# -- extraction / validation --------------------------------------------------

def test_entity_key_is_case_and_whitespace_insensitive():
    assert gx.entity_key("  Written   Evidence. ") == gx.entity_key("written evidence") == "written evidence"


def test_validate_normalises_type_spelling():
    [row] = gx.validate([_Rel("Union fees", "deduction", "requires", "Written evidence", "REQUIREMENT")])
    assert (row["s_type"], row["relation"], row["t_type"]) == ("Deduction", "REQUIRES", "Requirement")


def test_validate_drops_off_schema_types_and_relations():
    rows = gx.validate([
        _Rel("a", "Deduction", "REQUIRES", "b", "Planet"),       # unknown node type
        _Rel("a", "Deduction", "LOVES", "b", "Requirement"),     # unknown relation
        _Rel("", "Deduction", "REQUIRES", "b", "Requirement"),   # unnamed
        _Rel("Same", "Deduction", "RELATED_TO", "same", "Deduction"),  # self-loop
    ])
    assert rows == []


def test_extraction_failure_yields_nothing_not_an_exception(mocker):
    class _Boom:
        def invoke(self, prompt):
            raise RuntimeError("all targets failed")
    mocker.patch.object(gx, "_get_llm", return_value=_Boom())
    assert gx.extract("Gifts and donations", "passage") == []


def test_write_groups_by_label_combination_and_records_provenance():
    driver = _FakeDriver()
    rows = gx.validate([
        _Rel("Union fees", "Deduction", "REQUIRES", "written evidence", "Requirement"),
        _Rel("Gym fees", "Deduction", "REQUIRES", "written evidence", "Requirement"),
        _Rel("Political party gifts", "Deduction", "HAS_LIMIT", "$1,500 cap on political party gifts", "Limit"),
    ])

    gx.write(driver, rows, source="page.md", parent_id="p-1")

    assert len(driver.calls) == 2  # (Deduction, REQUIRES, Requirement) and (Deduction, HAS_LIMIT, Limit)
    query, params = driver.calls[0]
    assert params["parent_id"] == "p-1" and params["source"] == "page.md"
    assert len(params["rows"]) == 2


def test_only_allow_listed_labels_ever_reach_cypher():
    # Labels are interpolated into the query (Cypher can't parameterise them),
    # so a hostile "type" from the LLM must never get that far.
    driver = _FakeDriver()
    hostile = "Deduction) DETACH DELETE (n"
    rows = gx.validate([_Rel("x", hostile, "REQUIRES", "y", "Requirement")])
    gx.write(driver, rows, source="s", parent_id="p")
    assert rows == [] and driver.calls == []


# -- graph retrieval ------------------------------------------------------------

def test_lucene_escape_neutralises_query_syntax():
    assert gr.lucene_escape('$1,500 (singles) AND "x"') == r'$1,500 \(singles\) AND \"x\"'


def test_graph_search_none_when_graph_unavailable(mocker):
    mocker.patch.object(gr, "get_driver", return_value=None)
    assert gr.graph_search(["donations"], "q") is None


def test_graph_search_empty_when_no_entity_matches(mocker):
    mocker.patch.object(gr, "get_driver", return_value=_FakeDriver(replies=[[]]))
    assert gr.graph_search(["astrology"], "q") == {"entities": [], "facts": [], "parent_ids": []}


def test_graph_search_facts_direct_first_and_deduped(mocker):
    matches = [{"key": "donations", "name": "Donations"}]
    neighbourhood = [
        {"subject": "Deductions", "relation": "REQUIRES", "object": "written evidence", "detail": None, "evidence": ["p9"], "hops": 2},
        {"subject": "Donations", "relation": "REQUIRES", "object": "DGR status", "detail": "at time of gift", "evidence": ["p1"], "hops": 1},
        {"subject": "Donations", "relation": "REQUIRES", "object": "DGR status", "detail": "at time of gift", "evidence": ["p1"], "hops": 1},
    ]
    driver = _FakeDriver(replies=[matches, neighbourhood])
    mocker.patch.object(gr, "get_driver", return_value=driver)

    out = gr.graph_search(["donations"], "q")

    assert out["entities"] == ["Donations"]
    assert out["facts"][0] == "Donations —REQUIRES→ DGR status (at time of gift)"  # 1-hop before 2-hop
    assert len(out["facts"]) == 2  # duplicate fact collapsed
    assert driver.calls[1][1]["keys"] == ["donations"]


def test_evidence_behind_direct_facts_outranks_evidence_behind_many_indirect_ones(mocker):
    neighbourhood = [
        {"subject": "Donations", "relation": "REQUIRES", "object": "DGR status", "detail": None, "evidence": ["direct"], "hops": 1},
        {"subject": "Deductions", "relation": "REQUIRES", "object": "written evidence", "detail": None, "evidence": ["indirect"], "hops": 2},
        {"subject": "Deductions", "relation": "EXCLUDES", "object": "private expenses", "detail": None, "evidence": ["indirect"], "hops": 2},
        {"subject": "Deductions", "relation": "HAS_LIMIT", "object": "some cap", "detail": None, "evidence": ["indirect"], "hops": 2},
    ]
    mocker.patch.object(gr, "get_driver", return_value=_FakeDriver(replies=[[{"key": "d", "name": "Donations"}], neighbourhood]))

    assert gr.graph_search(["donations"], "q")["parent_ids"] == ["direct", "indirect"]


def test_among_direct_evidence_the_passage_backing_more_facts_wins(mocker):
    neighbourhood = [
        {"subject": "A", "relation": "REQUIRES", "object": "B", "detail": None, "evidence": ["once"], "hops": 1},
        {"subject": "A", "relation": "HAS_LIMIT", "object": "C", "detail": None, "evidence": ["twice"], "hops": 1},
        {"subject": "A", "relation": "EXCLUDES", "object": "D", "detail": None, "evidence": ["twice"], "hops": 1},
    ]
    mocker.patch.object(gr, "get_driver", return_value=_FakeDriver(replies=[[{"key": "a", "name": "A"}], neighbourhood]))

    assert gr.graph_search(["a"], "q")["parent_ids"] == ["twice", "once"]


def test_graph_search_uses_query_text_when_router_gave_no_entities(mocker):
    driver = _FakeDriver(replies=[[]])
    mocker.patch.object(gr, "get_driver", return_value=driver)
    gr.graph_search([], "deductions requiring written evidence")
    assert driver.calls[0][1]["q"] == "deductions requiring written evidence"


def test_graph_query_failure_falls_back_to_none(mocker):
    mocker.patch.object(gr, "get_driver", return_value=_FakeDriver(fail=True))
    assert gr.graph_search(["donations"], "q") is None


# -- connection -----------------------------------------------------------------

@pytest.fixture
def _clean_client(monkeypatch):
    monkeypatch.setattr(nc, "_driver", None)
    monkeypatch.setattr(nc, "_last_failure", 0.0)
    yield


def test_no_uri_means_graph_disabled(_clean_client, monkeypatch):
    monkeypatch.setattr(settings, "NEO4J_URI", None)
    assert nc.get_driver() is None


def test_failed_connection_backs_off_instead_of_retrying_every_call(_clean_client, monkeypatch, mocker):
    monkeypatch.setattr(settings, "NEO4J_URI", "neo4j+s://example")
    ctor = mocker.patch.object(nc.GraphDatabase, "driver", side_effect=RuntimeError("unreachable"))

    assert nc.get_driver() is None
    assert nc.get_driver() is None  # within cooldown — no second connection attempt
    assert ctor.call_count == 1


def test_connection_retried_after_cooldown(_clean_client, monkeypatch, mocker):
    monkeypatch.setattr(settings, "NEO4J_URI", "neo4j+s://example")
    good = mocker.MagicMock()
    mocker.patch.object(nc.GraphDatabase, "driver", side_effect=[RuntimeError("paused"), good])

    assert nc.get_driver() is None
    monkeypatch.setattr(nc, "_last_failure", 0.0)  # cooldown elapsed
    assert nc.get_driver() is good
