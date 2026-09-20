"""
Tests for retrieval routing — router, retriever strategies, grader, rewriter,
and the compiled graph end to end. Every LLM, search, and rerank call is
mocked; what's pinned down is which path the graph takes and why.
"""
import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.agents.nodes import router, retriever, grader, rewriter, responder
from app.agents.workflow import build_workflow
from app.config import settings


class _Decision:
    def __init__(self, route, search_query="q", sub_queries=None, topics=None, income_year=""):
        self.route = route
        self.search_query = search_query
        self.sub_queries = sub_queries or []
        self.topics = topics or []
        self.income_year = income_year


class _FakeLLM:
    def __init__(self, result):
        self.result = result
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.result


def _state(**kw):
    base = {"messages": [{"role": "user", "content": "question"}], "plan": [], "current_query": "q"}
    base.update(kw)
    return base


# -- router -------------------------------------------------------------------

def test_router_conversational_skips_retrieval_and_resets_turn_state(mocker):
    mocker.patch.object(router, "structured_llm", _FakeLLM(_Decision("conversational")))
    out = router.router_node(_state())
    assert out["route"] == "conversational"
    assert out["current_query"] == "CONVERSATIONAL"
    assert out["documents"] == [] and out["retrieval_attempts"] == 0
    assert "Intent: Conversational/Memory" in out["plan"]  # eval detect_tool marker


def test_router_unknown_route_defaults_to_searching_not_conversational(mocker):
    mocker.patch.object(router, "structured_llm", _FakeLLM(_Decision("banana", "gym fees")))
    out = router.router_node(_state())
    assert out["route"] == "explanatory"
    assert "Intent: Technical" in out["plan"]  # eval detect_tool marker


def test_router_multi_part_keeps_sub_queries(mocker):
    mocker.patch.object(router, "structured_llm", _FakeLLM(_Decision("multi_part", "q", ["a", "b", "c", "d", "e"])))
    out = router.router_node(_state())
    assert out["route"] == "multi_part"
    assert out["sub_queries"] == ["a", "b", "c", "d"]  # capped at MAX_SUB_QUERIES


def test_router_multi_part_with_one_part_becomes_explanatory(mocker):
    mocker.patch.object(router, "structured_llm", _FakeLLM(_Decision("multi_part", "q", ["only one"])))
    out = router.router_node(_state())
    assert out["route"] == "explanatory" and out["sub_queries"] == []


def test_router_drops_sub_queries_for_non_multi_part_routes(mocker):
    mocker.patch.object(router, "structured_llm", _FakeLLM(_Decision("lookup", "q", ["stray"])))
    assert router.router_node(_state())["sub_queries"] == []


def test_router_validates_filters(mocker):
    mocker.patch.object(router, "structured_llm", _FakeLLM(_Decision("lookup", "q", topics=["Deductions", "fake"], income_year="2025")))
    out = router.router_node(_state())
    assert out["search_filters"] == {"topics": ["deductions"], "income_year": ""}


# -- retriever strategies ---------------------------------------------------

def _docs(prefix, n):
    return [{"content": f"{prefix} {i}", "title": "", "section": ""} for i in range(n)]


def test_lookup_keeps_five(mocker):
    search = mocker.patch.object(retriever, "search_enterprise_knowledge", return_value=_docs("d", 20))
    mocker.patch.object(retriever, "rerank_with_scores", side_effect=lambda q, docs, top_n: [(d, 0.9) for d in docs[:top_n]])
    out = retriever.retrieve_node(_state(route="lookup", retrieval_attempts=0))
    assert search.call_args.kwargs["limit"] == 20
    assert len(out["documents"]) == 5
    assert out["retrieval_attempts"] == 1


def test_explanatory_keeps_eight(mocker):
    mocker.patch.object(retriever, "search_enterprise_knowledge", return_value=_docs("d", 20))
    mocker.patch.object(retriever, "rerank_with_scores", side_effect=lambda q, docs, top_n: [(d, 0.9) for d in docs[:top_n]])
    assert len(retriever.retrieve_node(_state(route="explanatory"))["documents"]) == 8


def test_multi_part_searches_each_part_and_covers_all_of_them(mocker):
    search = mocker.patch.object(
        retriever, "search_enterprise_knowledge",
        side_effect=lambda q, limit, filters: _docs(q, limit),
    )
    mocker.patch.object(retriever, "rerank_with_scores", side_effect=lambda q, docs, top_n: [(d, 0.9) for d in docs[:top_n]])

    out = retriever.retrieve_node(_state(route="multi_part", sub_queries=["alpha", "beta"]))

    assert [c.args[0] for c in search.call_args_list] == ["alpha", "beta"]
    docs = out["documents"]
    assert any("alpha" in d for d in docs) and any("beta" in d for d in docs)
    assert "alpha" in docs[0] and "beta" in docs[1]  # interleaved, not one part first


def test_top_relevance_is_the_best_score(mocker):
    mocker.patch.object(retriever, "search_enterprise_knowledge", return_value=_docs("d", 3))
    mocker.patch.object(retriever, "rerank_with_scores", return_value=[("a", 0.2), ("b", 0.7)])
    assert retriever.retrieve_node(_state(route="lookup"))["top_relevance"] == 0.7


def test_reranker_failure_means_unscored_not_zero(mocker):
    mocker.patch.object(retriever, "search_enterprise_knowledge", return_value=_docs("d", 3))
    mocker.patch.object(retriever, "rerank_with_scores", return_value=[("a", None)])
    assert retriever.retrieve_node(_state(route="lookup"))["top_relevance"] is None


def test_nothing_retrieved_scores_zero(mocker):
    mocker.patch.object(retriever, "search_enterprise_knowledge", return_value=[])
    mocker.patch.object(retriever, "rerank_with_scores", return_value=[])
    assert retriever.retrieve_node(_state(route="lookup"))["top_relevance"] == 0.0


# -- grader -------------------------------------------------------------------

@pytest.mark.parametrize("top, attempts, expected", [
    (0.99, 1, "relevant"),
    (settings.RELEVANCE_THRESHOLD, 1, "relevant"),   # boundary is inclusive
    (0.05, 1, "retry"),
    (0.05, settings.MAX_RETRIEVAL_ATTEMPTS, "insufficient"),
    (None, 1, "relevant"),                            # reranker down: don't refuse
])
def test_grade(top, attempts, expected, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_SUFFICIENCY_CHECK", False)  # stage 1 only
    out = grader.grade_node(_state(top_relevance=top, retrieval_attempts=attempts, documents=["x"]))
    assert out["retrieval_grade"] == expected


def test_insufficient_clears_sources():
    out = grader.grade_node(_state(top_relevance=0.0, retrieval_attempts=settings.MAX_RETRIEVAL_ATTEMPTS, documents=["x"]))
    assert out["documents"] == []


def test_route_after_grade():
    assert grader.route_after_grade({"retrieval_grade": "retry"}) == "rewriter"
    assert grader.route_after_grade({"retrieval_grade": "relevant"}) == "responder"
    assert grader.route_after_grade({"retrieval_grade": "insufficient"}) == "responder"
    assert grader.route_after_grade({"retrieval_grade": "hop"}) == "retriever"
    assert grader.route_after_grade({"retrieval_grade": "partial"}) == "responder"
    assert grader.route_after_grade({"retrieval_grade": "sufficient"}) == "responder"


# -- rewriter -----------------------------------------------------------------

def test_rewriter_broadens_and_drops_filters_and_sub_queries(mocker):
    class _R:
        search_query = "broader ATO term"
    llm = _FakeLLM(_R())
    mocker.patch.object(rewriter, "structured_llm", llm)

    out = rewriter.rewrite_node(_state(current_query="narrow", sub_queries=["a", "b"], search_filters={"topics": ["deductions"]}))

    assert out["current_query"] == "broader ATO term"
    assert out["sub_queries"] == [] and out["search_filters"] == {}
    assert "['a', 'b']" in llm.prompts[0]  # told which searches already failed


# -- the compiled graph, end to end ---------------------------------------------

class _Completion:
    def __init__(self, text):
        class _M: content = text
        class _C: message = _M()
        self.choices = [_C()]


class _Verdict:
    def __init__(self, verdict, missing="", follow_up_query=""):
        self.verdict, self.missing, self.follow_up_query = verdict, missing, follow_up_query


class _ScriptedLLM:
    def __init__(self, next_result):
        self.next_result = next_result
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.next_result()


@pytest.fixture
def graph(mocker):
    responder_prompts = []

    def fake_completion(messages, **kw):
        responder_prompts.append(messages[0]["content"])
        return _Completion("answer")

    mocker.patch.object(responder, "create_completion_with_fallback", side_effect=fake_completion)
    mocker.patch.object(retriever, "search_enterprise_knowledge",
                        side_effect=lambda q, limit=20, filters=None: _docs(q, 5))

    class _R:
        search_query = "rewritten"
    mocker.patch.object(rewriter, "structured_llm", _FakeLLM(_R()))

    app = build_workflow().compile(checkpointer=MemorySaver())

    def run(decision, scores_per_attempt, verdicts=None):
        mocker.patch.object(router, "structured_llm", _FakeLLM(decision))
        verdict_iter = iter(verdicts or [])
        mocker.patch.object(grader, "_get_llm", return_value=_ScriptedLLM(
            lambda: next(verdict_iter, _Verdict("sufficient"))))
        scores = iter(scores_per_attempt)
        mocker.patch.object(retriever, "rerank_with_scores", side_effect=lambda q, docs, top_n: [(docs[0], next(scores))])
        state = {"messages": [{"role": "user", "content": "question"}], "plan": [], "current_query": "question", "documents": []}
        return app.invoke(state, config={"configurable": {"thread_id": "t"}}), responder_prompts

    return run


def test_graph_relevant_first_try(graph):
    out, prompts = graph(_Decision("lookup", "gym fees"), [0.95])
    assert out["retrieval_attempts"] == 1
    assert out["retrieval_grade"] == "sufficient"
    assert out["documents"]
    assert "CONTEXT:" in prompts[-1]


def test_graph_weak_then_rewrite_recovers(graph):
    out, prompts = graph(_Decision("lookup", "narrow"), [0.02, 0.9])
    assert out["retrieval_attempts"] == 2
    assert out["current_query"] == "rewritten"
    assert out["retrieval_grade"] == "sufficient"
    assert any("Rewritten Search Term" in s for s in out["plan"])


def test_graph_never_covered_declines_honestly(graph):
    out, prompts = graph(_Decision("lookup", "FBT car parking"), [0.0, 0.01])
    assert out["retrieval_attempts"] == settings.MAX_RETRIEVAL_ATTEMPTS  # loop is bounded
    assert out["retrieval_grade"] == "insufficient"
    assert out["documents"] == []
    assert "contains nothing relevant" in prompts[-1]


def test_graph_conversational_never_retrieves(graph, mocker):
    search = retriever.search_enterprise_knowledge
    out, prompts = graph(_Decision("conversational"), [])
    search.assert_not_called()
    assert out["route"] == "conversational"


# -- relational route (knowledge graph) ---------------------------------------

def test_router_relational_keeps_entities(mocker):
    d = _Decision("relational", "deductions requiring written evidence")
    d.entities = ["written evidence"]
    mocker.patch.object(router, "structured_llm", _FakeLLM(d))
    out = router.router_node(_state())
    assert out["route"] == "relational" and out["entities"] == ["written evidence"]
    assert "Graph entities: written evidence" in out["plan"]


def test_router_drops_entities_for_other_routes(mocker):
    d = _Decision("lookup", "q")
    d.entities = ["stray"]
    mocker.patch.object(router, "structured_llm", _FakeLLM(d))
    assert router.router_node(_state())["entities"] == []


def test_relational_merges_graph_evidence_with_vector_results(mocker):
    mocker.patch.object(retriever, "graph_search", return_value={
        "entities": ["Donations"], "facts": ["Donations —REQUIRES→ DGR status"], "parent_ids": ["p1"],
    })
    fetch = mocker.patch.object(retriever, "fetch_parents", return_value=[{"content": "graph evidence", "title": "", "section": ""}])
    mocker.patch.object(retriever, "search_enterprise_knowledge", return_value=[
        {"content": "graph evidence", "title": "", "section": ""},  # also found by vector search — not duplicated
        {"content": "vector hit", "title": "", "section": ""},
    ])
    rerank = mocker.patch.object(retriever, "rerank_with_scores", side_effect=lambda q, docs, top_n: [(d, 0.9) for d in docs[:top_n]])

    out = retriever.retrieve_node(_state(route="relational", entities=["donations"]))

    assert fetch.call_args.args[0] == ["p1"]
    assert rerank.call_args.args[1] == ["graph evidence", "vector hit"]  # evidence first, deduped
    assert out["graph_facts"] == ["Donations —REQUIRES→ DGR status"]
    assert any(s.startswith("Graph: 1 entities matched, 1 relationships, 1 evidence passages") for s in out["plan"])


def test_relational_without_graph_falls_back_to_vector_search(mocker):
    mocker.patch.object(retriever, "graph_search", return_value=None)
    fetch = mocker.patch.object(retriever, "fetch_parents")
    mocker.patch.object(retriever, "search_enterprise_knowledge", return_value=_docs("d", 5))
    mocker.patch.object(retriever, "rerank_with_scores", side_effect=lambda q, docs, top_n: [(d, 0.9) for d in docs[:top_n]])

    out = retriever.retrieve_node(_state(route="relational"))

    fetch.assert_not_called()
    assert out["graph_facts"] == [] and len(out["documents"]) == 5
    assert "Graph: unavailable — vector search only" in out["plan"]


def test_rewriter_clears_graph_entities(mocker):
    class _R:
        search_query = "broader"
    mocker.patch.object(rewriter, "structured_llm", _FakeLLM(_R()))
    assert rewriter.rewrite_node(_state(entities=["x"]))["entities"] == []


def test_responder_includes_graph_facts_only_when_present(mocker):
    prompts = []
    mocker.patch.object(responder, "create_completion_with_fallback",
                        side_effect=lambda messages, **kw: prompts.append(messages[0]["content"]) or _Completion("a"))
    responder.generate_node(_state(documents=["CONTENT: x"], graph_facts=["A —REQUIRES→ B"]))
    responder.generate_node(_state(documents=["CONTENT: x"], graph_facts=[]))
    assert "A —REQUIRES→ B" in prompts[0] and "RELATIONSHIPS" in prompts[0]
    assert "RELATIONSHIPS" not in prompts[1]


# -- sufficiency grading + hops ------------------------------------------------

def _graded(mocker, verdict, **state):
    mocker.patch.object(grader, "_get_llm", return_value=_ScriptedLLM(lambda: verdict))
    base = dict(top_relevance=0.95, retrieval_attempts=1, documents=["CONTENT: x"])
    base.update(state)
    return grader.grade_node(_state(**base))


def test_sufficient_context_goes_to_responder(mocker):
    assert _graded(mocker, _Verdict("sufficient"))["retrieval_grade"] == "sufficient"


def test_partial_context_hops_with_the_follow_up_query(mocker):
    out = _graded(mocker, _Verdict("partial", "record-keeping rules", "records for political party gifts"))
    assert out["retrieval_grade"] == "hop"
    assert out["current_query"] == "records for political party gifts"
    assert out["retrieval_mode"] == "hop" and out["hops"] == 1
    assert out["hop_queries"] == ["records for political party gifts"]
    assert out["missing_info"] == "record-keeping rules"


def test_hops_are_bounded(mocker):
    out = _graded(mocker, _Verdict("partial", "x", "another search"), hops=settings.MAX_HOPS)
    assert out["retrieval_grade"] == "partial"


def test_a_follow_up_already_tried_is_not_repeated(mocker):
    out = _graded(mocker, _Verdict("partial", "x", "Records For Gifts"), hops=1, hop_queries=["records for gifts"])
    assert out["retrieval_grade"] == "partial"  # would otherwise loop on the same search


def test_partial_without_a_follow_up_query_answers_with_what_it_has(mocker):
    assert _graded(mocker, _Verdict("partial", "x", ""))["retrieval_grade"] == "partial"


def test_failed_sufficiency_check_does_not_block_the_answer(mocker):
    class _Boom:
        def invoke(self, prompt):
            raise RuntimeError("all targets failed")
    mocker.patch.object(grader, "_get_llm", return_value=_Boom())
    out = grader.grade_node(_state(top_relevance=0.95, retrieval_attempts=1, documents=["CONTENT: x"]))
    assert out["retrieval_grade"] == "sufficient"


def test_sufficiency_judge_sees_question_and_context(mocker):
    llm = _ScriptedLLM(lambda: _Verdict("sufficient"))
    mocker.patch.object(grader, "_get_llm", return_value=llm)
    grader.grade_node(_state(top_relevance=0.95, question="standalone q", documents=["CONTENT: the passage"]))
    assert "standalone q" in llm.prompts[0] and "the passage" in llm.prompts[0]


def test_hop_that_finds_nothing_answers_from_earlier_context(mocker):
    llm = _ScriptedLLM(lambda: _Verdict("sufficient"))
    mocker.patch.object(grader, "_get_llm", return_value=llm)
    out = grader.grade_node(_state(top_relevance=0.01, retrieval_mode="hop", retrieval_attempts=1,
                                   documents=["CONTENT: earlier relevant"]))
    assert out["retrieval_grade"] == "partial"   # not retry, not "insufficient"
    assert "documents" not in out                  # earlier relevant context kept
    assert llm.prompts == []                       # no LLM call spent on an empty hop


def test_hop_retrieval_adds_to_context_without_filters(mocker):
    search = mocker.patch.object(retriever, "search_enterprise_knowledge", return_value=_docs("new", 3))
    mocker.patch.object(retriever, "rerank_with_scores", side_effect=lambda q, docs, top_n: [(d, 0.8) for d in docs[:top_n]])

    out = retriever.retrieve_node(_state(retrieval_mode="hop", hops=1, current_query="follow up",
                                         search_filters={"topics": ["deductions"]},
                                         documents=["CONTENT: earlier", "CONTENT: new 0"]))

    assert "filters" not in search.call_args.kwargs
    assert out["documents"][:2] == ["CONTENT: earlier", "CONTENT: new 0"]  # kept, in order
    assert out["documents"].count("CONTENT: new 0") == 1                   # deduped
    assert len(out["documents"]) == 4


def test_responder_states_what_is_missing_on_partial(mocker):
    prompts = []
    mocker.patch.object(responder, "create_completion_with_fallback",
                        side_effect=lambda messages, **kw: prompts.append(messages[0]["content"]) or _Completion("a"))
    responder.generate_node(_state(documents=["CONTENT: x"], retrieval_grade="partial", missing_info="FBT rules"))
    assert "NOT FOUND IN THE KNOWLEDGE BASE" in prompts[0] and "FBT rules" in prompts[0]


def test_graph_partial_then_hop_completes_the_answer(graph):
    out, prompts = graph(
        _Decision("lookup", "donation with 1500 cap records"), [0.95, 0.9],
        verdicts=[_Verdict("partial", "record-keeping rules", "records for political party gifts"),
                  _Verdict("sufficient")],
    )
    assert out["hops"] == 1
    assert out["retrieval_grade"] == "sufficient"
    assert any(d.startswith("CONTENT: donation with 1500 cap records") for d in out["documents"])  # first pass kept
    assert any(d.startswith("CONTENT: records for political party gifts") for d in out["documents"])  # hop added
    assert any(s.startswith("Hop 1 retrieved") for s in out["plan"])


def test_graph_hops_stop_at_the_limit_and_say_what_is_missing(graph):
    verdicts = [_Verdict("partial", "missing piece", f"follow up {i}") for i in range(10)]
    out, prompts = graph(_Decision("lookup", "q"), [0.95] * 10, verdicts=verdicts)
    assert out["hops"] == settings.MAX_HOPS
    assert out["retrieval_grade"] == "partial"
    assert "missing piece" in prompts[-1]


def test_hop_that_adds_nothing_new_stops_even_when_relevant(mocker):
    # Live: a hop re-found only existing passages at relevance 1.00 and was
    # judged again for nothing. No new passages -> stop, no LLM call.
    llm = _ScriptedLLM(lambda: _Verdict("partial", "x", "yet another search"))
    mocker.patch.object(grader, "_get_llm", return_value=llm)
    out = grader.grade_node(_state(top_relevance=1.0, retrieval_mode="hop", hop_new_docs=0, hops=1,
                                   documents=["CONTENT: x"]))
    assert out["retrieval_grade"] == "partial"
    assert llm.prompts == []


def test_sufficiency_prompt_tells_judge_not_to_demand_unasked_detail():
    # Guards the fix for an over-strict judge observed live.
    prompt = grader.SUFFICIENCY_PROMPT
    assert "did NOT ask for never make it partial" in prompt
    assert 'When unsure, choose "sufficient"' in prompt
