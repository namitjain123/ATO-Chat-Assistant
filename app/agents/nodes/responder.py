import logfire
from app.agents import history
from app.agents.state import AgentState
from app.gateway.client import create_completion_with_fallback


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) via
    create_completion_with_fallback, which walks the Azure -> Groq target chain.
    """
    query = state["current_query"]

    # Reads the summary the router already wrote this turn — plan() returns
    # nothing left to compact here, so this costs no extra LLM call.
    _, verbatim = history.plan(state["messages"], state.get("summarized_msgs") or 0)
    history_str = history.format_for_prompt(state.get("history_summary") or "", verbatim)

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful assistant for the Australian Taxation
        Office (ATO) knowledge base — every question in this system is about
        Australian tax, unless the user is just making small talk.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    elif state.get("retrieval_grade") == "insufficient":
        # The grader found nothing relevant, even after a rewritten search.
        # Answering anyway would mean answering from the model's own memory —
        # exactly the ungrounded tax answer this system exists to avoid.
        logfire.info("Retrieval insufficient — declining to answer from general knowledge.")
        prompt = f"""
        You are the assistant for an Australian Taxation Office (ATO) tax
        knowledge base. The knowledge base was searched for the user's question
        (including a rephrased retry) and contains nothing relevant to it.

        Tell the user briefly and plainly that this isn't covered by the ATO
        pages you have access to. Do NOT answer from general knowledge, and do
        not state any rates, thresholds, or rules. Suggest ato.gov.au or a
        registered tax agent for this question.

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        # Relational route: how the concepts connect, from the knowledge graph.
        # Extracted from these same ATO pages — a map for connecting passages,
        # while exact figures and wording should come from the CONTEXT itself.
        # The grader searched (including follow-up hops) and still found part of
        # the answer missing — say so instead of filling it from memory.
        missing_block = ""
        if state.get("retrieval_grade") == "partial" and state.get("missing_info"):
            missing_block = f"""
        NOT FOUND IN THE KNOWLEDGE BASE (searched, including follow-up searches):
        {state['missing_info']}
        Answer what the CONTEXT supports, and state plainly that this part isn't covered.
"""

        graph_facts = state.get("graph_facts") or []
        graph_block = ""
        if graph_facts:
            facts = "\n".join(f"- {f}" for f in graph_facts)
            graph_block = f"""
        RELATIONSHIPS (knowledge graph extracted from the same ATO pages — use
        them to connect the pieces; take exact figures and wording from CONTEXT):
{facts}
"""

        # Explicit domain framing, added after observing gpt-5-mini (Azure
        # primary) hedge on questions like "tax slab for 2025" — asking which
        # country, even while the CONTEXT it was given was visibly ATO
        # material — instead of just answering from it. This isn't a content
        # restriction (that was deliberately removed earlier so the responder
        # would answer generically from whatever's retrieved) — it's telling
        # the model what system it's part of, so it stops treating an
        # obviously-Australian, ATO-sourced context as ambiguous.
        prompt = f"""
        You are the assistant for an Australian Taxation Office (ATO) tax
        knowledge base. Every question is about Australian tax law unless
        the CONTEXT says otherwise — never ask which country; assume
        Australia and answer directly from the CONTEXT provided. If the
        CONTEXT covers only part of the question, answer that part and say
        plainly which part it doesn't cover — don't fill the gap from memory.

        CONTEXT:
        {full_context}
        {graph_block}{missing_block}
        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            # Application-level primary/fallback (Azure OpenAI -> Groq) — see
            # app/gateway/client.py for why this isn't Portkey's own server-side
            # fallback strategy (confirmed broken for Azure targets specifically).
            response = create_completion_with_fallback(
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            logfire.info("✅ Response synthesised via LLM.")

            return {
                "final_answer": content,
                "status": "Response generated.",
                "plan": state["plan"],
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e