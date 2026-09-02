import logfire
from app.agents.state import AgentState
from app.gateway.client import create_completion_with_fallback


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) via
    create_completion_with_fallback, which walks the Azure -> Groq target chain.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

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
        Australia and answer directly from the CONTEXT provided.

        CONTEXT:
        {full_context}

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