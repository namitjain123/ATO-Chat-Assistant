import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # --- GEMINI EMBEDDINGS ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    # Configurable so a new schema can be built and tested in a separate
    # collection (e.g. enterprise_rag_v2) while production keeps reading the
    # old one — then go live by switching this, with instant rollback.
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "enterprise_rag")

    # Named vectors used by the hybrid (dense + sparse) schema — see
    # app/ingestion/processor.py and app/services/retrieval/qdrant_service.py.
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

 
    ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "false").lower() == "true"

    # Ingestion-only: one LLM call per child chunk (see app/ingestion/contextualizer.py).
    # Set to false for a fast, LLM-free re-ingest.
    ENABLE_CONTEXTUAL_RETRIEVAL = os.getenv("ENABLE_CONTEXTUAL_RETRIEVAL", "true").lower() == "true"

    # --- KNOWLEDGE GRAPH (NEO4J) — optional, see app/services/graph/ ---
    # Unset NEO4J_URI = graph disabled: relational questions fall back to
    # vector search, ingestion skips graph extraction.
    NEO4J_URI = os.getenv("NEO4J_URI")  # e.g. neo4j+s://xxxxxxxx.databases.neo4j.io
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    # --- CONVERSATION HISTORY (see app/agents/history.py) ---
    # Recent turns kept word-for-word in the prompt; older ones are compacted
    # into a running summary. Without a cap the prompt grows every turn —
    # messages accumulate in state and the checkpointer keeps a thread alive
    # across sessions.
    HISTORY_KEEP_TURNS = int(os.getenv("HISTORY_KEEP_TURNS", "8"))
    # Compact only once this many turns have aged out, so compaction isn't an
    # extra LLM call on every turn past the window.
    HISTORY_COMPACT_BATCH_TURNS = int(os.getenv("HISTORY_COMPACT_BATCH_TURNS", "4"))

    # --- CALL DEADLINES ---
    # Per-LLM-call ceiling. Without one, a hung provider call hangs the request
    # until the SDK gives up on its own — and one question can make ~10 calls
    # (guardrails, router, a grade per retrieval pass, rewriter, responder).
    LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    # Whole-graph ceiling: bounds the SUM of those calls, since each one's own
    # timeout doesn't. Past this the user gets an answer, not a hung request.
    REQUEST_DEADLINE_SECONDS = float(os.getenv("REQUEST_DEADLINE_SECONDS", "180"))

    # --- RETRIEVAL ROUTING (see app/agents/nodes/grader.py) ---
    # Top FlashRank score below this = "nothing relevant retrieved". Measured on
    # the live corpus: in-corpus questions scored >= 0.98, out-of-corpus <= 0.03.
    RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))
    # Total retrieval passes per question: the first search + rewrite retries.
    MAX_RETRIEVAL_ATTEMPTS = int(os.getenv("MAX_RETRIEVAL_ATTEMPTS", "2"))
    # After relevance passes, an LLM checks the context actually ANSWERS the
    # question; if part is missing, a follow-up search ("hop") adds to it.
    # One extra LLM call per retrieved question — false to skip it.
    ENABLE_SUFFICIENCY_CHECK = os.getenv("ENABLE_SUFFICIENCY_CHECK", "true").lower() == "true"
    MAX_HOPS = int(os.getenv("MAX_HOPS", "2"))

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    # "v1" (Azure's versionless surface) 404s on the classic AzureOpenAI SDK
    # pattern this project uses — verified directly: 2024-10-21 works, "v1"
    # doesn't, for this deployment/SDK combination.
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")


    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # When the guardrails check itself can't run (after a rebuild-and-retry):
    # false = pass the message through unchecked (availability; logged as an
    # error), true = block it with a "try again" message (safety).
    GUARDRAILS_FAIL_CLOSED = os.getenv("GUARDRAILS_FAIL_CLOSED", "false").lower() == "true"

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    # Unused by the app: a saved dashboard config has to be attached to a
    # client to take effect, and doing that re-applies Portkey's broken
    # Azure fallback routing (see app/gateway/client.py). Kept only so an
    # existing .env with this set doesn't look like a missing setting.
    PORTKEY_CONFIG_SLUG = os.getenv("PORTKEY_CONFIG_SLUG")
    AZURE_SLUG = "ragchatbot-azure"   # primary: Azure OpenAI (see step 2 in notes — added on the Portkey dashboard as an LLM Integration, not in code)
    GROQ_SLUG =  "ragchatbot"          # fallback 1: @rag/llama-3.3-70b-versatile
    GROQ_SLUG_2 = "enterprise-chatbot" # fallback 2: @brag/llama-3.1-8b-instant

    
    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()