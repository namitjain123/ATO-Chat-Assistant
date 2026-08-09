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
    QDRANT_COLLECTION = "enterprise_rag"

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    # --- CACHE (REDIS, L2 — see app/services/cache.py) ---
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # --- PRIMARY LLM (AZURE OPENAI) ---
    # Primary, not fallback: if Azure were fallback-only the system would never
    # touch it in normal operation. Groq is now the automatic fallback (see
    # app/gateway/client.py's GATEWAY_CONFIG and the Portkey dashboard config
    # referenced by PORTKEY_CONFIG_SLUG below, which is what's actually active
    # since it's set).
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    # "v1" (Azure's versionless surface) 404s on the classic AzureOpenAI SDK
    # pattern this project uses — verified directly: 2024-10-21 works, "v1"
    # doesn't, for this deployment/SDK combination.
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG_SLUG = os.getenv("PORTKEY_CONFIG_SLUG")  # saved config slug, e.g. "pc-xxxxxxxx"
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