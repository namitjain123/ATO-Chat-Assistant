# Enterprise Agentic RAG — ATO Tax Assistant

A production-grade RAG chatbot built with **LangGraph**, **Azure OpenAI + Groq via Portkey LLM Gateway**, and **Gemini Embeddings**, answering questions from a live-crawled Australian Taxation Office (ATO) knowledge base. The system combines semantic retrieval + reranking, history-aware planning, and NeMo Guardrails for input/output safety — deployed on Azure Container Apps with a GitHub Actions CI/CD pipeline.

**Live**: https://ragchatbot-ui.delightfulwater-01722cef.australiaeast.azurecontainerapps.io/ — running Azure OpenAI as primary LLM, Groq as automatic fallback (see [LLM Provider](#llm-provider-azure-openai-primary-groq-automatic-fallback) for the full story and how failover was verified).

![Chat UI](images/chat_ui.png)

## Key Features

- **Agentic Intelligence**: LangGraph for cyclic reasoning, multi-step planning, and conversation memory.
- **Durable Memory**: LangGraph checkpointer — a shared Postgres store when `POSTGRES_URL` is set (survives restarts / multiple replicas), falling back to in-process memory for local runs.
- **Guardrails**: NeMo Guardrails gate (Llama 3.3 70B + FastEmbed embeddings-only intent matching) blocks off-topic, jailbreak, and injection inputs before any retrieval — verified against paraphrased attacks, not just exact-match examples. Layered on top of Azure OpenAI's own default content filtering, so unsafe input faces two independent checks, not one.
- **LLM Gateway**: Portkey routes all LLM calls, with **Azure OpenAI (`gpt-5-mini`) as primary and Groq as automatic fallback** — implemented at the application level after finding a bug in Portkey's own server-side fallback for Azure targets (details + how failover was verified in [LLM Provider](#llm-provider-azure-openai-primary-groq-automatic-fallback)).
- **Two-Layer Cache**: In-process (`cachetools`) + Redis (shared/persistent) caching for embeddings and retrieval — the two pipeline layers Portkey's own gateway cache doesn't cover. Degrades to in-process-only if Redis isn't running, never a hard dependency.
- **Enterprise Search**: Qdrant Cloud for high-performance vector search + FlashRank for local semantic reranking.
- **Gemini Embeddings**: Google `gemini-embedding-2-preview` (3072-dim) via `langchain-google-genai`, with a local `sentence-transformers` fallback.
- **Live Knowledge Ingestion**: A `crawl4ai`-based deep crawler pulls real content from ato.gov.au — no static sample docs, with a pruning content filter that strips repeated site nav/footer chrome before it ever reaches the chunker.
- **Observability**: Full trace nesting with **Pydantic Logfire** and **LangSmith** across every agent node.
- **Evaluation Suite**: Auto-generated golden Q&A dataset (via `deepeval`) + a RAGAS-powered eval pipeline (6 metrics) with a dedicated Streamlit demo app.
- **Cloud Deployment**: Azure Container Apps (backend + UI), Postgres Flexible Server, and a GitHub Actions pipeline that builds both images on every push.
- **Test Suite**: `pytest` unit tests covering the pieces most worth pinning down — chunking (a real bug regression guard), the two-layer cache, crawl boilerplate-stripping, and the Azure/Groq fallback logic (mocked, no network calls) — run as a required CI gate before every build/deploy.

---

## Agent Intelligence Flow

```mermaid
graph TD
    User((User)) --> UI[Streamlit UI]
    UI --> API[FastAPI /query]
    API --> Guard{NeMo Guardrails}
    Guard -->|Blocked| UI
    Guard -->|Pass| Planner{Planner Node}
    Planner -->|Conversational| Responder[Responder Node]
    Planner -->|Technical| Retriever[Retriever Node]
    Retriever --> Reranker[FlashRank Local Reranker]
    Reranker --> Responder
    Responder --> UI
    Responder -.-> Memory[(Checkpointer: Postgres / in-memory)]
```

---

## Knowledge Base Pipeline

The chatbot's knowledge isn't a static bundled dataset — it's crawled live and re-ingested end to end:

```
crawl.py            → deep-crawls ato.gov.au (BFS, 25 pages, skips thin nav pages)
                       output: DATA/ato_deductions/*.md (+.docx)
        │
app/ingestion/       → parses, dedupes formats, chunks (~1500 char paragraphs),
processor.py           embeds with Gemini, upserts into Qdrant (collection: enterprise_rag)
        │
golden_synthetic.py  → deepeval Synthesizer auto-generates 75 Q&A pairs from the
                       same crawled pages → evals/golden_dataset.json
        │
evals/                → Phase 1: runs those 75 questions through the live backend;
                        Phase 2: RAGAS-style scoring (Faithfulness, Relevancy,
                        Context Precision/Recall, Correctness, Tool Correctness)
```

Run `python crawl.py` to (re-)crawl a source, `python -m app.ingestion.processor DATA/<folder> <tag> --wipe` to index it, and `python golden_synthetic.py` to regenerate the eval's golden dataset from whatever's currently crawled.

### Evaluation Results

Real numbers from the eval suite, captured against the live deployed instance after the Azure OpenAI migration:

![Final Summary](images/final_summary.png)

Faithfulness and Tool Correctness both perfect, Context Recall and Answer Correctness solidly "Good," Context Precision "Fair" — the one metric with room to improve. The per-sample breakdown below is left in, "None" cells included, deliberately: a couple of Faithfulness/Answer Correctness samples came back `None` in this run from the judge model (`openai/gpt-oss-20b` on Groq) hitting its daily quota mid-run — a real, known constraint of running eval judging on a free tier, not silently hidden.

<details>
<summary>Per-sample metric breakdown</summary>

![Faithfulness and Answer Relevancy](images/faithfulness.png)
![Context Precision and Context Recall](images/context_precision.png)
![Answer Correctness and Tool Correctness](images/answer_correctness.png)

</details>

---

## LLM Provider: Azure OpenAI (primary), Groq (automatic fallback)

The generation LLM (the responder node, and the planner node's routing decision) runs on **Azure OpenAI's `gpt-5-mini`** as primary, with **Groq automatically as fallback** if Azure fails. Primary, not fallback-only — a fallback-only integration would mean the system never actually touches Azure in normal operation, which undercuts "runs on Azure OpenAI" as a true claim. This mirrors a real enterprise pattern: a managed/compliant primary provider, with a commodity provider as the safety net.

### What moved, and what didn't

**Changed:**
- The generation LLM itself — `app/gateway/client.py`, `app/agents/nodes/responder.py`, `app/agents/nodes/planner.py`
- The Portkey configuration — a new `ragchatbot-azure` Azure OpenAI integration
- The config/env surface — `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT`, centralized in `app/config.py` alongside everything else

**Deliberately untouched:**
- **Gemini embeddings** — Azure's embedding models use a different vector dimension than Gemini's; switching would mean recreating the Qdrant collection at a new size, re-embedding the whole corpus, and re-validating retrieval quality regressed. Real cost, no payoff — nobody scans a resume for which embedding model backs a RAG system.
- **Qdrant** — vector store choice reads as commodity tooling; migrating to Azure AI Search would turn a one-day change into a week, for no visible benefit.
- **FlashRank** — local, free, zero-latency, already working. Its Azure-side equivalent is tied to the vector store, which isn't moving.
- **NeMo Guardrails** — kept deliberately, and it's now a *better* story than before: Azure OpenAI applies its own content filtering to every call by default, so keeping NeMo on top gives genuinely layered safety (platform-level filtering + this project's own jailbreak/injection detection), rather than either alone.
- **LangGraph orchestration, Logfire/LangSmith tracing** — both provider-agnostic; nothing to change.

### The bugs found along the way (and the real fix)

Getting Azure genuinely answering as primary — not silently falling back to Groq on every request — took more than a config change:

1. **API version**: Azure's versionless `v1` API surface 404s against this project's SDK pattern; `2024-10-21` is what actually works.
2. **`gpt-5-mini` rejects non-default `temperature`** — it's a reasoning model and only accepts the default (1). The same code path serves Groq too, so `temperature` overrides were removed entirely rather than special-cased.
3. **`gpt-5-mini` rejects `max_tokens`**, requiring `max_completion_tokens` instead — and needs generous headroom, since reasoning models spend tokens on internal chain-of-thought before writing the visible answer (verified directly: a one-word reply alone consumed 128 reasoning tokens; too small a limit silently returns empty content, not an error).
4. **The real blocker**: Portkey's own server-side fallback strategy has a confirmed bug for Azure OpenAI targets specifically. Every *direct-addressed* call to the Azure target succeeded (verified repeatedly, including through Portkey's own "Run Test Request" tool); every identical call routed through the saved fallback config failed with `"azure-openai error: Resource not found"` — despite the Azure resource, the Portkey integration, the deployment/alias/API-version mapping, and the config's target JSON all being verified correct, character-for-character. Most likely cause: Azure's REST API requires the deployment name in the URL path itself (unlike Groq/OpenAI-style providers, where `model` is just a body field), and Portkey's fallback-iteration code path doesn't appear to construct that URL correctly, while its simpler direct-request path does.

**The fix**: primary/fallback is implemented at the **application level** instead (`app/gateway/client.py`'s `create_completion_with_fallback` and `get_structured_llm_with_fallback`) — call Azure directly, catch any failure, retry Groq directly. One more trap along the way: even this initially still hit Groq, because the shared Portkey client had a `config` attached at the *instance* level, which kept applying the same broken routing underneath even with an explicit `model=` override. Fixed with a second, config-free Portkey client used specifically for this fallback path.

### How failover was actually verified, not just assumed

Three separate, real tests — not code review, actual forced runs:

- **Normal operation (local)**: confirmed the response model is genuinely `gpt-5-mini-2025-08-07` (Azure), not a Groq model name, across multiple live `/query` calls.
- **Forced failure (local)**: temporarily swapped in a Portkey integration slug that doesn't exist at all (no "default model" safety net to mask the test), called `create_completion_with_fallback` directly, confirmed the failure was caught and logged (`Target ... failed`), and that it correctly fell through to Groq — response model `llama-3.3-70b-versatile`, real content returned.
- **Normal operation (deployed)**: after redeploying via CI/CD, confirmed the same result holds on the live Container App, not just on a laptop — see the eval screenshots below for real scores captured against the deployed instance.

### Cost note

Azure OpenAI has no free tier — unlike everything else in this project's stack so far. At dev/test scale with a small model, this runs to cents, and this project's Azure for Students credit (~$100 USD) comfortably covers it — but it's the first component here that isn't free, worth budget-checking before treating it as a default choice for a larger deployment.

---

## Project Structure

```text
├── app/
│   ├── agents/
│   │   ├── graph.py     # LangGraph state machine + checkpointer (Postgres / in-memory)
│   │   └── nodes/       # Planner, Retriever, Responder LangGraph nodes
│   ├── gateway/         # Portkey LLM gateway — Azure OpenAI primary, Groq automatic fallback, caching
│   ├── guardrails/      # NeMo Guardrails input/output filtering (Llama 3.3 70B + FastEmbed)
│   ├── ingestion/
│   │   ├── chunking/    # Paragraph-based text splitter (~1500 char target)
│   │   └── loaders/     # Local parsers — PDF (pypdf), HTML, TXT/MD, DOCX, PPTX
│   ├── services/
│   │   ├── cache.py     # Two-layer (in-process + Redis) cache for embeddings/retrieval
│   │   └── retrieval/   # Gemini embeddings + Qdrant search + FlashRank reranking
│   ├── ui/              # Streamlit chat interface with reasoning step transparency
│   ├── config.py        # Centralized environment variable management
│   └── main.py          # FastAPI entrypoint — guardrails gate + /query endpoint
├── evals/               # Golden-dataset pipeline, RAGAS eval suite, Streamlit 3-tab demo
├── tests/                # pytest unit tests — chunking, cache, crawl utils, LLM fallback
├── crawl.py              # Deep-crawls a live source site into DATA/<name>/
├── golden_synthetic.py   # Generates evals/golden_dataset.json from crawled docs
├── processed_data/       # Auto-generated — parsed & chunked JSON output per document
├── DATA/                 # Crawled source pages (git-ignored; regenerate with crawl.py)
├── Dockerfile            # Backend image
├── Dockerfile.ui          # Streamlit UI image (lean — no ML deps)
├── .github/workflows/     # CI — runs tests, then builds/pushes/deploys both images on every push
└── requirements.txt      # Pinned dependencies
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangChain + LangGraph |
| LLMs | Azure OpenAI `gpt-5-mini` (primary) + Groq Llama 3.3 70B / 3.1 8B (fallback), via **Portkey** gateway |
| Guardrails | NeMo Guardrails (embeddings-only intent matching via FastEmbed) |
| Vector DB | Qdrant Cloud |
| Reranking | FlashRank (local, zero-latency) |
| Embeddings | Gemini `gemini-embedding-2-preview` (3072-dim) |
| Ingestion source | Live crawl via `crawl4ai` (deep BFS crawl) |
| Document Parsing | pypdf + pdfplumber (local, no OCR service) |
| Observability | Pydantic Logfire + LangSmith |
| Evaluation | `deepeval` (golden dataset generation) + RAGAS + custom Tool Correctness (Jaccard) |
| Deployment | Azure Container Apps + ACR + Postgres Flexible Server, via GitHub Actions |

---

## Getting Started

### 1. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

To also run the test suite (`pytest`), install `requirements-dev.txt` instead — it includes everything in `requirements.txt` plus test-only tooling (pytest, crawl4ai for `tests/test_crawl_utils.py`) that's deliberately kept out of the main file so the deployed Docker images don't carry it:

```powershell
pip install -r requirements-dev.txt
pytest
```

### 2. Configure environment

Create a `.env` file with the following keys:

```env
# Azure OpenAI (primary LLM — see "LLM Provider" section above)
AZURE_OPENAI_ENDPOINT = ""          # e.g. https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_API_VERSION = "2024-10-21"
AZURE_OPENAI_DEPLOYMENT = "gpt-5-mini"

# Groq Reasoning Engine (automatic fallback if Azure fails)
GROQ_API_KEY = ""
GROQ_FALLBACK_API_KEY = ""          # second Groq key, or same as primary

# Portkey LLM Gateway
PORTKEY_API_KEY = ""
PORTKEY_CONFIG_SLUG = ""            # saved dashboard config, e.g. "pc-xxxxxxxx"

# Qdrant Vector DB
QDRANT_API_KEY = ""
QDRANT_CLUSTER_ENDPOINT = ""        # e.g. https://your-cluster.cloud.qdrant.io:6333

# Durable conversation memory (optional — omit to use in-process memory locally)
POSTGRES_URL = ""                   # e.g. postgresql://user:pass@host:5432/db?sslmode=require

# Pydantic Logfire Observability
LOGFIRE_TOKEN = ""

# LangSmith
LANGSMITH_TRACING = true
LANGSMITH_ENDPOINT = https://api.smith.langchain.com
LANGSMITH_API_KEY = ""
LANGSMITH_PROJECT = ""

# Streamlit UI → FastAPI
BACKEND_URL = ""                    # e.g. http://localhost:8000

# Eval judge LLM (keep separate from main key to avoid rate-limiting the live app)
JUDGE_GROQ = ""

# Two-layer cache (optional — omit to run with in-process caching only)
REDIS_URL = "redis://localhost:6379/0"

# Gemini Embeddings
GEMINI_API_KEY = ""

# Golden dataset generation (evals/golden_synthetic.py — optional, only needed to regenerate goldens)
OPENAI_API_KEY = ""
```

### 3. Crawl a knowledge source

```powershell
python crawl.py
```

Edit `START_URL` / `OUTPUT_DIR` / `MAX_PAGES` at the top of `crawl.py` to point at a different site or section. Output lands in `DATA/<name>/` as `.md` + `.docx` per page.

### 4. Run data ingestion

Parses the crawled documents, chunks them, saves metadata to `processed_data/`, and indexes vectors into Qdrant.

```powershell
python -m app.ingestion.processor DATA/ato_deductions ato --wipe
```

> Pass `--wipe` to drop and recreate the Qdrant collection. Omit it to append to an existing collection.

### 5. (Optional) Start the local Redis cache

```powershell
docker compose up -d redis
```

Enables the shared (L2) layer of the two-layer cache. Skip this entirely if you don't have Docker — the app degrades to in-process-only caching automatically, no crash, no config change needed.

### 6. Launch the app

```powershell
# Terminal 1 — FastAPI backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run app/ui/app.py --server.port 8501
```

> Streamlit always defaults to port 8501 regardless of which app you run — it
> only moves to another port if 8501 is already taken by another running
> instance at that exact moment. If you plan to run the chat UI and the eval
> suite (below) at the same time, pin both ports explicitly as shown here,
> rather than relying on the default to "figure it out."

### 7. (Optional) Regenerate the golden dataset

Auto-generates realistic Q&A pairs from whatever's currently in `DATA/`, for use by the eval suite. Runs on Groq by default (free) rather than requiring `OPENAI_API_KEY` — batched with a persistent progress file (`DATA/golden_dataset/batch_progress.json`) so a large corpus can be generated across several runs without re-spending quota on pages already done.

```powershell
python golden_synthetic.py
```

### 8. Run the eval suite (optional)

```powershell
# Requires the FastAPI backend running on :8000
streamlit run evals/app.py --server.port 8502
```

Three tabs: review the golden dataset, run the questions live against the backend, then score the results with RAGAS. Full runs are slow by design (rate-limit-safe pacing) — set a sample limit via the "⚙️ Settings" panel in the app's sidebar (persisted to `evals/eval_config.json`, survives process restarts) to test against a small subset instead of the full golden set. The Step 3 metrics tab also lets you re-run just a subset of the 6 metrics, rather than all 6, to conserve judge-model quota on a partial retry.

To test the **deployed** backend instead of a local one, set `EVAL_TARGET_URL` before launching (Step 2's tab shows which target is active either way):

```powershell
$env:EVAL_TARGET_URL = "https://ragchatbot.delightfulwater-01722cef.australiaeast.azurecontainerapps.io/query"
streamlit run evals/app.py --server.port 8502
```

---

## Deployment

Deployed on **Azure Container Apps**: a backend Container App (FastAPI + LangGraph, 1 vCPU/2 GiB), a lean Streamlit UI Container App, and a Postgres Flexible Server for durable conversation memory — all built and pushed by a GitHub Actions workflow (`.github/workflows/`) on every push to `main`. See `Dockerfile` / `Dockerfile.ui` for the two images.

The CI/CD pipeline builds both images, pushes them to ACR, then explicitly redeploys both Container Apps (`az containerapp update`) — pushing to ACR alone doesn't redeploy a running app, and an update call with an unchanged image tag doesn't reliably force a fresh pull either, so each deploy step uses a unique `--revision-suffix` (the GitHub Actions run number) to guarantee a genuinely new revision every time. Verified end-to-end: the [LLM Provider](#llm-provider-azure-openai-primary-groq-automatic-fallback) change is confirmed live at the URL above.
