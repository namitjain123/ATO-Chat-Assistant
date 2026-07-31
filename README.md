# Enterprise Agentic RAG — ATO Tax Assistant

A production-grade RAG chatbot built with **LangGraph**, **Portkey LLM Gateway**, and **Gemini Embeddings**, answering questions from a live-crawled Australian Taxation Office (ATO) knowledge base. The system combines semantic retrieval + reranking, history-aware planning, and NeMo Guardrails for input/output safety — deployed on Azure Container Apps with a GitHub Actions CI/CD pipeline.

**Live**: https://ragchatbot-ui.delightfulwater-01722cef.australiaeast.azurecontainerapps.io/

## Key Features

- **Agentic Intelligence**: LangGraph for cyclic reasoning, multi-step planning, and conversation memory.
- **Durable Memory**: LangGraph checkpointer — a shared Postgres store when `POSTGRES_URL` is set (survives restarts / multiple replicas), falling back to in-process memory for local runs.
- **Guardrails**: NeMo Guardrails gate (Llama 3.3 70B + FastEmbed embeddings-only intent matching) blocks off-topic, jailbreak, and injection inputs before any retrieval — verified against paraphrased attacks, not just exact-match examples.
- **LLM Gateway**: Portkey routes all LLM calls with automatic fallback between primary and backup Groq keys, plus response caching.
- **Enterprise Search**: Qdrant Cloud for high-performance vector search + FlashRank for local semantic reranking.
- **Gemini Embeddings**: Google `gemini-embedding-2-preview` (3072-dim) via `langchain-google-genai`, with a local `sentence-transformers` fallback.
- **Live Knowledge Ingestion**: A `crawl4ai`-based deep crawler pulls real content from ato.gov.au — no static sample docs.
- **Observability**: Full trace nesting with **Pydantic Logfire** and **LangSmith** across every agent node.
- **Evaluation Suite**: Auto-generated golden Q&A dataset (via `deepeval`) + a RAGAS-powered eval pipeline (6 metrics) with a dedicated Streamlit demo app.
- **Cloud Deployment**: Azure Container Apps (backend + UI), Postgres Flexible Server, and a GitHub Actions pipeline that builds both images on every push.

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

---

## Project Structure

```text
├── app/
│   ├── agents/
│   │   ├── graph.py     # LangGraph state machine + checkpointer (Postgres / in-memory)
│   │   └── nodes/       # Planner, Retriever, Responder LangGraph nodes
│   ├── gateway/         # Portkey LLM gateway — primary + fallback Groq routing, caching
│   ├── guardrails/      # NeMo Guardrails input/output filtering (Llama 3.3 70B + FastEmbed)
│   ├── ingestion/
│   │   ├── chunking/    # Paragraph-based text splitter (~1500 char target)
│   │   └── loaders/     # Local parsers — PDF (pypdf), HTML, TXT/MD, DOCX, PPTX
│   ├── services/
│   │   └── retrieval/   # Gemini embeddings + Qdrant search + FlashRank reranking
│   ├── ui/              # Streamlit chat interface with reasoning step transparency
│   ├── config.py        # Centralized environment variable management
│   └── main.py          # FastAPI entrypoint — guardrails gate + /query endpoint
├── evals/               # Golden-dataset pipeline, RAGAS eval suite, Streamlit 3-tab demo
├── crawl.py              # Deep-crawls a live source site into DATA/<name>/
├── golden_synthetic.py   # Generates evals/golden_dataset.json from crawled docs
├── processed_data/       # Auto-generated — parsed & chunked JSON output per document
├── DATA/                 # Crawled source pages (git-ignored; regenerate with crawl.py)
├── Dockerfile            # Backend image
├── Dockerfile.ui          # Streamlit UI image (lean — no ML deps)
├── .github/workflows/     # CI — builds & pushes both images to ACR on every push
└── requirements.txt      # Pinned dependencies
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangChain + LangGraph |
| LLMs | Groq (Llama 3.3 70B) via **Portkey** gateway |
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

### 2. Configure environment

Create a `.env` file with the following keys:

```env
# Groq Reasoning Engine (Llama 3.3)
GROQ_API_KEY = ""
GROQ_FALLBACK_API_KEY = ""          # second Groq key, or same as primary

# Portkey LLM Gateway
PORTKEY_API_KEY = ""

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

### 5. Launch the app

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

### 6. (Optional) Regenerate the golden dataset

Auto-generates realistic Q&A pairs from whatever's currently in `DATA/`, for use by the eval suite.

```powershell
python golden_synthetic.py
```

### 7. Run the eval suite (optional)

```powershell
# Requires the FastAPI backend running on :8000
streamlit run evals/app.py --server.port 8502
```

Three tabs: review the golden dataset, run the 75 questions live against the backend, then score the results with RAGAS. Full runs are slow by design (rate-limit-safe pacing) — set `EVAL_SAMPLE_LIMIT=5` before launching to test against a small subset instead of all 75.

---

## Deployment

Deployed on **Azure Container Apps**: a backend Container App (FastAPI + LangGraph, 1 vCPU/2 GiB), a lean Streamlit UI Container App, and a Postgres Flexible Server for durable conversation memory — all built and pushed by a GitHub Actions workflow (`.github/workflows/`) on every push to `main`. See `Dockerfile` / `Dockerfile.ui` for the two images.
