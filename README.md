# Enterprise Agentic RAG — ATO Tax Assistant

A production-grade RAG chatbot built with **LangGraph**, **Azure OpenAI + Groq via Portkey LLM Gateway**, and **Gemini Embeddings**, answering questions from a live-crawled Australian Taxation Office (ATO) knowledge base. The system combines semantic retrieval + reranking, history-aware planning, and NeMo Guardrails for input/output safety — deployed on Azure Container Apps with a GitHub Actions CI/CD pipeline.

**Live**: https://ragchatbot-ui.delightfulwater-01722cef.australiaeast.azurecontainerapps.io/ — running Azure OpenAI as primary LLM, Groq as automatic fallback (see [LLM Provider](#llm-provider-azure-openai-primary-groq-automatic-fallback) for the full story and how failover was verified).

![Chat UI](images/chat_ui.png)

## Key Features

- **Agentic Intelligence**: LangGraph retrieval routing — a router picks a search strategy per question (lookup / explanatory / multi-part / relational / conversational), a grader checks whether retrieval actually found anything relevant, and weak results loop back through a query rewriter before the system will say a question isn't covered (see [Retrieval Routing](#retrieval-routing)).
- **Durable Memory**: LangGraph checkpointer — a shared Postgres store when `POSTGRES_URL` is set (survives restarts / multiple replicas), falling back to in-process memory for local runs.
- **Guardrails**: NeMo Guardrails gate (Llama 3.3 70B + FastEmbed embeddings-only intent matching) blocks off-topic, jailbreak, and injection inputs before any retrieval — verified against paraphrased attacks, not just exact-match examples. Layered on top of Azure OpenAI's own default content filtering, so unsafe input faces two independent checks, not one.
- **LLM Gateway**: Portkey routes all LLM calls, with **Azure OpenAI (`gpt-5-mini`) as primary and Groq as automatic fallback** — implemented at the application level after finding a bug in Portkey's own server-side fallback for Azure targets (details + how failover was verified in [LLM Provider](#llm-provider-azure-openai-primary-groq-automatic-fallback)).
- **Knowledge Graph (Neo4j)**: relationships between concepts (what a deduction *requires*, what it's *capped by*, what it *applies to*) extracted at ingestion into Neo4j, each linked back to the Qdrant passages it came from. Relationship-heavy questions are routed to graph + vector retrieval together; the graph is optional and falls back to vector search when unconfigured or unreachable (see [Knowledge Graph](#knowledge-graph-neo4j)).
- **Enterprise Search**: Qdrant Cloud, with a hybrid dense + sparse retrieval path (Gemini dense embeddings for semantic recall, BM25 sparse vectors for exact lexical matches — dollar thresholds, form/section names — fused via Reciprocal Rank Fusion) behind an `ENABLE_HYBRID_SEARCH` flag, plus FlashRank for local reranking (see [Hybrid Retrieval](#hybrid-retrieval-dense--sparse) for why it's flag-gated, not a flip-the-switch change).
- **Gemini Embeddings**: Google `gemini-embedding-2-preview` (3072-dim) via `langchain-google-genai`, with a local `sentence-transformers` fallback.
- **Live Knowledge Ingestion**: A `crawl4ai`-based deep crawler pulls real content from ato.gov.au — no static sample docs, with a pruning content filter that strips repeated site nav/footer chrome before it ever reaches the chunker.
- **Parent-Child Chunking**: small ~400-char child chunks get embedded and matched for precision; each match's larger ~2500-char parent chunk is what's actually returned to the LLM, so answers aren't built from an isolated fragment cut off mid-explanation (see [Hybrid Retrieval](#hybrid-retrieval-dense--sparse)).
- **Contextual Retrieval**: at ingestion, an LLM writes a 1-2 sentence note situating each child chunk within its full document, embedded (dense and sparse) alongside the chunk — so an ambiguous chunk like "not entitled – you can't claim a tax deduction" becomes findable by what it's actually about (DGR status).
- **Rich Metadata + Filtering**: every chunk carries title, page summary, section heading, LLM-tagged topics, and the income years it mentions; the planner extracts topic/income-year filters from the question and retrieval narrows to matching chunks — without ever excluding untagged ones, and topping up with unfiltered results if a filter leaves too few (see [Hybrid Retrieval](#hybrid-retrieval-dense--sparse)).
- **Observability**: Full trace nesting with **Pydantic Logfire** and **LangSmith** across every agent node.
- **Evaluation Suite**: Auto-generated golden Q&A dataset (via `deepeval`) + a RAGAS-powered eval pipeline (6 metrics) with a dedicated Streamlit demo app.
- **Cloud Deployment**: Azure Container Apps (backend + UI), Postgres Flexible Server, and a GitHub Actions pipeline that builds both images on every push.
- **Test Suite**: `pytest` unit tests covering the pieces most worth pinning down — chunking (a real bug regression guard), crawl boilerplate-stripping, and the Azure/Groq fallback logic (mocked, no network calls) — run as a required CI gate before every build/deploy.

---

## Agent Intelligence Flow

```mermaid
graph TD
    User((User)) --> UI[Streamlit UI]
    UI --> API[FastAPI /query]
    API --> Guard{NeMo Guardrails}
    Guard -->|Blocked| UI
    Guard -->|Pass| Router{Router Node}
    Router -->|conversational| Responder[Responder Node]
    Router -->|lookup / explanatory / multi_part| Retriever[Retriever Node<br/>strategy per route + FlashRank rerank]
    Router -->|relational| Retriever
    Retriever <-.->|relational: facts + evidence ids| Graph[(Neo4j knowledge graph)]
    Retriever <-.-> Qdrant[(Qdrant passages)]
    Retriever --> Grader{Grader<br/>1. relevance: rerank score<br/>2. sufficiency: LLM}
    Grader -->|sufficient / partial / insufficient| Responder
    Grader -->|nothing relevant, attempts left| Rewriter[Rewriter Node]
    Rewriter --> Retriever
    Grader -->|piece missing, hops left: hop| Retriever
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
app/ingestion/       → parses, dedupes formats, chunks parent-child (2500/400 char),
processor.py           contextualizes + topic-tags each child (LLM), extracts metadata
                       (title, section, income years), embeds heading+note+child with Gemini
                       (dense) + BM25 (sparse), upserts into
                       Qdrant (collection: enterprise_rag, named dense/sparse vectors)
        │
golden_synthetic.py  → deepeval Synthesizer auto-generates Q&A pairs from the same
                       crawled pages → evals/golden_dataset.json (63 RAG + 6 guardrails)
        │
evals/                → Phase 1: runs those questions through the live backend
                        (capped by eval_config.json's sample_limit — 5 by default);
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
- The generation LLM itself — `app/gateway/client.py`, `app/agents/nodes/responder.py`, `app/agents/nodes/planner.py` (since replaced by `router.py` — see [Retrieval Routing](#retrieval-routing))
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

## Knowledge Graph (Neo4j)

Some questions are about **how things connect**, not what one passage says: *"Which deductions require written evidence?"*, *"What do I need before I can claim a donation?"* Vector search finds passages that *sound like* the question; it can't follow "X requires Y, and Y is also required by Z" across pages. So relationships go in a graph, alongside the passages in Qdrant.

**Ingestion** (`app/ingestion/graph_extractor.py`): each parent chunk is sent to the LLM with a **fixed schema** — node types `Deduction, TaxOffset, IncomeType, Requirement, Record, Limit, Organisation, Occupation`; relationships `REQUIRES, HAS_LIMIT, APPLIES_TO, EXCLUDES, PART_OF, RELATED_TO`. Free-form extraction invents a new type and name for every passage and the graph fragments; off-schema output is dropped. Every relationship stores the `parent_id`s and pages it was extracted from.

**Retrieval** (`app/services/graph/graph_retrieval.py`), for questions the router sends to the `relational` route:
1. Find the starting entities with a full-text index (English analyzer, so "donations" matches "donation").
2. Traverse with **fixed, parameterised Cypher** — never LLM-generated Cypher, which is fragile and an injection surface. Node labels, which Cypher can't parameterise, come only from the validated allow-list.
3. Traversal is deliberately narrow: every relationship one hop from the starting entities, plus what they inherit from a broader category via `PART_OF`. Open two-hop traversal would route through hubs like "written evidence" and pull in half the graph.
4. The facts' `parent_id`s pull the **original passages** back out of Qdrant (evidence behind direct facts first). Those are merged with a normal vector search and reranked together — the graph only knows what extraction captured.
5. The responder gets the passages as CONTEXT plus a RELATIONSHIPS block, told to take exact figures and wording from the passages.

**Optional by design**: no `NEO4J_URI` → graph off, relational questions use vector search, ingestion skips extraction. An unreachable database (e.g. a paused free-tier instance) is retried after a 60s cooldown rather than timing out every request. `--wipe` clears the graph with the Qdrant collection — the graph's evidence pointers are that collection's `parent_id`s, regenerated on every ingest.

**Bug found while building it** (and fixed for every structured LLM call, not just this one): extraction of one passage returned *nothing*. The raw response showed `finish_reason: length`, `reasoning_tokens: 2048` of `2048`, zero tool calls — `gpt-5-mini` spent its whole budget reasoning. Worse, LangChain's structured output returns `None` rather than raising in that case, so the Azure → Groq fallback never fired. Now the token budget is per call (8192 for extraction) and a missing result raises, triggering the fallback. After the fix the same passage produced 8 relationships.

**Known limitation — entity resolution**: extraction sometimes names one concept differently across passages ("deduction for gifts **and** donations" / "**or** donations" / "gifts to DGRs"), creating separate nodes. Query-time lookup gathers the top 4 name matches per term to compensate; a proper fix is an entity-merge pass (e.g. merging near-duplicate names by embedding similarity) after ingestion.

**Setup**: create a free instance at [Neo4j AuraDB](https://console.neo4j.io), put its URI/username/password in `.env` (and as Container App secrets for the deployed backend), then re-ingest with `--wipe`. Extraction is one LLM call per parent chunk (62 for the current corpus, ~30s each, 8 at a time) — reasoning-heavy, so check the Azure cost page after the first run.

---

## Retrieval Routing

The graph used to make one decision: search, or don't. Everything after that was a straight line — retrieve 20, rerank to 8, answer — whether or not the 8 had anything to do with the question. Measured on the live corpus, a question the knowledge base doesn't cover (*"What are the FBT rules for car parking?"*) still returned 8 passages, all reranked at **0.00** relevance, and the responder was handed them as context.

Now the graph makes decisions *based on intermediate results*, which is what makes it agentic rather than a pipeline:

| Node | Decides | Cost |
|---|---|---|
| **Router** (`router.py`) | Which strategy: `conversational` (no search), `relational` (knowledge graph + vector search — see [Knowledge Graph](#knowledge-graph-neo4j)), `lookup` (a figure/threshold/limit — 20 candidates, keep 5 tight passages), `explanatory` (20 → 8 for fuller coverage), `multi_part` (split into sub-queries, search each, interleave so every part is represented); plus topic/year filters | 1 LLM call — replaces the old planner's call, not added |
| **Grader** (`grader.py`) | Two stages. (1) Did retrieval find anything relevant? — thresholds the reranker's top score. (2) Does the context actually *answer* what was asked? — an LLM judge; if a specific piece is missing it writes a follow-up search and the graph **hops** back to the retriever, adding to the context | (1) free; (2) 1 LLM call per retrieved question |
| **Rewriter** (`rewriter.py`) | Only when the grade is weak: reformulate with ATO terminology, drop filters and sub-query splits, search again | 1 LLM call, unhappy path only |
| **Responder** | Answers from context; if the loop ran out of attempts, says plainly the knowledge base doesn't cover it instead of answering from memory | — |

**Retry vs hop** — two different loops, for two different failures:
- **Retry** (nothing relevant found): rewrite the query and search again, *replacing* the context. Bounded by `MAX_RETRIEVAL_ATTEMPTS=2`.
- **Hop** (relevant, but a piece is missing): search for exactly the missing piece and *add* it to the context. Bounded by `MAX_HOPS=2`, never repeats a follow-up it already tried, and stops early when a hop only re-finds passages already in context.

This is what handles questions that need two lookups, where the second depends on the first. Live, *"What records do I need to keep for the donation type that has a $1,500 yearly cap?"*: the first retrieval identified the $1,500 cap as political-party donations but held nothing on records for them; the judge named exactly that as missing, one hop added 3 passages (relevance 0.98) on political-donation records, and the answer — keep a written record such as a receipt — came from both passes combined. If hops run out with a piece still missing, the responder says what isn't covered instead of filling it in.

**Two bugs found by running it live, not by the tests:**
- The first judge prompt asked whether anything needed for a *complete* answer was missing — and demanded examples and edge cases nobody asked for. *"Can I claim union fees?"* was answered by the first retrieval yet spent both hops and 43s. The judge now asks whether *what was asked* is answered, defaulting to sufficient when unsure: the same question now takes one pass and 20s.
- A Qdrant read timeout was being swallowed into "no results" — and two in a row would have told the user *"not covered by the knowledge base"*, a false refusal caused by infrastructure. The query client now has an explicit timeout and retries once.

**Latency cost, honestly**: the sufficiency check is one more `gpt-5-mini` call on every retrieved question, and reasoning models take seconds per call — a straightforward question measured ~20s end to end, a two-hop one ~66s. `ENABLE_SUFFICIENCY_CHECK=false` turns stage 2 off (retry still works) if latency matters more than completeness.

**Threshold, measured not guessed**: in-corpus questions reranked at **≥ 0.98**, out-of-corpus at **≤ 0.03** — `RELEVANCE_THRESHOLD=0.3` sits in a wide gap. Measured on the pre-migration chunks; worth re-checking after the parent-child re-ingest. The loop is bounded by `MAX_RETRIEVAL_ATTEMPTS=2`. If the reranker itself fails, the grade is "unscored", not zero — the system answers from what it retrieved rather than refusing because a local model didn't load.

**Verified live** on three questions: a lookup (*political party gifts* → route `lookup`, 5 passages, relevance 0.97, "$1,500"), a two-part question (*union fees + donation records* → `multi_part`, two searches, both parts answered), and the FBT question (0.00 → rewritten → 0.00 → "not covered by the ATO pages I have access to", no figures invented). The UI's reasoning steps show every routing decision.

---

## Hybrid Retrieval (Dense + Sparse)

Retrieval was pure dense vector search — Gemini embeddings, cosine similarity. That's strong on semantic recall but weak on exact terms: a user asking about "the $97,000 threshold" and a chunk that says "$97,000" should match on that number directly, not hope the embedding space happens to place them close together. Tax content is full of exactly this — dollar thresholds, form and section names, ATO-specific terminology — so a lexical (keyword) retrieval path was added alongside the semantic one.

**What it is**: each chunk now gets a second, sparse vector via FastEmbed's `Qdrant/bm25` — a statistical (non-neural) BM25 encoder, so it's CPU-cheap with no GPU and no new heavyweight dependency (FastEmbed was already in the stack for NeMo Guardrails' intent classifier). At query time, the dense and sparse rankings are fetched independently and fused with **Reciprocal Rank Fusion** — combining by rank position rather than raw score, which sidesteps dense and sparse scores living on incomparable scales.

**Parent-child chunking, at the same time**: ingestion also stopped using one flat ~1500-char chunk size (`app/ingestion/chunking/splitter.py`'s `chunk_parent_child`). A single size was always a tradeoff in both directions — small enough to embed precisely, it's too small to answer from; big enough to answer from, similarity gets diluted across unrelated sentences sharing one vector. Now small ~400-char **child** chunks get embedded and matched (both dense and sparse), but each match's larger ~2500-char **parent** chunk is what's actually returned as content — so an answer isn't built from an isolated fragment cut off mid-explanation. Because several children of one parent can each score highly, retrieval over-fetches (3x the caller's limit) and deduplicates by `parent_id`, keeping only the best-scoring child per parent.

**Contextual retrieval, on top of both**: small child chunks are precise but often ambiguous in isolation — *"The most you can claim in an income year is $1,500"* doesn't say for what. At ingestion (`app/ingestion/contextualizer.py`), an LLM is given the full document plus each child chunk and writes a 1-2 sentence note situating it; that note is prepended to the chunk before **both** dense and sparse embedding, and stored in the payload as `context`. Query-side is unchanged. Generation failures degrade per-chunk (that chunk is indexed without context) rather than failing the file. Cost is ingestion-only: ~318 LLM calls for the current corpus, run 8 at a time; `ENABLE_CONTEXTUAL_RETRIEVAL=false` skips it for a quick re-ingest.

Verified live on real chunks before shipping — e.g. the chunk *"not entitled – you can't claim a tax deduction"* came back with *"Explains how to check an organisation's deductible gift recipient (DGR) status... you can only claim a tax deduction if the recipient had DGR status at the time of your gift."*

**Rich metadata + filtering, for precision**: each point's payload carries `title` (from the page slug), `summary` (the page's opening line), `section` (nearest heading above the chunk), `topics` (tagged by the contextualizer's same LLM call, from a fixed taxonomy in `app/ingestion/metadata.py`), `income_years` (normalised from `2024–25` / `2024-2025` forms), plus `chunk_index` and `ingested_at`. `topics`, `income_years`, `source` and `source_type` get Qdrant keyword indexes at collection creation. The `Title › Section` path is also embedded with each chunk and shown as a header on every retrieved passage.

At query time the planner extracts optional `topics` and `income_year` filters — only from the fixed taxonomy, and a year only when the user explicitly names a financial year (*"tax slab for 2025"* is ambiguous between 2024-25 and 2025-26, so it isn't filtered). Three guards keep a filter from costing recall:
- **Untagged chunks always pass.** Each clause is "tagged with this value, *or* not tagged at all" — most chunks mention no income year, and they stay eligible for a year-specific question. A filter only excludes chunks positively tagged with something else.
- **Thin results are topped up.** If a filtered search returns fewer than 5 parents, it's backfilled with unfiltered results — filtered hits first, so the precision gain survives.
- **Filter errors fall back** to an unfiltered search rather than returning nothing, and filters are skipped entirely on a legacy collection with no metadata.

The topic taxonomy is the one domain-specific piece of the pipeline — swap it with the knowledge base.

### Why this is a flag (`ENABLE_HYBRID_SEARCH`), not a flip-the-switch change

This is a **schema change**, not just a code change. The existing Qdrant collection was built with a single unnamed dense vector; the migrated schema uses named `dense`/`sparse` vectors instead. Naively deploying new query code tied to one schema risks the exact failure pattern this project already hit once with the Groq model deprecation — code and live data silently disagreeing, breaking every technical query in production, discovered only after the fact.

So the two halves ship decoupled:
- **Ingestion** (`app/ingestion/processor.py`) always builds the new named dense+sparse schema for any *freshly created* collection — i.e. running `--wipe` and re-ingesting is the migration path.
- **Retrieval** (`app/services/retrieval/qdrant_service.py`) decides whether to attempt sparse+RRF fusion from `ENABLE_HYBRID_SEARCH`, but decides the actual **request shape** — named vector or not — by checking the live collection's schema at runtime (`_has_named_vectors`, cached once per process), never from the flag alone.

That second point is a fix, not the original design: the first version of this gated the request shape on `ENABLE_HYBRID_SEARCH` directly, which has a real gap — the collection rejects an unnamed query once migrated, and rejects a named one before migration (both verified directly against a real Qdrant instance), so the window between running `--wipe` and remembering to flip the flag would have broken every single search. Detecting the schema at runtime instead of trusting a manually-set flag closes that gap — the request shape always matches the data, regardless of migration/flag ordering, and `ENABLE_HYBRID_SEARCH` is left controlling only a genuine preference (spend the extra sparse-fusion work or not) rather than doubling as a correctness switch.

**Rollout, in order:**
1. Deploy this code. Nothing changes yet — schema detection sees the unmigrated collection and both the flag on or off behave exactly as before.
2. Re-run ingestion with `--wipe` (`python -m app.ingestion.processor DATA/ato_deductions ato --wipe`) — rebuilds the collection with named dense+sparse vectors, parent-child chunking, and BM25 sparse vectors for every child chunk. Retrieval keeps working immediately after this step even with the flag still off, now correctly naming the dense vector.
3. Set `ENABLE_HYBRID_SEARCH=true` and redeploy whenever you want fusion — no ordering risk either way.

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
│   │   ├── chunking/    # Parent-child text splitter (2500 char parent / 400 char child)
│   │   └── loaders/     # Local parsers — PDF (pypdf), HTML, TXT/MD, DOCX, PPTX
│   ├── services/
│   │   └── retrieval/   # Gemini embeddings + Qdrant search + FlashRank reranking
│   ├── ui/              # Streamlit chat interface with reasoning step transparency
│   ├── config.py        # Centralized environment variable management
│   └── main.py          # FastAPI entrypoint — guardrails gate + /query endpoint
├── evals/               # Golden-dataset pipeline, RAGAS eval suite, Streamlit 3-tab demo
├── tests/                # pytest unit tests — chunking, crawl utils, eval helpers, LLM fallback
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
| Vector DB | Qdrant Cloud — hybrid dense + sparse (RRF fusion), flag-gated |
| Reranking | FlashRank (local, zero-latency) |
| Embeddings | Gemini `gemini-embedding-2-preview` (3072-dim, dense) + FastEmbed `Qdrant/bm25` (sparse) |
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
PORTKEY_CONFIG_SLUG = ""            # unused: attaching a gateway config re-applies Portkey's broken Azure fallback routing
LLM_TIMEOUT_SECONDS = "60"          # ceiling per LLM call
REQUEST_DEADLINE_SECONDS = "180"    # ceiling for a whole question (per-call timeouts don't bound their sum)

# Qdrant Vector DB
QDRANT_API_KEY = ""
QDRANT_CLUSTER_ENDPOINT = ""        # e.g. https://your-cluster.cloud.qdrant.io:6333
ENABLE_HYBRID_SEARCH = "false"      # true only after re-ingesting with --wipe under the new schema — see Hybrid Retrieval section
# Knowledge graph (optional — omit NEO4J_URI to run without it)
# Username/database default to "neo4j" in Aura's docs, but newer AuraDB Free
# instances actually provision the initial DB and user under the INSTANCE ID
# (the xxxxxxxx in the URI below) — using "neo4j" for either then fails with
# a generic "Unauthorized"/"permission denied" that looks like a bad password.
# Get the real values from the instance's own "Connect" dialog in the Aura
# console, not assumed defaults.
NEO4J_URI = ""                      # e.g. neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME = "neo4j"            # verify against the Aura console — see note above
NEO4J_DATABASE = "neo4j"            # verify against the Aura console — see note above
NEO4J_PASSWORD = ""
ENABLE_SUFFICIENCY_CHECK = "true"   # LLM judges whether context answers the question; enables hops
MAX_HOPS = "2"
ENABLE_CONTEXTUAL_RETRIEVAL = "true" # ingestion only — one LLM call per child chunk; false for a fast, LLM-free re-ingest

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

Auto-generates realistic Q&A pairs from whatever's currently in `DATA/`, for use by the eval suite. Runs on Groq by default (free) rather than requiring `OPENAI_API_KEY` — batched with a persistent progress file (`DATA/golden_dataset/batch_progress.json`) so a large corpus can be generated across several runs without re-spending quota on pages already done.

```powershell
python golden_synthetic.py
```

### 7. Run the eval suite (optional)

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
