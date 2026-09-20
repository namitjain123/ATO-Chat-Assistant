import os
import sys
import time
import uuid
import json
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from app.config import settings
from app.services.retrieval.embedding import embed_texts, get_embedding_dim
from app.services.retrieval.sparse_embedding import embed_documents_sparse
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_parent_child
from app.ingestion.contextualizer import contextualize_chunks, build_embedding_inputs, embedding_prefix
from app.ingestion.graph_extractor import index_document_graph
from app.services.graph.neo4j_client import get_driver, ensure_schema, wipe_graph
from app.ingestion.metadata import (
    title_from_filename, page_summary, heading_positions, section_at, locate_chunks, income_years,
)

# Payload fields query-time filters run against (see qdrant_service._build_filter).
KEYWORD_INDEXED_FIELDS = ("topics", "income_years", "source", "source_type", "parent_id")  # parent_id: graph evidence lookups

logfire.configure(service_name="enterprise-ingestion-service", token=os.getenv("LOGFIRE_TOKEN"), send_to_logfire="if-token-present")

# Local folder where parsed + chunked JSON metadata is saved (replaces GCS processed bucket)
PROCESSED_DATA_DIR = "processed_data"

# Initialize Qdrant Client (generous timeout — Qdrant Cloud free-tier clusters can be slow to wake)
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    timeout=60,
)


def _with_retry(fn, *args, attempts: int = 4, **kwargs):
    """Call fn with exponential backoff (1s -> 2s -> 4s) on Qdrant timeouts."""
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except ResponseHandlingException as e:
            if attempt < attempts - 1:
                wait = 2 ** attempt
                logfire.warning(
                    f"Qdrant request timed out — retrying in {wait}s "
                    f"(attempt {attempt + 1}/{attempts})."
                )
                time.sleep(wait)
            else:
                logfire.error(f"Qdrant request failed after {attempts} attempts: {e}")
                raise


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(file_path: str, filename: str, source_type: str):
    """Parse → chunk → save locally → embed → index in Qdrant."""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext in ("txt", "md"):
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return

            # 2. Chunk text — parent-child ("small-to-big"): small child chunks get
            # embedded and searched for precise matching, but each child's larger
            # parent chunk is what's actually stored as retrievable content — see
            # chunk_parent_child's docstring for why a single flat chunk size was
            # always a tradeoff in both directions.
            pairs = chunk_parent_child(full_text)
            if not pairs:
                return

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": pairs,  # each: {parent_id, parent_text, child_text}
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data → {local_path}")

            # 4. Embed (dense + sparse) and index in Qdrant
            with logfire.span("Vectorizing & Indexing"):
                child_texts = [pair["child_text"] for pair in pairs]
                if settings.ENABLE_CONTEXTUAL_RETRIEVAL:
                    annotations = contextualize_chunks(full_text, child_texts)
                else:
                    annotations = [{"context": "", "topics": []} for _ in child_texts]

                title = title_from_filename(filename)
                summary = page_summary(full_text)
                headings = heading_positions(full_text)
                sections = [section_at(headings, pos) for pos in locate_chunks(full_text, child_texts)]

                # Heading path is embedded too — deterministic context that still
                # helps when ENABLE_CONTEXTUAL_RETRIEVAL is off.
                prefixes = [embedding_prefix(title, section, ann["context"]) for section, ann in zip(sections, annotations)]
                embed_inputs = build_embedding_inputs(prefixes, child_texts)
                dense_vectors = embed_texts(embed_inputs)
                sparse_vectors = embed_documents_sparse(embed_inputs)
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            settings.DENSE_VECTOR_NAME: dense_vec,
                            settings.SPARSE_VECTOR_NAME: sparse_vec,
                        },
                        payload={
                            "text": pair["parent_text"],       # returned to the LLM — full parent context
                            "child_text": pair["child_text"],  # what was matched (embedded with prefix)
                            "context": ann["context"],         # contextual-retrieval note, "" if disabled/failed
                            "parent_id": pair["parent_id"],    # dedupe key — see qdrant_service.py
                            "chunk_index": i,                  # position within the document
                            "source": filename,
                            "source_type": source_type,
                            "title": title,
                            "summary": summary,
                            "section": section,
                            "topics": ann["topics"],           # filterable — LLM-tagged, [] if unknown
                            "income_years": income_years(f"{pair['child_text']} {ann['context']}"),  # filterable
                            "ingested_at": int(time.time()),
                        },
                    )
                    for i, (pair, ann, section, dense_vec, sparse_vec) in enumerate(
                        zip(pairs, annotations, sections, dense_vectors, sparse_vectors)
                    )
                ]

                UPSERT_BATCH_SIZE = 100
                for i in range(0, len(points), UPSERT_BATCH_SIZE):
                    batch = points[i : i + UPSERT_BATCH_SIZE]
                    _with_retry(
                        qdrant_client.upsert,
                        collection_name=settings.QDRANT_COLLECTION,
                        points=batch,
                    )
                num_parents = len({pair["parent_id"] for pair in pairs})
                logfire.info(f"Indexed {len(points)} child points ({num_parents} parent chunks) to Qdrant from {filename}.")

            # 5. Knowledge graph (Neo4j) — per parent chunk: bigger than a child,
            # so each extraction call sees whole statements, and ~5x fewer calls.
            driver = get_driver()
            if driver is not None:
                parents = {pair["parent_id"]: pair["parent_text"] for pair in pairs}
                try:
                    index_document_graph(driver, title, filename, parents)
                except Exception as e:
                    # Qdrant indexing above already succeeded — don't report the file as failed.
                    logfire.error(f"Graph indexing failed for {filename} (vector index unaffected): {e}")

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")


# Preferred order when the same page exists in multiple formats (cleanest text first).
_FORMAT_PREFERENCE = [".md", ".txt", ".pdf", ".docx", ".pptx", ".html", ".htm"]


def _format_rank(filename: str) -> int:
    ext = os.path.splitext(filename)[1].lower()
    return _FORMAT_PREFERENCE.index(ext) if ext in _FORMAT_PREFERENCE else len(_FORMAT_PREFERENCE)


def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory, keeping one file per basename.

    A crawl often saves the same page as .md/.docx/.pdf; ingesting all of them
    would index the same content multiple times, so we pick the cleanest format
    per page (e.g. .md over .docx).
    """
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

        by_stem: dict[str, str] = {}
        for f in files:
            stem = os.path.splitext(f)[0]
            if stem not in by_stem or _format_rank(f) < _format_rank(by_stem[stem]):
                by_stem[stem] = f
        chosen = sorted(by_stem.values())

        skipped = len(files) - len(chosen)
        logfire.info(f"Found {len(files)} files in {dir_path}; ingesting {len(chosen)} (skipped {skipped} duplicate formats).")
        for filename in chosen:
            process_file(os.path.join(dir_path, filename), filename, source_type)


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if _with_retry(qdrant_client.collection_exists, settings.QDRANT_COLLECTION):
                    _with_retry(qdrant_client.delete_collection, settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")
                # The graph's evidence pointers are this collection's parent_ids,
                # which are regenerated on every ingest — wipe both or neither.
                driver = get_driver()
                if driver is not None:
                    wipe_graph(driver)
                    logfire.info("Knowledge graph wiped.")

        driver = get_driver()
        if driver is not None:
            ensure_schema(driver)
        elif settings.NEO4J_URI:
            logfire.warning("NEO4J_URI is set but Neo4j is unreachable — ingesting WITHOUT the knowledge graph.")

        # Recreate collection — dimension resolved at runtime after embedding model probe.
        # Named dense + sparse vectors (hybrid retrieval): a collection created before
        # this had a single unnamed dense vector, which the hybrid query code (see
        # qdrant_service.py, gated by settings.ENABLE_HYBRID_SEARCH) cannot query by
        # name. This branch only runs on a fresh collection or after --wipe, so it's
        # also the migration path — running --wipe + re-ingesting is what upgrades an
        # existing collection to the hybrid schema.
        if not _with_retry(qdrant_client.collection_exists, settings.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            _with_retry(
                qdrant_client.create_collection,
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config={
                    settings.DENSE_VECTOR_NAME: models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    settings.SPARSE_VECTOR_NAME: models.SparseVectorParams(),
                },
            )
            for field in KEYWORD_INDEXED_FIELDS:
                _with_retry(
                    qdrant_client.create_payload_index,
                    collection_name=settings.QDRANT_COLLECTION,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}' "
                f"(dense: {dim}-dim Cosine, sparse: BM25, keyword indexes: {', '.join(KEYWORD_INDEXED_FIELDS)})."
            )

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type)


if __name__ == "__main__":
    # Usage:
    #   python -m app.ingestion.processor DATA --wipe
    #   python -m app.ingestion.processor DATA/true_data true
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    logfire.info("Ingestion job completed.")