"""
Knowledge-graph extraction: turns each parent chunk into typed relationships
("gifts to political parties" -HAS_LIMIT-> "$1,500 cap on political party
gifts") and writes them to Neo4j. Qdrant keeps the passages; Neo4j keeps how
the concepts in them connect.

Every relationship stores the parent_ids and pages it was extracted from, so
a graph answer can always be backed by the real passage (fetched back from
Qdrant) rather than an unverifiable triple.

The schema is fixed, not free-form: open-ended LLM extraction produces the
same concept under several names and relationship types, and the graph
fragments. Like metadata.TOPICS, it's the domain-specific part — swap it
with the knowledge base.
"""
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import logfire
from pydantic import BaseModel, Field

from app.config import settings
from app.gateway.client import get_structured_llm_with_fallback

NODE_TYPES = {
    "Deduction": "a deductible expense or category (e.g. 'work-related car expenses', 'gifts to DGRs')",
    "TaxOffset": "a tax offset (e.g. 'seniors and pensioners tax offset')",
    "IncomeType": "a kind of income (e.g. 'salary and wages', 'lump sum payment in arrears')",
    "Requirement": "a condition or rule that must be met (e.g. 'written evidence', 'recipient has DGR status')",
    "Record": "a record or document to keep (e.g. 'receipt', 'logbook')",
    "Limit": "a cap, threshold, or amount — name it WITH what it limits (e.g. '$1,500 cap on political party gifts'), never a bare figure",
    "Organisation": "a type of organisation or party (e.g. 'deductible gift recipient', 'registered tax agent')",
    "Occupation": "an occupation or industry (e.g. 'nurses', 'building and construction')",
}
RELATIONS = {
    "REQUIRES": "the source needs the target (a condition, evidence, or record) to be claimed/met",
    "HAS_LIMIT": "the source is capped/thresholded by the target Limit",
    "APPLIES_TO": "the source applies to / is available to the target (an occupation, organisation, income type)",
    "EXCLUDES": "the source explicitly does NOT cover the target (e.g. a deduction you can't claim for it)",
    "PART_OF": "the source is a specific case of the broader target category",
    "RELATED_TO": "a clearly stated connection that fits none of the above",
}


class Relationship(BaseModel):
    source: str = Field(description="Short canonical name of the source entity, as the ATO would name it.")
    source_type: str = Field(description=f"One of: {', '.join(NODE_TYPES)}")
    relation: str = Field(description=f"One of: {', '.join(RELATIONS)}")
    target: str = Field(description="Short canonical name of the target entity.")
    target_type: str = Field(description=f"One of: {', '.join(NODE_TYPES)}")
    detail: str = Field(description="A short qualifier stated in the text (e.g. 'per income year'), or empty string.")


class GraphExtraction(BaseModel):
    relationships: list[Relationship] = Field(description="Relationships explicitly stated in the passage. Empty list if none.")


EXTRACTION_PROMPT = """Extract a knowledge graph from this passage of the ATO page "{title}".

Entity types:
{node_types}

Relationship types:
{relations}

Rules:
- Only relationships the passage states explicitly — never infer or use outside knowledge.
- Use short, canonical names so the same concept gets the same name across passages
  (e.g. always "written evidence", not "evidence in writing").
- Skip navigation text, links, and examples' personal names.

<passage>
{passage}
</passage>"""

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_structured_llm_with_fallback(
            GraphExtraction, feature="graph-extraction", method="function_calling",
            max_completion_tokens=8192,  # many relationships per passage, after reasoning; 2048 ran out (see gateway)
        )
    return _llm


def entity_key(name: str) -> str:
    """Merge key: case/whitespace-insensitive, so "Written Evidence" and
    "written  evidence" become one node."""
    return re.sub(r"\s+", " ", (name or "").strip().strip(".,;:").lower())


def _type(value: str, allowed) -> str | None:
    """Match an LLM-supplied type/relation against the allow-list (tolerating
    case and spacing). None = reject. The allow-list is also what makes it safe
    to put labels into Cypher, which can't parameterise them."""
    norm = re.sub(r"[\s_-]", "", (value or "")).lower()
    for option in allowed:
        if option.replace("_", "").lower() == norm:
            return option
    return None


def validate(relationships) -> list[dict]:
    """Drop anything off-schema, unnamed, or self-referential."""
    valid = []
    for rel in relationships:
        s_type, t_type = _type(rel.source_type, NODE_TYPES), _type(rel.target_type, NODE_TYPES)
        relation = _type(rel.relation, RELATIONS)
        s_key, t_key = entity_key(rel.source), entity_key(rel.target)
        if not (s_type and t_type and relation and s_key and t_key) or s_key == t_key:
            continue
        valid.append({
            "s_key": s_key, "s_name": rel.source.strip(), "s_type": s_type,
            "relation": relation,
            "t_key": t_key, "t_name": rel.target.strip(), "t_type": t_type,
            "detail": (rel.detail or "").strip()[:200],
        })
    return valid


def extract(title: str, passage: str) -> list[dict]:
    prompt = EXTRACTION_PROMPT.format(
        title=title,
        node_types="\n".join(f"- {k}: {v}" for k, v in NODE_TYPES.items()),
        relations="\n".join(f"- {k}: {v}" for k, v in RELATIONS.items()),
        passage=passage,
    )
    try:
        return validate(_get_llm().invoke(prompt).relationships)
    except Exception as e:
        logfire.warning(f"Graph extraction failed for a passage of {title!r}: {e}")
        return []


def write(driver, rows: list[dict], source: str, parent_id: str) -> None:
    """MERGE entities and relationships, appending provenance. One query per
    (source label, relation, target label) combination — labels and
    relationship types can't be Cypher parameters, so they're interpolated,
    and only ever from the validated allow-list."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["s_type"], row["relation"], row["t_type"])].append(row)

    for (s_type, relation, t_type), group in groups.items():
        driver.execute_query(
            f"""
            UNWIND $rows AS row
            MERGE (s:Entity {{key: row.s_key}}) ON CREATE SET s.name = row.s_name
            SET s:{s_type}
            MERGE (t:Entity {{key: row.t_key}}) ON CREATE SET t.name = row.t_name
            SET t:{t_type}
            MERGE (s)-[r:{relation}]->(t)
            SET r.detail = CASE WHEN coalesce(r.detail, '') = '' THEN row.detail ELSE r.detail END,
                r.parent_ids = CASE WHEN $parent_id IN coalesce(r.parent_ids, [])
                                    THEN r.parent_ids ELSE coalesce(r.parent_ids, []) + $parent_id END,
                r.sources = CASE WHEN $source IN coalesce(r.sources, [])
                                 THEN r.sources ELSE coalesce(r.sources, []) + $source END
            """,
            rows=group, parent_id=parent_id, source=source,
            database_=settings.NEO4J_DATABASE,
        )


def index_document_graph(driver, title: str, source: str, parents: dict[str, str], max_workers: int = 8) -> int:
    """Extract (concurrently) and write the graph for one document's parent
    chunks ({parent_id: text}). Returns the number of relationships written."""
    with logfire.span("Graph Extraction", source=source, parents=len(parents)):
        items = list(parents.items())
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            extracted = list(pool.map(lambda item: extract(title, item[1]), items))
        total = 0
        for (parent_id, _), rows in zip(items, extracted):
            if rows:
                write(driver, rows, source, parent_id)
                total += len(rows)
        logfire.info(f"Graph: {total} relationships from {len(items)} passages of {source}.")
        return total
