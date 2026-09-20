"""
Neo4j connection — optional infrastructure. get_driver() returns None when
NEO4J_URI is unset or the database is unreachable, and every caller treats
None as "graph unavailable, fall back to vector search".

Failures aren't cached forever (a paused free-tier instance resumes), but
retries wait out FAILURE_COOLDOWN_S — otherwise every relational question
would pay a connection timeout while the database is down.
"""
import time

import logfire
from neo4j import GraphDatabase

from app.config import settings

FAILURE_COOLDOWN_S = 60
CONNECT_TIMEOUT_S = 5

_driver = None
_last_failure = 0.0


def get_driver():
    global _driver, _last_failure
    if not settings.NEO4J_URI:
        return None
    if _driver is not None:
        return _driver
    if time.time() - _last_failure < FAILURE_COOLDOWN_S:
        return None
    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
            connection_timeout=CONNECT_TIMEOUT_S,
        )
        driver.verify_connectivity()
        _driver = driver
        logfire.info("Neo4j knowledge graph connected.")
    except Exception as e:
        _last_failure = time.time()
        logfire.warning(f"Neo4j unavailable ({e}) — graph retrieval disabled, using vector search only.")
    return _driver


def ensure_schema(driver) -> None:
    """Uniqueness on entity keys (what MERGE matches on) and a full-text index
    on names (how questions find their starting entities). 'english' analyzer:
    stemming + stop words, so "donations" matches "donation"."""
    driver.execute_query(
        "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
        database_=settings.NEO4J_DATABASE,
    )
    driver.execute_query(
        "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (e:Entity) ON EACH [e.name] "
        "OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}",
        database_=settings.NEO4J_DATABASE,
    )


def wipe_graph(driver) -> None:
    """Delete this project's graph only (nodes labelled :Entity), not the whole database."""
    driver.execute_query("MATCH (e:Entity) DETACH DELETE e", database_=settings.NEO4J_DATABASE)
