"""
graph/neo4j_client.py
Neo4j 백엔드 (config.yaml의 graph.backend: "neo4j" 일 때 사용)
Neo4j Desktop 또는 Docker 설치 후 전환
"""

import logging
import re
from typing import List, Dict

from graph.base import GraphBackend
from utils.config import get_config

logger = logging.getLogger(__name__)

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        cfg = get_config()
        _driver = GraphDatabase.driver(
            cfg["neo4j"]["uri"],
            auth=(cfg["neo4j"]["user"], cfg["neo4j"]["password"]),
        )
    return _driver


class Neo4jGraph(GraphBackend):

    def merge_entity(self, label: str, name: str) -> None:
        safe = _safe_label(label)
        with _get_driver().session() as s:
            s.run(f"MERGE (n:`{safe}` {{name: $name}})", name=name.strip())

    def merge_relation(self, source: str, relation: str, target: str) -> None:
        safe_rel = _safe_rel(relation)
        cypher = f"""
        MATCH (a {{name: $source}})
        MATCH (b {{name: $target}})
        MERGE (a)-[:`{safe_rel}`]->(b)
        """
        with _get_driver().session() as s:
            s.run(cypher, source=source.strip(), target=target.strip())

    def search_nodes(self, query: str, limit: int = 10) -> List[Dict]:
        cypher = "MATCH (n) WHERE n.name CONTAINS $q RETURN labels(n) AS labels, n.name AS name LIMIT $limit"
        with _get_driver().session() as s:
            return [dict(r) for r in s.run(cypher, q=query, limit=limit)]

    def search_related(self, name: str, depth: int = 2) -> List[Dict]:
        cypher = """
        MATCH path = (a {name: $name})-[*1..2]-(b)
        RETURN a.name AS source,
               [r IN relationships(path) | type(r)] AS relations,
               b.name AS target,
               labels(b) AS target_labels
        LIMIT 20
        """
        with _get_driver().session() as s:
            return [dict(r) for r in s.run(cypher, name=name)]

    def get_stats(self) -> Dict:
        with _get_driver().session() as s:
            nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return {"nodes": nodes, "relations": rels}

    def get_labels(self) -> List[str]:
        with _get_driver().session() as s:
            result = s.run("CALL db.labels() YIELD label RETURN label")
            return [r["label"] for r in result]

    def clear(self) -> None:
        with _get_driver().session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    def test_connection(self) -> bool:
        try:
            with _get_driver().session() as s:
                s.run("RETURN 1")
            return True
        except Exception as e:
            logger.error(f"Neo4j 연결 실패: {e}")
            return False


def _safe_label(label: str) -> str:
    return re.sub(r"[`'\"]", "", label).strip() or "Unknown"


def _safe_rel(relation: str) -> str:
    return re.sub(r"[^A-Z_가-힣0-9]", "_", relation.upper()) or "RELATED_TO"
