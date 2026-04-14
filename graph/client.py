"""
graph/client.py
설정에 따라 NetworkX 또는 Neo4j 백엔드를 반환하는 팩토리

사용법:
    from graph.client import get_graph
    g = get_graph()
    g.merge_entity("Person", "홍길동")

config.yaml에서 전환:
    graph:
      backend: "networkx"   ← 개발/소규모 (Neo4j 불필요)
      backend: "neo4j"      ← 운영/대규모
"""

from utils.config import get_config
from graph.base import GraphBackend

_instance: GraphBackend = None


def get_graph() -> GraphBackend:
    global _instance
    if _instance is None:
        cfg = get_config()
        backend = cfg.get("graph", {}).get("backend", "networkx").lower()

        if backend == "neo4j":
            from graph.neo4j_client import Neo4jGraph
            _instance = Neo4jGraph()
        else:
            from graph.networkx_client import NetworkXGraph
            _instance = NetworkXGraph()

    return _instance


def reset_instance():
    """테스트 또는 백엔드 전환 시 인스턴스 초기화"""
    global _instance
    _instance = None
