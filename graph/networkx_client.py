"""
graph/networkx_client.py
NetworkX 기반 로컬 그래프 백엔드
- 설치 불필요 (networkx는 sentence-transformers 의존성으로 자동 설치됨)
- data/graph/graph.pkl 파일로 영속 저장
- Neo4j 없이 완전 동작
- 추후 Neo4j로 마이그레이션 가능
"""

import pickle
import logging
import re
from pathlib import Path
from typing import List, Dict

import networkx as nx

from graph.base import GraphBackend
from utils.config import get_config

logger = logging.getLogger(__name__)


class NetworkXGraph(GraphBackend):

    def __init__(self):
        cfg = get_config()
        self._db_path = Path(cfg["paths"]["graph_db"]) / "graph.pkl"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._G: nx.MultiDiGraph = self._load()

    # ──────────────────────────────────────────────────
    # 영속성
    # ──────────────────────────────────────────────────

    def _load(self) -> nx.MultiDiGraph:
        if self._db_path.exists():
            with open(self._db_path, "rb") as f:
                G = pickle.load(f)
            logger.info(f"그래프 로드: 노드 {G.number_of_nodes()}개, 엣지 {G.number_of_edges()}개")
            return G
        return nx.MultiDiGraph()

    def _save(self):
        with open(self._db_path, "wb") as f:
            pickle.dump(self._G, f)

    # ──────────────────────────────────────────────────
    # GraphBackend 구현
    # ──────────────────────────────────────────────────

    def merge_entity(self, label: str, name: str) -> None:
        name = name.strip()
        if not name:
            return
        if name not in self._G:
            self._G.add_node(name, label=label)
        else:
            # 레이블 업데이트 (기존 노드면 레이블만 갱신)
            self._G.nodes[name]["label"] = label
        self._save()

    def merge_relation(self, source: str, relation: str, target: str) -> None:
        source, target = source.strip(), target.strip()
        if not source or not target:
            return
        # 노드가 없으면 Unknown으로 자동 추가
        if source not in self._G:
            self._G.add_node(source, label="Unknown")
        if target not in self._G:
            self._G.add_node(target, label="Unknown")

        # 같은 source→target 간 동일 relation이 이미 있으면 스킵
        existing = [
            d["relation"]
            for _, _, d in self._G.edges(source, data=True)
            if d.get("relation") == relation
               and list(self._G.successors(source))  # target 확인은 아래에서
        ]
        # 정확한 중복 확인
        already = any(
            d.get("relation") == relation
            for u, v, d in self._G.edges(data=True)
            if u == source and v == target
        )
        if not already:
            self._G.add_edge(source, target, relation=relation)
            self._save()

    def search_nodes(self, query: str, limit: int = 10) -> List[Dict]:
        query = query.strip()
        results = []
        for node, data in self._G.nodes(data=True):
            if query in node:
                results.append({
                    "name": node,
                    "labels": [data.get("label", "Unknown")],
                })
                if len(results) >= limit:
                    break
        return results

    def search_related(self, name: str, depth: int = 2) -> List[Dict]:
        if name not in self._G:
            return []

        results = []
        # BFS로 depth 내 이웃 탐색
        try:
            neighbors = nx.single_source_shortest_path(self._G, name, cutoff=depth)
        except Exception:
            return []

        for neighbor in list(neighbors.keys())[:20]:
            if neighbor == name:
                continue
            # name → neighbor 경로의 관계 수집
            try:
                paths = list(nx.all_simple_paths(self._G, name, neighbor, cutoff=depth))
            except Exception:
                continue
            for path in paths[:2]:
                for i in range(len(path) - 1):
                    edges = self._G.get_edge_data(path[i], path[i + 1]) or {}
                    for _, edge_data in edges.items():
                        rel = edge_data.get("relation", "RELATED_TO")
                        target_label = self._G.nodes[path[i + 1]].get("label", "Unknown")
                        results.append({
                            "source": path[i],
                            "relations": [rel],
                            "target": path[i + 1],
                            "target_labels": [target_label],
                        })
        return results[:20]

    def get_stats(self) -> Dict:
        return {
            "nodes": self._G.number_of_nodes(),
            "relations": self._G.number_of_edges(),
        }

    def get_labels(self) -> List[str]:
        labels = set()
        for _, data in self._G.nodes(data=True):
            lbl = data.get("label")
            if lbl:
                labels.add(lbl)
        return sorted(labels)

    def clear(self) -> None:
        self._G = nx.MultiDiGraph()
        self._save()
        logger.warning("NetworkX 그래프 초기화 완료")

    def test_connection(self) -> bool:
        return True  # 로컬 파일 기반이므로 항상 가능

    # ──────────────────────────────────────────────────
    # 추가 유틸리티
    # ──────────────────────────────────────────────────

    def export_to_json(self, path: str) -> None:
        """그래프를 JSON으로 내보내기 (Neo4j 마이그레이션용)"""
        import json
        data = nx.node_link_data(self._G)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"그래프 JSON 내보내기 완료: {path}")

    def get_subgraph_text(self) -> str:
        """그래프 전체를 텍스트로 요약 (RAG 컨텍스트용)"""
        lines = []
        for u, v, data in self._G.edges(data=True):
            rel = data.get("relation", "RELATED_TO")
            lines.append(f"{u} --[{rel}]--> {v}")
        return "\n".join(lines)
