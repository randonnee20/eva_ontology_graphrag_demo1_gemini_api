"""
scripts/migrate_to_neo4j.py
NetworkX 그래프 → Neo4j 마이그레이션

실행 전:
1. Neo4j Desktop 또는 Docker 설치 및 실행
2. config.yaml의 neo4j.password 수정
3. 아직 config.yaml의 graph.backend는 "networkx" 상태 유지

실행:
python scripts/migrate_to_neo4j.py

완료 후:
config.yaml의 graph.backend를 "neo4j"로 변경
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def migrate():
    print("NetworkX → Neo4j 마이그레이션")
    print("=" * 40)

    # 1. NetworkX 그래프 로드
    from utils.config import get_config
    cfg = get_config()

    if cfg.get("graph", {}).get("backend") != "networkx":
        print("❌ config.yaml의 graph.backend가 'networkx'여야 합니다.")
        return

    from graph.networkx_client import NetworkXGraph
    nx_graph = NetworkXGraph()
    stats = nx_graph.get_stats()
    print(f"NetworkX 그래프: 노드 {stats['nodes']}개, 관계 {stats['relations']}개")

    if stats["nodes"] == 0:
        print("마이그레이션할 데이터 없음")
        return

    # 2. Neo4j 연결 테스트
    from graph.neo4j_client import Neo4jGraph
    neo4j = Neo4jGraph()
    if not neo4j.test_connection():
        print("❌ Neo4j 연결 실패. Neo4j를 먼저 실행하세요.")
        return

    print(f"✅ Neo4j 연결 성공")

    # 3. 노드 마이그레이션
    import networkx as nx
    G = nx_graph._G
    for node, data in G.nodes(data=True):
        label = data.get("label", "Unknown")
        neo4j.merge_entity(label, node)
    print(f"✅ 노드 {G.number_of_nodes()}개 마이그레이션 완료")

    # 4. 관계 마이그레이션
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "RELATED_TO")
        neo4j.merge_relation(u, rel, v)
    print(f"✅ 관계 {G.number_of_edges()}개 마이그레이션 완료")

    neo4j_stats = neo4j.get_stats()
    print(f"\n마이그레이션 결과: 노드 {neo4j_stats['nodes']}개, 관계 {neo4j_stats['relations']}개")
    print("\n다음 단계:")
    print("  config.yaml에서 graph.backend: 'neo4j' 로 변경")


if __name__ == "__main__":
    migrate()
