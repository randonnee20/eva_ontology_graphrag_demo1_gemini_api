"""
scripts/verify_ontology.py
온톨로지가 실제로 작동하는지 단계별 검증

실행: python scripts/verify_ontology.py
실행: python scripts/verify_ontology.py --query "고소작업 안전 장비"
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def sep(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print('─'*50)


# ──────────────────────────────────────────────────
# [1] 온톨로지 파일 상태
# ──────────────────────────────────────────────────
def check_ontology():
    sep("1️⃣  온톨로지 상태 (classes.yaml)")
    from pipeline.ontology_updater import get_ontology
    from utils.config import get_config

    cfg = get_config()
    onto_path = cfg["paths"]["ontology"]
    onto = get_ontology()

    if not onto:
        print("  ❌ 온톨로지 비어 있음")
        print("  → 문서를 먼저 ingestion 하세요")
        return False

    total = sum(len(v) for v in onto.values() if isinstance(v, list))
    print(f"  ✅ 클래스 수: {len(onto)}개  |  총 엔티티: {total}개")
    print(f"  📄 파일: {onto_path}\n")

    for cls, vals in onto.items():
        if isinstance(vals, list):
            filled = "✅" if vals else "⬜ (비어있음)"
            print(f"  {filled}  {cls} ({len(vals)}개)")
            if vals:
                # 최대 5개만 표시
                preview = vals[:5]
                more = f" ... +{len(vals)-5}개" if len(vals) > 5 else ""
                print(f"          → {', '.join(preview)}{more}")

    return True


# ──────────────────────────────────────────────────
# [2] 그래프 노드가 온톨로지 타입과 일치하는지
# ──────────────────────────────────────────────────
def check_graph_alignment():
    sep("2️⃣  그래프 ↔ 온톨로지 정합성")
    from graph.client import get_graph
    from pipeline.ontology_updater import get_ontology

    g = get_graph()
    onto = get_ontology()
    stats = g.get_stats()

    if stats["nodes"] == 0:
        print("  ❌ 그래프 노드 없음")
        print("  → 문서를 먼저 ingestion 하세요")
        return False

    print(f"  그래프: 노드 {stats['nodes']}개  |  관계 {stats['relations']}개\n")

    # NetworkX인 경우 노드별 레이블 분포 확인
    from graph.networkx_client import NetworkXGraph
    if not isinstance(g, NetworkXGraph):
        print("  ℹ️  Neo4j 백엔드 - db.labels() 로 확인하세요")
        return True

    G = g._G
    onto_types = set(onto.keys())

    # 레이블별 노드 수 집계
    from collections import Counter
    label_counter = Counter()
    unknown_nodes = []

    for node, data in G.nodes(data=True):
        lbl = data.get("label", "Unknown")
        label_counter[lbl] += 1
        if lbl not in onto_types:
            unknown_nodes.append((node, lbl))

    print("  레이블별 노드 분포:")
    for lbl, cnt in label_counter.most_common():
        in_onto = "✅" if lbl in onto_types else "⚠️ (온톨로지 외부)"
        print(f"    {in_onto}  {lbl}: {cnt}개")

    if unknown_nodes:
        print(f"\n  ⚠️  온톨로지에 없는 타입의 노드 ({len(unknown_nodes)}개):")
        for node, lbl in unknown_nodes[:5]:
            print(f"    - '{node}' (타입: {lbl})")
        if len(unknown_nodes) > 5:
            print(f"    ... +{len(unknown_nodes)-5}개")
        print("  → 이 노드들은 LLM이 새로 만든 타입 (정상 동작)")
    else:
        print("\n  ✅ 모든 노드가 온톨로지 타입 내에 있음")

    # 관계 샘플 출력
    print("\n  관계 샘플 (최대 10개):")
    count = 0
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "?")
        u_lbl = G.nodes[u].get("label", "?")
        v_lbl = G.nodes[v].get("label", "?")
        print(f"    ({u_lbl}){u}  --[{rel}]-->  ({v_lbl}){v}")
        count += 1
        if count >= 10:
            remaining = G.number_of_edges() - 10
            if remaining > 0:
                print(f"    ... +{remaining}개 더")
            break

    return True


# ──────────────────────────────────────────────────
# [3] 실제 질의 시 온톨로지가 답변에 기여하는지
# ──────────────────────────────────────────────────
def check_retrieval(query: str):
    sep(f"3️⃣  하이브리드 검색 분해  |  질의: '{query}'")
    from rag.retriever import retrieve_with_detail
    from pipeline.embedder import get_index_stats

    faiss_stats = get_index_stats()
    if faiss_stats["total_vectors"] == 0:
        print("  ❌ FAISS 인덱스 없음 → 문서 ingestion 먼저")
        return

    detail = retrieve_with_detail(query)

    # 벡터 검색 결과
    v_chunks = detail["vector_chunks"]
    v_dists  = detail["vector_distances"]
    print(f"\n  [벡터 검색]  {len(v_chunks)}개 청크 반환")
    for i, (chunk, dist) in enumerate(zip(v_chunks, v_dists), 1):
        print(f"    {i}. 거리={dist:.4f}  |  {chunk[:80].strip()}...")

    # 그래프 검색 결과
    g_facts = detail["graph_facts"]
    if g_facts:
        lines = g_facts.strip().split("\n")
        print(f"\n  [그래프 검색]  {len(lines)}개 관계 반환")
        for line in lines:
            print(f"    {line}")
        print("\n  ✅ 온톨로지 기반 그래프가 검색에 기여하고 있음")
    else:
        print("\n  [그래프 검색]  결과 없음")
        print("  → 질의 키워드와 일치하는 노드가 그래프에 없거나")
        print("    아직 문서 ingestion 전임")

    # 최종 컨텍스트 길이
    ctx = detail["combined_context"]
    v_only = len("\n".join(v_chunks))
    g_only = len(g_facts)
    print(f"\n  최종 LLM 컨텍스트: {len(ctx)}자")
    print(f"    벡터 기여: {v_only}자  |  그래프 기여: {g_only}자")

    if g_only > 0:
        ratio = g_only / len(ctx) * 100
        print(f"    그래프 비중: {ratio:.1f}%  ← 이 만큼 온톨로지가 답변에 영향")


# ──────────────────────────────────────────────────
# [4] ingestion 전후 온톨로지 diff 시뮬레이션
# ──────────────────────────────────────────────────
def check_ontology_growth():
    sep("4️⃣  온톨로지 성장 기록")
    from utils.config import get_config
    from pipeline.ontology_updater import get_ontology
    import os

    cfg = get_config()
    onto_path = Path(cfg["paths"]["ontology"])

    mtime = onto_path.stat().st_mtime
    import datetime
    modified = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"  마지막 수정: {modified}")

    onto = get_ontology()
    total = sum(len(v) for v in onto.values() if isinstance(v, list))

    if total == 0:
        print("  ⬜ 아직 엔티티 없음 (문서 ingestion 후 자동으로 채워짐)")
    else:
        print(f"  ✅ 현재까지 학습된 엔티티: {total}개 (문서 추가할수록 증가)")
        print("  → 새 문서 ingestion 시 이 숫자가 올라가면 온톨로지가 성장한 것")


# ──────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="온톨로지 작동 검증")
    parser.add_argument(
        "--query", "-q",
        default="고소작업 안전 장비",
        help="검색 테스트에 사용할 질의 (기본: '고소작업 안전 장비')"
    )
    parser.add_argument(
        "--step", "-s", type=int, default=0,
        help="특정 단계만 실행 (1~4, 기본: 0=전체)"
    )
    args = parser.parse_args()

    print("\n🔍 온톨로지 작동 검증 시작")

    steps = {
        1: check_ontology,
        2: check_graph_alignment,
        4: check_ontology_growth,
    }

    if args.step == 0:
        check_ontology()
        check_graph_alignment()
        check_retrieval(args.query)
        check_ontology_growth()
    elif args.step == 3:
        check_retrieval(args.query)
    elif args.step in steps:
        steps[args.step]()
    else:
        print(f"유효한 step: 1, 2, 3, 4")

    print("\n✅ 검증 완료\n")


if __name__ == "__main__":
    main()
