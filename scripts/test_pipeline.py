"""
scripts/test_pipeline.py
파이프라인 단계별 테스트 스크립트
실행: python scripts/test_pipeline.py

각 단계를 순서대로 테스트하고 결과를 출력합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def sep(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print("="*50)


def test_config():
    sep("1. 설정 확인")
    from utils.config import get_config, get_root
    cfg = get_config()
    print(f"✅ 프로젝트 루트: {get_root()}")
    print(f"   LLM 경로: {cfg['llm']['model_path']}")
    print(f"   임베딩 모델: {cfg['embedding']['model']}")
    print(f"   그래프 백엔드: {cfg.get('graph', {}).get('backend', 'networkx')}")
    from pathlib import Path as P
    llm_exists = P(cfg['llm']['model_path']).exists()
    print(f"   LLM 파일 존재: {'✅' if llm_exists else '❌ (config.yaml에서 경로 확인 필요)'}")


def test_graph():
    sep("2. 그래프 DB 테스트")
    from graph.client import get_graph
    g = get_graph()
    print(f"✅ 백엔드 연결: {g.test_connection()}")

    g.merge_entity("TestClass", "테스트엔티티A")
    g.merge_entity("TestClass", "테스트엔티티B")
    g.merge_relation("테스트엔티티A", "RELATED_TO", "테스트엔티티B")

    nodes = g.search_nodes("테스트")
    print(f"✅ 노드 검색: {nodes}")

    stats = g.get_stats()
    print(f"✅ 통계: 노드 {stats['nodes']}개, 관계 {stats['relations']}개")


def test_chunker():
    sep("3. 청킹 테스트")
    from pipeline.chunker import split_text
    sample = "이것은 테스트 텍스트입니다. 문장 분리가 올바르게 되어야 합니다. 두 번째 문단입니다.\n\n세 번째 단락이 여기 있습니다. 충분히 긴 텍스트가 필요합니다. 마지막 문장."
    chunks = split_text(sample, chunk_size=50, overlap=10)
    print(f"✅ 청크 수: {len(chunks)}개")
    for i, c in enumerate(chunks, 1):
        print(f"   [{i}] {c[:60]}...")


def test_embedder():
    sep("4. FAISS 임베딩 테스트")
    from pipeline.embedder import update_faiss, search_faiss, get_index_stats

    test_chunks = [
        "고소작업 시에는 반드시 안전벨트를 착용해야 합니다.",
        "밀폐공간 작업 전 산소 농도를 측정해야 합니다.",
        "전기작업 시 절연장갑을 착용하세요.",
    ]
    update_faiss(test_chunks)
    stats = get_index_stats()
    print(f"✅ FAISS 상태: {stats}")

    chunks, dists = search_faiss("안전벨트 착용", top_k=2)
    print(f"✅ 검색 결과 ({len(chunks)}개):")
    for c, d in zip(chunks, dists):
        print(f"   거리={d:.3f}  {c[:60]}")


def test_ontology():
    sep("5. 온톨로지 자동 확장 테스트")
    from pipeline.ontology_updater import update_ontology, get_ontology

    new_ents = [
        {"type": "TestConcept", "name": "자동학습엔티티1"},
        {"type": "TestConcept", "name": "자동학습엔티티2"},
        {"type": "NewType", "name": "신규타입엔티티"},
    ]
    added = update_ontology(new_ents)
    print(f"✅ 추가된 항목: {added}개")

    onto = get_ontology()
    print(f"✅ 현재 온톨로지 클래스: {list(onto.keys())}")


def test_graph_builder():
    sep("6. 그래프 빌더 테스트")
    from pipeline.graph_builder import build_graph
    from graph.client import get_graph

    data = {
        "entities": [
            {"type": "Concept", "name": "개념A"},
            {"type": "Concept", "name": "개념B"},
        ],
        "relations": [
            {"source": "개념A", "relation": "RELATED_TO", "target": "개념B"},
        ]
    }
    build_graph(data)
    g = get_graph()
    stats = g.get_stats()
    print(f"✅ 그래프 현재 상태: 노드 {stats['nodes']}개, 관계 {stats['relations']}개")


def test_retriever():
    sep("7. 리트리버 테스트")
    from rag.retriever import retrieve
    ctx = retrieve("안전벨트")
    print(f"✅ 컨텍스트 길이: {len(ctx)}자")
    print(f"   앞 200자: {ctx[:200]}")


def cleanup():
    """테스트 후 정리"""
    sep("정리 (테스트 데이터 삭제)")
    from graph.client import get_graph
    from utils.config import get_config

    # 온톨로지에서 테스트 항목 제거
    from pipeline.ontology_updater import get_ontology
    import yaml
    from pathlib import Path
    cfg = get_config()
    onto = get_ontology()
    onto.pop("TestConcept", None)
    onto.pop("NewType", None)
    Path(cfg["paths"]["ontology"]).write_text(
        yaml.dump(onto, allow_unicode=True, default_flow_style=False), encoding="utf-8"
    )
    print("✅ 테스트 온톨로지 항목 제거 완료")

    # 그래프는 유지 (실제 데이터와 섞이지 않으므로)
    print("ℹ️  그래프 테스트 노드는 유지됩니다 (리셋 필요 시: python main.py reset)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="파이프라인 단계별 테스트")
    parser.add_argument("--step", type=int, default=0, help="특정 단계만 실행 (0=전체)")
    parser.add_argument("--no-cleanup", action="store_true", help="테스트 후 정리 생략")
    args = parser.parse_args()

    steps = [
        (1, test_config),
        (2, test_graph),
        (3, test_chunker),
        (4, test_embedder),
        (5, test_ontology),
        (6, test_graph_builder),
        (7, test_retriever),
    ]

    for num, fn in steps:
        if args.step == 0 or args.step == num:
            try:
                fn()
            except Exception as e:
                print(f"❌ 오류: {e}")
                import traceback
                traceback.print_exc()

    if not args.no_cleanup:
        try:
            cleanup()
        except Exception:
            pass

    print("\n✅ 테스트 완료\n")
