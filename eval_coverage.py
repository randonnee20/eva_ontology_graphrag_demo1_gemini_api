"""
eval_coverage.py — 문서 학습 완전성 검증 도구 (v2)
실행: python eval_coverage.py
"""

import sys
import json
import logging
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.WARNING)

from utils.config import get_config
from pipeline.embedder import search_faiss
from graph.client import get_graph

# ── 문서별 검증 질문 + 예상 키워드 ──────────────────
# expected: 이 단어가 검색 결과 청크에 있어야 정상
DOC_QUERIES = {
    "산업안전보건관리비 해설집": [
        ("산업안전보건관리비 계상 기준",   ["계상", "산업안전보건관리비"]),
        ("건설업 안전관리비 사용 항목",     ["안전관리비", "사용"]),
        ("안전관리비 요율",               ["요율", "관리비"]),
    ],
    "안전보건 교육 기본서(건설업)": [
        ("건설업 안전보건 교육 시간",       ["교육시간", "교육 시간", "건설업"]),
        ("건설공사 안전관리",             ["건설공사", "안전관리"]),
        ("추락 방지 안전조치",            ["추락", "방지", "안전난간"]),
    ],
    "안전보건 교육 기본서(서비스업)": [
        ("서비스업 안전관리",             ["서비스업", "안전관리"]),
        ("도소매업 안전보건",             ["도소매", "소비자용품"]),
        ("뇌심혈관질환 예방",             ["뇌", "심혈관", "뇌ㆍ심혈관"]),
    ],
    "안전보건 교육 기본서(제조업)": [
        ("제조업 안전보건 교육",           ["제조업", "안전보건"]),
        ("기계 방호장치",                ["방호장치", "방호"]),
        ("화학물질 취급 안전",            ["화학물질", "취급"]),
    ],
    "고양시청소년재단 중대산업재해": [
        ("중대재해처벌법",               ["중대재해", "처벌"]),
        ("고양시청소년재단 안전보건",      ["고양", "청소년재단"]),
        ("중대산업재해 예방",            ["중대산업재해", "예방"]),
    ],
    "산업안전보건교육 가이드북": [
        ("안전보건교육 가이드",           ["가이드", "교육"]),
        ("교육 대상 및 시간",            ["교육대상", "교육 대상", "교육시간"]),
        ("교육 실시 방법",              ["실시", "교육방법"]),
    ],
    "산업안전보건법 시행규칙": [
        ("안전보건교육 시행규칙",         ["시행규칙", "교육"]),
        ("과태료 부과 기준",            ["과태료", "부과"]),
        ("보호구 착용 기준",            ["보호구", "착용"]),
    ],
    "산업안전보건법 시행령": [
        ("안전보건관리책임자",           ["안전보건관리책임자", "관리책임자"]),
        ("산업안전보건위원회",           ["안전보건위원회", "위원회"]),
        ("도급인의 안전조치",           ["도급", "도급인"]),
    ],
    "산업안전보건법(법률)": [
        ("산업안전보건법 목적",          ["목적", "산업안전보건법"]),
        ("사업주 의무",               ["사업주", "의무"]),
        ("근로자 보호",               ["근로자", "보호"]),
    ],
    "신규입사자 안전보건 교재": [
        ("신규입사자 안전교육",          ["신규입사자", "신규", "채용"]),
        ("채용 시 교육",              ["채용", "배치 전", "직무 배치"]),
        ("안전수칙",                 ["안전수칙", "안전ㆍ보건수칙"]),
    ],
}


def check_vector(query: str, keywords: list, top_k: int = 5) -> dict:
    chunks, dists = search_faiss(query, top_k=top_k)
    # 키워드 중 하나라도 청크에 있으면 hit
    hit_chunk = ""
    hit = False
    for chunk in chunks:
        for kw in keywords:
            if kw in chunk:
                hit = True
                hit_chunk = chunk[:100]
                break
        if hit:
            break
    return {
        "query": query,
        "hit": hit,
        "top_dist": round(float(dists[0]), 1) if dists else None,
        "hit_preview": hit_chunk,
        "miss_preview": chunks[0][:80] if chunks and not hit else "",
    }


def check_graph(query: str) -> dict:
    g = get_graph()
    keywords = [w for w in query.split() if len(w) >= 2]
    found = []
    for kw in keywords[:4]:
        nodes = g.search_nodes(kw, limit=3)
        found.extend([n.get("name", "") for n in nodes])
    found = [n for n in set(found) if len(n) <= 30]  # 긴 문장 노드 제외
    return {"query": query, "found_nodes": found[:5], "hit": len(found) > 0}


def run():
    g = get_graph()
    stats = g.get_stats()
    print(f"\n{'='*65}")
    print(f"  문서 학습 완전성 검증 보고서 v2")
    print(f"  그래프: 노드 {stats['nodes']}개 / 관계 {stats['relations']}개")
    print(f"{'='*65}")

    all_results = {}
    total_q = 0
    vec_hits = 0
    graph_hits = 0

    for doc, queries in DOC_QUERIES.items():
        print(f"\n📄 {doc}")
        doc_vec_hits = 0
        doc_graph_hits = 0
        doc_results = []

        for q, kws in queries:
            total_q += 1
            v = check_vector(q, kws)
            gr = check_graph(q)

            v_icon  = "✅" if v["hit"]  else "❌"
            g_icon  = "✅" if gr["hit"] else "⚠️ "

            print(f"  {v_icon}벡터 {g_icon}그래프 | {q}")
            if not v["hit"] and v["miss_preview"]:
                print(f"    벡터 미스: {v['miss_preview'][:60]}")
            if gr["hit"]:
                print(f"    그래프 노드: {', '.join(gr['found_nodes'][:3])}")

            if v["hit"]:
                vec_hits += 1
                doc_vec_hits += 1
            if gr["hit"]:
                graph_hits += 1
                doc_graph_hits += 1

            doc_results.append({"query": q, "vector": v, "graph": gr})

        n = len(queries)
        vp = int(doc_vec_hits/n*100)
        gp = int(doc_graph_hits/n*100)
        bar_v = "█"*(vp//10) + "░"*(10-vp//10)
        bar_g = "█"*(gp//10) + "░"*(10-gp//10)
        print(f"  벡터  [{bar_v}] {vp}%")
        print(f"  그래프[{bar_g}] {gp}%")
        all_results[doc] = doc_results

    print(f"\n{'='*65}")
    vt = int(vec_hits/total_q*100)
    gt = int(graph_hits/total_q*100)
    print(f"  전체 벡터  커버리지: {vt}% ({vec_hits}/{total_q})")
    print(f"  전체 그래프커버리지: {gt}% ({graph_hits}/{total_q})")

    grade = "✅ 양호" if vt >= 70 else ("⚠️  보통" if vt >= 40 else "❌ 불량")
    print(f"  판정: {grade}")

    # 그래프 노드 품질 확인 (짧은 노드만)
    print(f"\n📌 그래프 핵심 노드 (연결 많은 순 top30, 30자 이내):")
    nodes = [(n, d, g._G.degree(n)) for n, d in g._G.nodes(data=True)]
    nodes_filtered = [(n, d, deg) for n, d, deg in nodes if len(n) <= 30]
    for name, data, deg in sorted(nodes_filtered, key=lambda x: x[2], reverse=True)[:30]:
        label = data.get("label", "?")
        print(f"  [{label:12s}] {name} (연결수: {deg})")

    # 저장
    out = Path("data/eval")
    out.mkdir(parents=True, exist_ok=True)
    with open(out/"coverage_report_v2.json", "w", encoding="utf-8") as f:
        json.dump({"vector_pct": vt, "graph_pct": gt, "docs": all_results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n💾 저장: data/eval/coverage_report_v2.json")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run()