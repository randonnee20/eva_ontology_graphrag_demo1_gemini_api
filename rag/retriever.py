"""
rag/retriever.py
Hybrid Retriever: FAISS 벡터 검색 + 그래프 검색
"""

import logging
from typing import List, Tuple

from utils.config import get_config
from pipeline.embedder import search_faiss
from graph.client import get_graph

logger = logging.getLogger(__name__)


def retrieve(query: str) -> str:
    cfg = get_config()
    top_k = cfg["embedding"].get("top_k", 5)

    vector_ctx = _vector_search(query, top_k)
    graph_ctx = _graph_search(query)

    parts = []
    if vector_ctx:
        parts.append(f"[문서 검색 결과]\n{vector_ctx}")
    if graph_ctx:
        parts.append(f"[지식 그래프 검색 결과]\n{graph_ctx}")

    return "\n\n".join(parts) if parts else "관련 정보를 찾을 수 없습니다."


def retrieve_with_detail(query: str) -> dict:
    cfg = get_config()
    top_k = cfg["embedding"].get("top_k", 5)
    chunks, dists = search_faiss(query, top_k=top_k)
    graph_ctx = _graph_search(query)

    return {
        "vector_chunks": chunks,
        "vector_distances": dists,
        "graph_facts": graph_ctx,
        "combined_context": retrieve(query),
    }


def _vector_search(query: str, top_k: int = 5) -> str:
    chunks, dists = search_faiss(query, top_k=top_k)
    if not chunks:
        return ""

    # [수정] 유사 청크 중복 제거 - 앞 60자가 같으면 중복으로 판단
    deduped = []
    seen_prefixes = set()
    for chunk in chunks:
        prefix = chunk.strip()[:60]
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            deduped.append(chunk)

    return "\n---\n".join(deduped)


def _graph_search(query: str) -> str:
    g = get_graph()
    if not g.test_connection():
        return ""
    try:
        keywords = _extract_keywords(query)
        all_nodes = []
        seen_names = set()

        for kw in keywords:
            # 1차: 정확한 키워드 검색
            nodes = g.search_nodes(kw, limit=3)
            # 2차: 키워드 부분 매칭 (search_nodes가 빈 결과일 때)
            if not nodes:
                try:
                    all_graph_nodes = g.get_all_nodes() if hasattr(g, 'get_all_nodes') else []
                    nodes = [n for n in all_graph_nodes if kw in n.get("name", "")][:3]
                except Exception:
                    pass
            for node in nodes:
                name = node.get("name", "")
                if name and name not in seen_names:
                    seen_names.add(name)
                    all_nodes.append(node)

        # 마지막: 2글자 단위 매칭
        if not all_nodes:
            try:
                all_graph_nodes = g.get_all_nodes() if hasattr(g, 'get_all_nodes') else []
                for node in all_graph_nodes:
                    name = node.get("name", "")
                    if name and any(kw[:2] in name for kw in keywords if len(kw) >= 2):
                        if name not in seen_names:
                            seen_names.add(name)
                            all_nodes.append(node)
                    if len(all_nodes) >= 5:
                        break
            except Exception:
                pass

        if not all_nodes:
            return ""

        facts = []
        seen_facts = set()
        for node in all_nodes[:5]:
            name = node.get("name", "")
            if not name:
                continue
            related = g.search_related(name, depth=2)
            for r in related[:5]:
                src = r.get("source", "")
                tgt = r.get("target", "")
                rels = r.get("relations", [])
                labels = r.get("target_labels", [])
                if src and tgt and rels:
                    fact_key = f"{src}-{rels[0]}-{tgt}"
                    if fact_key not in seen_facts:
                        seen_facts.add(fact_key)
                        lbl = f"({labels[0]})" if labels else ""
                        facts.append(f"{src} --[{' → '.join(rels)}]--> {tgt} {lbl}")

        return "\n".join(facts) if facts else ""
    except Exception as e:
        logger.error(f"그래프 검색 오류: {e}")
        return ""


def _extract_keywords(query: str) -> List[str]:
    """질문에서 의미 있는 키워드 추출"""
    stop_words = {"은", "는", "이", "가", "을", "를", "의", "에", "에서",
                  "으로", "로", "와", "과", "도", "만", "어떻게", "어떤",
                  "무엇", "뭐", "어디", "언제", "왜", "하지", "하는", "대한",
                  "위한", "관한", "있는", "없는", "하고", "이고", "이며",
                  "어느", "어느것", "정리", "설명", "알려", "해줘", "해주세요"}

    import re
    tokens = re.split(r"[\s\?\!\.\,\;\:\(\)\"\' ]+", query)
    keywords = [t for t in tokens if len(t) >= 2 and t not in stop_words]

    # [수정] 복합어 분리 추가
    # "제조업안전보건" → ["제조업", "안전보건"] 처리
    expanded = []
    split_pairs = [
        ("제조업", ["제조업", "제조"]),
        ("건설업", ["건설업", "건설"]),
        ("서비스업", ["서비스업", "서비스"]),
        ("도소매업", ["도소매업", "도소매"]),
        ("안전보건", ["안전보건", "안전"]),
        ("산업안전", ["산업안전", "안전"]),
        ("교육시간", ["교육시간", "교육"]),
        ("과태료", ["과태료"]),
        ("시행령", ["시행령"]),
        ("시행규칙", ["시행규칙"]),
    ]
    for kw in keywords:
        expanded.append(kw)
        for trigger, additions in split_pairs:
            if trigger in kw and kw != trigger:
                expanded.extend(additions)

    # 원본 query도 포함
    if query.strip() not in expanded:
        expanded.insert(0, query.strip())

    # 중복 제거, 순서 유지
    seen = set()
    result = []
    for k in expanded:
        if k not in seen:
            seen.add(k)
            result.append(k)

    return result[:8]