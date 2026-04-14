"""pipeline/graph_builder.py — 추출 결과 → 그래프 저장"""

import logging
from typing import Dict, List
from graph.client import get_graph

logger = logging.getLogger(__name__)


def build_graph(data: Dict[str, List]) -> None:
    g = get_graph()
    entities = data.get("entities", [])
    relations = data.get("relations", [])

    for ent in entities:
        t = ent.get("type", "Unknown")
        n = ent.get("name", "").strip()
        if n:
            try:
                g.merge_entity(t, n)
            except Exception as e:
                logger.error(f"엔티티 저장 실패 ({t}:{n}): {e}")

    saved = 0
    for rel in relations:
        src = rel.get("source", "").strip()
        relation = rel.get("relation", "RELATED_TO")
        tgt = rel.get("target", "").strip()
        if src and tgt:
            try:
                g.merge_relation(src, relation, tgt)
                saved += 1
            except Exception as e:
                logger.error(f"관계 저장 실패: {e}")

    logger.info(f"그래프 저장: 엔티티 {len(entities)}개, 관계 {saved}개")
