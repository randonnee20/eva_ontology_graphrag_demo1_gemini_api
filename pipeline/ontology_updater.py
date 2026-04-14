"""
pipeline/ontology_updater.py
범용 온톨로지 자동 확장
- 도메인 고정 없음
- LLM이 추출한 모든 엔티티 타입을 그대로 수용
"""

import yaml
import logging
from pathlib import Path
from typing import List, Dict

from utils.config import get_config

logger = logging.getLogger(__name__)


def update_ontology(new_entities: List[Dict]) -> int:
    """새 엔티티를 온톨로지에 추가. 반환: 추가된 항목 수"""
    if not new_entities:
        return 0

    cfg = get_config()
    onto_path = Path(cfg["paths"]["ontology"])

    # 현재 온톨로지 로드 (빈 파일이면 빈 dict)
    content = onto_path.read_text(encoding="utf-8")
    ontology = yaml.safe_load(content) or {}

    added = 0
    for ent in new_entities:
        t = str(ent.get("type", "")).strip()
        n = str(ent.get("name", "")).strip()
        if not t or not n or len(n) < 2:
            continue

        if t not in ontology:
            ontology[t] = []
            logger.info(f"새 온톨로지 클래스: {t}")

        if n not in ontology[t]:
            ontology[t].append(n)
            added += 1

    if added > 0:
        onto_path.write_text(
            yaml.dump(ontology, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
        logger.info(f"온톨로지 업데이트: {added}개 추가")

    return added


def get_ontology() -> dict:
    cfg = get_config()
    content = Path(cfg["paths"]["ontology"]).read_text(encoding="utf-8")
    return yaml.safe_load(content) or {}


def get_all_entities() -> List[str]:
    """온톨로지의 모든 엔티티 이름 목록"""
    onto = get_ontology()
    entities = []
    for values in onto.values():
        if isinstance(values, list):
            entities.extend(values)
    return entities
