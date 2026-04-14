"""
pipeline/pipeline_runner.py
단일 파일 / 전체 폴더 ingestion 파이프라인

raw 문서 → 텍스트 → 청킹 → FAISS → LLM 추출 → 온톨로지 → 그래프
"""

import pickle
import hashlib
import logging
from pathlib import Path
from typing import Optional, Callable

from utils.config import get_config
from pipeline.converter import convert_to_text
from pipeline.chunker import chunk_text
from pipeline.embedder import update_faiss
from pipeline.llm_extractor import extract_entities_relations
from pipeline.graph_builder import build_graph
from pipeline.ontology_updater import update_ontology

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".hwp", ".hwpx"}

# 파일당 샘플링할 최대 청크 수 (균등 샘플링)
_MAX_SAMPLE_CHUNKS = 10


def _processed_log_path() -> Path:
    cfg = get_config()
    return Path(cfg["paths"]["embeddings"]) / "processed_files.pkl"


def _load_processed() -> set:
    p = _processed_log_path()
    return pickle.loads(p.read_bytes()) if p.exists() else set()


def _save_processed(s: set):
    p = _processed_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(s))


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sample_chunks(chunks: list, n: int = _MAX_SAMPLE_CHUNKS) -> list:
    """청크 전체에서 균등하게 n개 샘플링 (앞부분만 쓰는 문제 해결)"""
    if len(chunks) <= n:
        return chunks
    step = len(chunks) // n
    return [chunks[i * step] for i in range(n)]


def process_single_file(
    filename: str,
    force: bool = False,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    단일 파일 전체 파이프라인 실행
    progress_callback(step: str, pct: int)
    """
    cfg = get_config()
    raw_path = Path(cfg["paths"]["raw"]) / filename

    if not raw_path.exists():
        return {"success": False, "reason": f"파일 없음: {raw_path}"}
    if raw_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return {"success": False, "reason": f"지원하지 않는 형식: {raw_path.suffix}"}

    # 중복 체크
    processed = _load_processed()
    file_hash = _file_hash(str(raw_path))
    hash_key = f"{filename}:{file_hash}"
    if hash_key in processed and not force:
        logger.info(f"이미 처리됨 (스킵): {filename}")
        return {"success": True, "skipped": True, "chunks": 0, "entities": 0, "relations": 0}

    stem = raw_path.stem
    text_path = Path(cfg["paths"]["text"]) / f"{stem}.txt"
    chunk_path = Path(cfg["paths"]["chunks"]) / f"{stem}.json"

    def _p(step, pct):
        logger.info(f"[{pct:3d}%] {step}")
        if progress_callback:
            progress_callback(step, pct)

    try:
        _p("텍스트 변환", 10)
        convert_to_text(str(raw_path), str(text_path))

        _p("청킹", 25)
        chunk_size = cfg["embedding"].get("chunk_size", 500)
        overlap = cfg["embedding"].get("chunk_overlap", 50)
        chunks = chunk_text(str(text_path), str(chunk_path), chunk_size, overlap)

        if not chunks:
            return {"success": False, "reason": "텍스트 추출 실패 또는 빈 문서"}

        _p("벡터 임베딩", 45)
        update_faiss(chunks)

        _p("엔티티/관계 추출 (LLM)", 65)
        # [수정] 앞 10개만 쓰던 방식 → 전체에서 균등 샘플링 후 청크별 개별 추출
        sampled = _sample_chunks(chunks, _MAX_SAMPLE_CHUNKS)
        all_entities, all_relations = [], []
        seen_entities = set()
        seen_relations = set()

        for i, chunk in enumerate(sampled):
            logger.info(f"청크 {i+1}/{len(sampled)} 추출 중...")
            result = extract_entities_relations(chunk)

            # 엔티티 중복 제거
            for e in result["entities"]:
                key = (e.get("type", ""), e.get("name", ""))
                if key not in seen_entities:
                    seen_entities.add(key)
                    all_entities.append(e)

            # 관계 중복 제거
            for r in result["relations"]:
                key = (r.get("source", ""), r.get("relation", ""), r.get("target", ""))
                if key not in seen_relations:
                    seen_relations.add(key)
                    all_relations.append(r)

        extracted = {"entities": all_entities, "relations": all_relations}
        logger.info(f"총 추출: 엔티티 {len(all_entities)}개, 관계 {len(all_relations)}개")

        _p("온톨로지 업데이트", 80)
        onto_added = update_ontology(extracted["entities"])

        _p("그래프 구축", 90)
        build_graph(extracted)

        processed.add(hash_key)
        _save_processed(processed)

        _p("완료", 100)
        return {
            "success": True,
            "chunks": len(chunks),
            "entities": len(extracted["entities"]),
            "relations": len(extracted["relations"]),
            "ontology_added": onto_added,
        }

    except Exception as e:
        logger.exception(f"처리 실패: {filename}")
        return {"success": False, "reason": str(e)}


def process_all_files(force: bool = False) -> dict:
    cfg = get_config()
    raw_dir = Path(cfg["paths"]["raw"])
    files = [f.name for f in raw_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()]

    results = {"total": len(files), "success": 0, "failed": 0, "details": {}}
    for fname in files:
        r = process_single_file(fname, force=force)
        results["success" if r.get("success") else "failed"] += 1
        results["details"][fname] = r

    logger.info(f"전체 처리 완료: {results['success']}/{results['total']}")
    return results