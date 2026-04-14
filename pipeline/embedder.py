"""pipeline/embedder.py — FAISS 벡터 인덱스 (Incremental)"""

import pickle
import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import get_config

logger = logging.getLogger(__name__)
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        cfg = get_config()
        logger.info(f"임베딩 모델 로딩: {cfg['embedding']['model']}")
        _model = SentenceTransformer(cfg["embedding"]["model"])
    return _model


def _paths():
    cfg = get_config()
    d = Path(cfg["paths"]["embeddings"])
    return d / "index.faiss", d / "chunks.pkl"


def update_faiss(new_chunks: List[str]) -> None:
    """기존 인덱스에 새 청크 추가 (Incremental)"""
    if not new_chunks:
        return

    index_path, chunk_store = _paths()
    embeddings = _get_model().encode(new_chunks, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")

    if index_path.exists():
        index = faiss.read_index(str(index_path))
        with open(chunk_store, "rb") as f:
            old_chunks = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        old_chunks = []

    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    with open(chunk_store, "wb") as f:
        pickle.dump(old_chunks + new_chunks, f)

    logger.info(f"FAISS 업데이트: 총 {index.ntotal}개 벡터")


def search_faiss(query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
    index_path, chunk_store = _paths()
    if not index_path.exists():
        return [], []

    index = faiss.read_index(str(index_path))
    with open(chunk_store, "rb") as f:
        chunks = pickle.load(f)

    q_emb = _get_model().encode([query], convert_to_numpy=True).astype("float32")
    top_k = min(top_k, index.ntotal)
    distances, indices = index.search(q_emb, top_k)

    result_chunks = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
    result_dists = [float(d) for d in distances[0]]
    return result_chunks, result_dists


def get_index_stats() -> dict:
    index_path, chunk_store = _paths()
    if not index_path.exists():
        return {"status": "없음", "total_vectors": 0, "total_chunks": 0}
    index = faiss.read_index(str(index_path))
    with open(chunk_store, "rb") as f:
        chunks = pickle.load(f)
    return {"status": "정상", "total_vectors": index.ntotal, "total_chunks": len(chunks)}
