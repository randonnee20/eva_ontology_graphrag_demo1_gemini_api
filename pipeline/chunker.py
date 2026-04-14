"""pipeline/chunker.py — 텍스트 청킹 (문장 경계 + overlap)"""

import json
import re
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def chunk_text(input_path: str, output_path: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    text = Path(input_path).read_text(encoding="utf-8")
    chunks = split_text(text, chunk_size, overlap)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"청킹 완료: {len(chunks)}개 → {output_path}")
    return chunks


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks, current, prev_tail = [], "", ""
    for sent in sentences:
        if len(current) + len(sent) > chunk_size and current:
            chunks.append((prev_tail + current).strip())
            prev_tail = current[-overlap:] if overlap > 0 else ""
            current = sent
        else:
            current += sent

    if current.strip():
        chunks.append((prev_tail + current).strip())

    return [c for c in chunks if len(c.strip()) > 20]


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text)
    parts = re.split(r"(?<=[.。?！!])\s+", text)
    sentences = []
    for part in parts:
        for sub in part.split("\n\n"):
            if sub.strip():
                sentences.append(sub + " ")
    return sentences
