"""rag/llm.py — Gemini API 래퍼
llama-cpp 완전 대체. generate() 인터페이스 동일 유지.

환경변수:
  GEMINI_API_KEY  : Google AI Studio API 키
  DAILY_LIMIT     : 일일 최대 호출 수 (기본 20, 로컬 개발 시 9999)
"""

import os
import json
import logging
from datetime import date
from pathlib import Path

import google.generativeai as genai

logger = logging.getLogger(__name__)

# 모델 폴백 순서 (무료 API 기준)
_MODELS = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
]

_USAGE_FILE = Path("data/usage.json")
_configured = False


def _ensure_configured():
    global _configured
    if not _configured:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "GEMINI_API_KEY 환경변수가 없습니다.\n"
                "로컬: .env 파일에 GEMINI_API_KEY=your_key 추가\n"
                "Render: Environment 탭에서 설정"
            )
        genai.configure(api_key=key)
        _configured = True


def _get_daily_limit() -> int:
    return int(os.environ.get("DAILY_LIMIT", "20"))


def _check_and_increment():
    """일일 사용량 확인 및 증가. 한도 초과 시 RuntimeError."""
    limit = _get_daily_limit()
    if limit >= 9999:
        return  # 로컬 무제한 모드

    today = str(date.today())
    _USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)

    usage = {"date": today, "count": 0}
    if _USAGE_FILE.exists():
        try:
            usage = json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 날짜 바뀌면 리셋
    if usage.get("date") != today:
        usage = {"date": today, "count": 0}

    if usage["count"] >= limit:
        raise RuntimeError(
            f"⛔ 일일 사용 한도 {limit}회 초과. 내일 다시 이용해 주세요.\n"
            f"현재: {usage['count']}/{limit}"
        )

    usage["count"] += 1
    _USAGE_FILE.write_text(json.dumps(usage, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Gemini 사용량: {usage['count']}/{limit} ({today})")


def generate(
    prompt: str,
    max_tokens: int = None,
    temperature: float = None,
    mode: str = "chat",
) -> str:
    """
    mode: "chat"    → 대화 응답
          "extract" → 엔티티/관계 추출 (낮은 temperature)
    """
    _ensure_configured()
    _check_and_increment()

    temp = temperature if temperature is not None else (0.1 if mode == "extract" else 0.7)
    max_tok = max_tokens or (400 if mode == "extract" else 1024)

    gen_config = genai.GenerationConfig(
        temperature=temp,
        max_output_tokens=max_tok,
    )

    last_err = None
    for model_name in _MODELS:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=gen_config,
            )
            response = model.generate_content(prompt)
            logger.debug(f"모델 사용: {model_name}")
            return response.text.strip()
        except Exception as e:
            logger.warning(f"모델 {model_name} 실패: {e}")
            last_err = e
            continue

    raise RuntimeError(f"모든 Gemini 모델 실패. 마지막 오류: {last_err}")


def get_usage() -> dict:
    """현재 사용량 조회 (UI 표시용)"""
    today = str(date.today())
    limit = _get_daily_limit()
    if not _USAGE_FILE.exists():
        return {"date": today, "count": 0, "limit": limit}
    try:
        usage = json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
        if usage.get("date") != today:
            return {"date": today, "count": 0, "limit": limit}
        usage["limit"] = limit
        return usage
    except Exception:
        return {"date": today, "count": 0, "limit": limit}


def is_loaded() -> bool:
    """llama-cpp 호환 인터페이스 유지"""
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())