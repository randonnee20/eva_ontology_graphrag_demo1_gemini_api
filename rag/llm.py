"""rag/llm.py — 로컬 llama-cpp (Gemma 4 E2B GGUF) 우선, Gemini API fallback

동작 방식:
  로컬 PC : config.yaml llm.model_path GGUF 파일 존재 → llama-cpp 자동 사용
  Render  : GGUF 없음 → GEMINI_API_KEY 환경변수로 Gemini API fallback

환경변수:
  LLAMA_MODEL_PATH : GGUF 경로 덮어쓰기 (선택)
  GEMINI_API_KEY   : Render 배포 시 필수
  DAILY_LIMIT      : Gemini 일일 한도 (기본 20)
"""

import os
import json
import logging
from pathlib import Path
from datetime import date

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────
# 전역 상태
# ──────────────────────────────────────────────────
_llama = None        # llama-cpp Llama 인스턴스
_gemini_client = None  # google.genai Client
_use_gemini = False  # True = Gemini fallback 모드
_configured = False

_USAGE_FILE = Path("data/usage.json")

_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
]


# ──────────────────────────────────────────────────
# 초기화
# ──────────────────────────────────────────────────
def _get_config_model_path() -> str:
    try:
        from utils.config import get_config
        return get_config().get("llm", {}).get("model_path", "")
    except Exception:
        return ""


def _init():
    global _llama, _use_gemini, _configured
    if _configured:
        return

    model_path = os.environ.get("LLAMA_MODEL_PATH", "").strip() or _get_config_model_path()

    if model_path and Path(model_path).exists():
        # ── 로컬 llama-cpp 모드 ──────────────────
        try:
            from llama_cpp import Llama
            from utils.config import get_config
            cfg = get_config().get("llm", {})

            logger.info(f"[LLM] llama-cpp 로드 중: {model_path}")
            _llama = Llama(
                model_path=model_path,
                n_ctx=cfg.get("n_ctx", 4096),
                n_gpu_layers=cfg.get("n_gpu_layers", -1),
                verbose=False,
            )
            _use_gemini = False
            logger.info("[LLM] llama-cpp 로드 완료 ✅ (Gemma 4 E2B GGUF)")

        except ImportError:
            logger.warning("[LLM] llama-cpp-python 미설치 → Gemini fallback")
            _use_gemini = True
        except Exception as e:
            logger.warning(f"[LLM] llama-cpp 로드 실패: {e} → Gemini fallback")
            _use_gemini = True
    else:
        if model_path:
            logger.info(f"[LLM] GGUF 파일 없음: {model_path}")
        logger.info("[LLM] Gemini API 모드")
        _use_gemini = True

    if _use_gemini:
        _init_gemini()

    _configured = True


def _init_gemini():
    global _gemini_client
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY 환경변수가 없습니다.\n"
            "  Render : Environment 탭에서 설정\n"
            "  로컬   : .env 파일에 GEMINI_API_KEY=your_key 추가"
        )
    from google import genai
    _gemini_client = genai.Client(api_key=key)
    logger.info("[LLM] Gemini API 초기화 완료")


# ──────────────────────────────────────────────────
# Gemini 일일 사용량 추적
# ──────────────────────────────────────────────────
def _get_daily_limit() -> int:
    return int(os.environ.get("DAILY_LIMIT", "20"))


def _check_and_increment():
    limit = _get_daily_limit()
    if limit >= 9999:
        return

    today = str(date.today())
    _USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)

    usage = {"date": today, "count": 0}
    if _USAGE_FILE.exists():
        try:
            usage = json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    if usage.get("date") != today:
        usage = {"date": today, "count": 0}

    if usage["count"] >= limit:
        raise RuntimeError(
            f"⛔ Gemini 일일 한도 {limit}회 초과. 내일 다시 이용해 주세요.\n"
            f"현재: {usage['count']}/{limit}"
        )

    usage["count"] += 1
    _USAGE_FILE.write_text(json.dumps(usage, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[Gemini] 사용량: {usage['count']}/{limit} ({today})")


# ──────────────────────────────────────────────────
# 공개 인터페이스 (기존과 완전 동일)
# ──────────────────────────────────────────────────
def generate(
    prompt: str,
    max_tokens: int = None,
    temperature: float = None,
    mode: str = "chat",
) -> str:
    """
    mode: "chat"    → 대화 응답 (temperature 0.7)
          "extract" → 엔티티/관계 추출 (temperature 0.1)
    """
    _init()

    temp = temperature if temperature is not None else (0.1 if mode == "extract" else 0.7)
    max_tok = max_tokens or (400 if mode == "extract" else 1024)

    if not _use_gemini and _llama is not None:
        return _generate_llama(prompt, max_tok, temp)
    return _generate_gemini(prompt, max_tok, temp)


def _generate_llama(prompt: str, max_tokens: int, temperature: float) -> str:
    """llama-cpp 로컬 추론"""
    output = _llama(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<end_of_turn>", "<eos>", "\n\n\n"],
        echo=False,
    )
    return output["choices"][0]["text"].strip()


def _generate_gemini(prompt: str, max_tokens: int, temperature: float) -> str:
    """Gemini API fallback (google.genai 신패키지)"""
    from google.genai import types
    _check_and_increment()

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    last_err = None
    for model_name in _GEMINI_MODELS:
        try:
            response = _gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            logger.debug(f"[Gemini] 모델 사용: {model_name}")
            return response.text.strip()
        except Exception as e:
            logger.warning(f"[Gemini] 모델 {model_name} 실패: {e}")
            last_err = e

    raise RuntimeError(f"모든 Gemini 모델 실패: {last_err}")


# ──────────────────────────────────────────────────
# 상태 조회 (UI 표시용)
# ──────────────────────────────────────────────────
def get_usage() -> dict:
    today = str(date.today())
    limit = _get_daily_limit()

    if not _use_gemini and _llama is not None:
        # 로컬 모드: API 호출 없음
        return {"date": today, "count": 0, "limit": 99999, "mode": "llama-cpp (로컬)"}

    # Gemini 모드
    if not _USAGE_FILE.exists():
        return {"date": today, "count": 0, "limit": limit, "mode": "gemini"}
    try:
        usage = json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
        if usage.get("date") != today:
            return {"date": today, "count": 0, "limit": limit, "mode": "gemini"}
        usage["limit"] = limit
        usage["mode"] = "gemini"
        return usage
    except Exception:
        return {"date": today, "count": 0, "limit": limit, "mode": "gemini"}


def is_loaded() -> bool:
    """사이드바 상태 표시용"""
    if _configured:
        return (_llama is not None) or _use_gemini

    # 초기화 전: 경로 존재 여부로 예측
    mp = _get_config_model_path()
    if mp and Path(mp).exists():
        return True
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())