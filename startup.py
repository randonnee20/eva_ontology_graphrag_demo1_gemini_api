"""
startup.py — Render 시작 시 자동 실행
1. .env 로드 (로컬 개발용)
2. data/embeddings가 비어있으면 파이프라인 자동 실행
3. Streamlit 앱 실행은 render.yaml startCommand에서 처리
"""

import os
import sys
import logging
from pathlib import Path

# ── .env 로드 (로컬 개발) ──────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env 로드 완료")
except ImportError:
    pass  # Render 환경에서는 환경변수가 직접 주입됨

# ── API 키 확인 ────────────────────────────────────────────
api_key = os.environ.get("GEMINI_API_KEY", "").strip()
if not api_key:
    print("❌ GEMINI_API_KEY가 설정되지 않았습니다.")
    print("   로컬: .env 파일에 GEMINI_API_KEY=your_key 추가")
    print("   Render: 대시보드 > Environment > Add Environment Variable")
    sys.exit(1)

# ── 파이프라인 자동 실행 (임베딩 없을 때) ──────────────────
_ROOT = Path(__file__).resolve().parent
emb_dir = _ROOT / "data" / "embeddings"
raw_dir = _ROOT / "data" / "raw"

# pkl 파일 확인 (처리 이력)
pkl_files = list(emb_dir.glob("*.pkl")) if emb_dir.exists() else []
raw_files = list(raw_dir.iterdir()) if raw_dir.exists() else []

if raw_files and not pkl_files:
    print(f"📂 data/raw에 파일 {len(raw_files)}개 발견. 파이프라인 자동 실행...")
    sys.path.insert(0, str(_ROOT))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    from pipeline.pipeline_runner import process_all_files
    result = process_all_files(force=False)
    print(f"✅ 파이프라인 완료: {result['success']}/{result['total']} 성공")
else:
    if pkl_files:
        print(f"✅ 임베딩 존재 ({len(pkl_files)}개). 파이프라인 스킵.")
    elif not raw_files:
        print("⚠️  data/raw 폴더가 비어있습니다. 문서를 추가하세요.")

print("🚀 Streamlit 앱 시작 준비 완료")
