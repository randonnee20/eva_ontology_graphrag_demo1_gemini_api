"""
main.py — CLI 진입점

python main.py              # data/raw 전체 처리
python main.py file doc.pdf # 단일 파일
python main.py watch        # 폴더 자동 감시
python main.py stats        # 상태 확인
python main.py reset        # 전체 초기화
python main.py export       # 그래프 JSON 내보내기

LibreOffice 설치 : https://www.libreoffice.org/download/libreoffice-fresh/


1. Conda 환경 생성
conda create -n eva_graphrag python=3.12 -y
conda activate eva_graphrag

2. 캐시 삭제
Remove-Item E:\Develop\eva_ontology_graphrag_demo1\data\embeddings\processed_files.pkl
Remove-Item E:\Develop\eva_ontology_graphrag_demo1\data\graph\* -ErrorAction SilentlyContinue

3. 실행
streamlit run ui/app.py

  Local URL: http://localhost:8501
  Network URL: http://192.168.0.12:8501

검증 :   python eval_coverage.py



"""

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import sys
import argparse
import logging
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_run(args):
    from pipeline.pipeline_runner import process_all_files
    result = process_all_files(force=args.force)
    print(f"\n✅ 완료: {result['success']}/{result['total']} 성공, {result['failed']} 실패")
    for fname, r in result.get("details", {}).items():
        icon = "✅" if r.get("success") else ("⏭️" if r.get("skipped") else "❌")
        print(f"   {icon} {fname}")


def cmd_file(args):
    from pipeline.pipeline_runner import process_single_file

    def show(step, pct):
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}% {step:<25}", end="", flush=True)

    result = process_single_file(args.filename, force=args.force, progress_callback=show)
    print()
    if result.get("success"):
        if result.get("skipped"):
            print("⏭️  이미 처리된 파일. 강제 재처리: --force")
        else:
            print(f"✅ 완료: 청크 {result['chunks']}  엔티티 {result['entities']}  관계 {result['relations']}")
    else:
        print(f"❌ 실패: {result.get('reason')}")


def cmd_watch(args):
    from pipeline.pipeline_runner import process_single_file, SUPPORTED_EXTENSIONS
    from utils.config import get_config

    cfg = get_config()
    raw_dir = Path(cfg["paths"]["raw"])
    interval = cfg.get("ingestion", {}).get("watch_interval", 2)

    logger.info(f"폴더 감시 시작: {raw_dir}  (Ctrl+C로 종료)")
    known = set(f.name for f in raw_dir.iterdir() if f.is_file())

    try:
        while True:
            current = set(f.name for f in raw_dir.iterdir() if f.is_file())
            for fname in current - known:
                if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                    logger.info(f"새 파일: {fname}")
                    r = process_single_file(fname)
                    logger.info(f"처리 결과: {r}")
            known = current
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("감시 종료")


def cmd_stats(args):
    from pipeline.embedder import get_index_stats
    from pipeline.ontology_updater import get_ontology
    from graph.client import get_graph

    faiss = get_index_stats()
    g = get_graph()
    g_stats = g.get_stats()
    onto = get_ontology()
    total = sum(len(v) for v in onto.values() if isinstance(v, list))

    print("\n📊 시스템 상태")
    print("=" * 40)
    print(f"\n🔢 FAISS   벡터: {faiss['total_vectors']}  청크: {faiss['total_chunks']}")
    print(f"🔗 그래프  노드: {g_stats['nodes']}  관계: {g_stats['relations']}")
    print(f"📖 온톨로지 클래스: {len(onto)}  엔티티: {total}")
    if onto:
        print()
        for cls, vals in onto.items():
            if isinstance(vals, list):
                sample = ", ".join(vals[:5])
                more = f" (+{len(vals)-5})" if len(vals) > 5 else ""
                print(f"   {cls}: {sample}{more}")
    print()


def cmd_reset(args):
    from utils.config import get_config
    from graph.client import get_graph

    confirm = input("⚠️  그래프 + FAISS 전체 초기화. 계속? (yes/no): ")
    if confirm.lower() != "yes":
        print("취소")
        return

    cfg = get_config()
    emb_dir = Path(cfg["paths"]["embeddings"])
    for f in emb_dir.iterdir():
        if f.is_file():
            f.unlink()
    print("✅ FAISS 초기화 완료")

    get_graph().clear()
    print("✅ 그래프 초기화 완료")


def cmd_export(args):
    """그래프를 JSON으로 내보내기 (Neo4j 마이그레이션 또는 분석용)"""
    from graph.client import get_graph
    from utils.config import get_config

    g = get_graph()
    cfg = get_config()

    # NetworkX만 지원
    from graph.networkx_client import NetworkXGraph
    if not isinstance(g, NetworkXGraph):
        print("export는 networkx 백엔드에서만 지원됩니다.")
        return

    out = Path(cfg["paths"]["graph_db"]) / "graph_export.json"
    g.export_to_json(str(out))
    print(f"✅ 내보내기 완료: {out}")


def main():
    p = argparse.ArgumentParser(description="GraphRAG CLI")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("run", help="전체 파일 처리")
    s.add_argument("--force", action="store_true")

    s = sub.add_parser("file", help="단일 파일 처리")
    s.add_argument("filename")
    s.add_argument("--force", action="store_true")

    sub.add_parser("watch", help="폴더 자동 감시")
    sub.add_parser("stats", help="시스템 상태")
    sub.add_parser("reset", help="전체 초기화")
    sub.add_parser("export", help="그래프 JSON 내보내기")

    args = p.parse_args()
    dispatch = {
        "run": cmd_run, "file": cmd_file, "watch": cmd_watch,
        "stats": cmd_stats, "reset": cmd_reset, "export": cmd_export,
    }

    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        # 기본: 전체 처리
        class _A:
            force = False
        cmd_run(_A())


if __name__ == "__main__":
    main()
