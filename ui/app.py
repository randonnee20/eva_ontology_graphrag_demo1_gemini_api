"""
ui/app.py — GraphRAG Streamlit UI
탭 1: 문서 업로드 & Ingestion
탭 2: Hybrid RAG 채팅
탭 3: 시스템 상태 (그래프/벡터/온톨로지)
"""

import sys
import logging
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(
    page_title="GraphRAG 시스템",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.config import get_config, get_root
from pipeline.pipeline_runner import process_single_file, process_all_files, SUPPORTED_EXTENSIONS
from rag.graph_rag import answer, answer_with_sources
from pipeline.embedder import get_index_stats
from pipeline.ontology_updater import get_ontology
from graph.client import get_graph

logging.basicConfig(level=logging.INFO)


# ──────────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────────
def sidebar():
    # 로고 이미지 (ui/static/logo.png)
    logo_path = Path(__file__).parent / "static" / "logo.png"
    if logo_path.exists():
        # 여백 제거 CSS
        st.sidebar.markdown(
            "<style>[data-testid='stSidebarContent'] img {margin-bottom:-30px;}</style>",
            unsafe_allow_html=True,
        )
        st.sidebar.image(str(logo_path), width=240)
    st.sidebar.title("🧠 GraphRAG 시스템")
    st.sidebar.markdown("---")

    g = get_graph()
    backend = get_config().get("graph", {}).get("backend", "networkx")

    if g.test_connection():
        st.sidebar.success(f"✅ 그래프 DB ({backend})")
    else:
        st.sidebar.error(f"❌ 그래프 DB ({backend}) 연결 실패")

    faiss_stats = get_index_stats()
    if faiss_stats["status"] == "정상":
        st.sidebar.success(f"✅ FAISS: {faiss_stats['total_vectors']}개 벡터")
    else:
        st.sidebar.warning("⚠️ FAISS 인덱스 없음")

    from rag.llm import is_loaded
    st.sidebar.info("✅ LLM 로드됨" if is_loaded() else "💤 LLM 대기 중 (첫 질문 시 로드)")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"root: {get_root()}")
    st.sidebar.caption(f"백엔드: {backend}")


# ──────────────────────────────────────────────────
# 탭 1: 업로드
# ──────────────────────────────────────────────────
def tab_upload():
    st.header("📂 문서 업로드 & Ingestion")
    st.caption("PDF, DOCX, HWP, HWPX 파일을 업로드하면 자동으로 지식 그래프와 벡터 DB에 반영됩니다.")

    col_up, col_list = st.columns([3, 2])

    with col_up:
        uploaded_files = st.file_uploader(
            "문서 업로드",
            type=["pdf", "docx", "hwp", "hwpx"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            cfg = get_config()
            raw_dir = Path(cfg["paths"]["raw"])

            for uf in uploaded_files:
                save_path = raw_dir / uf.name
                save_path.write_bytes(uf.getbuffer())
                st.info(f"💾 저장: {uf.name}")

                bar = st.progress(0)
                status = st.empty()

                def cb(step, pct, _bar=bar, _status=status):
                    _bar.progress(pct / 100)
                    _status.text(f"⚙️ {step}...")

                with st.spinner(f"처리 중: {uf.name}"):
                    result = process_single_file(uf.name, progress_callback=cb)

                bar.progress(1.0)

                if result.get("success"):
                    if result.get("skipped"):
                        status.warning("⏭️ 이미 처리된 파일 (변경 없음)")
                    else:
                        status.success(
                            f"✅ 완료 | 청크 {result['chunks']}개 | "
                            f"엔티티 {result['entities']}개 | 관계 {result['relations']}개"
                        )
                else:
                    status.error(f"❌ 실패: {result.get('reason')}")

    with col_list:
        st.subheader("처리된 파일")
        cfg = get_config()
        raw_dir = Path(cfg["paths"]["raw"])
        files = sorted(
            f for f in raw_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()
        ) if raw_dir.exists() else []

        if files:
            for f in files:
                st.text(f"📄 {f.name} ({f.stat().st_size // 1024}KB)")
        else:
            st.caption("아직 파일 없음")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 전체 재처리"):
                with st.spinner("처리 중..."):
                    r = process_all_files(force=True)
                st.success(f"{r['success']}/{r['total']} 완료")
        with col_b:
            if st.button("🗑️ 인덱스 초기화", type="secondary"):
                cfg = get_config()
                # 벡터 인덱스 삭제
                emb_dir = Path(cfg["paths"]["embeddings"])
                if emb_dir.exists():
                    for p in emb_dir.iterdir():
                        if p.is_file():
                            p.unlink()
                # 그래프 초기화
                get_graph().clear()
                # raw 파일 삭제 (파일 목록도 초기화)
                raw_dir = Path(cfg["paths"]["raw"])
                if raw_dir.exists():
                    for p in raw_dir.iterdir():
                        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                            p.unlink()
                st.success("초기화 완료 (파일 목록 포함)")
                st.rerun()


# ──────────────────────────────────────────────────
# 탭 2: 채팅
# ──────────────────────────────────────────────────
def tab_chat():
    st.header("💬 GraphRAG 채팅")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = False

    col_ctrl, _ = st.columns([1, 4])
    with col_ctrl:
        st.session_state.show_sources = st.toggle("출처 표시", value=st.session_state.show_sources)
        if st.button("🗑️ 초기화"):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if st.session_state.show_sources and msg.get("sources"):
                with st.expander("📎 출처"):
                    src = msg["sources"]
                    if src.get("vector"):
                        st.markdown("**벡터 검색:**")
                        for i, c in enumerate(src["vector"][:3], 1):
                            st.caption(f"{i}. {c[:250]}...")
                    if src.get("graph"):
                        st.markdown("**그래프 검색:**")
                        st.code(src["graph"])

    if query := st.chat_input("질문을 입력하세요..."):
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                try:
                    if st.session_state.show_sources:
                        result = answer_with_sources(query)
                        response = result["answer"]
                        sources = {"vector": result["vector_sources"], "graph": result["graph_sources"]}
                    else:
                        response = answer(query, st.session_state.messages[:-1])
                        sources = {}

                    st.markdown(response)

                    if sources and st.session_state.show_sources:
                        with st.expander("📎 출처"):
                            if sources.get("vector"):
                                st.markdown("**벡터 검색:**")
                                for i, c in enumerate(sources["vector"][:3], 1):
                                    st.caption(f"{i}. {c[:250]}...")
                            if sources.get("graph"):
                                st.markdown("**그래프 검색:**")
                                st.code(sources["graph"])

                except Exception as e:
                    response = f"❌ 오류: {e}"
                    st.error(response)
                    sources = {}

        st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})


# ──────────────────────────────────────────────────
# 그래프 시각화 (pyvis)
# ──────────────────────────────────────────────────
def _render_graph_pyvis(G, max_nodes: int = 150) -> str:
    """NetworkX 그래프 → pyvis HTML 문자열"""
    try:
        from pyvis.network import Network
    except ImportError:
        return ""

    # 노드 수 제한 (성능)
    if G.number_of_nodes() > max_nodes:
        # 엣지 많은 노드 우선
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes)

    # 레이블별 색상
    color_map = {
        "Law":          "#e74c3c",   # 빨강
        "Organization": "#3498db",   # 파랑
        "Person":       "#2ecc71",   # 초록
        "Regulation":   "#f39c12",   # 주황
        "Concept":      "#9b59b6",   # 보라
        "Product":      "#1abc9c",   # 청록
        "Unknown":      "#95a5a6",   # 회색
    }

    net = Network(
        height="600px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222",
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

    for node, data in G.nodes(data=True):
        label = data.get("label", "Unknown")
        color = color_map.get(label, "#95a5a6")
        degree = G.degree(node)
        net.add_node(
            node,
            label=node[:20],
            title=f"<b>{node}</b><br>타입: {label}<br>연결: {degree}",
            color={"background": color, "border": "#333333", "highlight": {"background": color, "border": "#000000"}},
            size=18 + degree * 3,
            font={"size": 13, "color": "#111111", "bold": True if degree > 2 else False},
            borderWidth=2,
        )

    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "")
        net.add_edge(u, v, label=rel, title=rel, arrows="to",
                     color={"color": "#666666", "highlight": "#ff0000"},
                     font={"size": 9, "color": "#444444"})

    net.set_options("""
    {
      "nodes": {"borderWidth": 2, "shadow": {"enabled": true, "color": "rgba(0,0,0,0.2)"}},
      "edges": {"smooth": {"type": "continuous"}, "shadow": false},
      "interaction": {"hover": true, "navigationButtons": true, "keyboard": true, "tooltipDelay": 100},
      "physics": {"enabled": true, "stabilization": {"iterations": 150}}
    }
    """)

    return net.generate_html()


def tab_status():
    st.header("📊 시스템 상태")

    faiss = get_index_stats()
    g = get_graph()
    graph_stats = g.get_stats()
    onto = get_ontology()
    total_ents = sum(len(v) for v in onto.values() if isinstance(v, list))
    cfg = get_config()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("벡터 수", faiss["total_vectors"])
    col2.metric("청크 수", faiss["total_chunks"])
    col3.metric("그래프 노드", graph_stats["nodes"])
    col4.metric("그래프 관계", graph_stats["relations"])

    col5, col6 = st.columns(2)
    col5.metric("온톨로지 클래스", len(onto))
    col6.metric("온톨로지 엔티티", total_ents)

    st.markdown("---")

    # ── 그래프 시각화 ──────────────────────────────
    st.subheader("🗺️ 지식 그래프 시각화")

    backend = cfg.get("graph", {}).get("backend", "networkx")
    if backend == "networkx" and graph_stats["nodes"] > 0:
        from graph.networkx_client import NetworkXGraph
        if isinstance(g, NetworkXGraph):

            # 컨트롤
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1:
                max_nodes = st.slider("최대 노드 수", 20, 200, 100, step=10)
            with col_ctrl2:
                filter_label = st.selectbox(
                    "타입 필터",
                    ["전체"] + g.get_labels(),
                )
            with col_ctrl3:
                show_text = st.checkbox("텍스트 목록도 표시", value=False)

            # 그래프 필터링
            import networkx as nx
            G = g._G
            if filter_label != "전체":
                nodes = [n for n, d in G.nodes(data=True) if d.get("label") == filter_label]
                G = G.subgraph(nodes)

            if G.number_of_nodes() == 0:
                st.info("해당 타입의 노드가 없습니다.")
            else:
                # pyvis 렌더링 시도
                try:
                    html = _render_graph_pyvis(G, max_nodes=max_nodes)
                    if html:
                        import streamlit.components.v1 as components
                        components.html(html, height=620, scrolling=False)

                        # 범례
                        st.markdown("**노드 색상 범례**")
                        legend_cols = st.columns(7)
                        legend = {
                            "Law (법률)": "🔴", "Organization (기관)": "🔵", "Person (인물)": "🟢",
                            "Regulation (규정)": "🟠", "Concept (개념)": "🟣", "Product (제품)": "🩵", "Unknown": "⚫"
                        }
                        for i, (lbl, icon) in enumerate(legend.items()):
                            legend_cols[i % 7].caption(f"{icon} {lbl}")
                    else:
                        st.warning("pyvis 미설치. 설치: pip install pyvis --break-system-packages")
                        st.code(g.get_subgraph_text(), language=None)

                except Exception as e:
                    st.error(f"시각화 오류: {e}")
                    if show_text:
                        st.code(g.get_subgraph_text(), language=None)

            if show_text:
                with st.expander("📋 텍스트 목록"):
                    st.code(g.get_subgraph_text(), language=None)
    else:
        st.info("그래프 데이터가 없습니다. 문서를 먼저 업로드하세요.")

    st.markdown("---")
    st.subheader("📖 현재 온톨로지 (자동 학습됨)")

    if onto:
        for cls, vals in onto.items():
            if isinstance(vals, list) and vals:
                with st.expander(f"{cls} ({len(vals)}개)"):
                    st.write(", ".join(vals))
    else:
        st.info("온톨로지가 비어 있습니다. 문서를 업로드하면 자동으로 채워집니다.")

    st.subheader("🔗 그래프 레이블")
    labels = g.get_labels()
    st.write(", ".join(labels) if labels else "데이터 없음")

    with st.expander("⚙️ 설정"):
        st.json({
            "graph_backend": cfg.get("graph", {}).get("backend"),
            "llm_model": cfg["llm"]["model_path"],
            "embedding_model": cfg["embedding"]["model"],
            "chunk_size": cfg["embedding"].get("chunk_size"),
            "chunk_overlap": cfg["embedding"].get("chunk_overlap"),
        })


# ──────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────
def main():
    sidebar()
    t1, t2, t3 = st.tabs(["📂 문서 업로드", "💬 채팅", "📊 시스템 상태"])
    with t1:
        tab_upload()
    with t2:
        tab_chat()
    with t3:
        tab_status()


if __name__ == "__main__":
    main()