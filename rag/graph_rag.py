"""rag/graph_rag.py — GraphRAG QA 엔진"""

import logging
from rag.retriever import retrieve, retrieve_with_detail
from rag.llm import generate

logger = logging.getLogger(__name__)

_SYSTEM = """당신은 산업안전보건 문서 검색 AI입니다.
아래 규칙을 반드시 지키세요:
1. [참고 정보]에 있는 내용만 사용해서 답변하세요.
2. 같은 내용을 절대 반복하지 마세요.
3. "별표 4", "별표 5" 같은 참조 번호만 나열하지 말고 실제 내용을 설명하세요.
4. 정보가 없으면 "해당 정보를 찾을 수 없습니다"라고 하세요.
5. 답변은 핵심만 간결하게 작성하세요."""


def _detect_query_type(query: str) -> str:
    """질문 유형 감지 → 프롬프트 전략 선택"""
    list_keywords = ["정리", "목록", "나열", "업종별", "종류", "모두", "전부", "리스트"]
    compare_keywords = ["차이", "비교", "다른점", "공통점", "vs"]
    if any(k in query for k in list_keywords):
        return "list"
    if any(k in query for k in compare_keywords):
        return "compare"
    return "qa"


def _build_prompt(query: str, context: str, history_text: str = "") -> str:
    q_type = _detect_query_type(query)

    if q_type == "list":
        # 목록형 질문: 구체적 내용 요구, 반복 금지 강조
        instruction = """[답변 형식]
- 각 항목을 한 번씩만 작성하세요
- 실제 교육 내용(시간, 과목, 대상)을 구체적으로 적으세요
- "별표 N" 같은 참조만 적지 말고 실제 내용을 쓰세요
- 중복 항목 금지"""
    elif q_type == "compare":
        instruction = """[답변 형식]
- 공통점과 차이점을 명확히 구분해서 답변하세요
- 출처 문서를 구분해서 설명하세요"""
    else:
        instruction = """[답변 형식]
- 핵심 내용을 간결하게 답변하세요
- 관련 법조문이나 기준이 있으면 포함하세요"""

    return f"""{_SYSTEM}
{history_text}

[참고 정보]
{context}

[질문]
{query}

{instruction}

[답변]
"""


def answer(query: str, conversation_history: list = None) -> str:
    context = retrieve(query)

    history_text = ""
    if conversation_history:
        recent = conversation_history[-4:]
        lines = []
        for t in recent:
            role = "사용자" if t["role"] == "user" else "AI"
            content = t["content"][:200]
            lines.append(f"{role}: {content}")
        if lines:
            history_text = "\n\n[이전 대화]\n" + "\n".join(lines)

    prompt = _build_prompt(query, context, history_text)

    try:
        resp = generate(prompt, mode="chat")
        return resp if resp and len(resp.strip()) > 3 else "답변 생성에 실패했습니다. 다시 시도해주세요."
    except Exception as e:
        logger.error(f"LLM 오류: {e}")
        return f"오류 발생: {e}"


def answer_with_sources(query: str) -> dict:
    detail = retrieve_with_detail(query)
    prompt = _build_prompt(query, detail["combined_context"])

    try:
        resp = generate(prompt, mode="chat")
    except Exception as e:
        resp = f"오류: {e}"

    return {
        "answer": resp,
        "vector_sources": detail["vector_chunks"],
        "graph_sources": detail["graph_facts"],
    }