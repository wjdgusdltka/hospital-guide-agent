"""RAG 검색기 (Day 2): 한국어 증상 -> 영어 검색어 변환 -> Chroma 벡터 검색.

- MedlinePlus(영문) 문서: 한국어를 그대로 임베딩하면 못 찾아서(시험 0/12) 영어 키워드로 바꿔 검색한다.
- 별표 1(한국어) 문서: 한국어 그대로 검색한다.
"""
import threading

from langchain_chroma import Chroma

from . import config
from .llm_chain import structured_chain
from .prompts import QUERY_SYSTEM
from .rag.build_index import get_vectorstore
from .rag.index_check import ensure_index
from .schemas import QueryPlan

TOP_K_PER_QUERY = 3  # 영어 검색어 하나당 가져올 문서 수
MAX_DOCS = 5  # 최종적으로 에이전트에게 넘기는 MedlinePlus 문서 수


_STORE: Chroma | None = None
_LOCK = threading.Lock()  # 첫 요청 여러 개가 동시에 인덱스를 열다 실패하는 것을 막는다


def _store() -> Chroma:
    """Chroma 연결을 한 번만 만든다 (인덱스가 없으면 안내와 함께 오류). 스레드 안전."""
    global _STORE
    if _STORE is None:
        with _LOCK:
            if _STORE is None:
                ensure_index()
                _STORE = get_vectorstore()
    return _STORE


def _to_dict(doc, score: float) -> dict:
    """LangChain Document 를 에이전트가 쓰는 dict 로 바꾼다 (doc_id, text, title, url ...)."""
    m = doc.metadata
    return {
        "doc_id": m["doc_id"], "text": doc.page_content, "title": m.get("title", ""),
        "url": m.get("url", ""), "source": m.get("source", ""), "score": round(float(score), 3),
    }


def translate_query(history_text: str) -> list[str]:
    """대화 전체를 영어 검색 키워드 1~3개로 바꾼다 (빠른 모델, 구조화 출력)."""
    plan = structured_chain("query_plan", QueryPlan, QUERY_SYSTEM, fast=True).invoke({"input": history_text})
    return [q.strip() for q in plan.english_queries if q.strip()][:3]


def search_medlineplus(queries: list[str]) -> list[dict]:
    """영어 검색어마다 MedlinePlus 문서를 찾아 합치고, 문서별 최고 점수(낮을수록 가까움)로 정렬한다."""
    best: dict[str, dict] = {}
    for q in queries:
        hits = _store().similarity_search_with_score(q, k=TOP_K_PER_QUERY, filter={"source": "MedlinePlus"})
        for doc, score in hits:
            d = _to_dict(doc, score)
            if d["doc_id"] not in best or d["score"] < best[d["doc_id"]]["score"]:
                best[d["doc_id"]] = d
    return sorted(best.values(), key=lambda d: d["score"])[:MAX_DOCS]


def search_annex(korean_text: str, k: int = 2) -> list[dict]:
    """한국어 증상 설명으로 별표 1(응급증상 목록) 청크를 찾는다."""
    hits = _store().similarity_search_with_score(korean_text, k=k, filter={"source": "별표 1"})
    return [_to_dict(doc, score) for doc, score in hits]
