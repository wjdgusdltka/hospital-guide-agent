"""FastAPI server: POST /query (course API spec) + static chat UI from web/."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config  # noqa: F401  (loads .env)
from .rag.index_check import NOT_READY_MESSAGE, ensure_index, index_status

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MOCK_MODE = os.getenv("MOCK_MODE", "1") == "1"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Real mode: refuse to start without the search index (answers without evidence are not allowed).
    # Mock mode (UI development): only warn.
    if MOCK_MODE:
        st = index_status()
        if not st["ready"]:
            print(NOT_READY_MESSAGE.format(reason=st["reason"]))
    else:
        ensure_index()
    yield


app = FastAPI(title="병원 길잡이 Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None  # maps to the LangGraph thread_id; omitted = new conversation


class Context(BaseModel):
    doc_id: str
    text: str


class TraceStep(BaseModel):
    step: str
    input: str
    output: str


class QueryResponse(BaseModel):
    answer: str
    contexts: list[Context]
    trace: list[TraceStep]


@app.get("/health")
def health():
    st = index_status()
    return {"status": "ok", "mock": MOCK_MODE, "index_ready": st["ready"], "index_docs": st["docs"], "index_note": st["reason"]}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if MOCK_MODE:
        return _mock_response(req)
    # 실제 에이전트 실행 (동기 함수라 FastAPI가 스레드풀에서 돌린다). 같은 session_id 면 같은 대화로 이어진다.
    from .agent import run_agent

    result = run_agent(req.question, req.session_id)
    return QueryResponse(
        answer=result["answer"],
        contexts=[Context(**c) for c in result["contexts"]],
        trace=[TraceStep(**t) for t in result["trace"]],
    )


DISCLAIMER = "※ 참고용 안내이며 근거가 확실하지 않을 수 있습니다. 정확한 판단은 의사에게 받으세요."


def _mock_response(req: QueryRequest) -> QueryResponse:
    """UI-development stand-in. Picks one of three answer shapes by keyword so every UI state can be tried."""
    q = req.question
    if any(k in q for k in ("가슴", "숨", "의식", "쓰러", "피를")):
        answer = (
            "🚨 응급 신호가 의심됩니다. 지금 바로 119에 연락하거나 가까운 응급실을 이용하세요.\n\n"
            "지역(시/도, 시/군/구)을 알려주시면 해당 지역의 응급의료기관을 안내해 드릴게요.\n\n" + DISCLAIMER
        )
        contexts = [Context(doc_id="annex1-3", text="(mock) [별표 1] 응급증상 - 심혈관계: 급성 호흡곤란, 심장질환으로 인한 급성 흉통 ...")]
        trace = [
            TraceStep(step="check_red_flags", input=q, output="emergency=true (mock)"),
            TraceStep(step="retrieve_guidelines", input="chest pain", output="Chest Pain, annex1-3 (mock)"),
        ]
    elif any(k in q for k in ("배", "두통", "머리", "열", "설사", "토")):
        answer = (
            "정확히 안내드리기 위해 몇 가지 여쭤볼게요.\n\n"
            "1. 증상이 갑자기 시작됐나요, 서서히 시작됐나요?\n"
            "2. 열이 있거나 구토, 목 뻣뻣함이 함께 있나요?\n\n"
            "모르시면 '모르겠어요'라고 답하셔도 됩니다."
        )
        contexts = [Context(doc_id="mplus-273-0", text="(mock) Headache: ... Get medical help right away if you have a headache after a head injury ...")]
        trace = [
            TraceStep(step="check_red_flags", input=q, output="emergency=false, need_more_info (mock)"),
            TraceStep(step="ask_followup", input=q, output="2 questions (mock)"),
        ]
    else:
        answer = (
            "[MOCK] 응급 신호는 보이지 않습니다.\n\n"
            "**관련될 수 있는 상태(참고)**\n- 긴장성 두통 (예시)\n- 수면 부족 (예시)\n\n"
            "**진료과 후보**\n- 가정의학과 또는 내과\n\n"
            "증상이 심해지거나 위험 신호가 생기면 바로 응급실을 이용하세요.\n\n" + DISCLAIMER
        )
        contexts = [Context(doc_id="mock-doc-1", text="(mock) 근거 문서 발췌 예시")]
        trace = [TraceStep(step="mock", input=q, output="canned response")]
    return QueryResponse(answer=answer, contexts=contexts, trace=trace)


# Mount last so API routes take priority.
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
