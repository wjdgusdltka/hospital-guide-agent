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
        return QueryResponse(
            answer=(
                "[MOCK] 이 답변은 UI 개발용 고정 응답입니다. "
                "실제 에이전트가 연결되면 응급 여부와 진료과 추천이 표시됩니다.\n\n"
                "※ 참고용 안내이며 근거가 확실하지 않을 수 있습니다. 정확한 판단은 의사에게 받으세요."
            ),
            contexts=[Context(doc_id="mock-doc-1", text="(mock) 응급의료법 시행규칙 별표 1 발췌 예시")],
            trace=[TraceStep(step="mock", input=req.question, output="canned response")],
        )
    raise NotImplementedError("Agent is not wired yet (set MOCK_MODE=1 for now).")


# Mount last so API routes take priority.
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
