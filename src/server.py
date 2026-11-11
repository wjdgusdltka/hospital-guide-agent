"""FastAPI server: POST /query (course API spec) + static chat UI from web/."""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config  # noqa: F401  (loads .env)

app = FastAPI(title="Symptom Triage Agent")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MOCK_MODE = os.getenv("MOCK_MODE", "1") == "1"


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
    return {"status": "ok", "mock": MOCK_MODE}


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
