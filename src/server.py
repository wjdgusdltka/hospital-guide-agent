"""FastAPI server: POST /query (course API spec) + static chat UI from web/."""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from pydantic import BaseModel

from . import config  # noqa: F401  (loads .env)
from . import memory, warmup
from .rag.index_check import NOT_READY_MESSAGE, ensure_index, index_status

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
# Mock is OFF unless explicitly enabled with MOCK_MODE=1 (never ship fixed fake answers by accident).
MOCK_MODE = os.getenv("MOCK_MODE", "0").strip() == "1"
logger = logging.getLogger("uvicorn.error")
MOCK_WARNING = (
    "!!! WARNING: MOCK_MODE=1 -- 모든 답변이 고정 가짜 응답입니다 (실제 에이전트가 아님). "
    "개발용으로만 쓰고 운영에서는 MOCK_MODE=0 으로 두세요. !!!"
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Real mode: refuse to start without the search index (answers without evidence are not allowed).
    # Mock mode (UI development): only warn.
    if MOCK_MODE:
        print(MOCK_WARNING, flush=True)
        logger.warning(MOCK_WARNING)
        st = index_status()
        if not st["ready"]:
            print(NOT_READY_MESSAGE.format(reason=st["reason"]))
    else:
        ensure_index()
        if config.WARMUP:  # 첫 질문이 초기화 비용을 치르지 않도록 백그라운드로 미리 데운다
            warmup.start()
    yield


app = FastAPI(title="병원 길잡이 Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None  # maps to the LangGraph thread_id; omitted = new conversation
    user_name: str | None = None  # 선택: 로그인한 사용자 이름. 있으면 장기 메모리(상담 이력)를 쓴다. 없으면 기존과 동일


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
    # model_cooldowns: 한도에 걸려 잠시 건너뛰는 모델(서버를 켠 뒤 기록). 비어 있으면 모두 정상
    # quota_alert.all_daily_limited 가 true 면 설정된 모든 모델이 하루 한도에 도달한 것이므로 한도 증설이 필요하다
    from .llm import cooldown_status, quota_alert

    st = index_status()
    return {"status": "ok", "mock": MOCK_MODE, "index_ready": st["ready"], "index_docs": st["docs"], "index_note": st["reason"],
            "model_cooldowns": cooldown_status(), "quota_alert": quota_alert()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if MOCK_MODE:
        return _mock_response(req)
    # 실제 에이전트 실행 (동기 함수라 FastAPI가 스레드풀에서 돌린다). 같은 session_id 면 같은 대화로 이어진다.
    from botocore.exceptions import BotoCoreError, ClientError

    from .agent import run_agent

    try:
        result = run_agent(req.question, req.session_id, memory.normalize_name(req.user_name))
    except (ClientError, BotoCoreError) as e:  # 모든 모델이 한도·장애로 답하지 못할 때: 오류 대신 안전 안내로 응답
        return QueryResponse(
            answer=MODEL_UNAVAILABLE_ANSWER, contexts=[],
            trace=[TraceStep(step="model_unavailable", input="(생략)", output=f"{type(e).__name__}: 모든 모델이 한도 또는 장애로 응답하지 못함")],
        )
    except Exception as e:  # 예상하지 못한 오류도 500 대신 안전 안내로 응답한다(원인은 서버 로그에 남긴다). 대화는 끊긴 지점에 보존된다
        logger.exception("unexpected error while answering")
        return QueryResponse(
            answer=MODEL_UNAVAILABLE_ANSWER, contexts=[],
            trace=[TraceStep(step="agent_error", input="(생략)", output=f"{type(e).__name__}: 답변 생성 중 오류, 대화는 보존됨")],
        )
    return QueryResponse(
        answer=result["answer"],
        contexts=[Context(**c) for c in result["contexts"]],
        trace=[TraceStep(**t) for t in result["trace"]],
    )


class NameRequest(BaseModel):
    name: str  # 이름은 URL 에 싣지 않으려고 POST 본문으로 받는다


class HistoryResponse(BaseModel):
    name: str
    greeting: str | None  # 최근 상담이 있으면 안부 인사, 없으면 None
    consultations: list[dict]  # 최신순 상담 요약


def _user(req: NameRequest) -> tuple[str, str]:
    uid = memory.normalize_name(req.name)
    if not uid:
        raise HTTPException(status_code=400, detail=f"이름은 1~{memory.MAX_NAME_CHARS}자로 입력해 주세요.")
    return uid, " ".join(req.name.split())


@app.post("/memory/history", response_model=HistoryResponse)
def history(req: NameRequest) -> HistoryResponse:
    """로그인 직후 호출: 이 사용자의 과거 상담 요약 목록과 안부 인사를 돌려준다 (LLM 호출 없음)."""
    uid, display = _user(req)
    rows = memory.list_consultations(uid)
    return HistoryResponse(name=display, greeting=memory.greeting(display, rows[0]) if rows else None,
                           consultations=[memory.summarize(r) for r in rows])


class DeleteRequest(NameRequest):
    id: str  # 지울 상담의 id (목록의 id 값)


@app.post("/memory/delete")
def delete_history(req: DeleteRequest) -> dict:
    """이 사용자의 상담 한 건을 지운다."""
    uid, _ = _user(req)
    return {"deleted": 1 if memory.delete_one(uid, req.id) else 0}


@app.post("/memory/clear")
def clear_history(req: NameRequest) -> dict:
    """이 사용자의 저장된 상담 이력을 모두 지운다."""
    uid, _ = _user(req)
    return {"deleted": memory.clear(uid)}


MODEL_UNAVAILABLE_ANSWER = (
    "지금은 답변을 만들지 못했어요(AI 모델 사용 한도나 일시적인 오류). 잠시 뒤 \"다시 답변해 주세요\"라고 보내시면 "
    "하던 대화를 이어서 답변드려요.\n\n"
    "응급이 의심되면 기다리지 말고 바로 119에 연락하세요."
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
            "[MOCK] 말씀해 주신 내용만으로는 응급 신호가 뚜렷하게 보이지 않아요.\n\n"
            "**관련될 수 있는 상태(참고)**\n- 긴장성 두통 (예시)\n- 수면 부족 (예시)\n\n"
            "**진료과 후보**\n- 가정의학과 또는 내과\n\n"
            "증상이 심해지거나 위험 신호가 생기면 바로 응급실을 이용하세요.\n\n" + DISCLAIMER
        )
        contexts = [Context(doc_id="mock-doc-1", text="(mock) 근거 문서 발췌 예시")]
        trace = [TraceStep(step="mock", input=q, output="canned response")]
    trace.append(TraceStep(step="mock_mode", input="(mock)", output="고정 응답입니다 (실제 에이전트가 아님)"))
    return QueryResponse(answer=answer, contexts=contexts, trace=trace)


# Mount last so API routes take priority.
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
