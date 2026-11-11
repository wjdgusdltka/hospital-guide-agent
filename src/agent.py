"""병원 길잡이 에이전트 그래프 (LangGraph).

한 번의 요청이 지나가는 길:
  begin(개인정보 마스킹) -> classify(범위 분류)
     ├ 범위 밖/제공 불가            -> refuse
     ├ (응급 안내 직후) 지역 답변   -> locate -> 응급의료기관 목록
     └ 증상 -> triage(응급 점검 + 근거 검색, 병렬)
            ├ 응급                  -> emergency
            ├ 인젝션                -> refuse_injection
            └ assess(정보 충분?)
                   ├ 부족(3턴 이내) -> followup (되묻기)
                   └ 충분/한도 도달 -> recommend (관련 상태 + 진료과)
  모든 답변은 finalize(출력 가드레일)를 거쳐 끝난다.

멀티턴 상태는 체크포인터(InMemorySaver)가 session_id(=thread_id)별로 보관한다. (서버 메모리에만 저장)
"""
import operator
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from . import answers, retriever, tools
from .guardrails import check_scope, mask_pii, output_guard

MAX_FOLLOWUP_TURNS = 3  # 되묻기 최대 횟수 (SERVICE.md 정책)
MAX_TRACE_CHARS = 400  # trace 한 칸의 최대 길이


class State(TypedDict, total=False):
    question: str  # 이번 턴 사용자 입력 (마스킹 후)
    history: Annotated[list[str], operator.add]  # 지금까지의 대화(사용자 발화 + 서비스 질문). 턴마다 누적
    followup_turns: int  # 지금까지 되묻기를 한 횟수
    awaiting_location: bool  # 응급 안내 후 지역 답변을 기다리는 중인가
    last_kind: str  # 직전 응답 유형: followup / emergency / refuse / answer
    scope: dict  # 입력 분류 결과
    red: dict  # 응급 점검 결과
    docs: list[dict]  # 검색된 근거 문서
    notes: list[str]  # 답변 앞에 붙일 안내(약 용량 불가 등)
    answer: str  # 이번 턴 최종 답변
    contexts: list[dict]  # 이번 턴 근거 문서 (API 응답의 contexts)
    trace: list[dict]  # 이번 턴 처리 과정 (API 응답의 trace)


# ------------------------------------------------------------------ 보조 함수
def _clip(x) -> str:
    """trace 에 넣을 문자열을 길이 제한에 맞춘다."""
    s = x if isinstance(x, str) else str(x)
    return s if len(s) <= MAX_TRACE_CHARS else s[:MAX_TRACE_CHARS] + "…"


def _step(state: State, name: str, inp, out) -> list[dict]:
    """기존 trace 에 한 단계를 덧붙인 새 목록을 돌려준다 (step 이름 = 도구 이름)."""
    return state.get("trace", []) + [{"step": name, "input": _clip(inp), "output": _clip(out)}]


def _history_text(state: State) -> str:
    """대화 기록을 프롬프트용 문자열로 합친다."""
    return "\n".join(state["history"])


def _contexts(docs: list[dict]) -> list[dict]:
    """검색 문서를 API 응답 형식(doc_id, text)으로 바꾼다."""
    return [{"doc_id": d["doc_id"], "text": d["text"]} for d in docs]


# ------------------------------------------------------------------ 노드
def begin(state: State) -> dict:
    """입력 가드레일 1단계: 개인정보를 마스킹하고 이번 턴 상태를 초기화한다."""
    masked, found = mask_pii(state["question"])
    out = f"마스킹: {', '.join(found)}" if found else "개인정보 없음"
    return {
        "question": masked, "history": [f"사용자: {masked}"], "answer": "", "contexts": [], "docs": [],
        "notes": [], "red": {}, "trace": [{"step": "input_guardrail", "input": _clip(masked), "output": out}],
    }


def classify(state: State) -> dict:
    """입력 가드레일 2단계: 범위(증상/범위 밖/제공 불가/인젝션)와 약 용량·확률 요구를 분류한다."""
    scope = check_scope(state["question"], state.get("last_kind", ""))
    notes = []
    if scope.wants_dosage:
        notes.append(answers.DOSAGE_NOTE)
    if scope.wants_diagnosis:
        notes.append(answers.DIAGNOSIS_NOTE)
    if scope.category == "injection":
        notes.append(answers.INJECTION_NOTE)
    return {"scope": scope.model_dump(), "notes": notes,
            "trace": _step(state, "scope_check", state["question"], f"category={scope.category}")}


def route_after_classify(state: State) -> str:
    """분류 결과로 다음 노드를 고른다."""
    cat = state["scope"]["category"]
    if cat in ("out_of_scope", "unavailable"):
        return "refuse"
    if state.get("awaiting_location") and cat == "symptom":
        return "locate"
    return "triage"


def refuse(state: State) -> dict:
    """범위 밖이거나 제공하지 않는 정보 요청을 정중히 거절한다."""
    text = answers.REFUSE_UNAVAILABLE if state["scope"]["category"] == "unavailable" else answers.REFUSE_OUT_OF_SCOPE
    return {"answer": text, "last_kind": "refuse"}


def locate(state: State) -> dict:
    """응급 안내 뒤에 받은 지역 답변으로 응급의료기관 목록을 조회한다. 지역이 없으면 다시 증상 흐름으로 보낸다."""
    loc = tools.extract_location(state["question"])
    if not (loc.found and loc.sido):
        return {"awaiting_location": False, "trace": _step(state, "find_emergency_rooms", state["question"], "지역 없음 -> 증상 흐름으로")}
    rooms = tools.find_emergency_rooms(loc.sido, loc.sigungu)
    names = ", ".join(r["name"] for r in rooms) or "결과 없음"
    return {
        "awaiting_location": False, "last_kind": "answer",
        "answer": answers.compose_rooms(loc.sido, loc.sigungu, rooms),
        "trace": _step(state, "find_emergency_rooms", f"{loc.sido} {loc.sigungu or ''}", f"{len(rooms)}곳: {names}"),
    }


def route_after_locate(state: State) -> str:
    """지역 조회로 답이 만들어졌으면 마무리, 아니면 증상 흐름으로 이어간다."""
    return "finalize" if state.get("answer") else "triage"


def triage(state: State) -> dict:
    """도구 check_red_flags(응급 점검)와 retrieve_guidelines(근거 검색)를 동시에 실행한다.

    둘은 서로 입력이 독립이라 병렬로 돌려 시간을 줄인다. 응급이면 검색 결과는 쓰지 않고 버린다.
    - check_red_flags: 별표 1 기준 응급 여부 (매 턴 대화 전체를 다시 본다)
    - retrieve_guidelines: 한국어 증상을 영어 키워드로 바꿔 MedlinePlus 근거 문서를 검색
    """
    text = _history_text(state)

    def search():
        queries = retriever.translate_query(text)
        return queries, retriever.search_medlineplus(queries)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_red, f_search = pool.submit(tools.check_red_flags, text), pool.submit(search)
        res, (queries, docs) = f_red.result(), f_search.result()
    red_out = f"emergency={res.emergency}; 항목={res.matched_items}; {res.reason}"
    docs_out = ", ".join(f"{d['doc_id']}({d['title']})" for d in docs) or "문서 없음"
    trace = _step(state, "check_red_flags", text, red_out)
    trace += [{"step": "retrieve_guidelines", "input": _clip(" | ".join(queries)), "output": _clip(docs_out)}]
    return {"red": res.model_dump(), "docs": docs, "trace": trace}


def route_after_triage(state: State) -> str:
    """응급이면 응급 안내, 인젝션이면 거절, 아니면 정보 충분성 판단으로."""
    if state["red"]["emergency"]:
        return "emergency"
    if state["scope"]["category"] == "injection":
        return "refuse_injection"
    return "assess"


def emergency(state: State) -> dict:
    """응급 권고: 별표 1 근거를 보여주고 지역을 묻는다. 다음 턴의 지역 답변을 기다린다."""
    red = state["red"]
    annex = retriever.search_annex(_history_text(state))
    text = answers.compose_emergency(red["matched_items"], red["reason"], state.get("notes", []))
    return {"answer": text, "contexts": _contexts(annex), "awaiting_location": True, "followup_turns": 0, "last_kind": "emergency"}


def refuse_injection(state: State) -> dict:
    """증상 없는 프롬프트 인젝션 시도를 거절한다 (응급 신호가 있었다면 위에서 이미 응급 안내로 갔다)."""
    return {"answer": answers.REFUSE_INJECTION, "last_kind": "refuse"}


def assess(state: State) -> dict:
    """도구 ask_followup: 정보가 충분한지 보고, 부족하고 한도 안이면 되묻기 질문을 만든다."""
    turns = state.get("followup_turns", 0)
    res = tools.ask_followup(_history_text(state), state["docs"])
    if not res.enough_info and turns < MAX_FOLLOWUP_TURNS:
        text = answers.compose_followup(res.questions, state.get("notes", []))
        asked = "서비스 질문: " + " / ".join(res.questions)
        return {"answer": text, "history": [asked], "followup_turns": turns + 1, "last_kind": "followup",
                "contexts": _contexts(state["docs"]), "scope": {**state["scope"], "conservative": False},
                "trace": _step(state, "ask_followup", f"{turns + 1}번째 되묻기", " / ".join(res.questions))}
    # 충분하거나 되묻기 한도에 도달: 한도 도달인데 부족하면 보수적으로 안내한다
    conservative = not res.enough_info
    label = "추가 질문 불필요" if res.enough_info else f"되묻기 {MAX_FOLLOWUP_TURNS}턴 한도 도달 -> 보수적 안내"
    return {"contexts": _contexts(state["docs"]), "scope": {**state["scope"], "conservative": conservative},
            "trace": _step(state, "assess_info", "정보 충분성 판단", label)}


def route_after_assess(state: State) -> str:
    """되묻기 답이 만들어졌으면 마무리, 아니면 추천 단계로."""
    return "finalize" if state.get("answer") else "recommend"


def recommend(state: State) -> dict:
    """도구 find_related_conditions 와 recommend_department 를 동시에 실행해 비응급 안내를 만든다."""
    text, docs = _history_text(state), state["docs"]
    with ThreadPoolExecutor(max_workers=2) as pool:  # 두 LLM 호출을 병렬로
        f_cond = pool.submit(tools.find_related_conditions, text, docs)
        f_dept = pool.submit(tools.recommend_department, text, docs)
        conditions, dept = f_cond.result(), f_dept.result()
    answer = answers.compose_recommendation(
        conditions, dept.departments, dept.reason, dept.watch_signs,
        state["scope"].get("conservative", False), state.get("notes", []),
    )
    trace = _step(state, "find_related_conditions", "근거 문서에서 후보 추출", ", ".join(c["name_ko"] for c in conditions) or "없음")
    trace = trace + [{"step": "recommend_department", "input": "증상과 관련 상태", "output": _clip(f"{dept.departments}: {dept.reason}")}]
    return {"answer": answer, "trace": trace, "followup_turns": 0, "last_kind": "answer"}


def finalize(state: State) -> dict:
    """출력 가드레일: 금지 표현(확률·안심·병명 단정·약 용량)이 든 줄을 지우고 trace 에 기록한다."""
    cleaned, removed = output_guard(state["answer"])
    out = f"제거: {', '.join(removed)}" if removed else "통과"
    return {"answer": cleaned, "trace": _step(state, "output_guardrail", "최종 답변 검사", out)}


# ------------------------------------------------------------------ 그래프 조립
def build_graph():
    """노드와 분기를 연결해 컴파일한다. 체크포인터가 session_id 별 대화 상태를 보관한다."""
    g = StateGraph(State)
    for fn in (begin, classify, refuse, locate, triage, emergency, refuse_injection, assess, recommend, finalize):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "begin")
    g.add_edge("begin", "classify")
    g.add_conditional_edges("classify", route_after_classify, ["refuse", "locate", "triage"])
    g.add_conditional_edges("locate", route_after_locate, ["finalize", "triage"])
    g.add_conditional_edges("triage", route_after_triage, ["emergency", "refuse_injection", "assess"])
    g.add_conditional_edges("assess", route_after_assess, ["finalize", "recommend"])
    for node in ("refuse", "emergency", "refuse_injection", "recommend"):
        g.add_edge(node, "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=InMemorySaver())


_GRAPH = None
_GRAPH_LOCK = threading.Lock()  # 그래프(와 그 체크포인터)는 프로세스에 하나만 있어야 세션이 유지된다


def run_agent(question: str, session_id: str | None = None) -> dict:
    """질문 하나를 처리해 {answer, contexts, trace} 를 돌려준다. session_id 가 같으면 같은 대화로 이어진다."""
    global _GRAPH
    if _GRAPH is None:
        with _GRAPH_LOCK:
            if _GRAPH is None:
                _GRAPH = build_graph()
    thread = session_id or uuid.uuid4().hex  # session_id 가 없으면 매번 새 대화
    out = _GRAPH.invoke(
        {"question": question},
        config={"configurable": {"thread_id": thread}, "run_name": "hospital-guide-agent", "tags": ["hospital-guide"]},
    )
    return {"answer": out["answer"], "contexts": out.get("contexts", []), "trace": out.get("trace", [])}
