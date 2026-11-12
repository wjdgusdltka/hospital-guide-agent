"""병원 길잡이 에이전트 그래프 (LangGraph).

한 번의 요청이 지나가는 길:
  begin(개인정보 마스킹) -> classify(범위 분류)
     ├ 범위 밖/제공 불가            -> refuse
     ├ 응급실 목록 요청 / 지역 답 / 응급 안내 이후의 대화 -> locate: 지역이면 응급의료기관 목록,
     │      지역 없는 목록 요청이면 지역을 되묻고, 응급 대화 중이면 emergency_followup(짧은 이어가기 답변)
     └ 증상 -> triage(응급 점검 + 근거 검색, 병렬)
            ├ 긴급(별표 1 응급증상)  -> emergency (검정 띠, 지금 바로 119/응급실)
            ├ 응급일 수 있음        -> advise_er (부드러운 응급실 권유 + 진료과 후보)
            ├ 인젝션              -> refuse_injection
            └ assess(정보 충분?)
                   ├ 부족(3턴 이내) -> followup (되묻기)
                   └ 충분/한도 도달 -> recommend (관련 상태 + 진료과)
  모든 답변은 finalize(출력 가드레일)를 거쳐 끝난다.

장기 메모리(로그인한 사용자만): begin 이 새 대화 첫 턴에 최근 상담을 불러와(load_memory) 안부 인사의 답으로 읽고,
  지난 상담 안부에 '나아졌어요'라고만 답하면 checkin 으로 짧게 맺는다.
  emergency / advise_er / recommend 로 상담이 끝나면 save_memory 가 요약을 저장한 뒤 finalize 로 간다.

HITL(interrupt)은 꼭 필요한 한 곳에만 쓴다: 응급의료기관을 찾으려는데 지역이 없을 때(ask_region).
되묻기 질문이나 응급 안내 뒤의 대화는 interrupt 없이 '다음 요청이 새 턴'으로 처리한다. 응급 점검을 매 답마다
처음부터 다시 해야 하는 흐름이라, 멈췄다 이어가는 것보다 새 턴으로 처리하는 편이 안전하고 단순하기 때문이다.

멀티턴 상태는 체크포인터(InMemorySaver)가 session_id(=thread_id)별로 보관한다. (서버 메모리에만 저장)
"""
import operator
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import answers, memory, retriever, tools
from .guardrails import check_scope, mask_pii, output_guard

MAX_FOLLOWUP_TURNS = 3  # 되묻기 최대 횟수 (SERVICE.md 정책)
MAX_TRACE_CHARS = 400  # trace 한 칸의 최대 길이


class State(TypedDict, total=False):
    question: str  # 이번 턴 사용자 입력 (마스킹 후)
    history: Annotated[list[str], operator.add]  # 지금까지의 대화(사용자 발화 + 서비스 질문). 턴마다 누적
    followup_turns: int  # 지금까지 되묻기를 한 횟수
    emergency_active: bool  # 이 대화에서 응급 안내가 나갔는가 (이후 턴은 응급 점검을 반복하지 않고 대화를 이어간다)
    awaiting_location: bool  # 응급 안내 후 지역 답변을 기다리는 중인가
    region: dict | None  # 조회할 지역 {sido, sigungu}. 없으면 None
    last_kind: str  # 직전 응답 유형: greeting / followup / emergency / refuse / answer
    user_id: str | None  # 로그인한 사용자(이름 정규화값). 없으면 장기 메모리를 쓰지 않는다
    session_id: str  # 이번 대화 id (상담 요약의 저장 키)
    consult: dict | None  # 이번 턴에 끝난 상담 요약(저장 대기)
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
    history = [f"사용자: {masked}"]
    trace = [{"step": "input_guardrail", "input": _clip(masked), "output": out}]
    extra = {}
    uid = state.get("user_id")
    if uid and not state.get("history"):  # 로그인 사용자의 새 대화 첫 턴: 장기 메모리에서 최근 상담을 불러온다
        rec = _recall(uid)
        if rec:
            history = [memory.context_line(rec)] + history  # '서비스 질문:' 줄이라 안전망 규칙에 걸리지 않고, 프롬프트는 과거 기록으로만 본다
            extra["last_kind"] = "greeting"  # UI가 건넨 안부 인사에 대한 답으로 읽는다
            trace.append({"step": "load_memory", "input": "최근 상담 이력 조회(장기 메모리)",
                          "output": _clip(f"{memory.LEVEL_LABEL[rec['level']]}: {memory.symptom_text(rec)}")})
    return {
        "question": masked, "history": history, "answer": "", "contexts": [], "docs": [],
        "notes": [], "red": {}, "consult": None, "trace": trace, **extra,
    }


def _recall(uid: str) -> dict | None:
    """최근 상담을 읽되, 저장소 오류가 상담 자체를 막지 않게 한다."""
    try:
        return memory.latest(uid)
    except Exception:
        return None


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
    if cat == "checkin_ok" and state.get("last_kind") == "greeting":  # 안부 인사에 '나아졌어요'라고만 답함
        return "checkin"
    # 응급실 목록 요청, 지역 답변을 기다리는 중의 답, 응급 안내 뒤의 대화는 모두 locate 에서 먼저 처리한다
    if cat == "er_lookup" or (cat == "symptom" and (state.get("emergency_active") or state.get("awaiting_location"))):
        return "locate"
    return "triage"


def checkin(state: State) -> dict:
    """안부 인사에 '나아졌어요/진료 받았어요'라고만 답했을 때: 새 증상이 없으니 점검 없이 짧게 맺고 언제든 다시 말하라고 안내한다."""
    return {"answer": answers.CHECKIN_REPLY, "last_kind": "answer",
            "trace": _step(state, "checkin_reply", state["question"], "새 증상 없음 -> 안내만")}


def refuse(state: State) -> dict:
    """범위 밖이거나 제공하지 않는 정보 요청을 정중히 거절한다."""
    text = answers.REFUSE_UNAVAILABLE if state["scope"]["category"] == "unavailable" else answers.REFUSE_OUT_OF_SCOPE
    return {"answer": text, "last_kind": "refuse"}


def locate(state: State) -> dict:
    """도구 find_emergency_rooms 의 입구: 메시지에서 지역(시/도, 시/군/구)을 찾는다.

    응급 안내 뒤의 지역 답이든, 처음부터 '서울 종로구 응급실 알려줘'라고 지역과 함께 물었든 같은 방식으로 찾는다.
    """
    loc = tools.extract_location(state["question"])
    if loc.found and loc.sido:
        return {"region": {"sido": loc.sido, "sigungu": loc.sigungu}}
    trace = _step(state, "find_emergency_rooms", state["question"], "지역 없음")
    return {"region": None, "trace": trace}


def route_after_locate(state: State) -> str:
    """지역이 있으면 조회, 응급실 목록을 직접 물었는데 지역이 없으면 지역을 묻고(HITL), 아니면 대화 흐름으로."""
    if state.get("region"):
        return "lookup"
    if state["scope"]["category"] == "er_lookup":
        return "ask_region"
    return "emergency_followup" if state.get("emergency_active") else "triage"


def ask_region(state: State) -> dict:
    """HITL: 응급의료기관 조회에 꼭 필요한 값(지역)이 없을 때만 interrupt() 로 멈추고 사용자에게 묻는다.

    위치는 자동으로 알 수 없어서 사용자가 직접 알려 줘야 하고, 그 값 없이는 다음 단계(조회)로 갈 수 없다.
    같은 session_id 의 다음 요청이 Command(resume=답변) 으로 이 지점에서 이어진다.
    답에 지역이 있으면 곧바로 조회로 가고, 지역이 아닌 다른 말이면 새 입력으로 처음부터 정상 흐름을 다시 탄다.
    """
    reply = interrupt({"kind": "region", "prompt": answers.compose_ask_region()})
    masked, _ = mask_pii(reply)
    loc = tools.extract_location(masked)
    if loc.found and loc.sido:
        step = {"step": "human_in_the_loop", "input": "사용자 응답 수신(재개)", "output": f"지역: {loc.sido} {loc.sigungu or ''}"}
        return {"question": masked, "region": {"sido": loc.sido, "sigungu": loc.sigungu}, "trace": [step]}
    return {"question": reply, "region": None}  # 지역이 아니면 begin 이 마스킹하고 새 입력으로 처리한다


def route_after_ask_region(state: State) -> str:
    """재개된 답에 지역이 있으면 조회로, 아니면 새 입력으로 begin 부터."""
    return "lookup" if state.get("region") else "begin"


def lookup(state: State) -> dict:
    """도구 find_emergency_rooms: 국립중앙의료원 API 로 해당 지역의 응급의료기관 목록을 조회해 안내한다."""
    r = state["region"]
    rooms = tools.find_emergency_rooms(r["sido"], r["sigungu"])
    names = ", ".join(x["name"] for x in rooms) or "결과 없음"
    return {
        "awaiting_location": False, "last_kind": "answer", "region": None,
        "answer": answers.compose_rooms(r["sido"], r["sigungu"], rooms),
        "trace": _step(state, "find_emergency_rooms", f"{r['sido']} {r['sigungu'] or ''}", f"{len(rooms)}곳: {names}"),
    }


def emergency_followup(state: State) -> dict:
    """응급 안내 뒤 사용자가 덧붙인 말('쉬면 나아질까요?', '설사도 해요')에 짧게 답한다.

    응급 점검을 다시 돌리지 않으므로 같은 안내문이 반복되지 않는다. 응급 권고는 답변 안에서 유지된다.
    """
    reply = tools.reply_in_emergency(_history_text(state), state["question"])
    text = answers.compose_emergency_reply(reply, state.get("notes", []), state.get("awaiting_location", False))
    return {"answer": text, "history": [f"서비스 답변: {reply}"], "last_kind": "emergency",
            "trace": _step(state, "emergency_followup", state["question"], reply)}


def triage(state: State) -> dict:
    """도구 check_red_flags(응급 점검)와 retrieve_guidelines(근거 검색)를 동시에 실행한다.

    둘은 서로 입력이 독립이라 병렬로 돌려 시간을 줄인다. 응급이면 검색 결과는 쓰지 않고 버린다.
    - check_red_flags: 별표 1 기준 응급 여부 (매 턴 대화 전체를 다시 본다)
    - retrieve_guidelines: 한국어 증상을 영어 키워드로 바꿔 MedlinePlus 근거 문서를 검색

    단, 누가 봐도 응급인 표현(안전망 규칙)이면 판정이 즉시 나오고 응급 답변은 MedlinePlus 문서를 쓰지 않으므로
    MedlinePlus 검색은 시작하지 않는다(영어 변환 LLM 호출과 임베딩 검색을 아낀다).
    """
    text = _history_text(state)
    if tools.hard_red_flags(text):
        res = tools.check_red_flags(text)  # 규칙 판정이라 LLM 호출 없이 즉시 돌아온다
        red_out = f"emergency={res.emergency}; level={res.level}; 항목={res.matched_items}; {res.reason}"
        return {"red": res.model_dump(), "docs": [], "trace": _step(state, "check_red_flags", text, red_out)}

    def search():
        queries = retriever.translate_query(text)
        return queries, retriever.search_medlineplus(queries)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_red, f_search = pool.submit(tools.check_red_flags, text), pool.submit(search)
        res, (queries, docs) = f_red.result(), f_search.result()
    red_out = f"emergency={res.emergency}; level={res.level}; 항목={res.matched_items}; {res.reason}"
    docs_out = ", ".join(f"{d['doc_id']}({d['title']})" for d in docs) or "문서 없음"
    trace = _step(state, "check_red_flags", text, red_out)
    trace += [{"step": "retrieve_guidelines", "input": _clip(" | ".join(queries)), "output": _clip(docs_out)}]
    return {"red": res.model_dump(), "docs": docs, "trace": trace}


def route_after_triage(state: State) -> str:
    """긴급이면 응급 안내, 응급일 수 있으면 응급실 권유(+진료과), 인젝션이면 거절, 아니면 정보 충분성 판단으로."""
    if state["red"]["emergency"]:
        return "emergency" if state["red"].get("level", "urgent") == "urgent" else "advise_er"
    if state["scope"]["category"] == "injection":
        return "refuse_injection"
    return "assess"


def _safe_memo(history_text: str):
    """도구 make_visit_memo 를 호출하되, 실패해도 본 답변은 막지 않는다(메모만 생략)."""
    try:
        return tools.make_visit_memo(history_text)
    except Exception:
        return None


def emergency(state: State) -> dict:
    """응급 권고: 별표 1 근거를 보여주고, 병원에서 보여 줄 진료 메모를 붙이고, 지역을 묻는다. 다음 턴의 지역 답변을 기다린다."""
    red = state["red"]
    text_in = _history_text(state)
    with ThreadPoolExecutor(max_workers=2) as pool:  # 근거 검색과 메모 생성은 서로 독립이라 동시에 실행
        f_annex, f_memo = pool.submit(retriever.search_annex, text_in), pool.submit(_safe_memo, text_in)
        annex, memo = f_annex.result(), f_memo.result()
    guidance = "응급 신호 의심" + (f" (별표 1: {', '.join(red['matched_items'])})" if red["matched_items"] else "")
    memo_text = answers.compose_memo(memo, guidance)
    text = answers.compose_emergency(red["matched_items"], red["reason"], state.get("notes", []), memo_text)
    trace = state.get("trace", [])
    if not any(s["step"] == "retrieve_guidelines" for s in trace):  # 규칙으로 바로 응급이 된 경우: 응급 답변의 근거(별표 1) 검색을 기록
        trace = trace + [{"step": "retrieve_guidelines", "input": "별표 1(응급증상) 검색", "output": _clip(", ".join(d["doc_id"] for d in annex) or "문서 없음")}]
    trace = trace + [{"step": "make_visit_memo", "input": "대화에서 사용자가 말한 내용 정리", "output": "메모 생성" if memo_text else "메모 생략"}]
    return {"answer": text, "contexts": _contexts(annex), "emergency_active": True, "awaiting_location": True,
            "followup_turns": 0, "last_kind": "emergency", "trace": trace,
            "consult": memory.build_record("urgent", memo, [], text_in)}


def _safe_dept(history_text: str, docs: list[dict]):
    """도구 recommend_department 를 호출하되, 실패해도 응급실 권유는 막지 않는다(진료과만 생략)."""
    try:
        return tools.recommend_department(history_text, docs)
    except Exception:
        return None


def advise_er(state: State) -> dict:
    """응급실 방문 권유(level=advise): 응급일 수 있는 경우. 부드러운 어조로 응급실을 권하고 진료과 후보를 함께 안내한다.

    결론(응급실 방문)은 긴급 안내와 같고 어조만 다르다. 별표 1 근거 검색, 진료과 추천, 진료 메모를 동시에 실행한다.
    지역 질문과 이후 대화(emergency_followup)는 긴급 안내와 같은 흐름을 탄다.
    """
    red, docs = state["red"], state["docs"]
    text_in = _history_text(state)
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_annex = pool.submit(retriever.search_annex, text_in)
        f_dept, f_memo = pool.submit(_safe_dept, text_in, docs), pool.submit(_safe_memo, text_in)
        annex, dept, memo = f_annex.result(), f_dept.result(), f_memo.result()
    depts = dept.departments if dept else []
    guidance = "응급실 방문 권유" + (f" (별표 1: {', '.join(red['matched_items'])})" if red["matched_items"] else "")
    if depts:
        guidance += "; 진료과 후보: " + ", ".join(depts)
    memo_text = answers.compose_memo(memo, guidance)
    text = answers.compose_advise(red["matched_items"], red["reason"], depts, dept.reason if dept else "",
                                  dept.watch_signs if dept else [], state.get("notes", []), memo_text)
    trace = state.get("trace", []) + [
        {"step": "recommend_department", "input": "증상과 관련 문서(응급실 권유와 함께 안내)", "output": _clip(f"{depts}: {dept.reason}" if dept else "생략")},
        {"step": "make_visit_memo", "input": "대화에서 사용자가 말한 내용 정리", "output": "메모 생성" if memo_text else "메모 생략"},
    ]
    return {"answer": text, "contexts": _contexts(annex) + _contexts(docs), "emergency_active": True, "awaiting_location": True,
            "followup_turns": 0, "last_kind": "emergency", "trace": trace,
            "consult": memory.build_record("advise", memo, depts, text_in)}


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
    """도구 find_related_conditions, recommend_department, make_visit_memo 를 동시에 실행해 비응급 안내를 만든다."""
    text, docs = _history_text(state), state["docs"]
    with ThreadPoolExecutor(max_workers=3) as pool:  # 세 LLM 호출은 서로 독립이라 병렬로
        f_cond = pool.submit(tools.find_related_conditions, text, docs)
        f_dept = pool.submit(tools.recommend_department, text, docs)
        f_memo = pool.submit(_safe_memo, text)
        conditions, dept, memo = f_cond.result(), f_dept.result(), f_memo.result()
    guidance = "진료과 후보: " + ", ".join(dept.departments)  # '응급 아님'이라고 단정하는 문구는 의사가 보는 메모에 넣지 않는다
    memo_text = answers.compose_memo(memo, guidance)
    answer = answers.compose_recommendation(
        conditions, dept.departments, dept.reason, dept.watch_signs,
        state["scope"].get("conservative", False), state.get("notes", []), memo_text,
    )
    trace = _step(state, "find_related_conditions", "근거 문서에서 후보 추출", ", ".join(c["name_ko"] for c in conditions) or "없음")
    trace = trace + [
        {"step": "recommend_department", "input": "증상과 관련 상태", "output": _clip(f"{dept.departments}: {dept.reason}")},
        {"step": "make_visit_memo", "input": "대화에서 사용자가 말한 내용 정리", "output": "메모 생성" if memo_text else "메모 생략"},
    ]
    return {"answer": answer, "trace": trace, "followup_turns": 0, "last_kind": "answer",
            "consult": memory.build_record("recommend", memo, dept.departments, text)}


def save_memory(state: State) -> dict:
    """장기 메모리(패턴 10): 끝난 상담의 요약(증상·기간·안내 결과)을 사용자별로 저장한다. 로그인하지 않았으면 건너뛴다.

    저장 실패가 답변을 막지 않는다. 같은 대화는 한 건으로 덮어쓰므로 대화가 이어지면 최신 결론이 남는다.
    """
    uid, rec = state.get("user_id"), state.get("consult")
    if not uid or not rec:
        return {}
    try:
        memory.save_consultation(uid, state.get("session_id") or "default", rec)
        out = f"{memory.LEVEL_LABEL[rec['level']]}: {memory.symptom_text(rec)}"
    except Exception as e:
        out = f"저장 실패({type(e).__name__}): 답변은 그대로 진행"
    return {"trace": _step(state, "save_memory", "상담 요약 저장(장기 메모리)", out)}


def finalize(state: State) -> dict:
    """출력 가드레일: 금지 표현(확률·안심·병명 단정·약 용량)이 든 줄을 지우고 trace 에 기록한다."""
    cleaned, removed = output_guard(state["answer"])
    out = f"제거: {', '.join(removed)}" if removed else "통과"
    return {"answer": cleaned, "trace": _step(state, "output_guardrail", "최종 답변 검사", out)}


# ------------------------------------------------------------------ 그래프 조립
def build_graph():
    """노드와 분기를 연결해 컴파일한다. 체크포인터가 session_id 별 대화 상태를 보관한다."""
    g = StateGraph(State)
    for fn in (begin, classify, checkin, refuse, locate, ask_region, lookup, emergency_followup, triage, emergency, advise_er,
               refuse_injection, assess, recommend, save_memory, finalize):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "begin")
    g.add_edge("begin", "classify")
    g.add_conditional_edges("classify", route_after_classify, ["refuse", "checkin", "locate", "triage"])
    g.add_conditional_edges("locate", route_after_locate, ["lookup", "ask_region", "emergency_followup", "triage"])
    g.add_conditional_edges("ask_region", route_after_ask_region, ["lookup", "begin"])  # HITL 재개 후: 지역이면 조회, 아니면 새 입력
    g.add_edge("lookup", "finalize")
    g.add_edge("emergency_followup", "finalize")
    g.add_conditional_edges("triage", route_after_triage, ["emergency", "advise_er", "refuse_injection", "assess"])
    g.add_conditional_edges("assess", route_after_assess, ["finalize", "recommend"])
    for node in ("checkin", "refuse", "refuse_injection"):
        g.add_edge(node, "finalize")
    for node in ("emergency", "advise_er", "recommend"):  # 상담이 끝난 답변은 요약을 저장한 뒤 마무리
        g.add_edge(node, "save_memory")
    g.add_edge("save_memory", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=InMemorySaver())


_GRAPH = None
_GRAPH_LOCK = threading.Lock()  # 그래프(와 그 체크포인터)는 프로세스에 하나만 있어야 세션이 유지된다


def run_agent(question: str, session_id: str | None = None, user_id: str | None = None) -> dict:
    """질문 하나를 처리해 {answer, contexts, trace} 를 돌려준다. session_id 가 같으면 같은 대화로 이어진다.

    user_id 가 있으면(로그인한 사용자) 새 대화 첫 턴에 지난 상담을 불러오고, 상담이 끝나면 요약을 저장한다(장기 메모리).
    """
    global _GRAPH
    if _GRAPH is None:
        with _GRAPH_LOCK:
            if _GRAPH is None:
                _GRAPH = build_graph()
    thread = session_id or uuid.uuid4().hex  # session_id 가 없으면 매번 새 대화
    config = {"configurable": {"thread_id": thread}, "run_name": "hospital-guide-agent", "tags": ["hospital-guide"]}
    # 직전 요청이 ask_region 의 interrupt() 로 멈춘 세션이면 그 지점에서 재개하고, 아니면 새 입력으로 시작한다.
    # (이전 실행이 오류로 중단된 세션은 interrupt 가 아니므로 새 입력으로 다시 시작한다.)
    waiting = any(t.interrupts for t in _GRAPH.get_state(config).tasks)
    payload = {"question": question, "user_id": user_id, "session_id": thread}
    out = _GRAPH.invoke(Command(resume=question) if waiting else payload, config=config)
    paused = out.get("__interrupt__")
    if paused:  # 지역을 기다리며 멈춤: interrupt 에 담긴 질문 문구를 응답으로 돌려주고, trace 에 멈춘 사실을 남긴다
        trace = out.get("trace", []) + [{"step": "human_in_the_loop", "input": "응급의료기관 조회에 필요한 지역이 없음",
                                          "output": "interrupt() 로 대기, 같은 session_id 의 다음 요청이 재개"}]
        return {"answer": paused[0].value["prompt"], "contexts": [], "trace": trace}
    return {"answer": out["answer"], "contexts": out.get("contexts", []), "trace": out.get("trace", [])}
