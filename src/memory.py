"""장기 메모리 (패턴 10): 사용자별 상담 이력을 LangGraph Store 에 보관하고, 다음 방문 때 안부 인사에 쓴다.

- 저장소: LangGraph `SqliteStore` (파일 `data/memory/memory.sqlite`). 서버를 다시 켜도 남는다.
  (수업의 InMemoryStore 와 같은 Store 인터페이스라 put/get/search/delete 를 그대로 쓰고, 파일에 남기만 다르다.)
- 대화 안의 단기 기억(체크포인터, session_id)과 별개로, 이 모듈은 '대화가 끝난 뒤에도 남는 요약'만 다룬다.
- 저장하는 것: 진료 메모에서 가져온 요약(증상·기간·안내 결과)과 진료 메모 원문(이력에서 진료 카드를 다시 보려고),
  그리고 상담이 끝나기 전 '진행 중' 기록(사용자가 쓴 문장 최대 6개, 개인정보는 마스킹된 것). AI 답변 원문은 저장하지 않는다.
- 진행 중 기록은 증상 점검이 시작될 때 바로 저장되고(LLM 호출 없음), 상담이 끝나면 같은 키로 최종 기록이 덮어쓴다.
- 이름은 로그인 같은 인증이 아니다. 같은 이름이면 같은 이력을 본다(학습용). 이름은 LLM 에 보내지 않는다.
"""
import threading
import unicodedata
from datetime import datetime

from . import config

_STORE = None
_LOCK = threading.Lock()

LEVEL_LABEL = {"urgent": "응급실 안내(긴급)", "advise": "응급실 방문 권유", "recommend": "진료과 안내",
               "in_progress": "상담 중 (이어서 가능)"}
MAX_NAME_CHARS = 20
MAX_HISTORY = 50  # 화면에 보여 주는 최근 상담 건수
MAX_STATEMENTS = 6  # 진행 중 상담에 남기는 사용자 발화 수(마스킹된 것만)


def get_store():
    """프로세스에 하나뿐인 SqliteStore 를 처음 쓸 때 만든다 (스레드 안전)."""
    global _STORE
    if _STORE is None:
        with _LOCK:
            if _STORE is None:
                import sqlite3

                from langgraph.store.sqlite import SqliteStore

                config.MEMORY_DB.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(config.MEMORY_DB, check_same_thread=False, isolation_level=None)
                store = SqliteStore(conn)
                store.setup()
                _STORE = store
    return _STORE


# ------------------------------------------------------------------ 사용자 식별
def normalize_name(name: str | None) -> str | None:
    """입력한 이름을 사용자 식별자로 정리한다(공백·대소문자 차이 흡수). 비었거나 너무 길면 None."""
    if not name:
        return None
    s = unicodedata.normalize("NFC", name)
    s = " ".join(s.split()).casefold()
    if not s or len(s) > MAX_NAME_CHARS or any(unicodedata.category(c).startswith("C") for c in s):
        return None
    return s


def _ns(uid: str) -> tuple[str, str]:
    return ("consultations", uid)


# ------------------------------------------------------------------ 기록 만들기 / 저장 / 조회
def _statements(history_text: str) -> list[str]:
    """대화 기록에서 사용자가 한 말만 뽑는다(이미 개인정보가 마스킹된 문장)."""
    return [l[4:].strip()[:120] for l in history_text.split("\n") if l.startswith("사용자:") and l[4:].strip()]


def build_record(level: str, memo, departments: list[str], history_text: str, memo_text: str = "") -> dict:
    """응급 안내·권유·진료과 안내가 끝난 시점의 상담 요약. 진료 메모(사용자가 말한 내용)에서 가져오고, 없으면 첫 발화 앞부분.

    memo_text 는 화면에 보인 '진료 메모' 문단 그대로라서, 이력에서 진료 카드를 다시 열 때 쓴다.
    """
    if memo is not None and memo.chief_complaints:
        symptoms = list(memo.chief_complaints)[:3]
        onset = memo.onset_duration or ""
    else:
        first = next(iter(_statements(history_text)), "")
        symptoms, onset = ([first[:40]] if first else []), ""
    return {"ts": datetime.now().astimezone().isoformat(timespec="seconds"), "level": level,
            "symptoms": symptoms, "onset": onset, "departments": list(departments)[:2], "memo": memo_text or ""}


def build_progress(history_text: str) -> dict | None:
    """상담이 끝나기 전, 사용자가 말한 내용만 모은 '진행 중' 기록(LLM 호출 없음). 말한 내용이 없으면 None."""
    sts = _statements(history_text)
    if not sts:
        return None
    return {"ts": datetime.now().astimezone().isoformat(timespec="seconds"), "level": "in_progress",
            "symptoms": [sts[0][:40]], "onset": "", "departments": [], "memo": "", "statements": sts[-MAX_STATEMENTS:]}


def save_consultation(uid: str, session_id: str, record: dict) -> None:
    """같은 대화(session_id)는 한 건으로 덮어쓴다 — 대화가 이어지면 최신 결론이 남는다."""
    get_store().put(_ns(uid), session_id, record)


def save_progress(uid: str, session_id: str, record: dict) -> bool:
    """진행 중 기록을 저장한다. 이미 결론이 난 기록은 덮어쓰지 않는다(결론이 더 쓸모 있다). 저장했으면 True."""
    cur = get_store().get(_ns(uid), session_id)
    if cur is not None and cur.value.get("level") != "in_progress":
        return False
    get_store().put(_ns(uid), session_id, record)
    return True


def list_consultations(uid: str, limit: int = MAX_HISTORY) -> list[dict]:
    """최신순 상담 요약 목록 (각 항목에 id=session_id 포함)."""
    items = get_store().search(_ns(uid), limit=1000)
    rows = sorted(({**it.value, "id": it.key} for it in items), key=lambda r: r["ts"], reverse=True)
    return rows[:limit]


def latest(uid: str) -> dict | None:
    rows = list_consultations(uid, 1)
    return rows[0] if rows else None


def delete_one(uid: str, session_id: str) -> bool:
    """상담 한 건을 지운다. 있었으면 True."""
    store = get_store()
    existed = store.get(_ns(uid), session_id) is not None
    if existed:
        store.delete(_ns(uid), session_id)
    return existed


def clear(uid: str) -> int:
    """사용자의 상담 이력을 모두 지운다. 지운 건수를 돌려준다."""
    store = get_store()
    items = store.search(_ns(uid), limit=1000)
    for it in items:
        store.delete(_ns(uid), it.key)
    return len(items)


# ------------------------------------------------------------------ 요약·인사 문구 (LLM 없이 코드로 만든다)
def _when(rec: dict) -> str:
    d = datetime.fromisoformat(rec["ts"])
    return f"{d.month}월 {d.day}일"


def symptom_text(rec: dict) -> str:
    s = ", ".join(rec.get("symptoms") or []) or "불편한 증상"
    return f"{s}({rec['onset']})" if rec.get("onset") else s


def summarize(rec: dict) -> dict:
    """화면에 보여 줄 한 건 요약."""
    return {"id": rec.get("id", ""), "date": rec["ts"][:10], "when": _when(rec), "level": rec["level"],
            "level_label": LEVEL_LABEL.get(rec["level"], ""), "symptoms": symptom_text(rec),
            "departments": rec.get("departments") or [], "memo": rec.get("memo") or "",
            "statements": rec.get("statements") or []}


def greeting(display_name: str, rec: dict | None) -> str:
    """첫 화면 인사: 가장 최근 상담을 바탕으로 안부와 확인 질문을 건넨다. 이력이 없으면 짧은 첫 인사."""
    if rec is None:
        return f"{display_name}님, 안녕하세요. 어디가 불편하신가요?"
    head, sym = f"{display_name}님, 안녕하세요. {_when(rec)}에", symptom_text(rec)
    if rec["level"] == "in_progress":
        return f"{head} {sym} 때문에 상담하시다가 안내를 받기 전에 끝났어요. 이어서 도와드릴게요. 그 뒤로 증상이 어떠세요?"
    if rec["level"] in ("urgent", "advise"):
        return f"{head} {sym} 때문에 응급실 방문을 안내드렸어요. 응급실에서 진료는 받으셨나요? 지금은 어떠세요?"
    dept = " 또는 ".join(rec.get("departments") or [])
    return f"{head} {sym} 증상으로 {dept + ' ' if dept else ''}진료를 안내드렸어요. 증상은 좀 나아지셨나요?"


def context_line(rec: dict) -> str:
    """에이전트 대화 기록 맨 앞에 넣는 한 줄. '서비스 질문:' 접두사라 안전망 규칙(사용자 발화에만 적용)에 걸리지 않는다."""
    if rec["level"] == "in_progress":
        said = " / ".join(rec.get("statements") or rec.get("symptoms") or [])
        return (f"서비스 질문: (지난 상담 참고, 지금 증상이 아님) {_when(rec)}에 상담하던 중 안내를 마치기 전에 끝났다. "
                f"그때 말한 내용: {said}. 지금은 어떠신가요?")
    outcome = "응급실 방문을 안내" if rec["level"] in ("urgent", "advise") else "진료과를 안내"
    return f"서비스 질문: (지난 상담 참고, 지금 증상이 아님) {_when(rec)}에 {symptom_text(rec)}로 {outcome}했다. 지금은 어떠신가요?"
