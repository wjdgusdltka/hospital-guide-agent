"""도메인 도구 (Day 3~4). 각 함수가 SERVICE.md 의 도구 하나이며, trace 의 step 이름과 같다.

check_red_flags / ask_followup / find_related_conditions / recommend_department 는 LLM 판단 도구이고,
find_emergency_rooms 는 국립중앙의료원 공공 API 도구다. (검색 도구 retrieve_guidelines 는 retriever.py)
"""
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from . import config
from .llm_chain import structured_chain
from .prompts import (ASSESS_SYSTEM, DEPARTMENT_SYSTEM, DEPARTMENTS, EMERGENCY_REPLY_SYSTEM, LOCATION_SYSTEM,
                      RELATED_SYSTEM, REDFLAG_SYSTEM)
from .schemas import (Assessment, DepartmentRec, EmergencyReply, LocationExtract, RedFlagResult, RelatedConditions)


# ------------------------------------------------------------ check_red_flags
def _load_annex() -> tuple[str, str]:
    """별표 1 파일에서 '응급증상'(A)과 '준하는 증상'(B) 목록을 읽어 프롬프트용 문자열로 만든다."""
    a, b, cur = [], [], None
    for line in config.ANNEX1_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("## A."):
            cur = a
        elif line.startswith("## B."):
            cur = b
        elif line.startswith("### ") and cur is not None:
            cur.append(f"[{line[4:].strip()}]")
        elif line.startswith("- ") and cur is not None:
            cur.append(line)
    return "\n".join(a), "\n".join(b)


# LLM 판단 앞에 두는 안전망: 누가 봐도 응급인 표현은 모델 판단과 무관하게 응급으로 본다.
_HARD_FLAGS = [
    (re.compile(r"의식(이|을)?\s*(없|잃|흐려)|쓰러[졌지져]|정신을\s*잃"), "급성 의식장애"),
    (re.compile(r"피를\s*토|토혈|각혈|피를\s*쏟"), "계속되는 각혈·토혈 등 출혈"),
    (re.compile(r"숨(을|이)?\s*(못|안)\s*쉬|호흡\s*(정지|곤란이\s*심)"), "급성 호흡곤란"),
    (re.compile(r"경련"), "경련성 장애"),
    (re.compile(r"한쪽\s*(팔|다리|얼굴)[^.\n]{0,8}(힘이\s*(빠|없)|마비)|말이\s*(어눌|안\s*나)"), "급성 신경학적 이상"),
    (re.compile(r"심정지|심폐소생"), "심폐소생술이 필요한 증상"),
]


def check_red_flags(history_text: str) -> RedFlagResult:
    """대화 전체에서 응급 신호를 점검한다: 1) 확실한 표현은 규칙으로 즉시 응급 2) 나머지는 LLM이 별표 1에 비춰 판단."""
    for pattern, label in _HARD_FLAGS:
        if pattern.search(history_text):
            return RedFlagResult(emergency=True, matched_items=[label], reason="입력 내용에 응급을 강하게 시사하는 표현이 있어요.")
    annex_a, annex_b = _load_annex()
    system = REDFLAG_SYSTEM.format(annex_a=annex_a, annex_b=annex_b)
    return structured_chain("red_flags", RedFlagResult, system).invoke({"input": history_text})


# --------------------------------------------------------------- ask_followup
def _docs_for_prompt(docs: list[dict], limit: int = 700) -> str:
    """근거 문서를 프롬프트에 넣을 짧은 형태(id, 제목, 앞부분)로 만든다."""
    return "\n\n".join(f"[{d['doc_id']}] {d['title']}\n{d['text'][:limit]}" for d in docs) or "(근거 문서 없음)"


def ask_followup(history_text: str, docs: list[dict]) -> Assessment:
    """정보가 충분한지 판단하고, 부족하면 위험 신호부터 1~2개 질문을 만든다."""
    payload = f"[대화]\n{history_text}\n\n[근거 문서]\n{_docs_for_prompt(docs)}"
    # 가벼운 판단이라 빠른 모델을 쓴다 (응급 판정 check_red_flags 는 안전이 걸려 있어 메인 모델 유지)
    result = structured_chain("assess", Assessment, ASSESS_SYSTEM, fast=True).invoke({"input": payload})
    result.questions = [q.strip() for q in result.questions if q.strip()][:2]
    if not result.questions:  # 질문이 없는데 부족하다고 하면 충분한 것으로 본다
        result.enough_info = True
    return result


# ---------------------------------------------------- find_related_conditions
def find_related_conditions(history_text: str, docs: list[dict]) -> list[dict]:
    """근거 문서에서 증상과 연결되는 '관련될 수 있는 상태' 후보를 뽑는다(확률·순위 없음). 문서에 없는 id는 버린다."""
    by_id = {d["doc_id"]: d for d in docs}
    payload = f"[대화]\n{history_text}\n\n[근거 문서]\n{_docs_for_prompt(docs)}"
    result = structured_chain("related", RelatedConditions, RELATED_SYSTEM, fast=True).invoke({"input": payload})
    out = []
    for c in result.conditions[:3]:
        doc = by_id.get(c.source_doc_id)
        if doc:  # 실제 검색된 문서에 근거한 항목만 인정
            out.append({"name_ko": c.name_ko, "name_en": c.name_en, "source": f'MedlinePlus "{doc["title"]}"', "doc_id": doc["doc_id"]})
    return out


# ------------------------------------------------------ recommend_department
def recommend_department(history_text: str, docs: list[dict]) -> DepartmentRec:
    """진료과 후보 1~2개와 이유, '바로 응급실' 신호를 제안한다. 허용 목록 밖의 과는 걸러낸다."""
    system = DEPARTMENT_SYSTEM.format(departments=", ".join(DEPARTMENTS))
    payload = f"[대화]\n{history_text}\n\n[근거 문서]\n{_docs_for_prompt(docs)}"
    result = structured_chain("department", DepartmentRec, system, fast=True).invoke({"input": payload})
    result.departments = [d for d in result.departments if d in DEPARTMENTS][:2] or ["가정의학과"]
    result.watch_signs = [s.strip() for s in result.watch_signs if s.strip()][:3]
    return result


# ----------------------------------------------------------- emergency_followup
def reply_in_emergency(history_text: str, message: str) -> str:
    """응급 안내 뒤 이어지는 사용자 말에 짧게 답한다 (응급 점검을 다시 돌리지 않고 대화를 이어간다)."""
    payload = f"[대화]\n{history_text}\n\n[사용자의 새 말]\n{message}"
    return structured_chain("emergency_reply", EmergencyReply, EMERGENCY_REPLY_SYSTEM, fast=True).invoke({"input": payload}).reply.strip()


# ------------------------------------------------------ find_emergency_rooms
def extract_location(text: str) -> LocationExtract:
    """사용자 메시지에서 시/도·시/군/구를 뽑는다 (빠른 모델)."""
    return structured_chain("location", LocationExtract, LOCATION_SYSTEM, fast=True).invoke({"input": text})


def find_emergency_rooms(sido: str, sigungu: str | None = None, limit: int = 8) -> list[dict]:
    """국립중앙의료원 '응급의료기관 목록정보 조회'를 호출해 기관명·구분·주소·전화를 돌려준다. 실패하면 빈 목록.

    API는 '수원시 영통구'(띄어쓰기)를 못 찾고 '수원시영통구'나 '영통구'는 찾는다. 그래서 표기를 바꿔가며 결과가 나올 때까지 시도한다.
    """
    if not sigungu:
        return _query_rooms(sido, None, limit)
    words = sigungu.split()
    for candidate in dict.fromkeys([sigungu.replace(" ", ""), words[-1]]):  # 띄어쓰기 제거 -> 마지막 단어 순
        rooms = _query_rooms(sido, candidate, limit)
        if rooms:
            return rooms
    return []


def _query_rooms(sido: str, sigungu: str | None, limit: int) -> list[dict]:
    """API 한 번 호출 (시/도, 시/군/구 문자열을 그대로 사용)."""
    endpoint = os.getenv("GOV_DATA_API_ENDPOINT", "").rstrip("/")
    key = os.getenv("GOV_DATA_API_KEY", "")
    if not endpoint or not key:
        return []
    params = {"serviceKey": key, "Q0": sido, "pageNo": 1, "numOfRows": limit}
    if sigungu:
        params["Q1"] = sigungu
    url = f"{endpoint}/getEgytListInfoInqire?" + urllib.parse.urlencode(params, safe="%")
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            root = ET.fromstring(resp.read())
    except Exception:  # 네트워크·파싱 오류는 '조회 결과 없음'으로 처리 (키가 로그에 남지 않게 메시지는 버린다)
        return []
    return [
        {"name": it.findtext("dutyName", ""), "kind": it.findtext("dutyEmclsName", ""),
         "address": it.findtext("dutyAddr", ""), "tel": it.findtext("dutyTel3", "") or it.findtext("dutyTel1", "")}
        for it in root.iter("item")
    ]
