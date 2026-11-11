"""가드레일 (Day 5): 입력 쪽은 개인정보 마스킹과 범위 분류, 출력 쪽은 금지 표현 제거.

LLM이 어긴 표현은 코드가 마지막에 한 번 더 걸러낸다(정책을 프롬프트에만 맡기지 않는다).
"""
import re

from .llm_chain import structured_chain
from .prompts import SCOPE_SYSTEM
from .schemas import ScopeCheck

# --------------------------------------------------------------- 입력: 개인정보 마스킹
# (패턴, 대체 문구, 라벨). 마스킹된 문장만 이후 단계와 trace 에 남는다.
_PII_PATTERNS = [
    (re.compile(r"\d{6}[-\s]?[1-4]\d{6}"), "[주민번호]", "주민번호"),
    (re.compile(r"01[016789][-\s]?\d{3,4}[-\s]?\d{4}"), "[전화번호]", "전화번호"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[이메일]", "이메일"),
    (re.compile(r"(?:\d{4}[-\s]?){3}\d{4}"), "[카드번호]", "카드번호"),
    (re.compile(r"(이름은\s*)[가-힣]{2,4}"), r"\1[이름]", "이름"),
]


def mask_pii(text: str) -> tuple[str, list[str]]:
    """개인정보를 대체 문구로 바꾸고, 마스킹한 항목 라벨 목록을 함께 돌려준다."""
    found: list[str] = []
    for pattern, repl, label in _PII_PATTERNS:
        text, n = pattern.subn(repl, text)
        if n:
            found.append(label)
    return text, found


def check_scope(message: str, last_kind: str) -> ScopeCheck:
    """입력이 서비스 범위 안인지 분류한다 (빠른 모델). last_kind 는 직전 응답 유형으로, 짧은 답변을 증상으로 읽는 데 쓴다."""
    payload = f"[직전 서비스 응답 유형] {last_kind or '없음'}\n[사용자 메시지]\n{message}"
    return structured_chain("scope", ScopeCheck, SCOPE_SYSTEM, fast=True).invoke({"input": payload})


# --------------------------------------------------------------- 출력: 금지 표현 제거
# 확률 수치, 안심 표현, 병명 단정, 약 용량 안내가 들어간 문장을 통째로 지운다.
_FORBIDDEN = [
    (re.compile(r"(확률|가능성)[^.\n]{0,12}\d+\s*(%|퍼센트)|\d+\s*(%|퍼센트)\s*(의\s*)?(확률|가능성)"), "확률 수치"),
    (re.compile(r"걱정\s*(하지\s*)?(않으셔도|마세요|안 하셔도)|괜찮(습니다|아요|으니)|문제\s*없|안심\s*(하세요|해도)|별일\s*아|심각하지\s*않"), "안심 표현"),
    (re.compile(r"[가-힣]{1,8}(증|염|암|병|감기|경색|출혈|졸중)(입니다|이에요|이네요|으로\s*(진단|확진))"), "병명 단정"),
    (re.compile(r"전형적(인|입니다)|틀림없|확실히\s|분명히\s|정확한\s*진단과\s*치료"), "단정 표현"),
    (re.compile(r"\d+\s*(mg|밀리그램|ml|mL)[^.\n]{0,10}(복용|드세요|먹)|(복용|드세요|먹)[^.\n]{0,10}\d+\s*(mg|밀리그램|ml|mL)"), "약 용량 안내"),
]

DISCLAIMER = "※ 참고용 안내이며 근거가 확실하지 않을 수 있습니다. 정확한 판단은 의사에게 받으세요."
_FALLBACK = "정확한 판단을 위해 의사의 진료를 받아 보세요. 응급이 의심되면 즉시 119에 연락하세요.\n\n" + DISCLAIMER


def output_guard(text: str) -> tuple[str, list[str]]:
    """금지 표현이 든 줄을 지우고, 지운 사유 목록을 돌려준다. 남는 내용이 없으면 안전한 기본 문구로 바꾼다."""
    removed: list[str] = []
    kept_lines = []
    for line in text.split("\n"):
        hit = next((label for pattern, label in _FORBIDDEN if pattern.search(line)), None)
        if hit:
            removed.append(hit)
        else:
            kept_lines.append(line)
    cleaned = "\n".join(kept_lines).strip()
    if not cleaned:
        return _FALLBACK, removed
    return cleaned, removed
