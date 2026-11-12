"""최종 답변 문장 만들기. 고정 문구(고지·거절·안전망)는 LLM에 맡기지 않고 여기서 코드로 붙인다."""
from .guardrails import DISCLAIMER

# 거절 문구 (범위 밖 / 제공하지 않는 정보 / 프롬프트 인젝션)
REFUSE_OUT_OF_SCOPE = (
    "저는 증상을 듣고 응급실에 가야 할지, 외래라면 어느 진료과가 좋을지 안내하는 서비스예요. "
    "건강과 관련 없는 질문에는 답해 드리기 어려워요.\n\n불편한 증상이 있으시면 말씀해 주세요."
)
REFUSE_UNAVAILABLE = (
    "요청하신 정보(실시간 병상·대기시간, 진료·검사 비용, 특정 병원이나 의사 추천 등)는 제공하지 않아요. "
    "병원에 직접 전화하거나 건강보험 관련 기관에 문의해 주세요. 응급 상황이면 바로 119에 연락하세요.\n\n"
    "증상을 말씀해 주시면 응급실이 필요한지, 어느 진료과가 좋을지는 안내해 드릴 수 있어요."
)
REFUSE_INJECTION = (
    "지시를 바꾸거나 내부 설정을 알려 드릴 수는 없어요. 안전을 위해 응급 신호가 있으면 언제나 응급 안내를 먼저 드려요.\n\n"
    "불편한 증상이 있으시면 말씀해 주세요. " + "응급이 의심되면 즉시 119에 연락하세요."
)
INJECTION_NOTE = "참고로 안내 방식을 바꿔 달라는 요청은 따를 수 없어요."
DOSAGE_NOTE = "약의 종류나 복용량은 안내해 드릴 수 없어요. 약은 의사나 약사와 상담해 주세요."
DIAGNOSIS_NOTE = "병명을 확정하거나 확률로 말씀드릴 수는 없어요."


CHECKIN_REPLY = (
    "말씀해 주셔서 고마워요. 증상이 다시 생기거나 새로 불편한 곳이 있으면 언제든 말씀해 주세요. "
    "응급이 의심되면 바로 119에 연락하세요.\n\n" + DISCLAIMER
)


def compose_memo(memo, guidance: str) -> str:
    """진료 메모 문단. 사용자가 말한 내용만 항목별로 적고, 마지막에 서비스의 참고용 안내 내용을 덧붙인다.

    UI 는 '**진료 메모'로 시작하는 문단을 찾아 복사 버튼이 있는 카드로 보여준다. 주 증상이 없으면 빈 문자열(메모 생략).
    """
    if memo is None or not memo.chief_complaints:
        return ""
    lines = ["**진료 메모 (병원에서 의사에게 보여 주세요)**", "- 주 증상: " + ", ".join(memo.chief_complaints)]
    if memo.onset_duration:
        lines.append(f"- 시작·기간: {memo.onset_duration}")
    if memo.related_symptoms:
        lines.append("- 함께 있는 증상: " + ", ".join(memo.related_symptoms))
    if memo.denied_symptoms:
        lines.append("- 없다고 한 증상: " + ", ".join(memo.denied_symptoms))
    if memo.background:
        lines.append("- 배경(나이·임신·기저질환·복용 약): " + ", ".join(memo.background))
    lines.append(f"- 서비스 안내(참고용): {guidance}")
    lines.append("- 이 메모는 사용자가 말한 내용만 정리한 것이며 진단이 아닙니다.")
    return "\n".join(lines)


def compose_emergency(matched: list[str], reason: str, notes: list[str], memo_text: str = "") -> str:
    """응급 권고. 첫 문단이 핵심 메시지(UI가 크게 보여준다). 별표 1 근거와 지역 질문을 덧붙인다."""
    parts = ["🚨 응급 신호가 의심됩니다. 지금 바로 119에 연락하거나 가까운 응급실을 이용하세요."]
    basis = ""
    # 한 항목이 다른 항목에 포함되면('호흡곤란' / '급성 호흡곤란') 겹치므로 짧은 쪽을 뺀다
    matched = [m for m in matched if not any(m != o and m in o for o in matched)]
    if matched:
        basis += "판단 근거(응급의료법 시행규칙 별표 1)\n" + "\n".join(f"- {m}" for m in matched)
    if reason:
        basis += ("\n" if basis else "") + reason
    if basis:
        parts.append(basis)
    if notes:
        parts.append(" ".join(notes))
    parts.append("지역(시/도, 시/군/구)을 알려주시면 해당 지역의 응급의료기관을 안내해 드릴게요.")
    if memo_text:
        parts.append(memo_text)
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def compose_advise(matched: list[str], reason: str, departments: list[str], dept_reason: str, watch_signs: list[str],
                   notes: list[str], memo_text: str = "") -> str:
    """응급실 방문 권유(응급일 수 있는 경우). 긴급 안내보다 부드러운 어조지만 '응급실 방문'이라는 결론은 같다.

    순서: 응급실 권유 -> 참고한 기준 -> 지역 질문 -> (보조) 진료과 후보 -> 바로 119 신호 -> 메모.
    진료과는 응급실보다 뒤에 둔다(외래로 가 버리지 않도록). UI 는 '⚠️'로 시작하는 답변을 호박색 안내 띠로 보여준다.
    """
    parts = ["⚠️ 응급실 방문을 권해요. 증상이 가볍게 시작됐더라도 빠르게 나빠질 수 있어서, 오늘 안에 확인받는 게 안전해요."]
    matched = [m for m in matched if not any(m != o and m in o for o in matched)]
    basis = ""
    if matched:
        basis += "참고한 기준(응급의료법 시행규칙 별표 1)\n" + "\n".join(f"- {m}" for m in matched)
    if reason:
        basis += ("\n" if basis else "") + reason
    if basis:
        parts.append(basis)
    if notes:
        parts.append(" ".join(notes))
    parts.append("지역(시/도, 시/군/구)을 알려주시면 해당 지역의 응급의료기관을 안내해 드릴게요.")
    if departments:
        parts.append("**진료과 후보 (응급실 진료 뒤, 또는 증상이 가라앉았을 때 참고)**\n- " + " 또는 ".join(departments)
                     + (f"\n{dept_reason}" if dept_reason else ""))
    if watch_signs:
        parts.append("**이런 경우에는 기다리지 말고 바로 119에 연락하세요**\n" + "\n".join(f"- {s}" for s in watch_signs))
    if memo_text:
        parts.append(memo_text)
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def compose_emergency_reply(reply: str, notes: list[str], ask_location: bool) -> str:
    """응급 안내 뒤 이어지는 대화 답변. 안내문을 통째로 반복하지 않고, 아직 지역을 못 받았으면 한 줄만 다시 묻는다."""
    parts = [reply]
    if notes:
        parts.append(" ".join(notes))
    if ask_location:
        parts.append("지역(시/도, 시/군/구)을 알려주시면 가까운 응급의료기관을 안내해 드릴게요.")
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def compose_followup(questions: list[str], notes: list[str]) -> str:
    """되묻기. 질문 1~2개와 '모르겠어요'도 괜찮다는 안내."""
    lines = [" ".join(notes)] if notes else []
    lines.append("정확히 안내드리기 위해 몇 가지 여쭤볼게요.")
    lines.append("\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1)))
    lines.append("모르시면 '모르겠어요'라고 답하셔도 됩니다.")
    return "\n\n".join(lines)


def compose_recommendation(conditions: list[dict], departments: list[str], reason: str,
                           watch_signs: list[str], conservative: bool, notes: list[str], memo_text: str = "") -> str:
    """비응급 안내: 관련 상태(참고) + 진료과 후보 + 응급실 안전망 + 고지."""
    parts = []
    if notes:
        parts.append(" ".join(notes))
    # 응급이 아니라고 단정하지 않는다: 입력된 내용만으로는 뚜렷하지 않다고만 말하고, 아래 안전망으로 이어간다
    parts.append("말씀해 주신 내용만으로는 응급 신호가 뚜렷하게 보이지 않아요.")
    if conditions:
        lines = [f"- {c['name_ko']} ({c['name_en']})" for c in conditions]  # 출처는 본문에 적지 않고 '근거 문서'(contexts)로 제공
        parts.append("**관련될 수 있는 상태(참고)**\n" + "\n".join(lines))
    dept = " 또는 ".join(departments)
    parts.append(f"**진료과 후보**\n- {dept}" + (f"\n{reason}" if reason else ""))
    if watch_signs:
        parts.append("**이런 경우에는 바로 응급실을 이용하세요**\n" + "\n".join(f"- {s}" for s in watch_signs))
    if conservative:
        parts.append("확인하지 못한 정보가 있어 보수적으로 안내드렸어요. 증상이 심해지거나 걱정되면 가까운 의료기관에 문의하거나 응급실을 이용하세요.")
    if memo_text:
        parts.append(memo_text)
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def compose_ask_region() -> str:
    """응급실 목록을 물었는데 지역이 없을 때: 위치는 자동으로 알 수 없으니 지역을 묻는다."""
    return (
        "가까운 응급의료기관을 찾아 드릴게요. 위치는 자동으로 알 수 없어서, 계신 지역(시/도, 시/군/구)을 알려 주세요. "
        "예: 서울 종로구, 경기도 수원시 영통구\n\n"
        "병상이나 대기 정보는 제공하지 않아요. 응급 상황이면 기다리지 말고 바로 119에 연락하세요.\n\n"
        + DISCLAIMER
    )


def compose_rooms(sido: str, sigungu: str | None, rooms: list[dict]) -> str:
    """응급의료기관 목록 안내(병상 정보는 제공하지 않는다)."""
    place = f"{sido} {sigungu}".strip() if sigungu else sido
    if not rooms:
        return (f"{place}에서 조회된 응급의료기관이 없어요. 지역 이름을 다시 알려 주시거나, 급하면 바로 119에 연락하세요.\n\n"
                + DISCLAIMER)
    lines = [f"- {r['name']} ({r['kind']}) / {r['address']} / 전화 {r['tel']}" for r in rooms]
    return (f"{place}의 응급의료기관이에요.\n\n" + "\n".join(lines)
            + "\n\n병상이나 대기 정보는 제공하지 않아요. 방문 전에 전화로 확인하시고, 급하면 119가 가장 빠르고 안전해요.\n\n" + DISCLAIMER)
