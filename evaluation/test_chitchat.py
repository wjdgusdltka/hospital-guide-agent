"""인사·감사·경과 보고 응답 시험. LLM 호출이 없어 토큰을 쓰지 않는다(분류기와 답변 모델은 가짜로 바꾼다).

실행: python evaluation/test_chitchat.py   (실패가 있으면 종료 코드 1)
  1) 인사·감사로 분류된 짧은 말은 점검 없이 짧게 맞받는 노드로 간다
  2) 지난 상담 뒤 경과 보고('응급실 가서 꿰매고 왔어요, 고마워요')는 안부 답(checkin) 노드로 간다 — '응급실' 같은 말에 걸려 점검으로 돌아가지 않는다
  3) '지금' 증상이 섞였거나 너무 길면 분류가 인사여도 정상 점검(triage)으로 보낸다 (안전망)
  4) 답변 = 모델이 만든 맞장구 + 코드가 붙이는 안전 문구(119·재방문 안내). 모델이 실패하면 고정 문구로 대신한다
  5) 실제 그래프 실행: 응급 점검·검색 단계가 trace 에 없다
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import agent, answers, tools  # noqa: E402
from src.guardrails import output_guard  # noqa: E402
from src.schemas import ScopeCheck  # noqa: E402

state = {"category": "chitchat", "social_fail": False}
agent.check_scope = lambda message, last_kind: ScopeCheck(category=state["category"], wants_dosage=False, wants_diagnosis=False)


def fake_social(history_text, message):
    if state["social_fail"]:
        raise RuntimeError("시험용 오류")
    return "응급실에서 잘 처치받고 오셨다니 다행이에요."


tools.reply_social = fake_social


def route(question: str, cat: str = "chitchat", last_kind: str = "answer") -> str:
    return agent.route_after_classify({"scope": {"category": cat}, "question": question, "last_kind": last_kind})


def main() -> int:
    bad = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal bad
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    check("'고마워!' 는 인사 노드로", route("고마워!") == "chitchat")
    report = "저번에 바로 응급실 가라고 해서 잘 꿰매고 왔어 고마워 ㅠ"
    check("경과 보고는 안부 답 노드로(응급실·꿰매 같은 말에 걸리지 않음)", route(report, "checkin_ok", "greeting") == "checkin")
    check("안부 인사 상태가 아닌 경과 보고는 인사 노드로", route(report, "checkin_ok", "answer") == "chitchat")
    check("경과 보고라도 지금 증상이 있으면 점검으로", route("잘 꿰맸는데 다시 열이 나요", "checkin_ok", "greeting") == "triage"
          and route("고마워요 근데 아직도 아파요", "checkin_ok", "greeting") == "triage")
    check("증상이 섞인 감사 인사도 점검으로", route("고마워요, 그런데 열이 나요") == "triage" and route("감사합니다 배가 아파요") == "triage")
    check("누가 봐도 응급인 표현이 섞이면 점검으로", route("가슴이 쥐어짜듯 아프고 식은땀이 나요. 고마워요") == "triage")
    check("증상 없는 긴 감사 인사는 인사로", route("정말 고마워요 덕분에 많은 도움이 됐어요") == "chitchat")
    check("너무 긴 말은 점검으로", route("고마워요 " + "그런데 어제부터 계속 이상한 느낌이 들어서 하는 말인데 " * 2) == "triage")

    reply = answers.compose_social("다행이에요.", "answer")
    reply_er = answers.compose_social("다행이에요.", "emergency")
    check("맞장구 뒤에 안전 문구(119·재방문)를 코드가 붙임", reply.startswith("다행이에요.") and "119" in reply and "다시 말씀" in reply, reply[-30:])
    check("응급 안내 직후에는 응급실 이용을 상기시킴", "응급실 이용" in reply_er and "119" in reply_er, reply_er[-40:])
    hello_reply = answers.compose_social("안녕하세요.", "", "안녕하세요")
    check("첫 인사 뒤에는 '다시 생기면'이 아니라 증상을 말해 달라는 문구", "불편한 증상이 있으시면" in hello_reply and "다시 생기거나" not in hello_reply, hello_reply[-40:])
    canned = [answers.compose_chitchat(m, k) for m, k in (("고마워!", "answer"), ("이만 나갈게요", "answer"), ("안녕하세요", ""), ("고마워요", "emergency"), ("네 알겠어요", "answer"))]
    check("고정 문구·조합 문구 모두 금지 표현(안심·단정)이 없음", all(output_guard(t)[1] == [] for t in canned + [reply, reply_er]))

    state["category"] = "checkin_ok"
    r = agent.run_agent(report, "chit-greet")
    steps = [t["step"] for t in r["trace"]]
    check("실제 실행(경과 보고): 점검 없이 맞장구 + 안전 문구",
          "check_red_flags" not in steps and "retrieve_guidelines" not in steps and "잘 처치받고 오셨다니" in r["answer"] and "119" in r["answer"], str(steps))

    state["category"], state["social_fail"] = "chitchat", True
    r = agent.run_agent("고마워!", "chit-fail")
    check("답변 모델이 실패해도 고정 문구로 정상 답변", "도움이 되었다니" in r["answer"], r["answer"][:30])

    print(f"\n{14 - bad}/14 통과")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
