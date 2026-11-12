"""인사·감사·작별 응답 시험. LLM 호출이 없어 토큰을 쓰지 않는다(분류기는 가짜로 바꾼다).

실행: python evaluation/test_chitchat.py   (실패가 있으면 종료 코드 1)
  1) 인사·감사로 분류된 짧은 말은 점검 없이 짧게 맞받는 노드로 간다
  2) 분류가 인사여도 누가 봐도 응급인 표현이 섞였거나 40자를 넘으면 정상 점검(triage)으로 보낸다 (안전망)
  3) 답변 문구: 감사 / 작별 / 첫 인사 / 응급 안내 직후 / 그 외, 금지 표현(안심·단정)이 없다
  4) 실제 그래프 실행: 응급 점검·검색 단계가 trace 에 없고 chitchat_reply 가 있다
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import agent, answers  # noqa: E402
from src.guardrails import output_guard  # noqa: E402
from src.schemas import ScopeCheck  # noqa: E402

agent.check_scope = lambda message, last_kind: ScopeCheck(category="chitchat", wants_dosage=False, wants_diagnosis=False)


def route(question: str) -> str:
    return agent.route_after_classify({"scope": {"category": "chitchat"}, "question": question})


def main() -> int:
    bad = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal bad
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    check("'고마워!' 는 인사 노드로", route("고마워!") == "chitchat")
    check("응급 표현이 섞이면 인사로 처리하지 않고 점검으로", route("가슴이 쥐어짜듯 아프고 식은땀이 나요. 고마워요") == "triage")
    check("증상이 섞인 감사 인사도 점검으로", route("고마워요, 그런데 열이 나요") == "triage" and route("감사합니다 배가 아파요") == "triage")
    check("증상 없는 긴 감사 인사는 인사로", route("정말 고마워요 덕분에 많은 도움이 됐어요") == "chitchat")
    check("40자를 넘는 긴 말은 점검으로", route("고마워요 " + "그런데 어제부터 계속 이상한 느낌이 들어서 하는 말인데 " * 2) == "triage")

    thanks = answers.compose_chitchat("고마워!", "answer")
    bye = answers.compose_chitchat("이만 나갈게요", "answer")
    hello = answers.compose_chitchat("안녕하세요", "")
    after_er = answers.compose_chitchat("고마워요", "emergency")
    other = answers.compose_chitchat("네 알겠어요", "answer")
    check("감사 답변", "도움이 되었다니" in thanks and "119" in thanks, thanks[:40])
    check("작별 답변", "몸조리" in bye, bye[:40])
    check("첫 인사 답변", hello.startswith("안녕하세요") and "증상" in hello, hello[:40])
    check("응급 안내 직후에는 응급실 이용을 상기시킴", "응급실을 이용" in after_er and "119" in after_er, after_er[:50])
    check("모든 답변에 금지 표현(안심·단정)이 없음", all(output_guard(t)[1] == [] for t in (thanks, bye, hello, after_er, other)))

    r = agent.run_agent("고마워!", "chit-1")
    steps = [t["step"] for t in r["trace"]]
    check("실제 실행: 점검·검색 없이 chitchat_reply", "chitchat_reply" in steps and "check_red_flags" not in steps and "retrieve_guidelines" not in steps and "도움이 되었다니" in r["answer"], str(steps))

    print(f"\n{11 - bad}/11 통과")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
