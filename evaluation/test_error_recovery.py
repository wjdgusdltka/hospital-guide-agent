"""오류 복구 시험: 답변 도중 오류가 나도 대화가 끊기지 않고 이어지는지 확인한다. LLM 호출이 없어 토큰을 쓰지 않는다.

실행: python evaluation/test_error_recovery.py   (실패가 있으면 종료 코드 1)
LLM 단계는 가짜로 바꾸고 '응급 점검(check_red_flags)이 오류를 내는' 상황을 만든다.
  1) 일시적 오류 1번: 자동으로 한 번 다시 시도해 정상 답변이 나온다
  2) 오류가 계속되면 호출자에게 예외가 전달된다(서버는 이를 안전 안내로 바꾼다)
  3) 그 뒤 '다시 답변을 주세요'를 보내면 새 질문이 아니라 끊긴 지점부터 이어서 처리한다(범위 분류를 다시 하지 않는다)
  4) 그 뒤 새 문장(증상 내용)을 보내도 오류 직전의 대화 기록이 함께 쓰인다
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import agent, retriever, tools  # noqa: E402
from src.schemas import Assessment, RedFlagResult, ScopeCheck  # noqa: E402

calls = {"scope": 0, "red": 0, "fail_times": 0}


def fake_scope(message, last_kind):
    calls["scope"] += 1
    return ScopeCheck(category="symptom", wants_dosage=False, wants_diagnosis=False)


def fake_red(history_text):
    calls["red"] += 1
    if calls["fail_times"] > 0:
        calls["fail_times"] -= 1
        raise RuntimeError("일시적 오류(시험용)")
    return RedFlagResult(emergency=False, level="none", matched_items=[], reason="")


agent.check_scope = fake_scope
tools.check_red_flags = fake_red
retriever.translate_query = lambda text: ["abdominal pain"]
retriever.search_medlineplus = lambda queries: []
tools.ask_followup = lambda history_text, docs: Assessment(enough_info=False, questions=["언제부터 아팠나요?"])


def main() -> int:
    bad = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal bad
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    # 1) 일시적 오류 1번 -> 자동 재시도로 정상 답변
    calls.update(red=0, scope=0, fail_times=1)
    r = agent.run_agent("배가 아파요", "rec-1")
    check("일시적 오류 1번은 자동 재시도로 답변", "여쭤볼게요" in r["answer"] and calls["red"] == 2, f"red 호출 {calls['red']}번")

    # 2) 오류가 계속되면 예외 전달
    calls.update(red=0, scope=0, fail_times=2)
    raised = False
    try:
        agent.run_agent("배가 아파요", "rec-2")
    except RuntimeError:
        raised = True
    check("오류가 계속되면 예외가 호출자에게 전달됨(서버가 안전 안내로 바꿈)", raised)

    # 3) 재시도 요청 -> 끊긴 지점부터 이어서 처리(범위 분류를 다시 하지 않음)
    scope_before = calls["scope"]
    r = agent.run_agent("다시 답변을 주세요", "rec-2")
    hist = agent._GRAPH.get_state({"configurable": {"thread_id": "rec-2"}}).values["history"]
    check("'다시 답변을 주세요'는 끊긴 질문을 이어서 답함", "여쭤볼게요" in r["answer"] and calls["scope"] == scope_before, str(hist))
    check("그 질문(배가 아파요)이 대화 기록에 있고 재시도 문장은 기록되지 않음",
          any("배가 아파요" in h for h in hist) and not any("다시 답변" in h for h in hist))

    # 4) 오류 뒤 새 문장을 보내도 오류 직전의 대화가 이어진다
    calls.update(red=0, scope=0, fail_times=2)
    try:
        agent.run_agent("배가 아파요", "rec-3")
    except RuntimeError:
        pass
    calls["fail_times"] = 0
    agent.run_agent("아래쪽, 아래 배에요", "rec-3")
    hist = agent._GRAPH.get_state({"configurable": {"thread_id": "rec-3"}}).values["history"]
    check("새 문장을 보내도 오류 직전 질문이 대화 기록에 남아 있음",
          any("배가 아파요" in h for h in hist) and any("아래쪽" in h for h in hist), str(hist))

    print(f"\n{5 - bad}/5 통과")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
