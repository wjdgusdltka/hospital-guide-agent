"""모델 폴백 시험: 하루 한도에서 막 풀린 모델을 재시도 없이 한 번만 찔러 보는지 확인한다. LLM 호출이 없어 토큰을 쓰지 않는다.

실행: python evaluation/test_llm_fallback.py   (실패가 있으면 종료 코드 1)
  1) 처음 한도에 걸리면 평소 재시도 설정으로 부르고, 실패하면 쿨다운에 들어가 다음 모델이 답한다
  2) 쿨다운이 풀린 뒤 첫 호출은 재시도 없는 프로브만 쓰고, 한도가 그대로면 바로 다음 모델로 넘어간다
  3) 프로브가 성공하면(한도가 풀림) '하루 한도' 표시가 지워져 다시 평소 설정으로 돌아간다
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from botocore.exceptions import ClientError  # noqa: E402
from langchain_core.runnables import Runnable  # noqa: E402

from src import llm  # noqa: E402


def daily_error() -> ClientError:
    return ClientError({"Error": {"Code": "ThrottlingException", "Message": "Too many tokens per day, please wait"}}, "Converse")


def rate_error() -> ClientError:  # 일반 스로틀링(하루 한도가 아님)
    return ClientError({"Error": {"Code": "ThrottlingException", "Message": "Too many requests, please wait before trying again."}}, "Converse")


class Fake(Runnable):
    """호출 횟수를 세고, fail=True 면 하루 한도 오류를 낸다."""

    def __init__(self, name, fail=False, err=daily_error):
        self.name, self.fail, self.err, self.calls = name, fail, err, 0

    def invoke(self, input, config=None, **kwargs):  # noqa: A002
        self.calls += 1
        if self.fail:
            raise self.err()
        return f"{self.name} 답변"


def main() -> int:
    bad = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal bad
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    llm._COOLDOWN.clear()
    a_norm, a_probe = Fake("A", fail=True), Fake("A-probe", fail=True)
    b_norm, b_probe = Fake("B"), Fake("B-probe")
    chain = llm.CooldownFallback(["A", "B"], [a_norm, b_norm], [a_probe, b_probe])

    # 1) 처음 한도: 평소 설정으로 부르고, 실패하면 쿨다운 + 다음 모델이 답한다
    out = chain.invoke("q")
    check("처음 한도에 걸린 호출은 평소 설정으로 부르고 다음 모델이 답함", out == "B 답변" and a_norm.calls == 1 and a_probe.calls == 0, f"A={a_norm.calls} probe={a_probe.calls}")
    check("A 가 하루 한도 쿨다운에 들어감(하루 한도 표시 포함)", llm._on_cooldown("A") and llm._needs_probe("A"))

    # 2) 쿨다운이 풀린 뒤 첫 호출: 재시도 없는 프로브만 쓰고, 한도가 그대로면 바로 다음 모델
    until, daily = llm._COOLDOWN["A"]
    llm._COOLDOWN["A"] = (time.time() - 1, daily)  # 쿨다운만 지나간 상태(하루 한도 표시는 남음)
    out = chain.invoke("q")
    check("풀린 뒤 첫 호출은 프로브(재시도 없음)만 쓰고 한도가 그대로면 바로 다음 모델로", out == "B 답변" and a_norm.calls == 1 and a_probe.calls == 1, f"A={a_norm.calls} probe={a_probe.calls}")
    check("프로브가 다시 한도에 걸리면 쿨다운이 다시 걸림", llm._on_cooldown("A"))

    # 3) 프로브가 성공하면 표시가 지워지고 평소 설정으로 돌아간다
    llm._COOLDOWN["A"] = (time.time() - 1, True)
    a_probe.fail = False
    out = chain.invoke("q")
    check("한도가 풀려 프로브가 성공하면 A 가 답함", out == "A-probe 답변", out)
    check("그 뒤 '하루 한도' 표시가 지워짐", not llm._needs_probe("A"))
    a_norm.fail = False
    out = chain.invoke("q")
    check("이후 호출은 평소 설정(재시도 포함)으로 부름", out == "A 답변" and a_probe.calls == 2, f"probe={a_probe.calls}")

    # 4) 일반 스로틀링(하루 한도가 아님)도 같은 규칙: 쿨다운이 풀린 첫 호출은 재시도 없는 프로브만 쓴다
    llm._COOLDOWN.clear()
    c_norm, c_probe = Fake("C", fail=True, err=rate_error), Fake("C-probe", fail=True, err=rate_error)
    chain2 = llm.CooldownFallback(["C", "D"], [c_norm, Fake("D")], [c_probe, Fake("D-probe")])
    chain2.invoke("q")
    check("일반 스로틀링도 쿨다운에 들어가고 하루 한도 표시는 아님", llm._on_cooldown("C") and llm._COOLDOWN["C"][1] is False)
    llm._COOLDOWN["C"] = (time.time() - 1, False)
    out = chain2.invoke("q")
    check("쿨다운이 풀린 뒤 첫 호출은 프로브(재시도 없음)만 쓰고 막혀 있으면 바로 다음 모델로", out == "D 답변" and c_norm.calls == 1 and c_probe.calls == 1, f"C={c_norm.calls} probe={c_probe.calls}")
    llm._COOLDOWN["C"] = (time.time() - 1, False)
    c_probe.fail = False
    check("프로브가 성공하면 표시가 지워져 평소 설정으로 돌아감", chain2.invoke("q") == "C-probe 답변" and not llm._needs_probe("C"))

    # 프로브 목록이 없는 체인(기존 방식)도 그대로 동작
    llm._COOLDOWN.clear()
    plain = llm.CooldownFallback(["A", "B"], [Fake("A"), Fake("B")])
    check("프로브 없이 만든 체인도 정상 동작", plain.invoke("q") == "A 답변")

    llm._COOLDOWN.clear()
    print(f"\n{11 - bad}/11 통과")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
