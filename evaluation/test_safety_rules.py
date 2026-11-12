"""안전망 규칙(결정적 응급 규칙)의 단위 시험. LLM 호출이 없어 토큰을 쓰지 않는다.

실행: python evaluation/test_safety_rules.py   (실패가 있으면 종료 코드 1)
핵심: 규칙은 '사용자가 말한 내용'에만 적용되고, 서비스가 한 질문·답변 문장에는 걸리지 않아야 한다.
(서비스가 "의식이 흐려지나요?"라고 물었는데 사용자가 "아니요"라고 답해도 응급으로 오판하면 안 된다.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools import hard_red_flags  # noqa: E402

ASKED = "서비스 질문: 의식이 흐려지거나 쓰러질 것 같나요? / 피를 토하거나 검은 변이 있나요?"

CASES = [
    # (설명, 대화, 기대: 응급 규칙에 걸려야 하는가)
    ("사용자가 직접 호흡 곤란을 말함", "사용자: 숨이 안 쉬어져요", True),
    ("서비스 질문 문장만 있음(사용자는 아직 답 안 함)", f"사용자: 배가 아파요\n{ASKED}", False),
    ("글로 '아니요 없어요'라고 답함(질문 속 단어에 걸리면 안 됨)", f"사용자: 배가 아파요\n{ASKED}\n사용자: 아니요 둘 다 없어요", False),
    ("서비스 답변(이어가기)에 위험 단어가 있어도 규칙에 안 걸림", "사용자: 좀 쉬면 나아질까요?\n서비스 답변: 숨이 안 쉬어지는 증상은 위험할 수 있어요", False),
    ("글로 직접 '피를 토했어요'", f"사용자: 배가 아파요\n{ASKED}\n사용자: 네 피를 토했어요", True),
    ("글로 직접 '의식을 잃었어요'", f"사용자: 어지러워요\n{ASKED}\n사용자: 갑자기 의식을 잃었어요", True),
]


def main() -> int:
    bad = 0
    for name, history, expected in CASES:
        got = bool(hard_red_flags(history))
        ok = got == expected
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  (기대={expected}, 결과={got})")
    print(f"\n{len(CASES) - bad}/{len(CASES)} 통과")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
