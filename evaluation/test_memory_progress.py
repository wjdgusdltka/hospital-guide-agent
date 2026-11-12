"""진행 중 상담 저장 + 이력 목록(최대 50건) 시험. LLM 호출이 없어 토큰을 쓰지 않는다. 임시 DB 파일을 쓴다.

실행: python evaluation/test_memory_progress.py   (실패가 있으면 종료 코드 1)
  1) 되묻기까지만 하고 끊겨도 '진행 중' 기록이 저장된다(상담이 끝나길 기다리지 않음)
  2) 응급 점검 단계에서 오류가 나 답변이 안 나와도 진행 중 기록이 남는다
  3) 진행 중 기록은 인사말·대화 맥락 줄에 쓰이고, 최종 결과가 나오면 같은 키로 덮어쓴다
  4) 결론이 난 기록은 진행 중 기록이 덮어쓰지 않는다
  5) 목록은 최신순 최대 50건이고, 요약에 진료 메모 원문이 들어 있다
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import agent, config, memory, retriever, tools  # noqa: E402
from src.schemas import Assessment, RedFlagResult, ScopeCheck  # noqa: E402

config.MEMORY_DB = Path(tempfile.mkdtemp()) / "t.sqlite"

state = {"fail": False}


def fake_red(history_text):
    if state["fail"]:
        raise RuntimeError("시험용 오류")
    return RedFlagResult(emergency=False, level="none", matched_items=[], reason="")


agent.check_scope = lambda message, last_kind: ScopeCheck(category="symptom", wants_dosage=False, wants_diagnosis=False)
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

    # 1) 되묻기에서 끊김
    r = agent.run_agent("배가 아파요", "s1", "길동")
    rec = memory.latest("길동")
    check("되묻기만 하고 끊겨도 진행 중 기록이 저장됨",
          rec is not None and rec["level"] == "in_progress" and rec["statements"] == ["배가 아파요"], str(rec))
    check("trace 에 save_memory 단계가 있음", any(t["step"] == "save_memory" for t in r["trace"]))

    # 2) 점검 단계 오류
    state["fail"] = True
    try:
        agent.run_agent("배가 아파요", "s2", "둘리")
    except RuntimeError:
        pass
    state["fail"] = False
    rec2 = memory.latest("둘리")
    check("점검 단계에서 오류가 나도 진행 중 기록이 남음", rec2 is not None and rec2["level"] == "in_progress", str(rec2))

    # 3) 인사말/맥락 줄, 최종 결과가 덮어씀
    g = memory.greeting("길동", {**rec, "id": "s1"})
    c = memory.context_line(rec)
    check("진행 중 인사말은 '이어서'를 안내", "중간" in g or "끝났어요" in g and "이어서" in g, g)
    check("맥락 줄은 '서비스 질문:' 접두사 + 그때 말한 내용", c.startswith("서비스 질문:") and "배가 아파요" in c, c)
    final = memory.build_record("recommend", None, ["내과"], "사용자: 배가 아파요", "**진료 메모 (병원에서)**\n- 주 증상: 복통")
    memory.save_consultation("길동", "s1", final)
    check("최종 기록이 같은 대화의 진행 중 기록을 덮어씀",
          memory.latest("길동")["level"] == "recommend" and len(memory.list_consultations("길동")) == 1)

    # 4) 결론이 난 기록은 보호
    kept = memory.save_progress("길동", "s1", memory.build_progress("사용자: 다시 배가 아파요"))
    check("결론이 난 기록은 진행 중 기록이 덮어쓰지 않음", kept is False and memory.latest("길동")["level"] == "recommend")

    # 5) 50건 제한 + 최신순 + 진료 메모 원문
    for i in range(55):
        memory.save_consultation("많음", f"x{i}", {**final, "ts": f"2026-01-{1 + i % 28:02d}T10:{i % 60:02d}:00+09:00"})
    rows = memory.list_consultations("많음")
    check("목록은 최대 50건", len(rows) == memory.MAX_HISTORY == 50, str(len(rows)))
    check("최신순 정렬", all(rows[i]["ts"] >= rows[i + 1]["ts"] for i in range(len(rows) - 1)))
    s = memory.summarize(memory.list_consultations("길동")[0])
    check("요약에 진료 메모 원문이 들어 있음", s["memo"].startswith("**진료 메모") and s["level_label"] == "진료과 안내", str(s))

    # 6) 개별 삭제
    gone = memory.delete_one("길동", "s1")
    check("상담 한 건을 지우면 목록에서 사라지고 없는 건은 False", gone and memory.list_consultations("길동") == []
          and memory.delete_one("길동", "s1") is False)

    print(f"\n{11 - bad}/11 통과")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
