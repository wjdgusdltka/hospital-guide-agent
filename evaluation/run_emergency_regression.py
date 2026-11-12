"""응급 회귀 테스트: check_red_flags 만 호출해 별표 1 기준의 응급 판정이 유지되는지 확인한다.

- 검색(RAG)과 무관하다. 검색 방식·프롬프트·모델을 바꾼 뒤에도 안전 기준이 같은지 매번 확인하는 용도.
- 케이스는 evaluation/emergency_regression.csv (별표 1 항목 ID 와 기준일이 연결되어 있다).
- 실행: python evaluation/run_emergency_regression.py [--workers 2]
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, tools  # noqa: E402
from src.annex import match_item_ids  # noqa: E402

CASES = ROOT / "evaluation" / "emergency_regression.csv"
OUT = ROOT / "evaluation" / "emergency_regression_last_run.json"


def history_from(text: str) -> str:
    """CSV 의 \\n 표기를 줄바꿈으로 바꾸고, 사용자 발화 접두사가 없으면 붙인다 (에이전트가 쓰는 대화 형식과 같게)."""
    lines = text.replace("\\n", "\n").split("\n")
    return "\n".join(l if l.startswith(("사용자:", "서비스 질문:")) else f"사용자: {l}" for l in lines)


def run_case(row: dict) -> dict:
    """케이스 하나를 check_red_flags 로 판정하고, 인용한 항목을 별표 1 ID 로 바꿔 기대 항목과 비교한다."""
    try:
        res = tools.check_red_flags(history_from(row["input"]))
    except Exception as e:  # 호출 실패는 실패로 기록 (한도·네트워크)
        return {**row, "got_emergency": None, "matched_ids": [], "error": f"{type(e).__name__}: {e}"[:200]}
    return {**row, "got_emergency": res.emergency, "matched_items": res.matched_items,
            "matched_ids": match_item_ids(res.matched_items), "reason": res.reason}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    rows = list(csv.DictReader(CASES.open(encoding="utf-8", newline="")))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_case, rows))

    rows_out, ok_n, gap_n, item_n, item_total = [], 0, 0, 0, 0
    for r in results:
        expected = r["expected_emergency"] == "true"
        passed = r["got_emergency"] == expected
        gap = r.get("known_gap") == "true"
        want_ids = [i for i in r["annex_item_ids"].split(";") if i and i[0] in "AB"]
        item_ok = None
        if expected and want_ids and r["got_emergency"]:  # 응급으로 판정했을 때 기대 항목을 인용했는지(법령 기반 케이스)
            item_ok = any(i in r["matched_ids"] for i in want_ids)
            item_total += 1
            item_n += bool(item_ok)
        if passed:
            ok_n += 1
        elif gap:
            gap_n += 1
        mark = "PASS" if passed else ("GAP " if gap else "FAIL")
        print(f"{mark} {r['id']:<4} [{r['basis']:<8}] 기대={expected!s:<5} 결과={r['got_emergency']!s:<5} "
              f"항목={'-' if item_ok is None else ('일치' if item_ok else '불일치')} {r['matched_ids']}")
        rows_out.append({"id": r["id"], "basis": r["basis"], "expected": expected, "got": r["got_emergency"],
                         "passed": passed, "known_gap": gap, "expected_item_ids": want_ids,
                         "matched_item_ids": r["matched_ids"], "item_match": item_ok, "error": r.get("error")})

    fails = [x["id"] for x in rows_out if not x["passed"] and not x["known_gap"]]
    summary = {"run_date": date.today().isoformat(), "law_version": config.ANNEX1_VERSION, "cases": len(rows_out),
               "passed": ok_n, "known_gaps": gap_n, "failed": fails, "item_match": f"{item_n}/{item_total}"}
    OUT.write_text(json.dumps({"summary": summary, "results": rows_out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
