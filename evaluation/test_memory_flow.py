"""장기 메모리 흐름 통합 시험 (실행 중인 서버 필요, LLM 호출 있음).

실행: python evaluation/test_memory_flow.py [--base http://localhost:8000]
확인: 저장 -> 이력·인사 조회 -> 새 대화에서 불러오기 -> 안부 답(checkin) -> 비로그인은 저장 안 함 -> 삭제.
시험용 이름으로만 쓰고 끝나면 그 이름의 이력을 지운다.
"""
import argparse
import json
import sys
import urllib.request
import uuid

RUN = uuid.uuid4().hex[:6]
NAME = "시험사용자-" + RUN  # 이름과 세션 id 를 실행마다 새로 만든다(서버 메모리의 이전 대화를 재사용하지 않도록)


def post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))


def steps(r: dict) -> list[str]:
    return [t["step"] for t in r["trace"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    base = ap.parse_args().base
    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    try:
        h = post(base, "/memory/history", {"name": NAME})
        check("처음에는 이력·인사가 없음", h["consultations"] == [] and h["greeting"] is None)

        a = post(base, "/query", {"question": "3일 전부터 목이 따갑고 콧물이 나요. 열은 없고 숨쉬는 건 괜찮아요.",
                                  "session_id": f"m-a-{RUN}", "user_name": NAME})
        check("상담이 끝나면 save_memory 가 기록됨", "save_memory" in steps(a), str(steps(a)))
        h = post(base, "/memory/history", {"name": NAME})
        check("이력 1건과 안부 인사", len(h["consultations"]) == 1 and bool(h["greeting"]), str(h["greeting"]))

        b = post(base, "/query", {"question": "많이 나아졌어요", "session_id": f"m-b-{RUN}", "user_name": NAME})
        check("새 대화 첫 턴에 load_memory, 안부 답은 checkin_reply", "load_memory" in steps(b) and "checkin_reply" in steps(b), str(steps(b)))

        c = post(base, "/query", {"question": "아직도 목이 아프고 더 심해졌어요", "session_id": f"m-c-{RUN}", "user_name": NAME})
        check("악화는 일반 상담으로 이어지고 과거 기록이 응급 오판을 만들지 않음",
              "load_memory" in steps(c) and "checkin_reply" not in steps(c) and not c["answer"].startswith("🚨"), str(steps(c)))

        d = post(base, "/query", {"question": "3일 전부터 목이 따갑고 콧물이 나요. 열은 없어요.", "session_id": f"m-d-{RUN}"})
        check("user_name 없이도 동작하고 저장하지 않음(기존 API 계약)", "save_memory" not in steps(d) and "load_memory" not in steps(d))

        post(base, "/memory/clear", {"name": NAME})
        h = post(base, "/memory/history", {"name": NAME})
        check("삭제 후 이력이 비어 있음", h["consultations"] == [])
    finally:
        try:
            post(base, "/memory/clear", {"name": NAME})
        except Exception:
            pass
    print(f"\n{sum(results)}/{len(results)} 통과")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
