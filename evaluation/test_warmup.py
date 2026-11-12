"""서버 시작 워밍업 시험. LLM·네트워크 호출이 없어 토큰을 쓰지 않는다(무거운 단계는 가짜로 바꾼다).

실행: python evaluation/test_warmup.py   (실패가 있으면 종료 코드 1)
  1) 모델 클라이언트는 (모델, 재시도 횟수) 마다 하나를 만들어 모든 체인이 공유한다
  2) 워밍업은 그래프·별표 1·메모리 저장소·색인+임베딩·빠른 모델·메인 모델을 한 번씩 부르고, 마지막에 실제 처리 경로(run_agent)를 한 번 돌린다
  3) 한 단계가 실패해도 나머지는 계속되고 예외가 밖으로 나오지 않는다 (서비스에 영향 없음)
  4) start() 는 백그라운드 스레드로 돌고 끝난다
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import agent, annex, config, llm, memory, retriever, warmup  # noqa: E402


class FakeModel:
    def __init__(self, calls, name):
        self.calls, self.name = calls, name

    def invoke(self, text):
        self.calls.append(self.name)


def main() -> int:
    bad = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal bad
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    mid = config.FAST_MODELS[0]
    a, b, c = llm.build_model(mid), llm.build_model(mid), llm.build_model(mid, attempts=1)
    check("같은 모델은 하나를 공유하고, 재시도 횟수가 다르면 따로 만든다", a is b and a is not c)

    calls: list[str] = []
    agent.get_graph = lambda: calls.append("graph")
    annex.annex_prompt_text = lambda: calls.append("annex")
    memory.get_store = lambda: calls.append("memory")
    retriever.search_medlineplus = lambda q: calls.append("index")
    agent.run_agent = lambda q, sid=None, uid=None: calls.append("pipeline")
    llm.build_model = lambda model_id, attempts=None: FakeModel(calls, "fast" if model_id == config.FAST_MODELS[0] else "main")

    warmup.warm()
    check("그래프·별표 1·저장소·색인·빠른 모델·메인 모델·실제 처리 경로를 한 번씩 부름", sorted(calls) == sorted(["graph", "annex", "memory", "index", "fast", "main", "pipeline"]), str(sorted(calls)))

    calls.clear()

    def boom():
        raise RuntimeError("시험용 오류")

    annex.annex_prompt_text = boom
    raised = False
    try:
        warmup.warm()
    except Exception:
        raised = True
    check("한 단계가 실패해도 예외가 나오지 않고 나머지는 계속됨", not raised and "graph" in calls and "main" in calls, str(sorted(calls)))

    calls.clear()
    annex.annex_prompt_text = lambda: calls.append("annex")
    th = warmup.start()
    th.join(timeout=10)
    check("start() 는 별도 스레드로 돌고 끝남", isinstance(th, threading.Thread) and th.daemon and not th.is_alive() and "graph" in calls)

    print(f"\n{4 - bad}/4 통과")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
