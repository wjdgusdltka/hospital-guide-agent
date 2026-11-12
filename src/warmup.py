"""서버 시작 직후 백그라운드로 '첫 요청의 초기화 비용'을 미리 치른다.

처음 쓰는 것들은 첫 요청 때 한꺼번에 만들어져서(그래프 조립, 색인 열기, 별표 1 파싱, 모델 클라이언트와 연결, 임베딩 클라이언트)
서버를 켠 직후 첫 질문이 이후보다 몇 초 더 걸렸다. 여기서 미리 해 두면 첫 사용자가 그 비용을 치르지 않는다.
(측정, 같은 질문 3번: 워밍업 없이 첫 질문 9.6~11.9초 / 연결만 데우면 8.4~8.9초 / 실제 처리 경로까지 데우면 6.1~6.3초로 이후 질문과 같음.
 단, 질문 시간이 6초와 10~17초로 오락가락한 주된 원인은 초기화가 아니라 us.* Sonnet 스로틀링이었다 — config.py 의 모델 순서 주석 참고.)

- 마지막에 실제 처리 경로를 한 번 통째로 돌려(가짜 세션, 개인 이름 없음이라 이력은 저장되지 않음) 각 단계의 체인과 첫 호출 비용까지 치른다.
  (연결만 데웠을 때는 첫 질문이 이후보다 약 2.5초 느렸다.) 토큰은 질문 1건 분량이다.
- 백그라운드 스레드라서 서버는 바로 요청을 받는다(데우는 중에 온 요청은 평소처럼 처리된다).
- 연결용 모델 호출은 한 단어짜리 아주 작은 호출이다. 한도 오류 등으로 어느 단계가 실패해도 서버와 나머지 단계에는 영향이 없다.
- 모델 클라이언트는 llm.build_model 이 모델별로 공유하므로, 여기서 연결한 클라이언트를 모든 체인이 그대로 쓴다.
- WARMUP=0 으로 끈다.
"""
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("uvicorn.error")  # 서버 로그에 소요 시간이 보이도록 uvicorn 로거를 쓴다


def _step(name: str, fn) -> float:
    t0 = time.time()
    try:
        fn()
        took = time.time() - t0
        log.info("warmup %-22s %.1fs", name, took)
        return took
    except Exception as e:  # noqa: BLE001 (미리 데우기는 실패해도 서비스에 영향이 없어야 한다)
        log.warning("warmup %s skipped (%s)", name, type(e).__name__)
        return -1.0


def warm() -> None:
    """초기화를 미리 치른다. 순서와 무관한 것들은 병렬로 한다."""
    from . import agent, annex, config, llm, memory, retriever

    t0 = time.time()
    ping = "한 단어로만 답해: 안녕"
    jobs = [
        ("graph", agent.get_graph),
        ("annex", annex.annex_prompt_text),
        ("memory store", memory.get_store),
        ("index + embedding", lambda: retriever.search_medlineplus(["fever"])),
        ("fast model", lambda: llm.build_model(config.FAST_MODELS[0]).invoke(ping)),
        ("main model", lambda: llm.build_model(config.MODELS[0]).invoke(ping)),
    ]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        list(pool.map(lambda j: _step(*j), jobs))
    # 위에서 연결·색인·그래프를 데웠으니, 실제 질문 한 번으로 각 단계의 체인(범위 분류·점검·검색·되묻기)까지 데운다
    _step("pipeline dry run", lambda: agent.run_agent("배가 아파요", "warmup-" + uuid.uuid4().hex[:8]))
    log.info("warmup done in %.1fs", time.time() - t0)


def start() -> threading.Thread:
    """백그라운드 스레드로 warm() 을 시작한다."""
    th = threading.Thread(target=warm, daemon=True, name="warmup")
    th.start()
    return th
