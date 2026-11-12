"""Probe every model the account can use and say whether they are at the daily token limit.

Run: python -m src.check_quota
Exit code 2 when ALL models are at the daily limit (time to ask for a quota increase).
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from . import config
from .llm import build_model


def probe(model_id: str) -> tuple[str, str, str]:
    """One tiny call. Returns (model id, status, note); status is OK / DAILY_LIMIT / THROTTLED / ERROR."""
    start = time.time()
    try:
        r = build_model(model_id).invoke("한 단어로만 답해: 안녕")
        retries = r.response_metadata.get("ResponseMetadata", {}).get("RetryAttempts")
        return model_id, "OK", f"{time.time() - start:.1f}s, retries={retries}"
    except Exception as e:  # noqa: BLE001 (report every kind of failure)
        msg = str(e)
        status = "DAILY_LIMIT" if "per day" in msg.lower() else ("THROTTLED" if "Throttl" in msg else "ERROR")
        return model_id, status, msg[-70:]


def main() -> int:
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(probe, config.KNOWN_MODELS))
    for mid, status, note in results:
        print(f"{status:<12} {mid:<52} {note}")
    ok = sum(1 for _, s, _ in results if s == "OK")
    daily = sum(1 for _, s, _ in results if s == "DAILY_LIMIT")
    print(f"\nOK {ok} / DAILY_LIMIT {daily} / total {len(results)}")
    if daily == len(results):
        print("!! 알려진 모든 모델이 하루 토큰 한도에 도달했습니다. 한도 증설을 요청해야 합니다.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
