"""Bedrock chat model factory: boto3 retries first, then model fallback with a cooldown (circuit breaker).

Order of defense for one call:
  1) boto3 retries the same model (adaptive backoff).
  2) If the model still fails, move on to the next model in the list.
  3) A model that failed because of throttling is put on cooldown, so the following calls skip it right away
     instead of spending several seconds of retries on it every time (a daily token limit does not recover soon).
"""
import contextvars
import logging
import threading
import time
from collections import Counter

from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.runnables import Runnable

from . import config

log = logging.getLogger("hospital.llm")

# Errors that should move us on to the next model (after boto3 retries are exhausted).
FALLBACK_EXCEPTIONS = (ClientError, BotoCoreError)

# ------------------------------------------------------------------ circuit breaker state
_COOLDOWN: dict[str, tuple[float, bool]] = {}  # model id -> (skipped until epoch seconds, caused by the daily token limit?)
_LOCK = threading.Lock()


def _on_cooldown(model_id: str) -> bool:
    """Is this model currently being skipped?"""
    with _LOCK:
        return _COOLDOWN.get(model_id, (0, False))[0] > time.time()


def _needs_probe(model_id: str) -> bool:
    """이 모델이 하루 한도로 쿨다운에 들어갔다가 막 풀린 상태인가? 그렇다면 재시도 없이 한 번만 찔러 본다(아래 CooldownFallback)."""
    with _LOCK:
        entry = _COOLDOWN.get(model_id)
        return bool(entry and entry[1])


def _clear_daily(model_id: str) -> None:
    """한도가 풀려 호출에 성공했으면 '하루 한도였던 모델' 표시를 지워 다시 평소 재시도로 돌아간다."""
    with _LOCK:
        entry = _COOLDOWN.get(model_id)
        if entry and entry[1]:
            del _COOLDOWN[model_id]


def _is_throttle(exc: Exception) -> bool:
    """True for throttling / quota errors (the ones worth a cooldown)."""
    code = exc.response.get("Error", {}).get("Code", "") if isinstance(exc, ClientError) else ""
    return code in ("ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException") or "Throttling" in str(exc)


def _trip(model_id: str, exc: Exception) -> None:
    """Put the model on cooldown: long for a daily token limit, short for ordinary throttling."""
    daily = "per day" in str(exc).lower()
    seconds = config.DAILY_LIMIT_COOLDOWN_S if daily else config.THROTTLE_COOLDOWN_S
    with _LOCK:
        _COOLDOWN[model_id] = (time.time() + seconds, daily)
    log.warning("model %s throttled (%s) -> skipped for %ss", model_id, "daily limit" if daily else "throttle", seconds)
    if daily and quota_alert()["all_daily_limited"]:  # every configured model is at its daily limit: someone must act
        log.critical("ALL configured models hit the daily token limit -> request a quota increase (python -m src.check_quota)")


def cooldown_status() -> dict[str, int]:
    """Models currently skipped and the seconds left (shown on /health for debugging)."""
    now = time.time()
    with _LOCK:
        return {m: int(t - now) for m, (t, _) in _COOLDOWN.items() if t > now}


def quota_alert() -> dict:
    """Are all configured models (main + fast lists) at the daily token limit right now? (shown on /health)"""
    chain = list(dict.fromkeys(config.MODELS + config.FAST_MODELS))
    now = time.time()
    with _LOCK:
        limited = [m for m in chain if _COOLDOWN.get(m, (0, False))[0] > now and _COOLDOWN[m][1]]
    return {"all_daily_limited": len(limited) == len(chain), "daily_limited": limited, "checked": len(chain)}


# ------------------------------------------------------------------ which model answered (per request)
# 요청마다 어떤 모델이 실제로 응답했는지 센다. 한도 때문에 Sonnet 대신 Haiku 가 답하면 평가 결과 해석이 달라지므로 trace 에 남긴다.
_USAGE: contextvars.ContextVar[Counter | None] = contextvars.ContextVar("llm_usage", default=None)


def start_usage() -> Counter:
    """요청 처리 시작 시 호출. 이 컨텍스트(와 여기서 복사된 스레드 컨텍스트)에서 성공한 호출을 센다."""
    c: Counter = Counter()
    _USAGE.set(c)
    return c


def usage_text(c: Counter) -> str:
    """'모델 id ×횟수' 목록 (많이 쓴 순)."""
    return ", ".join(f"{m} ×{n}" for m, n in c.most_common()) or "LLM 호출 없음"


def _note_model(model_id: str) -> None:
    c = _USAGE.get()
    if c is not None:
        with _LOCK:
            c[model_id] += 1


# ------------------------------------------------------------------ models
def build_model(model_id: str, attempts: int | None = None) -> ChatBedrockConverse:
    """One Bedrock model with adaptive retry/backoff for throttling. attempts=1 이면 재시도 없이 한 번만 호출한다."""
    return ChatBedrockConverse(
        model_id=model_id,
        region_name=config.AWS_REGION,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        config=Config(
            retries={"mode": "adaptive", "max_attempts": attempts or config.MAX_RETRY_ATTEMPTS},
            read_timeout=config.READ_TIMEOUT_S,
        ),
    )


class CooldownFallback(Runnable):
    """Try the models in order, skipping the ones on cooldown; a throttled model is put on cooldown.

    If every model is on cooldown it still tries them all as a last resort (the limit may have been lifted).

    하루 한도로 쉬던 모델이 쿨다운에서 풀린 뒤 처음 불릴 때는 `probes`(재시도 1번짜리 같은 모델)로 한 번만 찔러 본다.
    한도가 그대로면 재시도 대기 없이 바로 다음 모델로 넘어가고, 풀렸으면 성공하면서 평소 재시도 설정으로 돌아간다.
    (5분마다 풀린 첫 요청이 한도에 걸린 모델의 재시도를 기다리느라 느려지던 문제를 막는다.)
    """

    def __init__(self, ids: list[str], runnables: list[Runnable], probes: list[Runnable] | None = None):
        self.ids = ids
        self.runnables = runnables
        self.probes = probes

    def invoke(self, input, config=None, **kwargs):  # noqa: A002 (Runnable interface)
        order = [i for i, mid in enumerate(self.ids) if not _on_cooldown(mid)] or list(range(len(self.ids)))
        last: Exception | None = None
        for i in order:
            try:
                probing = self.probes is not None and _needs_probe(self.ids[i])
                result = (self.probes[i] if probing else self.runnables[i]).invoke(input, config, **kwargs)
                if result is None:  # 구조화 출력을 해석하지 못해 None 이 온 경우도 실패로 보고 다음 모델로 넘어간다
                    raise ValueError("model returned no structured output")
                if probing:
                    _clear_daily(self.ids[i])
                    log.info("model %s is back after a daily limit", self.ids[i])
                _note_model(self.ids[i])
                return result
            except FALLBACK_EXCEPTIONS as e:
                last = e
                if _is_throttle(e):
                    _trip(self.ids[i], e)
                log.warning("model %s failed (%s) -> next model", self.ids[i], type(e).__name__)
            except Exception as e:  # 형식에 맞지 않는 출력(검증·파싱 오류)도 다음 모델로 넘긴다. 쿨다운은 걸지 않는다
                last = e
                log.warning("model %s returned invalid output (%s) -> next model", self.ids[i], type(e).__name__)
        raise last  # every model failed


class ResilientLLM:
    """Several models behind one object: with_structured_output() gives a CooldownFallback over all of them."""

    def __init__(self, ids: list[str]):
        self.ids = ids
        self.models = [build_model(m) for m in ids]
        self.probes = [build_model(m, attempts=1) for m in ids]  # 한도에서 막 풀린 모델을 재시도 없이 찔러 보는 용도

    def with_structured_output(self, schema, **kwargs):
        return CooldownFallback(self.ids, [m.with_structured_output(schema, **kwargs) for m in self.models],
                                [m.with_structured_output(schema, **kwargs) for m in self.probes])

    def invoke(self, input, config=None, **kwargs):  # noqa: A002
        return CooldownFallback(self.ids, self.models, self.probes).invoke(input, config, **kwargs)


def get_llm(models: list[str] | None = None):
    """Primary model with ordered fallbacks and a cooldown for throttled models.

    Fallback models must support tool calling / structured output.
    """
    ids = models or config.MODELS
    if config.DISABLE_FALLBACK or len(ids) == 1:
        return build_model(ids[0])
    return ResilientLLM(ids)
