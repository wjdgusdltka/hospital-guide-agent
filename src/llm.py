"""Bedrock chat model factory: boto3 retries first, then model fallback with a cooldown (circuit breaker).

Order of defense for one call:
  1) boto3 retries the same model (adaptive backoff).
  2) If the model still fails, move on to the next model in the list.
  3) A model that failed because of throttling is put on cooldown, so the following calls skip it right away
     instead of spending several seconds of retries on it every time (a daily token limit does not recover soon).
"""
import logging
import threading
import time

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


# ------------------------------------------------------------------ models
def build_model(model_id: str) -> ChatBedrockConverse:
    """One Bedrock model with adaptive retry/backoff for throttling."""
    return ChatBedrockConverse(
        model_id=model_id,
        region_name=config.AWS_REGION,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        config=Config(
            retries={"mode": "adaptive", "max_attempts": config.MAX_RETRY_ATTEMPTS},
            read_timeout=config.READ_TIMEOUT_S,
        ),
    )


class CooldownFallback(Runnable):
    """Try the models in order, skipping the ones on cooldown; a throttled model is put on cooldown.

    If every model is on cooldown it still tries them all as a last resort (the limit may have been lifted).
    """

    def __init__(self, ids: list[str], runnables: list[Runnable]):
        self.ids = ids
        self.runnables = runnables

    def invoke(self, input, config=None, **kwargs):  # noqa: A002 (Runnable interface)
        order = [i for i, mid in enumerate(self.ids) if not _on_cooldown(mid)] or list(range(len(self.ids)))
        last: Exception | None = None
        for i in order:
            try:
                result = self.runnables[i].invoke(input, config, **kwargs)
                if result is None:  # 구조화 출력을 해석하지 못해 None 이 온 경우도 실패로 보고 다음 모델로 넘어간다
                    raise ValueError("model returned no structured output")
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

    def with_structured_output(self, schema, **kwargs):
        return CooldownFallback(self.ids, [m.with_structured_output(schema, **kwargs) for m in self.models])

    def invoke(self, input, config=None, **kwargs):  # noqa: A002
        return CooldownFallback(self.ids, self.models).invoke(input, config, **kwargs)


def get_llm(models: list[str] | None = None):
    """Primary model with ordered fallbacks and a cooldown for throttled models.

    Fallback models must support tool calling / structured output.
    """
    ids = models or config.MODELS
    if config.DISABLE_FALLBACK or len(ids) == 1:
        return build_model(ids[0])
    return ResilientLLM(ids)
