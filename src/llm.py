"""Bedrock chat model factory: boto3 retries first, then model fallback."""
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from langchain_aws import ChatBedrockConverse

from . import config

# Errors that should move us on to the next model (after boto3 retries are exhausted).
FALLBACK_EXCEPTIONS = (ClientError, BotoCoreError)


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


def get_llm(models: list[str] | None = None):
    """Primary model with ordered fallbacks.

    Order of defense: (1) boto3 retry on the same model, (2) next model in the list.
    Fallback models must support tool calling / structured output.
    """
    ids = models or config.MODELS
    primary = build_model(ids[0])
    if config.DISABLE_FALLBACK or len(ids) == 1:
        return primary
    fallbacks = [build_model(m) for m in ids[1:]]
    return primary.with_fallbacks(fallbacks, exceptions_to_handle=FALLBACK_EXCEPTIONS)
