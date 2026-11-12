"""Central settings, read from environment variables (.env)."""
import os

from dotenv import load_dotenv

load_dotenv()

# Fallback order: first = primary. Overridable via BEDROCK_MODELS (comma-separated).
DEFAULT_MODELS = [
    "global.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-6",
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
]

# Light steps (scope check, query translation, region extraction) use a faster/cheaper model first.
DEFAULT_FAST_MODELS = [
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "global.anthropic.claude-sonnet-4-6",
]

# Every model the account can use (given by the user). `python -m src.check_quota` probes all of them.
KNOWN_MODELS = [
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-sonnet-4-6", "global.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0", "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.amazon.nova-pro-v1:0", "us.amazon.nova-2-lite-v1:0", "global.amazon.nova-2-lite-v1:0", "us.amazon.nova-lite-v1:0",
]

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MODELS = [m.strip() for m in os.getenv("BEDROCK_MODELS", "").split(",") if m.strip()] or DEFAULT_MODELS
FAST_MODELS = [m.strip() for m in os.getenv("BEDROCK_FAST_MODELS", "").split(",") if m.strip()] or DEFAULT_FAST_MODELS

# Circuit breaker: a model that hit a limit is skipped for a while instead of being retried on every call.
DAILY_LIMIT_COOLDOWN_S = int(os.getenv("MODEL_DAILY_COOLDOWN_S", "300"))  # "Too many tokens per day": often recovers within minutes in practice, so re-test every 5 min
THROTTLE_COOLDOWN_S = int(os.getenv("MODEL_THROTTLE_COOLDOWN_S", "60"))  # ordinary throttling

# Retry (boto3, per model) happens BEFORE falling back to the next model.
MAX_RETRY_ATTEMPTS = int(os.getenv("BEDROCK_MAX_ATTEMPTS", "6"))
READ_TIMEOUT_S = int(os.getenv("BEDROCK_READ_TIMEOUT", "60"))

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# --- RAG (embeddings + Chroma) ---
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "data" / "chroma"))  # git-ignored: build it with `python -m src.rag.build_index`
COLLECTION_NAME = "hospital_guide"
SOURCES_DIR = ROOT_DIR / "data" / "sources"  # committed source data
MEDLINEPLUS_DATE = os.getenv("MEDLINEPLUS_DATE", "2026-11-10")  # pinned so evaluations are reproducible
ANNEX1_PATH = SOURCES_DIR / "emergency_symptoms_annex1.md"
# 응급 판단 기준(별표 1)의 법령 기준. 파일의 '기준 정보'와 같아야 하며, 인덱스 청크 메타데이터와 평가 케이스에 기록된다.
ANNEX1_VERSION = "보건복지부령 제1189호(2026-07-13 개정), 2026-08-14 시행"

# Set to 1 to disable fallbacks (use for evaluation runs so round1/round2 stay comparable).
DISABLE_FALLBACK = os.getenv("DISABLE_FALLBACK", "0") == "1"
