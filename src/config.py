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

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MODELS = [m.strip() for m in os.getenv("BEDROCK_MODELS", "").split(",") if m.strip()] or DEFAULT_MODELS

# Retry (boto3, per model) happens BEFORE falling back to the next model.
MAX_RETRY_ATTEMPTS = int(os.getenv("BEDROCK_MAX_ATTEMPTS", "6"))
READ_TIMEOUT_S = int(os.getenv("BEDROCK_READ_TIMEOUT", "60"))

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# Set to 1 to disable fallbacks (use for evaluation runs so round1/round2 stay comparable).
DISABLE_FALLBACK = os.getenv("DISABLE_FALLBACK", "0") == "1"
