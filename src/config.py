"""Central settings, read from environment variables (.env)."""
import os

from dotenv import load_dotenv

load_dotenv()

# 사용 리전: us-west-2 (오레곤). 아래 모델 id 는 이 리전에서 쓸 수 있는 것들이다(us.* 는 미국 추론 프로파일, global.* 는 글로벌 프로파일).
# 리전을 바꾸면 이 목록도 그 리전에서 쓸 수 있는 id 로 바꿔야 한다. BEDROCK_MODELS / BEDROCK_FAST_MODELS 로 덮어쓸 수 있다.
# Fallback order: first = primary. us.* 와 global.* 는 한도가 따로 잡히는 경우가 있어 번갈아 둔다.
# global.* 를 먼저 둔 이유(2026-11-12 측정): us.anthropic.claude-sonnet-4-6 는 같은 호출이 4번 중 2번 스로틀링 오류가 나서
# 재시도·대기 뒤에야 다음 모델로 넘어가 응급 점검이 2초에서 10초까지 흔들렸고, global.anthropic.claude-sonnet-4-6 는 4번 모두 1.3초였다.
# 안전 판단(응급 점검 등)에 쓰는 메인 체인은 Anthropic 모델만 둔다. Nova 계열은 구조화 출력 품질을 검증하지 않아 넣지 않았다.
DEFAULT_MODELS = [
    "global.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-6",
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
]

# Light steps (scope check, query translation, region extraction) use a faster/cheaper model first.
DEFAULT_FAST_MODELS = [
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "global.anthropic.claude-sonnet-4-6",
]

# Every model the account can use in this region (given by the user). `python -m src.check_quota` probes all of them.
KNOWN_MODELS = [
    "us.anthropic.claude-sonnet-4-6", "global.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0", "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.amazon.nova-pro-v1:0", "us.amazon.nova-lite-v1:0", "us.amazon.nova-micro-v1:0",
    "us.amazon.nova-2-lite-v1:0", "global.amazon.nova-2-lite-v1:0",
]

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-west-2")
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
MEMORY_DB = Path(os.getenv("MEMORY_DB", ROOT_DIR / "data" / "memory" / "memory.sqlite"))  # 장기 메모리(상담 이력). git-ignored, 개인 건강 정보
MEDLINEPLUS_DATE = os.getenv("MEDLINEPLUS_DATE", "2026-11-10")  # pinned so evaluations are reproducible
ANNEX1_PATH = SOURCES_DIR / "emergency_symptoms_annex1.md"
# 응급 판단 기준(별표 1)의 법령 기준. 파일의 '기준 정보'와 같아야 하며, 인덱스 청크 메타데이터와 평가 케이스에 기록된다.
ANNEX1_VERSION = "보건복지부령 제1189호(2026-07-13 개정), 2026-08-14 시행"

# Set to 1 to disable fallbacks (use for evaluation runs so round1/round2 stay comparable).
DISABLE_FALLBACK = os.getenv("DISABLE_FALLBACK", "0") == "1"
WARMUP = os.getenv("WARMUP", "1") == "1"  # 서버 시작 때 백그라운드로 첫 요청의 초기화 비용을 미리 치른다(0 이면 끔)
