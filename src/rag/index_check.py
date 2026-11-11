"""Check that the Chroma index has been built (it is not committed to git)."""
import chromadb

from .. import config

BUILD_COMMAND = "python -m src.rag.build_index"

NOT_READY_MESSAGE = f"""
============================================================
[인덱스 없음] 검색용 인덱스(data/chroma/)가 아직 만들어지지 않았습니다.

서버를 시작하기 전에 아래 명령을 먼저 한 번 실행하세요.
(약 2분 소요, .env 에 AWS 키 필요, 임베딩 비용은 수십 원 이하)

    {BUILD_COMMAND}

사유: {{reason}}
(Search index not found. Run `{BUILD_COMMAND}` once before starting the server.)
============================================================
"""


class IndexNotReadyError(RuntimeError):
    pass


def index_status() -> dict:
    """Return {"ready": bool, "docs": int, "reason": str}. Never creates an empty index."""
    if not (config.CHROMA_DIR / "chroma.sqlite3").exists():
        return {"ready": False, "docs": 0, "reason": "data/chroma/ 가 없습니다."}
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        col = client.get_collection(config.COLLECTION_NAME)
    except Exception:
        return {"ready": False, "docs": 0, "reason": f"컬렉션 '{config.COLLECTION_NAME}' 이 없습니다."}
    n = col.count()
    if n == 0:
        return {"ready": False, "docs": 0, "reason": "컬렉션이 비어 있습니다."}
    dim = len(col.get(limit=1, include=["embeddings"])["embeddings"][0])
    if dim != config.EMBED_DIM:
        return {
            "ready": False,
            "docs": n,
            "reason": f"임베딩 차원이 다릅니다(인덱스 {dim}, 설정 {config.EMBED_DIM}). --rebuild 로 다시 만드세요.",
        }
    return {"ready": True, "docs": n, "reason": ""}


def ensure_index() -> None:
    st = index_status()
    if not st["ready"]:
        raise IndexNotReadyError(NOT_READY_MESSAGE.format(reason=st["reason"]))
