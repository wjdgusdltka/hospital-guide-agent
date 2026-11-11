"""Build the Chroma index (data/chroma/). Run: python -m src.rag.build_index [--rebuild]

The index is NOT committed to git. Everyone must build it once before starting the server
(about 2 minutes, needs the AWS keys in .env, costs a few cents of Titan embedding calls).
"""
import argparse
import shutil
import time

from langchain_chroma import Chroma

from .. import config
from .corpus import load_annex1, load_medlineplus, medlineplus_zip
from .embeddings import TitanEmbeddings

BATCH = 200


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=TitanEmbeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="delete data/chroma first")
    ap.add_argument("--limit", type=int, default=0, help="only embed the first N medline chunks (testing)")
    args = ap.parse_args()

    if args.rebuild and config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)

    zip_path = medlineplus_zip()
    docs = load_annex1() + load_medlineplus(zip_path)
    if args.limit:
        docs = docs[: args.limit]
    print(f"documents (chunks): {len(docs)}  | source: {zip_path.name}")

    vs = get_vectorstore()
    existing = set(vs.get(include=[])["ids"])
    todo = [d for d in docs if d.metadata["doc_id"] not in existing]
    print(f"already indexed: {len(existing)}  | to embed: {len(todo)}")

    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        vs.add_documents(batch, ids=[d.metadata["doc_id"] for d in batch])
        print(f"  {min(i + BATCH, len(todo))}/{len(todo)}  ({time.time() - t0:.0f}s)")
    print("done. total in collection:", vs._collection.count())


if __name__ == "__main__":
    main()
