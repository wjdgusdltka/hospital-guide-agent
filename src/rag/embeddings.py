"""Titan Text Embeddings V2 on Bedrock, as a LangChain `Embeddings`.

Titan takes one text per call, so documents are embedded with a small thread pool.
boto3's adaptive retry handles throttling.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config
from langchain_core.embeddings import Embeddings

from .. import config


class TitanEmbeddings(Embeddings):
    def __init__(self, max_workers: int = 8):
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=config.AWS_REGION,
            config=Config(
                retries={"mode": "adaptive", "max_attempts": config.MAX_RETRY_ATTEMPTS},
                read_timeout=config.READ_TIMEOUT_S,
            ),
        )
        self._max_workers = max_workers

    def _embed(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text, "dimensions": config.EMBED_DIM, "normalize": True})
        resp = self._client.invoke_model(modelId=config.EMBED_MODEL_ID, body=body)
        return json.loads(resp["body"].read())["embedding"]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            return list(pool.map(self._embed, texts))
