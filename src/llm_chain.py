"""LCEL 체인 만들기 (Day 1): 프롬프트 | 모델.with_structured_output(스키마)."""
from langchain_core.prompts import ChatPromptTemplate

from . import config
from .llm import get_llm

_CHAINS: dict = {}  # 이름별로 한 번만 만들어 재사용


def structured_chain(name: str, schema, system: str, fast: bool = False):
    """시스템 프롬프트 + `{input}` 사용자 입력 -> 스키마로 구조화된 결과를 돌려주는 체인.

    fast=True 면 가벼운 판단용 모델(Haiku 우선)을, 아니면 메인 모델(Sonnet 우선)을 쓴다.
    """
    if name not in _CHAINS:
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])
        llm = get_llm(config.FAST_MODELS if fast else None)
        _CHAINS[name] = prompt | llm.with_structured_output(schema)
    return _CHAINS[name]
