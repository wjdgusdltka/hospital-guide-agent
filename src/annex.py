"""별표 1(응급증상 및 이에 준하는 증상) 파서. 코퍼스, 응급 점검 프롬프트, 응급 회귀 테스트가 모두 이 모듈로 같은 파일을 읽는다.

원본: data/sources/emergency_symptoms_annex1.md (공식 PDF 기준, 항목마다 ID 가 있다)
"""
import re
from dataclasses import dataclass

from . import config

# 항목 줄 형식: - `A-나-2` 급성호흡곤란
_ITEM = re.compile(r"^- `([AB]-[가-힣]-\d+)` (.+)$")
KIND = {"A": "응급증상", "B": "응급증상에 준하는 증상"}


@dataclass(frozen=True)
class AnnexItem:
    id: str  # 예: A-나-2
    kind: str  # 응급증상 / 응급증상에 준하는 증상
    category: str  # 예: 나. 심혈관계 응급증상
    text: str  # 공식 원문 표기


def load_annex_items() -> list[AnnexItem]:
    """별표 1 파일의 모든 항목을 순서대로 읽는다."""
    items, category = [], ""
    for line in config.ANNEX1_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            category = line[4:].strip()
        m = _ITEM.match(line)
        if m:
            items.append(AnnexItem(m.group(1), KIND[m.group(1)[0]], category, m.group(2).strip()))
    return items


def annex_prompt_text() -> tuple[str, str]:
    """응급 점검 프롬프트에 넣을 (응급증상, 준하는 증상) 목록 문자열. ID 는 빼고 원문 표기만 쓴다."""
    out = {"A": [], "B": []}
    last = {"A": "", "B": ""}
    for it in load_annex_items():
        k = it.id[0]
        if it.category != last[k]:
            out[k].append(f"[{it.category}]")
            last[k] = it.category
        out[k].append(f"- {it.text}")
    return "\n".join(out["A"]), "\n".join(out["B"])


def _norm(s: str) -> str:
    """비교용 정규화: 공백과 괄호 속 설명, 가운뎃점 등을 지워 표기 차이를 흡수한다."""
    s = re.sub(r"\(.*?\)", "", s)
    return re.sub(r"[\s·ㆍ,]", "", s)


def match_item_ids(texts: list[str]) -> list[str]:
    """응급 점검이 인용한 항목 문구를 별표 1 항목 ID 로 바꾼다 (표기 차이는 정규화해서 비교, 없으면 건너뜀)."""
    items = load_annex_items()
    ids = []
    for t in texts:
        nt = _norm(t)
        for it in items:
            ni = _norm(it.text)
            if nt and (nt == ni or nt in ni or ni in nt):
                ids.append(it.id)
                break
    return ids
