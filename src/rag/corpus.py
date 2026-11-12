"""Turn the raw sources into LangChain Documents (one per chunk).

Sources:
  - MedlinePlus health topics (English XML, public domain, attribution "MedlinePlus.gov")
  - Emergency symptom list from Enforcement Rule of the Emergency Medical Service Act, Annex 1
"""
import html
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from langchain_core.documents import Document

from .. import config
from ..annex import load_annex_items

MAX_CHUNK_CHARS = 1800
ANNEX1_URL = "https://www.law.go.kr/법령/응급의료에관한법률시행규칙"  # 국가법령정보센터


# ---------------------------------------------------------------- MedlinePlus
def medlineplus_zip(date: str = config.MEDLINEPLUS_DATE) -> Path:
    """Path of the pinned MedlinePlus topics zip (committed in data/sources/).

    MedlinePlus only keeps the 6 most recent files online, so the zip is committed to the repo
    and downloaded only if it is missing (which works only while that date is still online).
    """
    zip_path = config.SOURCES_DIR / f"mplus_topics_compressed_{date}.zip"
    if not zip_path.exists():
        url = f"https://medlineplus.gov/xml/mplus_topics_compressed_{date}.zip"
        print(f"{zip_path.name} not found locally, downloading {url}")
        urllib.request.urlretrieve(url, zip_path)
    return zip_path


def _html_to_text(raw: str) -> str:
    s = html.unescape(raw)
    s = re.sub(r"(?i)<li[^>]*>", "\n- ", s)
    s = re.sub(r"(?i)</(p|ul|ol|h\d|div)>|<br\s*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def _split(paragraphs: list[str], limit: int) -> list[str]:
    """Greedy-pack paragraphs into chunks of at most `limit` chars (long paragraphs stay whole)."""
    chunks, cur = [], ""
    for p in paragraphs:
        if cur and len(cur) + len(p) + 1 > limit:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def load_medlineplus(zip_path: Path) -> list[Document]:
    with zipfile.ZipFile(zip_path) as z:
        xml_name = next(n for n in z.namelist() if n.endswith(".xml"))
        with z.open(xml_name) as f:
            root = ET.parse(f).getroot()
    docs: list[Document] = []
    for t in root.findall("health-topic"):
        if t.get("language") != "English":
            continue
        title, topic_id, url = t.get("title"), t.get("id"), t.get("url")
        also = [a.text for a in t.findall("also-called") if a.text]
        groups = [g.text for g in t.findall("group") if g.text]
        body = _html_to_text(t.findtext("full-summary") or "")
        if not body:
            continue
        header = title + (f" (also called: {', '.join(also)})" if also else "")
        pieces = _split(body.split("\n"), MAX_CHUNK_CHARS)
        for i, piece in enumerate(pieces):
            docs.append(
                Document(
                    page_content=f"{header}\n\n{piece}",
                    metadata={
                        "doc_id": f"mplus-{topic_id}-{i}",
                        "source": "MedlinePlus",
                        "title": title,
                        "url": url,
                        "groups": "; ".join(groups),
                        "is_symptom": "Symptoms" in groups,
                        "chunk": i,
                    },
                )
            )
    return docs


# --------------------------------------------------------------------- Annex 1
def load_annex1() -> list[Document]:
    """별표 1 을 (구분, 범주)별 청크 하나씩으로 만든다. 항목마다 ID 를 붙이고 법령 기준(개정·시행일)을 메타데이터에 남긴다."""
    groups: dict[tuple[str, str], list] = {}
    for it in load_annex_items():  # 파서는 src/annex.py 한 곳에만 있다
        groups.setdefault((it.kind, it.category), []).append(it)
    docs = []
    for n, ((kind, category), items) in enumerate(groups.items()):
        text = (
            f"응급의료에 관한 법률 시행규칙 [별표 1] {kind} - {category} ({config.ANNEX1_VERSION})\n"
            + "\n".join(f"- [{it.id}] {it.text}" for it in items)
        )
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": f"annex1-{n}",
                    "source": "별표 1",
                    "title": f"{kind} / {category}",
                    "url": ANNEX1_URL,
                    "groups": "",
                    "is_symptom": False,
                    "chunk": 0,
                    "law_version": config.ANNEX1_VERSION,
                },
            )
        )
    return docs
