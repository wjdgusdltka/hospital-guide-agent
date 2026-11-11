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

MAX_CHUNK_CHARS = 1800
ANNEX1_URL = "https://www.amc.seoul.kr/asan/emergencytreatguide/emergencysymptom/emergencySymptom.do"


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
def load_annex1(path: Path = config.ANNEX1_PATH) -> list[Document]:
    """One chunk per category under '## A. 응급증상' / '## B. 응급증상에 준하는 증상'."""
    kind = None
    category = None
    items: dict[tuple[str, str], list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## A."):
            kind = "응급증상"
        elif line.startswith("## B."):
            kind = "응급증상에 준하는 증상"
        elif line.startswith("### ") and kind:
            category = line[4:].strip()
        elif line.startswith("- ") and kind and category:
            items.setdefault((kind, category), []).append(line[2:].strip())
    docs = []
    for n, ((kind, category), lst) in enumerate(items.items()):
        text = (
            f"응급의료에 관한 법률 시행규칙 [별표 1] {kind} - {category}\n"
            + "\n".join(f"- {x}" for x in lst)
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
                },
            )
        )
    return docs
