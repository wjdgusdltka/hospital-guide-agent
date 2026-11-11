# 병원 길잡이 Agent (hospital-guide-agent)

증상을 말하면 **응급실에 가야 할지, 외래라면 어느 진료과로 갈지** 근거와 함께 안내하는 에이전트입니다.

> ⚠️ **이 프로젝트는 개인 학습용으로 개발되었습니다.**
> 의료 서비스나 의료 기기가 아니며, 진단·처방·치료를 대신하지 않습니다. 제공되는 모든 내용은 **단순 참고용**이며 정확성·완전성을 보장하지 않습니다. 실제 건강 판단과 결정은 반드시 의사 등 전문가에게 받으세요. 응급 상황이 의심되면 즉시 119에 연락하세요.

## 무엇을 하나요

1. 증상에서 **응급 신호**를 점검합니다. 있으면 즉시 119/응급실을 안내합니다.
2. 판단에 필요한 정보가 부족하면 **구체적으로 되묻습니다.** (최대 3턴)
3. 응급이 아니라고 판단되면 **관련될 수 있는 상태(참고)** 와 **진료과 후보**를 근거 출처와 함께 안내합니다.
4. 응급 권고 시 사용자가 알려 준 지역의 **응급의료기관 목록**을 보여 줍니다.

확정적 진단, 약·용량 안내는 하지 않으며, 애매하면 더 높은 단계로 안내하는 보수적 정책을 따릅니다. 자세한 기획과 정책은 [SERVICE.md](SERVICE.md)를 참고하세요.

## 현재 상태

개발 중입니다. 현재는 API 서버와 UI 자리(mock 응답), Bedrock 모델 연결(재시도 후 폴백), **RAG 코퍼스(Chroma 인덱스)** 까지 준비되어 있습니다. 에이전트 로직과 평가 세트는 구현 예정입니다.

## 구성

```
src/          # 서버(FastAPI), LLM 설정, (예정) 에이전트·도구
web/          # 채팅 UI
data/         # 근거 자료 (data/sources/ 에 응급증상 목록)
evaluation/   # (예정) 평가 세트와 결과
SERVICE.md    # 서비스 기획
```

## 실행 (개발용)

Python 3.14에서 개발했습니다.

> ⚠️ **서버를 실행하기 전에 3번 "인덱스 생성"을 반드시 먼저 실행해야 합니다.**
> 검색용 인덱스(Chroma DB, `data/chroma/`)는 저장소에 들어 있지 않고 **직접 만들어야** 합니다. 만들지 않으면 서버가 시작되지 않고(`MOCK_MODE=0`일 때) 안내 메시지를 출력합니다.

**1. 가상환경과 패키지**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. `.env` 만들기** — 아래 "환경 변수" 표를 보고 AWS 키를 넣습니다. (인덱스 생성과 질의 모두 Bedrock 임베딩을 호출하므로 키가 필요합니다.)

**3. 인덱스 생성 (최초 1회, 약 2분, 임베딩 비용 수십 원 이하)**

```bash
python -m src.rag.build_index
```

완료 후 `done. total in collection: 1740` 이 출력되면 성공입니다. 코퍼스를 처음부터 다시 만들려면 `--rebuild` 를 붙입니다.

**4. 서버 실행**

```bash
uvicorn src.server:app --reload
```

브라우저에서 `http://127.0.0.1:8000` 을 열면 됩니다. 현재는 `MOCK_MODE=1`(기본값)이라 고정 응답이 나옵니다. `http://127.0.0.1:8000/health` 의 `index_ready` 로 인덱스 상태를 확인할 수 있습니다.

### RAG 코퍼스와 Chroma 인덱스

- 구성: MedlinePlus 건강 주제(영문, 2026-11-10 기준 1,725 청크) + 응급증상 목록(별표 1, 15 청크). 임베딩은 Amazon Titan Text Embeddings V2(512차원)입니다.
- 원본은 `data/sources/` 에 포함되어 있습니다(MedlinePlus는 최근 6개 날짜 파일만 공개하므로 압축 원본을 함께 보관합니다). 인덱스(`data/chroma/`, 약 24MB)는 위 3번 명령으로 만들며 git에서 제외됩니다.
- `data/chroma/chroma.sqlite3` 는 VS Code의 SQLite 확장으로 열어 볼 수 있습니다. 문서 내용과 메타데이터는 `embedding_metadata`, 청크 목록은 `embeddings` 테이블에서 확인할 수 있습니다.
- 검색 시험 결과: 한국어 질의를 그대로 임베딩하면 영문 문서가 거의 찾아지지 않았습니다(12개 중 0개). 영어 키워드로 바꾸면 12개 중 9개가 기대한 주제를 상위 5개 안에서 찾았습니다. 그래서 한국어 질의는 영어 키워드로 변환한 뒤 검색합니다.

### API

`POST /query`

```json
// 요청
{ "question": "어제부터 설사랑 두통이 있어요" }
// 응답
{ "answer": "...", "contexts": [{ "doc_id": "...", "text": "..." }], "trace": [{ "step": "...", "input": "...", "output": "..." }] }
```

### 환경 변수 (`.env`, 저장소에 올리지 않습니다)

| 변수 | 용도 |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | Amazon Bedrock 호출 |
| `BEDROCK_MODELS` (선택) | 모델 우선순위(쉼표 구분). 기본은 Claude Sonnet 계열 폴백 |
| `BEDROCK_MAX_ATTEMPTS`, `BEDROCK_READ_TIMEOUT` | 모델별 재시도 횟수, 읽기 제한 시간 |
| `DISABLE_FALLBACK` | `1`이면 폴백 없이 한 모델만 사용(평가 재현용) |
| `GOV_DATA_API_ENDPOINT`, `GOV_DATA_API_KEY` | 국립중앙의료원 응급의료정보 조회 서비스 |
| `LANGSMITH_*` (선택) | 트레이싱 |
| `MOCK_MODE` | `1`이면 고정 응답 |

## 데이터 출처와 이용 조건

- **응급증상 목록:** 「응급의료에 관한 법률 시행규칙」 [별표 1] 항목. 서울아산병원 안내 페이지를 출처로 정리했습니다. 공식 현행본과 대조하지 않았으며, 수집일은 파일에 적었습니다.
- **MedlinePlus 건강 주제:** 미국 국립의학도서관(NLM)의 [MedlinePlus.gov](https://medlineplus.gov/) 건강 주제 요약(영문). 출처: MedlinePlus, National Library of Medicine. 이 프로젝트는 NLM의 보증이나 후원을 받지 않았습니다.
- **응급의료기관 목록:** 국립중앙의료원 응급의료정보 조회 서비스(공공데이터포털).

## 기술 스택

Python, FastAPI, LangChain/LangGraph, Amazon Bedrock(Claude), (예정) 벡터 검색과 RAGAS 평가.

## 한계

- 증상에서 진료과로 이어지는 공식 매핑 자료가 없어 진료과 추천은 근거가 약합니다.
- 영문 자료를 한국어로 옮겨 쓰므로 번역 오류가 있을 수 있습니다.
- 의학적 정확성은 검증되지 않았습니다.

## 라이선스

(정해지면 추가)
