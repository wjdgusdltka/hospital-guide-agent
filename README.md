# 미니 PJT: 병원 길잡이 Agent (hospital-guide-agent)

> ⚠️ **이 프로젝트는 개인 학습용으로 개발되었습니다.**
> 의료 서비스나 의료 기기가 아니며, 진단·처방·치료를 대신하지 않습니다. 제공되는 모든 내용은 **단순 참고용**이며 정확성·완전성을 보장하지 않습니다. 실제 건강 판단과 결정은 반드시 의사 등 전문가에게 받으세요. 응급 상황이 의심되면 즉시 119에 연락하세요.

> ⚠️ **실행 전에 "검색 인덱스 생성"을 반드시 먼저 해야 합니다.**
> 검색용 인덱스(Chroma DB, `data/chroma/`)는 저장소에 들어 있지 않고 **직접 만들어야** 합니다. 아래 [실행 방법](#실행-방법)의 3번(`python -m src.rag.build_index`)을 서버 실행 전에 한 번 실행하세요. 만들지 않으면 서버가 시작되지 않고 안내 메시지를 출력합니다.

## 무엇을 푸나

몸이 아플 때 **"지금 응급실에 가야 하는지, 외래라면 어느 진료과로 가야 하는지"** 를 판단하지 못하는 일반 사용자에게, 법령 기준과 공개 의학 자료를 근거로 안내합니다. (기획 상세: [SERVICE.md](SERVICE.md))

- 응급 신호가 있으면 즉시 119/응급실을 안내합니다.
- 정보가 부족하면 위험 신호부터 구체적으로 되묻습니다(최대 3턴).
- 응급이 아니라고 판단되면 "관련될 수 있는 상태(참고)"와 진료과 후보를 근거 출처와 함께 안내합니다.
- 확정적 진단·약 용량 안내는 하지 않고, 애매하면 더 높은 단계로 안내합니다(보수적 트리아지).

## 활용한 패턴 (Day 1~7)

상태: ✅ 완료 · 🚧 구현 예정

- Day 2: **RAG** — MedlinePlus 건강 주제 + 응급증상 목록(별표 1)을 Titan 임베딩으로 Chroma에 저장. 한국어 증상을 영어 검색어로 바꿔 검색(쿼리 변환). ✅ 코퍼스·인덱스 / 🚧 검색 도구 연결
- Day 1: **LCEL chain (Pydantic 구조화 출력)** — 응급 여부·관련 상태·진료과 후보·출처를 구조화해 답변 생성 🚧
- Day 3: **ReAct** — `bind_tools`로 도메인 도구를 에이전트가 자율 선택 🚧
- Day 4: **도구 다중(외부 API 포함)** — 응급 신호 점검, 근거 검색, 관련 상태, 진료과 추천, 국립중앙의료원 응급의료기관 조회(API 호출 확인) 🚧
- Day 5: **가드레일·미들웨어** — 입력(개인정보 마스킹, 프롬프트 인젝션·범위 이탈 차단)과 출력(단정·확률·안심 표현 차단) 필터링 🚧
- Day 5: **HITL/멀티턴** — 되묻기 상태를 LangGraph 체크포인트(`session_id` → `thread_id`)로 유지 🚧
- Day 7: **Observability · Trace** — 응답의 `trace` 와 LangSmith 트레이스 🚧
- Day 7: **평가** — 자체 인-아웃 세트 + RAGAS + LLM-as-Judge 🚧
- (인프라) Bedrock 호출 재시도 후 모델 폴백 ✅ — [src/llm.py](src/llm.py)

## 아키텍처

```
사용자 (web/index.html 채팅 UI)
   │  POST /query {question, session_id?}
   ▼
FastAPI (src/server.py)
   ▼
[에이전트 그래프 — 구현 예정]
  1. 입력 가드레일 (PII 마스킹 · 인젝션/범위 이탈 차단)
  2. check_red_flags      응급 신호 → 있으면 즉시 119/응급실 안내 후 종료
  3. retrieve_guidelines  별표 1 + MedlinePlus 코퍼스 검색 (한국어 → 영어 검색어)
  4. ask_followup         정보 부족 시 위험 신호부터 되묻기 (최대 3턴, 답마다 2번 재점검)
  5. find_related_conditions → recommend_department   (응급이 아닐 때만)
  6. 출력 가드레일 (단정·확률·안심 표현 차단, 출처·고지 확인)
  7. 응급 권고 시 지역을 물어본 뒤 find_emergency_rooms
   ▼
{ answer, contexts[{doc_id,text}], trace[{step,input,output}] }

데이터: data/sources/ (별표 1 목록, MedlinePlus 원본 zip)
        → src/rag/build_index.py → data/chroma/ (Titan 임베딩, 로컬 Chroma)
```

## 실행 방법

Python 3.14에서 개발했습니다. 인덱스 생성과 질의 모두 Bedrock 임베딩을 호출하므로 AWS 키가 필요합니다.

**1. 가상환경과 패키지**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. `.env` 만들기** — 아래 "환경 변수" 표를 보고 값을 넣습니다. (`.env`는 저장소에 올리지 않습니다.)

**3. 검색 인덱스 생성 (최초 1회 · 약 2분 · 임베딩 비용 수십 원 이하) — 서버 실행 전에 반드시**

```bash
python -m src.rag.build_index
```

`done. total in collection: 1740` 이 출력되면 성공입니다. 처음부터 다시 만들려면 `--rebuild` 를 붙입니다.

**4. 서버 실행**

```bash
uvicorn src.server:app --reload
```

브라우저에서 `http://127.0.0.1:8000` 을 엽니다. 현재는 `MOCK_MODE=1`(기본값)이라 UI 개발용 고정 응답이 나옵니다. `http://127.0.0.1:8000/health` 의 `index_ready` 로 인덱스 상태를 확인할 수 있습니다.

### API

`POST /query`

```json
// 요청 (session_id 는 선택: 같은 값이면 같은 대화로 이어짐)
{ "question": "어제부터 설사랑 두통이 있어요", "session_id": "abc-123" }
// 응답
{ "answer": "...", "contexts": [{ "doc_id": "...", "text": "..." }], "trace": [{ "step": "...", "input": "...", "output": "..." }] }
```

### 환경 변수

| 변수 | 용도 |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | Amazon Bedrock 호출 |
| `BEDROCK_MODELS` (선택) | 모델 우선순위(쉼표 구분). 기본은 Claude Sonnet 계열 폴백 |
| `BEDROCK_MAX_ATTEMPTS`, `BEDROCK_READ_TIMEOUT` | 모델별 재시도 횟수, 읽기 제한 시간 |
| `DISABLE_FALLBACK` | `1`이면 폴백 없이 한 모델만 사용(평가 재현용) |
| `GOV_DATA_API_ENDPOINT`, `GOV_DATA_API_KEY` | 국립중앙의료원 응급의료정보 조회 서비스(공공데이터포털) |
| `LANGSMITH_*` (선택) | 트레이싱 |
| `MOCK_MODE` | `1`이면 고정 응답 |

### 검색 인덱스(Chroma) 참고

- 구성: MedlinePlus 건강 주제(영문, 2026-11-10 기준 1,725 청크) + 응급증상 목록(별표 1, 15 청크). 임베딩은 Amazon Titan Text Embeddings V2(512차원).
- 원본은 `data/sources/` 에 포함되어 있습니다(MedlinePlus는 최근 6개 날짜 파일만 공개하므로 압축 원본을 함께 보관). 인덱스(`data/chroma/`, 약 24MB)는 위 3번으로 만들며 git에서 제외됩니다.
- `data/chroma/chroma.sqlite3` 는 VS Code의 SQLite 확장으로 열어 볼 수 있습니다. 문서 내용과 메타데이터는 `embedding_metadata`, 청크 목록은 `embeddings` 테이블에서 확인합니다.

## RAGAS 평가 결과

아직 측정하지 않았습니다. (Day 9~10에 채웁니다)

- context_recall: -
- context_precision: -
- faithfulness: -
- answer_relevancy: -

## 인-아웃 세트 통과율 (자체 평가)

세트: [evaluation/test_queries.csv](evaluation/test_queries.csv) (20건, 초안)

- 1차 (Day 9 종료): - / 20 통과
- 2차 (Day 10 개선 후): - / 20 통과
- 개선폭: - 건 (주요 개선 사항: -)

## 트라이앤에러 회고

진행 중인 기록입니다. 평가 후 보강합니다.

**시도했지만 실패한 접근 · 왜 실패했는가**
- *한국어 질의를 그대로 임베딩해 영문 코퍼스를 검색*: 12개 증상 중 0개에서 기대한 주제가 상위 5개에 들었습니다. 영어 키워드로 바꾸면 9개가 찾아져서, 쿼리 변환 단계를 유지합니다.
- *국가건강정보포털을 한국어 코퍼스로 사용*: 증상·질환 목록이 자바스크립트 검색 필터로만 열려 일괄 수집이 어렵고, 공공누리 4유형(변경 금지)과 수집 허용 여부가 불명확했습니다. API는 별도 활용 신청·승인이 필요해 일정상 제외했습니다.
- *KTAS(한국형 응급환자 분류도구) 증상별 판정표*: 공개 페이지에는 단계 정의와 대표 증상뿐이고 상세표는 회원 자료이며 저작권 표시가 있어, 설계 참고 문헌으로만 사용합니다.
- *PMC 오픈액세스 논문*: 논문별 라이선스 필터링 부담에 비해 이득이 작다고 판단해 제외했습니다.

**최종 채택한 접근 · 왜 그것으로 갔는가**
- MedlinePlus 전체 건강 주제 XML을 내려받아 **로컬 벡터 인덱스**로 구성: 건강 주제 요약이 저작권이 없고(출처 표기), 호출 제한 없이 재현 가능한 평가가 가능합니다. 응급 판단의 한국 기준은 응급의료법 시행규칙 별표 1 항목을 사용합니다.

**남은 한계 · 향후 개선 방향**
- 증상에서 진료과로 이어지는 공식 매핑 자료가 없어 진료과 추천 근거가 약합니다(채점도 정답 하나가 아닌 허용 집합 방식).
- 영문 자료를 한국어로 옮겨 쓰므로 번역 오류 가능성이 있습니다.
- 별표 1은 병원 안내 페이지를 출처로 정리했으며 공식 현행본과 대조하지 않았습니다.
- 의학적 정확성은 전문가 검증을 받지 않았습니다.
- 향후: 하이브리드 검색·리랭킹, 한국어 의학 자료 추가.

## 핵심 코드 위치

- [src/rag/corpus.py](src/rag/corpus.py) — MedlinePlus·별표 1 파싱과 청킹
- [src/rag/build_index.py](src/rag/build_index.py) — 임베딩·Chroma 인덱스 생성
- [src/rag/embeddings.py](src/rag/embeddings.py) — Titan 임베딩 래퍼
- [src/rag/index_check.py](src/rag/index_check.py) — 인덱스 없음 안내와 시작 점검
- [src/llm.py](src/llm.py) — Bedrock 호출 재시도와 모델 폴백
- [src/server.py](src/server.py) — `POST /query` API
- `src/agent.py` — 메인 에이전트 그래프 (구현 예정)
- `src/tools.py` — 도메인 도구 (구현 예정)
- `src/retriever.py` — RAG 검색 파이프라인 (구현 예정)

## 데이터 출처와 이용 조건

- **응급증상 목록:** 「응급의료에 관한 법률 시행규칙」 [별표 1] 항목. 서울아산병원 안내 페이지를 출처로 정리했습니다(수집일은 파일에 기재).
- **MedlinePlus 건강 주제:** 미국 국립의학도서관(NLM)의 [MedlinePlus.gov](https://medlineplus.gov/) 건강 주제 요약(영문). 출처: MedlinePlus, National Library of Medicine. 이 프로젝트는 NLM의 보증이나 후원을 받지 않았습니다.
- **응급의료기관 목록:** 국립중앙의료원 응급의료정보 조회 서비스(공공데이터포털).

## 라이선스

(정해지면 추가)
