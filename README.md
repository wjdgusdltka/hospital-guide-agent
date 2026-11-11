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

개발 중입니다. 현재는 API 서버와 UI 자리(mock 응답), Bedrock 모델 연결(재시도 후 폴백)까지 준비되어 있습니다. RAG 코퍼스, 에이전트 로직, 평가 세트는 구현 예정입니다.

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

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn src.server:app --reload
```

브라우저에서 `http://127.0.0.1:8000` 을 열면 됩니다. 현재는 `MOCK_MODE=1`(기본값)이라 고정 응답이 나옵니다.

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
