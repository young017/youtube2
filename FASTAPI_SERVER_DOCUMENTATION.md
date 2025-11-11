# FastAPI Server 기능 문서

YouTube Analytics FastAPI 서버의 주요 기능과 API 엔드포인트를 정리한 문서입니다.

## 📋 목차

1. [서버 개요](#서버-개요)
2. [주요 기능](#주요-기능)
3. [API 엔드포인트](#api-엔드포인트)
4. [기술 스택](#기술-스택)

---

## 🎯 서버 개요

**FastAPI 기반 REST API 서버**로 YouTube 영상 분석 및 태그 추천 기능을 제공합니다.

- **서버 포트**: `8001` (기본값)
- **API 문서**: `/docs` (Swagger UI), `/redoc` (ReDoc)
- **데이터베이스**: SQLite (`youtube_analytics.db`)
- **CORS**: 활성화 (모든 오리진 허용)

---

## 🚀 주요 기능

### 1. 사용자 인증 시스템
- 회원가입, 로그인, 로그아웃
- 세션 기반 인증
- 프로필 조회 및 수정
- 사용자 활동 로깅

### 2. AI 기반 태그 추천
- **3가지 추천 방법**:
  - `hybrid`: SBERT + 유사도 기반 하이브리드 추천
  - `sbert`: SBERT 임베딩 기반 추천
  - `similarity`: 유사한 제목 기반 추천
- 태그 수정 및 보강 기능
- OpenAI를 활용한 태그 보강

### 3. 제목 추천
- OpenAI GPT-4o-mini를 활용한 제목 생성
- 유튜브 최적화 전략 반영
- 키워드 및 이미지 텍스트 기반 제목 생성

### 4. 조회수 예측
- **카테고리별 ML 모델 사용**:
  - CatBoost (카테고리 1, 15, 19)
  - LightGBM (카테고리 10, 22, 24, 26)
  - XGBoost (카테고리 17, 20, 23, 28)
- 분류 모델 + 회귀 모델 2단계 예측
- 인기도 확률 및 예상 조회수 제공

### 5. 영상 정보 관리
- 영상 정보 저장 (제목, 카테고리, 길이, 설명 등)
- 사용자별 영상 목록 조회
- 전체 영상 목록 조회

### 6. 시스템 통계
- 전체 사용자 수
- 오늘 가입한 사용자 수
- 활성 세션 수
- 역할별 사용자 분포

---

## 📡 API 엔드포인트

### 🔐 인증 API (`/api/auth/*`)

#### 1. **회원가입** 
- **엔드포인트**: `POST /api/auth/register`
- **요청 본문**:
  ```json
  {
    "email": "user@example.com",
    "password": "password123",
    "name": "사용자 이름",
    "role": "user",
    "profile_data": {}
  }
  ```
- **응답**: 사용자 정보 및 성공 메시지

#### 2. **로그인**
- **엔드포인트**: `POST /api/auth/login`
- **요청 본문**:
  ```json
  {
    "email": "user@example.com",
    "password": "password123"
  }
  ```
- **응답**: 사용자 정보 및 세션 토큰

#### 3. **로그아웃**
- **엔드포인트**: `POST /api/auth/logout`
- **요청 본문**:
  ```json
  {
    "session_token": "세션토큰"
  }
  ```

#### 4. **프로필 조회**
- **엔드포인트**: `GET /api/auth/profile?session_token=토큰`
- **응답**: 사용자 프로필 정보

#### 5. **프로필 수정**
- **엔드포인트**: `PUT /api/auth/profile?session_token=토큰`
- **요청 본문**:
  ```json
  {
    "profile_data": {
      "bio": "자기소개",
      "preferences": {}
    }
  }
  ```

---

### 🏷️ 태그 추천 API (`/api/tags/*`)

#### 1. **태그 추천**
- **엔드포인트**: `POST /api/tags/recommend`
- **요청 본문**:
  ```json
  {
    "title": "영상 제목",
    "top_k": 10,
    "method": "hybrid"  // "hybrid", "sbert", "similarity"
  }
  ```
- **응답**:
  ```json
  {
    "success": true,
    "title": "영상 제목",
    "recommended_tags": ["태그1", "태그2", ...],
    "method": "hybrid",
    "similar_titles": [...]
  }
  ```

#### 2. **태그 수정**
- **엔드포인트**: `POST /api/tags/refine`
- **요청 본문**:
  ```json
  {
    "title": "영상 제목",
    "candidate_tags": ["태그1", "태그2", ...]
  }
  ```
- **기능**: 후보 태그를 제목의 문맥에 맞게 수정

#### 3. **태그 보강 (OpenAI)**
- **엔드포인트**: `POST /api/tags/enrich`
- **요청 본문**:
  ```json
  {
    "title": "영상 제목",
    "description": "영상 설명",
    "top_k": 15,
    "title_sim_threshold": 0.30,
    "tag_abs_threshold": 0.30,
    "extra_k": 10,
    "api_key": "optional"
  }
  ```
- **기능**: 
  - SBERT 기반 태그 추천
  - OpenAI를 통한 태그 보강 및 추가 태그 생성
  - 유사도 기반 필터링

---

### 📝 제목 추천 API

#### **제목 생성**
- **엔드포인트**: `POST /api/titles/generate`
- **요청 본문**:
  ```json
  {
    "keyword": "주제 키워드",
    "imageText": "이미지 내용 요약 (선택)",
    "n": 5
  }
  ```
- **응답**:
  ```json
  {
    "success": true,
    "titles": ["제목1", "제목2", ...]
  }
  ```
- **기능**: 
  - OpenAI GPT-4o-mini 사용
  - 유튜브 최적화 전략 반영
  - 클릭률 향상을 위한 제목 생성

---

### 🎬 영상 관리 API (`/api/videos/*`)

#### 1. **영상 정보 저장 및 조회수 예측**
- **엔드포인트**: `POST /api/videos/create?session_token=토큰`
- **요청 본문**:
  ```json
  {
    "title": "영상 제목",
    "category": "1",
    "length": 10,
    "upload_time": "2024-01-01T12:00",
    "description": "영상 설명",
    "thumbnail_image": "base64...",
    "has_subtitles": "provided",
    "video_quality": "HD",
    "subscriber_count": 100000
  }
  ```
- **응답**:
  ```json
  {
    "success": true,
    "message": "영상 정보가 저장되었습니다.",
    "data": {
      "video": {...},
      "prediction": {
        "predicted_views": 50000,
        "pred_popular_prob": 0.75,
        "confidence": 75.0,
        "cls_model": "CatBoost Classifier",
        "reg_model": "CatBoost Regressor",
        "model_type": "catboost"
      }
    }
  }
  ```
- **기능**:
  - 영상 정보를 데이터베이스에 저장
  - 카테고리별 ML 모델로 조회수 예측
  - 인기도 확률 계산

#### 2. **영상 목록 조회**
- **엔드포인트**: `GET /api/videos/list?session_token=토큰&limit=100&offset=0`
- **기능**:
  - 로그인 시: 사용자별 영상 목록
  - 비로그인 시: 전체 영상 목록

---

### 📊 시스템 API

#### **시스템 통계**
- **엔드포인트**: `GET /api/stats`
- **응답**:
  ```json
  {
    "success": true,
    "data": {
      "stats": {
        "total_users": 100,
        "today_signups": 5,
        "active_sessions": 20,
        "role_distribution": {
          "user": 80,
          "creator": 15,
          "admin": 5
        }
      }
    }
  }
  ```

---

## 🔧 기술 스택

### Backend Framework
- **FastAPI**: 고성능 비동기 웹 프레임워크
- **Pydantic**: 데이터 검증 및 직렬화
- **Uvicorn**: ASGI 서버

### Machine Learning
- **태그 추천 모델**:
  - SBERT (Sentence-BERT) 임베딩
  - 유사도 기반 추천

- **조회수 예측 모델**:
  - **CatBoost**: 카테고리 1, 15, 19
  - **LightGBM**: 카테고리 10, 22, 24, 26
  - **XGBoost**: 카테고리 17, 20, 23, 28

### AI/LLM
- **OpenAI GPT-4o-mini**: 
  - 태그 보강 (`/api/tags/enrich`)
  - 제목 생성 (`/api/titles/generate`)

### 데이터베이스
- **SQLite**: 사용자 정보, 영상 정보 저장
- **데이터베이스 모듈**: `database.py`

### 보안
- **비밀번호 해시화**: PBKDF2 (SHA-256, 100,000회 반복)
- **세션 관리**: 토큰 기반 인증
- **CORS**: 크로스 오리진 요청 허용

### 데이터 처리
- **Pandas**: 데이터 전처리 및 변환
- **NumPy**: 수치 연산

---

## 📈 조회수 예측 프로세스

### 1단계: 데이터 전처리
```
사용자 입력 → 전처리 함수
- 업로드 시간 파싱 (월, 일, 시간, 요일)
- 시간/요일을 sin/cos 변환 (순환 특성)
- 구독자 수 로그 변환
- 자막/화질 이진 변환
```

### 2단계: 분류 모델 예측
```
전처리된 데이터 → 분류 모델 (Classifier)
→ 인기도 확률 (pred_popular_prob)
```

### 3단계: 회귀 모델 예측
```
전처리된 데이터 + pred_popular_prob → 회귀 모델 (Regressor)
→ 예상 조회수 (로그 스케일)
```

### 4단계: 스케일 변환
```
로그 스케일 조회수 → 실제 조회수
- 카테고리 10, 23: 100만 단위
- 기타: 10만 단위

최종 결과: 예상 조회수 + 인기도 확률
```

---

## 🎨 주요 특징

### 1. 한글 지원
- UTF-8 인코딩 보장
- `UTF8JSONResponse` 클래스로 한글 응답 처리

### 2. 자동 API 문서
- Swagger UI (`/docs`)
- ReDoc (`/redoc`)

### 3. 에러 처리
- HTTP 상태 코드 기반 에러 응답
- 상세한 에러 메시지 제공

### 4. 활동 로깅
- 모든 사용자 활동 기록
- IP 주소 및 User Agent 추적

### 5. 모델 캐싱
- 조회수 예측 모델 메모리 캐싱
- 태그 추천 모델 전역 로드

---

## 📝 사용 예시

### Python 클라이언트 예시

```python
import requests

BASE_URL = "http://localhost:8001"

# 1. 회원가입
response = requests.post(
    f"{BASE_URL}/api/auth/register",
    json={
        "email": "test@example.com",
        "password": "password123",
        "name": "테스트 사용자",
        "role": "user"
    }
)
print(response.json())

# 2. 로그인
response = requests.post(
    f"{BASE_URL}/api/auth/login",
    json={
        "email": "test@example.com",
        "password": "password123"
    }
)
data = response.json()
session_token = data["data"]["session_token"]

# 3. 태그 추천
response = requests.post(
    f"{BASE_URL}/api/tags/recommend",
    json={
        "title": "브이로그 일상",
        "top_k": 10,
        "method": "hybrid"
    }
)
print(response.json())

# 4. 제목 생성
response = requests.post(
    f"{BASE_URL}/api/titles/generate",
    json={
        "keyword": "먹방",
        "n": 5
    }
)
print(response.json())

# 5. 영상 저장 및 조회수 예측
response = requests.post(
    f"{BASE_URL}/api/videos/create",
    params={"session_token": session_token},
    json={
        "title": "맛있는 음식 먹방",
        "category": "1",
        "length": 15,
        "upload_time": "2024-12-25T18:00",
        "has_subtitles": "provided",
        "video_quality": "HD",
        "subscriber_count": 50000
    }
)
print(response.json())
```

---

## 🔍 모델 파일 구조

```
모델/
├── catboost_model_1_class.cbm      # 카테고리 1 분류 모델
├── catboost_model_1.cbm             # 카테고리 1 회귀 모델
├── catboost_model_15_class.cbm
├── catboost_model_15.cbm
├── catboost_model_19_class.cbm
├── catboost_model_19.cbm
├── lgbm_model_10_class.pkl           # 카테고리 10 분류 모델
├── lgbm_model_10.pkl                 # 카테고리 10 회귀 모델
├── lgbm_model_22_class.pkl
├── lgbm_model_22.pkl
├── lgbm_model_24_class.pkl
├── lgbm_model_24.pkl
├── lgbm_model_26_class.pkl
├── lgbm_model_26.pkl
├── xgb_model_17_class.pkl            # 카테고리 17 분류 모델
├── xgb_model_17.pkl                  # 카테고리 17 회귀 모델
├── xgb_model_20_class.pkl
├── xgb_model_20.pkl
├── xgb_model_23_class.pkl
├── xgb_model_23.pkl
├── xgb_model_28_class.pkl
└── xgb_model_28.pkl
```

---

## ⚙️ 환경 변수

필요한 환경 변수:
- `OPENAI_API_KEY`: OpenAI API 키 (태그 보강, 제목 생성에 사용)

`.env` 파일 예시:
```
OPENAI_API_KEY=sk-...
```

---

## 🚀 서버 실행

```bash
# 1. 데이터베이스 초기화
python init_database.py

# 2. 서버 실행
python fastapi_server.py

# 또는 uvicorn 직접 실행
uvicorn fastapi_server:app --host 0.0.0.0 --port 8001
```

---

## 📊 API 응답 형식

모든 API는 다음 형식을 따릅니다:

**성공 응답**:
```json
{
  "success": true,
  "message": "성공 메시지",
  "data": {...}
}
```

**에러 응답**:
```json
{
  "detail": "에러 메시지"
}
```

HTTP 상태 코드:
- `200`: 성공
- `400`: 잘못된 요청
- `401`: 인증 실패
- `500`: 서버 오류
- `503`: 서비스 사용 불가 (모델 로드 실패 등)

---

## 🎯 주요 기능 요약

| 기능 | 엔드포인트 | 설명 |
|------|-----------|------|
| 회원가입 | `POST /api/auth/register` | 새 사용자 등록 |
| 로그인 | `POST /api/auth/login` | 세션 토큰 발급 |
| 태그 추천 | `POST /api/tags/recommend` | 3가지 방법으로 태그 추천 |
| 태그 보강 | `POST /api/tags/enrich` | OpenAI로 태그 보강 |
| 제목 생성 | `POST /api/titles/generate` | GPT로 제목 생성 |
| 조회수 예측 | `POST /api/videos/create` | ML 모델로 조회수 예측 |
| 영상 목록 | `GET /api/videos/list` | 저장된 영상 조회 |
| 통계 | `GET /api/stats` | 시스템 통계 |

---

이 문서는 PPT 발표 자료 작성을 위한 참고 자료로 활용하실 수 있습니다.

