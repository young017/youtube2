# 🎬 1등 유튜버 되기 - 유튜브 영상 제작을 위한 맞춤형 피드백 서비스

배포 주소 : https://yudaag.github.io/youtube/index.html

2025 유튜브 영상 데이터를 기반으로, 분석부터 맞춤형 피드백까지 모든 유튜버가 성장할 수 있는 유튜브 생태계를 만듭니다.

YouTube 영상 분석을 통한 월별 트렌드, 조회수 예측, 태그 & 제목 추천을 제공하는 종합 분석 플랫폼입니다.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 빠른 시작

```bash
# 1. 저장소 클론
git clone <repository-url>
cd youtube

# 2. 가상 환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 환경 변수 설정 (.env 파일 생성)
OPENAI_API_KEY=your_key_here

# 5. 데이터베이스 초기화
python init_database.py

# 6. 서버 실행
python fastapi_server.py
```

서버 실행 후: http://localhost:8001/docs

## 📋 목차

- [주요 기능](#-주요-기능)
- [빠른 시작](#-빠른-시작)
- [기술 스택](#-기술-스택)
- [설치 및 실행](#-설치-및-실행)
- [API 사용 예시](#-api-사용-예시)
- [환경 변수 설정](#-환경-변수-설정)
- [문제 해결](#-문제-해결)
- [프로젝트 구조](#-프로젝트-구조)

## ✨ 주요 기능

### 1. 사용자 인증 및 관리
- 회원가입, 로그인, 로그아웃
- 세션 기반 인증
- 사용자 프로필 관리
- 활동 로그 기록

### 2. 조회수 예측
- 카테고리별 ML 모델을 통한 조회수 예측
- 지원 모델: CatBoost, LightGBM, XGBoost
- 인기 확률 및 예상 조회수 제공
- 영상 정보 저장 및 관리

### 3. 태그 추천 시스템
- **기본 태그 추천**: SBERT 기반 유사도 추천
- **하이브리드 추천**: 제목 유사도 + SBERT 결합
- **태그 보정**: OpenAI GPT를 활용한 태그 개선
- **태그 강화**: OpenAI 임베딩 + GPT를 통한 고품질 태그 생성

### 4. 제목 생성
- OpenAI GPT를 활용한 제목 자동 생성
- 키워드 및 이미지 설명 기반 제목 추천
- 클릭률 최적화 전략 반영

### 5. 트렌드 분석
- Kaggle 데이터를 활용한 월별 트렌드 분석
- YouTube API를 통한 카테고리 정보 수집
- 상위 카테고리 통계 제공

## 🛠 기술 스택

### Backend
- **FastAPI**: REST API 서버
- **SQLite**: 사용자 및 영상 데이터 저장
- **Python 3.11**: 개발 언어

### Machine Learning
- **CatBoost**: 카테고리 1, 15, 19 조회수 예측
- **LightGBM**: 카테고리 10, 22, 24, 26 조회수 예측
- **XGBoost**: 카테고리 17, 20, 23, 28 조회수 예측
- **SBERT**: 태그 추천을 위한 문장 임베딩

### AI/ML 서비스
- **OpenAI API**: 태그 보정, 제목 생성
- **Hugging Face Hub**: 모델 저장 및 다운로드
- **Kaggle API**: 트렌드 데이터 수집
- **YouTube Data API**: 영상 메타데이터 수집

### 기타 라이브러리
- pandas, numpy: 데이터 처리
- sentence-transformers: 문장 임베딩
- scikit-learn: 유사도 계산

## 📁 프로젝트 구조

```
youtube/
├── fastapi_server.py          # FastAPI 메인 서버
├── database.py                # SQLite 데이터베이스 관리
├── init_database.py           # 데이터베이스 초기화 스크립트
├── enrich_tags.py             # 태그 강화 파이프라인
├── requirements.txt           # Python 패키지 의존성
├── runtime.txt                # Python 버전 명시
├── railway.json               # Railway 배포 설정
│
├── tags/                      # 태그 추천 모듈
│   ├── __init__.py
│   ├── tag_recommendation_model.py  # 태그 추천 모델
│   ├── predict_tags.py
│   ├── enrich_tags.py
│   ├── enrich_tags_openai_embed.py
│   ├── fastapi_server.py
│   └── tag_recommendation_model.pkl  # 학습된 모델
│
├── 모델/                      # 조회수 예측 모델 (Hugging Face에서 다운로드)
│   ├── catboost_model_*.cbm
│   ├── lgbm_model_*.pkl
│   └── xgb_model_*.pkl
│
├── UI/                        # 웹 프론트엔드
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── mypage.html
│   ├── feedback.html
│   ├── feedback-result.html
│   └── trend.html
│
└── docs/                      # 문서 (UI와 동일)
    └── ...
```

## 🚀 설치 및 실행

### 필수 요구사항

- Python 3.11 이상
- pip 패키지 관리자
- (선택) OpenAI API 키 (태그/제목 기능 사용 시)
- (선택) Kaggle API 키 (트렌드 분석 사용 시)
- (선택) YouTube Data API 키 (트렌드 분석 사용 시)

### 단계별 설치

#### 1. 저장소 클론

```bash
git clone <repository-url>
cd youtube
```

#### 2. 가상 환경 생성 및 활성화

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

> **참고**: CatBoost 설치 시 Python 3.13 호환성 문제가 있을 수 있습니다. Python 3.11 또는 3.12 사용을 권장합니다.

#### 4. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 필요한 API 키를 설정하세요:

```env
# OpenAI API (태그 보정, 제목 생성) - 선택사항
OPENAI_API_KEY=your_openai_api_key

# Kaggle API (트렌드 분석) - 선택사항
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key

# YouTube Data API (트렌드 분석) - 선택사항
YOUTUBE_API_KEY=your_youtube_api_key
```

#### 5. 데이터베이스 초기화

```bash
python init_database.py
```

데모 계정이 자동으로 생성됩니다:
- 이메일: `demo@youtubeanalytics.com`
- 비밀번호: `demo123`

#### 6. 서버 실행

```bash
python fastapi_server.py
```

서버가 실행되면 다음 주소에서 접근할 수 있습니다:
- 🌐 **API 서버**: http://localhost:8001
- 📚 **Swagger UI**: http://localhost:8001/docs
- 📖 **ReDoc**: http://localhost:8001/redoc

## 📚 API 문서

서버 실행 후 다음 엔드포인트에서 인터랙티브 API 문서를 확인할 수 있습니다:

- **Swagger UI**: http://localhost:8001/docs (추천)
- **ReDoc**: http://localhost:8001/redoc

## 💻 API 사용 예시

### 1. 회원가입

```python
import requests

response = requests.post("http://localhost:8001/api/auth/register", json={
    "email": "user@example.com",
    "password": "password123",
    "name": "홍길동",
    "role": "creator"
})
print(response.json())
```

### 2. 로그인

```python
response = requests.post("http://localhost:8001/api/auth/login", json={
    "email": "user@example.com",
    "password": "password123"
})
data = response.json()
session_token = data["data"]["session_token"]
```

### 3. 태그 추천

```python
response = requests.post("http://localhost:8001/api/tags/recommend", json={
    "title": "맛있는 파스타 만들기",
    "top_k": 10,
    "method": "hybrid"  # hybrid, sbert, similarity
})
tags = response.json()["recommended_tags"]
print(f"추천 태그: {tags}")
```

### 4. 조회수 예측

```python
response = requests.post(
    "http://localhost:8001/api/videos/create",
    params={"session_token": session_token},
    json={
        "title": "맛있는 파스타 만들기",
        "category": "26",  # 카테고리 ID
        "length": 15.5,  # 분
        "upload_time": "2025-01-15T18:00",
        "has_subtitles": "provided",
        "video_quality": "HD",
        "subscriber_count": 100000
    }
)
prediction = response.json()["data"]["prediction"]
print(f"예상 조회수: {prediction['predicted_views']:,}")
print(f"인기 확률: {prediction['confidence']:.1f}%")
```

### 5. 제목 생성

```python
response = requests.post("http://localhost:8001/api/titles/generate", json={
    "keyword": "파스타 레시피",
    "imageText": "크림 파스타 요리 과정",
    "n": 5
})
titles = response.json()["titles"]
for i, title in enumerate(titles, 1):
    print(f"{i}. {title}")
```

### 주요 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/auth/register` | POST | 회원가입 |
| `/api/auth/login` | POST | 로그인 |
| `/api/auth/logout` | POST | 로그아웃 |
| `/api/auth/profile` | GET | 프로필 조회 |
| `/api/tags/recommend` | POST | 기본 태그 추천 |
| `/api/tags/enrich` | POST | 태그 강화 (OpenAI) |
| `/api/titles/generate` | POST | 제목 자동 생성 |
| `/api/videos/create` | POST | 영상 저장 및 조회수 예측 |
| `/api/videos/list` | GET | 영상 목록 조회 |
| `/api/trends/update-month` | POST | 월별 트렌드 분석 |

## 🔧 환경 변수 설정

### 환경 변수 목록

| 변수명 | 설명 | 필수 여부 | 기본값 |
|--------|------|----------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 (태그 보정, 제목 생성) | 태그/제목 기능 사용 시 | - |
| `KAGGLE_USERNAME` | Kaggle 사용자명 | 트렌드 분석 사용 시 | - |
| `KAGGLE_KEY` | Kaggle API 키 | 트렌드 분석 사용 시 | - |
| `YOUTUBE_API_KEY` | YouTube Data API 키 | 트렌드 분석 사용 시 | - |
| `PORT` | 서버 포트 | 선택 | 8001 |

### API 키 발급 방법

#### OpenAI API 키
1. https://platform.openai.com/api-keys 접속
2. 계정 생성 또는 로그인
3. "Create new secret key" 클릭
4. 생성된 키를 `.env` 파일에 추가

#### Kaggle API 키
1. https://www.kaggle.com/ 접속
2. 계정 설정 → API → "Create New Token" 클릭
3. 다운로드된 `kaggle.json` 파일에서 `username`과 `key` 추출
4. `.env` 파일에 추가

#### YouTube Data API 키
1. https://console.cloud.google.com/ 접속
2. 새 프로젝트 생성
3. YouTube Data API v3 활성화
4. 사용자 인증 정보 → API 키 생성
5. `.env` 파일에 추가

### 환경 변수 설정 방법

1. 프로젝트 루트에 `.env` 파일 생성
2. 위의 변수들을 설정
3. 서버 재시작

```bash
# .env 파일 예시
OPENAI_API_KEY=sk-...
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
YOUTUBE_API_KEY=your_youtube_key
```

## 🔍 문제 해결

### 자주 발생하는 문제

#### 1. CatBoost 설치 실패

**문제**: Python 3.13에서 CatBoost 설치 실패

**해결책**:
```bash
pip install catboost --no-build-isolation
# 또는 Python 3.11/3.12 사용 권장
```

#### 2. 모델 다운로드 실패

**문제**: Hugging Face에서 모델 다운로드 실패

**해결책**:
- 인터넷 연결 확인
- Hugging Face Hub 토큰 설정 (필요 시):
```bash
pip install huggingface_hub
huggingface-cli login
```

#### 3. 데이터베이스 오류

**문제**: `youtube_analytics.db` 파일이 없음

**해결책**:
```bash
python init_database.py
```

#### 4. 포트 충돌

**문제**: 포트 8001이 이미 사용 중

**해결책**:
```bash
# 다른 포트 사용
export PORT=8002  # Linux/Mac
set PORT=8002     # Windows
python fastapi_server.py
```

#### 5. 한글 인코딩 오류

**문제**: 한글이 깨져서 표시됨

**해결책**:
- 데이터베이스가 UTF-8로 설정되어 있는지 확인
- `init_database.py`를 다시 실행하여 데이터베이스 재생성

## 📖 주요 기능 상세

### 1. 조회수 예측

카테고리별로 최적화된 ML 모델을 사용하여 조회수를 예측합니다.

**입력 데이터:**
- 영상 제목
- 카테고리 ID (필수)
- 영상 길이 (분)
- 업로드 예정 시간
- 자막 제공 여부 (`provided` / `not_provided`)
- 해상도 품질 (`HD` / `SD`)
- 구독자 수

**출력:**
- 예상 조회수 (정수)
- 인기 확률 (0-1 사이의 실수)
- 사용된 모델 정보

**지원 카테고리 및 모델:**
- **CatBoost**: 카테고리 1, 15, 19
- **LightGBM**: 카테고리 10, 22, 24, 26
- **XGBoost**: 카테고리 17, 20, 23, 28

모델은 Hugging Face Hub에서 자동으로 다운로드됩니다.

### 2. 태그 추천 시스템

#### 기본 태그 추천 (`/api/tags/recommend`)
- **SBERT 기반**: 제목-태그 직접 유사도 계산
- **유사 제목 기반**: 유사한 제목의 태그 추천
- **하이브리드**: 두 방법을 결합한 추천 (기본값)

**사용 예시:**
```python
# 하이브리드 방식 (기본)
response = requests.post("/api/tags/recommend", json={
    "title": "맛있는 파스타 만들기",
    "top_k": 10,
    "method": "hybrid"
})
```

#### 태그 강화 (`/api/tags/enrich`)
OpenAI 임베딩과 GPT를 활용한 고품질 태그 생성

**처리 과정:**
1. 기본 모델로 후보 태그 생성
2. OpenAI 임베딩으로 제목-태그 유사도 재계산
3. 유사도 임계값(기본 0.30) 기반 필터링
4. GPT를 통한 최종 태그 보정 및 추가 태그 생성

**파라미터:**
- `top_k`: 후보 태그 개수 (기본 15)
- `title_sim_threshold`: 제목 유사도 임계값 (기본 0.30)
- `tag_abs_threshold`: 태그 유사도 임계값 (기본 0.30)
- `extra_k`: 추가 태그 개수 (기본 10)

### 3. 제목 생성

OpenAI GPT를 활용하여 키워드 기반 제목을 생성합니다.

**최적화 전략:**
- ✅ 키워드 앞부분 배치 (SEO)
- ✅ 숫자/괄호 활용 (클릭률 향상)
- ✅ 질문형, 가치 제시형 제목
- ✅ 홀수 숫자 활용 (7, 9 등)
- ✅ 감탄사/수식어 활용

**사용 예시:**
```python
response = requests.post("/api/titles/generate", json={
    "keyword": "파스타 레시피",
    "imageText": "크림 파스타 요리 과정",  # 선택사항
    "n": 5  # 생성할 제목 개수
})
```

### 4. 트렌드 분석

Kaggle의 YouTube 트렌드 데이터를 활용하여 월별 인기 카테고리를 분석합니다.

**기능:**
- 2025년 한국(KR) 데이터 자동 필터링
- 월별 상위 5개 카테고리 통계
- YouTube API를 통한 카테고리 정보 수집
- 상위 30% 영상 기준 분석

**사용 예시:**
```python
response = requests.post("/api/trends/update-month?month=1")
trends = response.json()["data"]["trends"]
```

## 🗄 데이터베이스

SQLite 데이터베이스를 사용하며, UTF-8 인코딩을 지원합니다.

### 테이블 구조

| 테이블명 | 설명 |
|---------|------|
| `users` | 사용자 정보 (이메일, 비밀번호 해시, 프로필 등) |
| `user_sessions` | 세션 관리 (토큰, 만료 시간 등) |
| `user_activity_logs` | 사용자 활동 로그 (로그인, 회원가입 등) |
| `videos` | 영상 정보 (제목, 카테고리, 조회수 예측 결과 등) |

**데이터베이스 파일**: `youtube_analytics.db` (프로젝트 루트)

### 데이터베이스 초기화

```bash
python init_database.py
```

초기화 시 데모 계정이 자동으로 생성됩니다:
- **데모 계정**: `demo@youtubeanalytics.com` / `demo123`
- **관리자 계정**: `admin@youtubeanalytics.com` / `admin123`

## 🚢 배포

### Railway 배포

`railway.json` 파일이 포함되어 있어 Railway에서 바로 배포할 수 있습니다.

**배포 단계:**
1. [Railway](https://railway.app/)에 로그인
2. "New Project" → "Deploy from GitHub repo" 선택
3. 이 저장소 연결
4. 환경 변수 설정 (`.env` 파일의 변수들)
5. 자동 배포 완료

**Railway 환경 변수 설정:**
- Railway 대시보드 → Variables 탭에서 설정
- 또는 `.env` 파일을 Railway에 업로드

### 기타 플랫폼

#### Heroku
```bash
# Procfile 생성
echo "web: uvicorn fastapi_server:app --host 0.0.0.0 --port \$PORT" > Procfile

# 배포
git push heroku main
```

#### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 일반 서버
```bash
# Python 3.11 필수
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
```

**주의사항:**
- 모든 환경 변수를 서버에 설정해야 합니다
- 모델은 첫 실행 시 Hugging Face에서 자동 다운로드됩니다
- 데이터베이스 파일(`youtube_analytics.db`)은 영구 저장소에 저장하세요

## 📁 프로젝트 구조

```
youtube/
├── fastapi_server.py          # FastAPI 메인 서버
├── database.py                # SQLite 데이터베이스 관리
├── init_database.py           # 데이터베이스 초기화 스크립트
├── enrich_tags.py             # 태그 강화 파이프라인
├── requirements.txt           # Python 패키지 의존성
├── runtime.txt                # Python 버전 명시
├── railway.json               # Railway 배포 설정
│
├── tags/                      # 태그 추천 모듈
│   ├── tag_recommendation_model.py
│   ├── predict_tags.py
│   └── ...
│
├── 모델/                      # 조회수 예측 모델 (Hugging Face에서 다운로드)
│   ├── catboost_model_*.cbm
│   ├── lgbm_model_*.pkl
│   └── xgb_model_*.pkl
│
├── UI/                        # 웹 프론트엔드
│   ├── index.html
│   ├── login.html
│   └── ...
│
└── docs/                      # 문서
    └── ...
```

## 🤝 기여

기여를 환영합니다! 다음 방법으로 기여할 수 있습니다:

1. **이슈 등록**: 버그 리포트나 기능 제안
2. **Pull Request**: 코드 개선 사항 제출
3. **문서 개선**: README나 문서 개선

### 개발 환경 설정

```bash
# 개발용 패키지 설치
pip install -r requirements.txt
pip install pytest  # 테스트용

# 코드 포맷팅 (선택사항)
pip install black isort
```

## 📝 라이선스

이 프로젝트의 라이선스 정보를 여기에 추가하세요.

## 📧 문의 및 지원

- **이슈 등록**: GitHub Issues에서 버그 리포트나 기능 제안
- **문서**: `/docs` 엔드포인트에서 API 문서 확인

## 🙏 감사의 말

이 프로젝트는 YouTube 크리에이터를 위한 도구로 개발되었습니다.

---

**Made with ❤️ for YouTube Creators**

> 💡 **팁**: API 문서는 서버 실행 후 http://localhost:8001/docs 에서 확인하세요!
