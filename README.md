# 🎬 1등 유튜버 되기 - 유튜브 영상 제작을 위한 맞춤형 피드백 서비스

2025 유튜브 영상 데이터를 기반으로, 분석부터 맞춤형 피드백까지 모든 유튜버가 성장할 수 있는 유튜브 생태계를 만듭니다.

YouTube 영상 분석을 통한 월별 트렌드, 조회수 예측, 태그 & 제목 추천을 제공하는 종합 분석 플랫폼입니다.

## 📋 목차

- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 실행](#설치-및-실행)
- [API 문서](#api-문서)
- [환경 변수 설정](#환경-변수-설정)
- [주요 기능 상세](#주요-기능-상세)

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

### 1. 저장소 클론

```bash
git clone <repository-url>
cd youtube
```

### 2. 가상 환경 생성 및 활성화

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성하고 다음 변수들을 설정하세요:

```env
# OpenAI API (태그 보정, 제목 생성)
OPENAI_API_KEY=your_openai_api_key

# Kaggle API (트렌드 분석)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key

# YouTube Data API (트렌드 분석)
YOUTUBE_API_KEY=your_youtube_api_key
```

### 5. 데이터베이스 초기화

```bash
python init_database.py
```

### 6. 서버 실행

```bash
python fastapi_server.py
```

서버가 실행되면 다음 주소에서 접근할 수 있습니다:
- API 서버: http://localhost:8001
- API 문서 (Swagger): http://localhost:8001/docs
- API 문서 (ReDoc): http://localhost:8001/redoc

## 📚 API 문서

서버 실행 후 다음 엔드포인트에서 API 문서를 확인할 수 있습니다:

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

### 주요 API 엔드포인트

#### 인증
- `POST /api/auth/register` - 회원가입
- `POST /api/auth/login` - 로그인
- `POST /api/auth/logout` - 로그아웃
- `GET /api/auth/profile` - 프로필 조회
- `PUT /api/auth/profile` - 프로필 업데이트

#### 태그 추천
- `POST /api/tags/recommend` - 기본 태그 추천
- `POST /api/tags/refine` - 태그 보정
- `POST /api/tags/enrich` - 태그 강화 (OpenAI 임베딩 + GPT)

#### 제목 생성
- `POST /api/titles/generate` - 제목 자동 생성

#### 영상 관리
- `POST /api/videos/create` - 영상 정보 저장 및 조회수 예측
- `GET /api/videos/list` - 영상 목록 조회

#### 트렌드 분석
- `POST /api/trends/update-month` - 월별 트렌드 분석
- `GET /api/trends/test-kaggle` - Kaggle 데이터 다운로드 테스트

## 🔧 환경 변수 설정

### 필수 환경 변수

| 변수명 | 설명 | 필수 여부 |
|--------|------|----------|
| `OPENAI_API_KEY` | OpenAI API 키 (태그 보정, 제목 생성) | 태그/제목 기능 사용 시 필수 |
| `KAGGLE_USERNAME` | Kaggle 사용자명 | 트렌드 분석 사용 시 필수 |
| `KAGGLE_KEY` | Kaggle API 키 | 트렌드 분석 사용 시 필수 |
| `YOUTUBE_API_KEY` | YouTube Data API 키 | 트렌드 분석 사용 시 필수 |

### 환경 변수 설정 방법

1. 프로젝트 루트에 `.env` 파일 생성
2. 위의 변수들을 설정
3. 서버 재시작

## 📖 주요 기능 상세

### 조회수 예측

카테고리별로 최적화된 ML 모델을 사용하여 조회수를 예측합니다.

**입력 데이터:**
- 영상 제목
- 카테고리 ID
- 영상 길이 (분)
- 업로드 예정 시간
- 자막 제공 여부
- 해상도 품질 (HD/SD)
- 구독자 수

**출력:**
- 예상 조회수
- 인기 확률 (0-1)
- 신뢰도 (%)

**지원 카테고리:**
- CatBoost: 1, 15, 19
- LightGBM: 10, 22, 24, 26
- XGBoost: 17, 20, 23, 28

### 태그 추천

#### 1. 기본 태그 추천 (`/api/tags/recommend`)
- SBERT 기반 제목-태그 유사도 계산
- 유사한 제목 기반 태그 추천
- 하이브리드 방식 지원

#### 2. 태그 보정 (`/api/tags/refine`)
- 후보 태그를 제목에 맞게 수정
- 간단한 규칙 기반 보정

#### 3. 태그 강화 (`/api/tags/enrich`)
- OpenAI 임베딩으로 제목-태그 유사도 재계산
- 유사도 임계값 기반 필터링
- GPT를 통한 최종 태그 보정 및 추가

### 제목 생성

OpenAI GPT를 활용하여 키워드 기반 제목을 생성합니다.

**특징:**
- 클릭률 최적화 전략 반영
- 숫자/괄호 활용
- 질문형, 가치 제시형 제목 생성
- 이미지 설명 기반 제목 생성 지원

### 트렌드 분석

Kaggle의 YouTube 트렌드 데이터를 활용하여 월별 인기 카테고리를 분석합니다.

**기능:**
- 2025년 한국(KR) 데이터 필터링
- 월별 상위 5개 카테고리 통계
- YouTube API를 통한 카테고리 정보 수집

## 🗄 데이터베이스

SQLite 데이터베이스를 사용하며, 다음 테이블을 포함합니다:

- `users`: 사용자 정보
- `user_sessions`: 세션 관리
- `user_activity_logs`: 활동 로그
- `videos`: 영상 정보

데이터베이스 파일: `youtube_analytics.db`

## 🚢 배포

### Railway 배포

`railway.json` 파일이 포함되어 있어 Railway에서 바로 배포할 수 있습니다.

1. Railway에 프로젝트 연결
2. 환경 변수 설정
3. 자동 배포

### 기타 플랫폼

다른 플랫폼에서 배포 시:
- Python 3.11 사용
- `uvicorn fastapi_server:app --host 0.0.0.0 --port $PORT` 명령어로 실행
- 환경 변수 설정 필수

## 📝 라이선스

이 프로젝트의 라이선스 정보를 여기에 추가하세요.

## 🤝 기여

기여를 환영합니다! 이슈를 등록하거나 Pull Request를 보내주세요.

## 📧 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 등록해주세요.

---

**Made with ❤️ for YouTube Creators**
