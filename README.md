# 🎬 YouTube Analytics - AI 기반 영상 분석 플랫폼

YouTube 영상 데이터를 AI로 분석하고 트렌드를 확인할 수 있는 웹 애플리케이션입니다.

## ✨ 주요 기능

- **🤖 AI 기반 분석**: 조회수 예측, 태그 추천, 제목 추천
- **📊 월별 트렌드 분석**: 카테고리별 인기도 변화 추적
- **👤 사용자 인증 시스템**: 회원가입, 로그인, 프로필 관리
- **💾 분석 결과 저장**: 마이페이지에서 저장된 분석 결과 확인
- **📱 반응형 웹 디자인**: 모바일과 데스크톱 모두 지원

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone https://github.com/[사용자명]/youtube-analytics.git
cd youtube-analytics
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 데이터베이스 초기화
```bash
python init_database.py
```

### 5. 환경 변수 설정
`.env` 파일을 생성하고 다음 환경 변수를 설정하세요:
```bash
OPENAI_API_KEY=your-openai-api-key
KAGGLE_USERNAME=your-kaggle-username
KAGGLE_KEY=your-kaggle-key
YOUTUBE_API_KEY=your-youtube-api-key
```

### 6. 서버 실행
```bash
python fastapi_server.py
```

### 7. 웹 페이지 접속
- **API 서버**: http://localhost:8001
- **API 문서**: http://localhost:8001/docs (Swagger UI)
- **ReDoc 문서**: http://localhost:8001/redoc
- **웹 인터페이스**: `UI/index.html` 파일을 브라우저에서 열기

## ⚠️ 주의사항

- **AI 모델 파일**: ML 모델들은 Hugging Face Hub에서 자동으로 다운로드됩니다 (`yudaag/youtube-view-predict-models`)
- **태그 추천 모델**: `tags/tag_recommendation_model.pkl` 파일이 존재해야 태그 추천이 정상 작동합니다
- **데이터베이스**: `youtube_analytics.db`는 `init_database.py` 실행 시 자동으로 생성됩니다
- **환경 변수**: OpenAI, Kaggle, YouTube API 키가 필요합니다 (`.env` 파일에 설정)
- **Python 버전**: Python 3.11 이상 권장 (CatBoost 호환성)

## 사용 가능한 페이지

- `UI/index.html` - 메인 페이지
- `UI/login.html` - 로그인 페이지
- `UI/signup.html` - 회원가입 페이지
- `UI/mypage.html` - 마이페이지 (저장된 분석 결과 확인)
- `UI/trend.html` - 월별 트렌드 분석
- `UI/feedback.html` - 영상 피드백 입력
- `UI/feedback-result.html` - 피드백 결과 확인

## 사용법

1. **메인 페이지**
   - `UI/index.html`에서 전체 서비스 개요를 확인할 수 있습니다
   - 월별 트렌드와 영상 분석 기능으로 이동할 수 있습니다

2. **영상 분석 및 조회수 예측**
   - `UI/feedback.html`에서 영상 정보(제목, 카테고리, 길이, 업로드 시간 등)를 입력할 수 있습니다
   - 입력한 정보를 기반으로 AI가 조회수를 예측합니다
   - `UI/feedback-result.html`에서 예측 결과를 확인할 수 있습니다
   - 로그인한 사용자는 분석 결과가 마이페이지에 저장됩니다

3. **월별 트렌드 분석**
   - `UI/trend.html`에서 2025년 월별 트렌드 데이터를 확인할 수 있습니다
   - 각 월별로 인기 카테고리 TOP5를 확인할 수 있습니다
   - 탭을 클릭하여 월별 데이터를 전환할 수 있습니다
   - 트렌드 데이터는 Kaggle API를 통해 최신 데이터로 업데이트할 수 있습니다

4. **마이페이지**
   - `UI/mypage.html`에서 저장된 영상 분석 결과를 확인할 수 있습니다
   - 로그인 후 사용한 모든 분석 결과를 조회할 수 있습니다

## 기술 스택

- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Backend**: Python, FastAPI, Uvicorn
- **Database**: SQLite
- **ML/AI**: 
  - CatBoost, LightGBM, XGBoost (조회수 예측)
  - Sentence-BERT (태그 추천)
  - OpenAI GPT (제목 생성, 태그 보정)
- **API**: 
  - Kaggle API (트렌드 데이터)
  - YouTube Data API v3 (카테고리 정보)
- **차트**: Chart.js
- **아이콘**: Bootstrap Icons
- **스타일**: 반응형 웹 디자인

### 데이터베이스 선택

본 프로젝트는 **SQLite**를 데이터베이스로 사용합니다. 초기에는 PostgreSQL을 고려했으나, 다음과 같은 이유로 SQLite를 선택했습니다:

- **간편한 설정**: 별도의 데이터베이스 서버 설치 및 설정이 불필요하여 개발 환경 구성이 간단합니다
- **빠른 프로토타이핑**: 개발 초기 단계에서 빠르게 프로토타입을 구축하고 테스트할 수 있습니다
- **프로젝트 규모 적합성**: 현재 프로젝트의 데이터 규모와 동시 접속자 수에 적합한 선택입니다
- **배포 용이성**: 단일 파일로 관리되어 배포 및 백업이 간편합니다
- **Python 표준 라이브러리**: 추가 패키지 설치 없이 바로 사용 가능합니다

향후 사용자 수 증가나 더 복잡한 쿼리 요구사항이 생길 경우, PostgreSQL로 마이그레이션을 고려할 수 있습니다.

## API 엔드포인트

### 인증 API
- `POST /api/auth/register` - 회원가입
- `POST /api/auth/login` - 로그인
- `POST /api/auth/logout` - 로그아웃
- `GET /api/auth/profile` - 프로필 조회
- `PUT /api/auth/profile` - 프로필 업데이트

### 태그 추천 API
- `POST /api/tags/recommend` - 제목 기반 태그 추천
- `POST /api/tags/refine` - 프롬프트 기반 태그 수정
- `POST /api/tags/enrich` - 제목/설명 기반 태그 보정 (OpenAI 사용)

### 제목 생성 API
- `POST /api/titles/generate` - 키워드 기반 제목 생성 (OpenAI GPT 사용)

### 영상 분석 API
- `POST /api/videos/create` - 영상 정보 저장 및 조회수 예측
- `GET /api/videos/list` - 영상 목록 조회

### 트렌드 분석 API
- `POST /api/trends/update-month` - 특정 월의 트렌드 분석 업데이트 (Kaggle + YouTube API 사용)
- `GET /api/trends/test-kaggle` - Kaggle 데이터 다운로드 테스트

### 시스템 API
- `GET /api/stats` - 시스템 통계

## 데모 계정

- **이메일**: demo@youtubeanalytics.com
- **비밀번호**: demo123
- **역할**: 크리에이터

- **관리자 계정**:
  - **이메일**: admin@youtubeanalytics.com
  - **비밀번호**: admin123
  - **역할**: 관리자

## 파일 구조

```
youtube-main배포완료 코드/
├── README.md                # 프로젝트 문서
├── requirements.txt         # Python 패키지 의존성
├── runtime.txt              # Python 런타임 버전
├── railway.json             # Railway 배포 설정
├── database.py              # 데이터베이스 관리 모듈
├── init_database.py         # 데이터베이스 초기화 스크립트
├── fastapi_server.py        # FastAPI 웹 서버 (메인)
├── enrich_tags.py           # 태그 보정 모듈
├── youtube_analytics.db      # SQLite 데이터베이스 파일 (자동 생성)
├── UI/                      # 웹 인터페이스
│   ├── index.html           # 메인 페이지
│   ├── login.html           # 로그인 페이지
│   ├── signup.html          # 회원가입 페이지
│   ├── mypage.html          # 마이페이지
│   ├── trend.html           # 월별 트렌드 분석
│   ├── feedback.html        # 피드백 입력 페이지
│   └── feedback-result.html # 피드백 결과 페이지
├── tags/                    # 태그 추천 모듈
│   ├── __init__.py
│   ├── fastapi_server.py    # 태그 관련 API (사용 안 함)
│   ├── tag_recommendation_model.py  # 태그 추천 모델
│   ├── tag_recommendation_model.pkl # 태그 추천 모델 파일
│   ├── predict_tags.py      # 태그 예측 스크립트
│   ├── enrich_tags.py       # 태그 보정 모듈
│   └── enrich_tags_openai_embed.py  # OpenAI 임베딩 태그 보정
├── 모델/                    # ML 모델 파일들 (Hugging Face Hub에서 다운로드)
│   ├── catboost_model_*.cbm # CatBoost 모델 (카테고리 1, 15, 19)
│   ├── lgbm_model_*.pkl     # LightGBM 모델 (카테고리 10, 22, 24, 26)
│   └── xgb_model_*.pkl      # XGBoost 모델 (카테고리 17, 20, 23, 28)
└── docs/                    # 문서용 HTML 파일들
    └── (UI와 동일한 구조)
```

## 보안 기능

- **비밀번호 해시화**: PBKDF2를 사용한 안전한 비밀번호 저장
- **세션 관리**: 토큰 기반 세션 인증
- **CORS 지원**: 크로스 오리진 요청 허용
- **입력 검증**: 서버 측 데이터 유효성 검사
- **활동 로깅**: 사용자 활동 추적 및 기록

## 주요 기능 상세

### 1. 조회수 예측
- 카테고리별 전용 ML 모델 사용 (CatBoost, LightGBM, XGBoost)
- 영상 정보(제목, 카테고리, 길이, 업로드 시간, 구독자 수 등)를 입력받아 조회수 예측
- 인기도 확률(pred_popular_prob)과 예상 조회수를 함께 제공

### 2. 태그 추천
- Sentence-BERT 기반 유사도 분석
- 제목과 유사한 영상의 태그를 기반으로 추천
- 하이브리드 방식 지원 (SBERT + 유사도 결합)

### 3. 태그 보정 (Enrich)
- OpenAI GPT를 사용한 태그 자동 보정
- 제목과 설명을 분석하여 관련성 높은 태그 추천
- 불필요한 태그 제거 및 관련 태그 추가

### 4. 제목 생성
- OpenAI GPT-4o-mini를 사용한 제목 자동 생성
- 키워드와 이미지 설명을 기반으로 클릭률 높은 제목 생성
- 유튜브 제목 최적화 전략 적용

### 5. 트렌드 분석
- Kaggle API를 통한 최신 트렌드 데이터 수집
- YouTube API를 통한 카테고리 정보 수집
- 월별 인기 카테고리 TOP 5 분석
