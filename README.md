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

### 5. 서버 실행
```bash
cd tags
python fastapi_server.py
```

### 6. 웹 페이지 접속
- **API 서버**: http://localhost:8001
- **웹 인터페이스**: `UI/index.html` 파일을 브라우저에서 열기

## ⚠️ 주의사항

- **AI 모델 파일**: `tag_recommendation_model.pkl`은 태그 추천 기능에 필수적으로 사용됩니다
- **태그 추천 기능**: `tags/tag_recommendation_model.pkl` 파일이 존재해야 태그 추천이 정상 작동합니다
- **데이터베이스**: `youtube_analytics.db`는 실행 시 자동으로 생성됩니다

### 4. 사용 가능한 페이지
- `index.html` - 메인 페이지
- `login.html` - 로그인 페이지
- `signup.html` - 회원가입 페이지
- `trend.html` - 월별 트렌드 분석
- `feedback.html` - 영상 피드백
- `feedback-result.html` - 피드백 결과

## 사용법

1. **메인 페이지**
   - `index.html`에서 전체 서비스 개요를 확인할 수 있습니다
   - 월별 트렌드와 피드백 기능으로 이동할 수 있습니다

2. **월별 트렌드 분석**
   - `trend.html`에서 4월~6월 데이터를 확인할 수 있습니다
   - 각 월별로 인기 카테고리 TOP5를 확인할 수 있습니다
   - 탭을 클릭하여 월별 데이터를 전환할 수 있습니다

3. **영상 피드백 (데모)**
   - `feedback.html`에서 영상 정보를 입력할 수 있습니다
   - 입력한 정보는 로컬 스토리지에 저장됩니다
   - `feedback-result.html`에서 데모 결과를 확인할 수 있습니다

## 기술 스택

- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Backend**: Python, Flask
- **Database**: SQLite
- **차트**: Chart.js
- **아이콘**: Bootstrap Icons
- **스타일**: 반응형 웹 디자인

## API 엔드포인트

### 인증 API
- `POST /api/auth/register/` - 회원가입
- `POST /api/auth/login/` - 로그인
- `POST /api/auth/logout/` - 로그아웃
- `GET /api/auth/profile/` - 프로필 조회
- `PUT /api/auth/profile/` - 프로필 업데이트

### 시스템 API
- `GET /api/stats/` - 시스템 통계

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
유튜브 데이터/
├── README.md                # 프로젝트 문서
├── requirements.txt         # Python 패키지 의존성
├── database.py              # 데이터베이스 관리 모듈
├── init_database.py         # 데이터베이스 초기화 스크립트
├── web_server.py            # Flask 웹 서버
├── run_server.py            # 서버 자동 실행 스크립트
├── youtube_analytics.db     # SQLite 데이터베이스 파일
├── UI/                      # 웹 인터페이스
│   ├── index.html           # 메인 페이지
│   ├── login.html           # 로그인 페이지
│   ├── signup.html          # 회원가입 페이지
│   ├── trend.html           # 월별 트렌드 분석
│   ├── feedback.html        # 피드백 입력 페이지
│   └── feedback-result.html # 피드백 결과 페이지
├── data/                    # 데이터 파일들
├── 트렌드 분석/             # 트렌드 분석 데이터
└── 기타 폴더들...           # 프로젝트 관련 파일들
```

## 보안 기능

- **비밀번호 해시화**: PBKDF2를 사용한 안전한 비밀번호 저장
- **세션 관리**: 토큰 기반 세션 인증
- **CORS 지원**: 크로스 오리진 요청 허용
- **입력 검증**: 서버 측 데이터 유효성 검사
- **활동 로깅**: 사용자 활동 추적 및 기록

## 주의사항

- 피드백 기능은 데모용으로 실제 예측을 수행하지 않습니다
- 사용자 데이터는 SQLite 데이터베이스에 안전하게 저장됩니다
- 트렌드 데이터는 2024년 4월~6월 기준입니다
- 개발 환경에서는 디버그 모드가 활성화되어 있습니다

## 문제 해결

### 웹 페이지가 제대로 표시되지 않는 경우
1. 브라우저에서 JavaScript가 활성화되어 있는지 확인
2. 인터넷 연결을 확인하여 CDN 리소스가 로드되는지 확인
3. 브라우저 콘솔에서 오류 메시지 확인

### 피드백 기능이 작동하지 않는 경우
1. 브라우저에서 로컬 스토리지가 활성화되어 있는지 확인
2. 필수 입력 항목이 모두 채워져 있는지 확인
3. 브라우저 콘솔에서 JavaScript 오류 확인
