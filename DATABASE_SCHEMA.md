# 데이터베이스 스키마 문서

이 문서는 YouTube Analytics 프로젝트의 SQLite 데이터베이스에 저장되는 데이터 구조를 설명합니다.

## 📊 테이블 구조

데이터베이스는 총 **4개의 테이블**로 구성되어 있습니다.

---

## 1. 👤 users (사용자 테이블)

사용자 계정 정보를 저장하는 테이블입니다.

| 필드명 | 타입 | 설명 | 제약조건 |
|--------|------|------|----------|
| `id` | INTEGER | 사용자 고유 ID | PRIMARY KEY, AUTOINCREMENT |
| `email` | TEXT | 이메일 주소 | UNIQUE, NOT NULL |
| `password_hash` | TEXT | 해시화된 비밀번호 (PBKDF2) | NOT NULL |
| `salt` | TEXT | 비밀번호 해시용 솔트 | NOT NULL |
| `name` | TEXT | 사용자 이름 | NOT NULL |
| `role` | TEXT | 사용자 역할 (user, creator, admin) | NOT NULL, DEFAULT 'user' |
| `created_at` | TIMESTAMP | 계정 생성 시간 | DEFAULT CURRENT_TIMESTAMP |
| `updated_at` | TIMESTAMP | 정보 수정 시간 | DEFAULT CURRENT_TIMESTAMP |
| `last_login` | TIMESTAMP | 마지막 로그인 시간 | NULL 가능 |
| `is_active` | BOOLEAN | 계정 활성화 여부 | DEFAULT 1 |
| `profile_data` | TEXT | 추가 프로필 데이터 (JSON 형식) | NULL 가능 |

**저장되는 데이터 예시:**
- 사용자 이메일, 이름
- 암호화된 비밀번호
- 사용자 역할 (일반 사용자, 크리에이터, 관리자)
- 프로필 정보 (바이오, 선호 설정 등 JSON 형식)

---

## 2. 🔐 user_sessions (세션 테이블)

사용자 로그인 세션 정보를 관리하는 테이블입니다.

| 필드명 | 타입 | 설명 | 제약조건 |
|--------|------|------|----------|
| `id` | INTEGER | 세션 고유 ID | PRIMARY KEY, AUTOINCREMENT |
| `user_id` | INTEGER | 사용자 ID | NOT NULL, FOREIGN KEY → users(id) |
| `session_token` | TEXT | 세션 토큰 (인증용) | UNIQUE, NOT NULL |
| `created_at` | TIMESTAMP | 세션 생성 시간 | DEFAULT CURRENT_TIMESTAMP |
| `expires_at` | TIMESTAMP | 세션 만료 시간 | NOT NULL |
| `is_active` | BOOLEAN | 세션 활성화 여부 | DEFAULT 1 |

**저장되는 데이터 예시:**
- 로그인한 사용자의 세션 토큰
- 세션 생성 및 만료 시간
- 현재 활성화된 세션 여부

**기능:**
- 로그인 시 새 세션 생성 (기존 세션 비활성화)
- 세션 토큰으로 사용자 인증
- 만료된 세션 자동 정리

---

## 3. 📝 user_activity_logs (사용자 활동 로그 테이블)

사용자의 활동 내역을 기록하는 테이블입니다.

| 필드명 | 타입 | 설명 | 제약조건 |
|--------|------|------|----------|
| `id` | INTEGER | 로그 고유 ID | PRIMARY KEY, AUTOINCREMENT |
| `user_id` | INTEGER | 사용자 ID | NOT NULL, FOREIGN KEY → users(id) |
| `activity_type` | TEXT | 활동 유형 | NOT NULL |
| `activity_data` | TEXT | 활동 상세 데이터 (JSON 형식) | NULL 가능 |
| `ip_address` | TEXT | 접속 IP 주소 | NULL 가능 |
| `user_agent` | TEXT | 브라우저/클라이언트 정보 | NULL 가능 |
| `created_at` | TIMESTAMP | 활동 발생 시간 | DEFAULT CURRENT_TIMESTAMP |

**저장되는 데이터 예시:**
- 활동 유형: 로그인, 로그아웃, 영상 분석 요청, 프로필 수정 등
- 활동 상세 정보 (JSON 형식)
- 접속 IP 주소 및 브라우저 정보
- 활동 발생 시간

**활용:**
- 사용자 행동 분석
- 보안 감사 (의심스러운 활동 추적)
- 서비스 개선을 위한 데이터 수집

---

## 4. 🎬 videos (영상 정보 테이블)

사용자가 분석한 영상 정보를 저장하는 테이블입니다.

| 필드명 | 타입 | 설명 | 제약조건 |
|--------|------|------|----------|
| `id` | INTEGER | 영상 고유 ID | PRIMARY KEY, AUTOINCREMENT |
| `user_id` | INTEGER | 소유자 사용자 ID | NULL 가능, FOREIGN KEY → users(id) |
| `title` | TEXT | 영상 제목 | NOT NULL |
| `category` | TEXT | 영상 카테고리 | NOT NULL |
| `length` | INTEGER | 영상 길이 (분 단위) | NOT NULL |
| `upload_time` | TIMESTAMP | 업로드 예정/실제 시간 | NULL 가능 |
| `description` | TEXT | 영상 설명 | NULL 가능 |
| `thumbnail_image` | TEXT | 썸네일 이미지 (Base64 또는 URL) | NULL 가능 |
| `created_at` | TIMESTAMP | 레코드 생성 시간 | DEFAULT CURRENT_TIMESTAMP |
| `updated_at` | TIMESTAMP | 레코드 수정 시간 | DEFAULT CURRENT_TIMESTAMP |

**저장되는 데이터 예시:**
- 영상 제목, 카테고리, 길이
- 영상 설명 및 썸네일
- 업로드 시간
- 영상을 분석한 사용자 정보

**기능:**
- 사용자별 영상 목록 조회 (`get_user_videos`)
- 전체 영상 목록 조회 (`get_all_videos`)
- 마이페이지에서 저장된 분석 결과 확인

---

## 🔗 테이블 관계도

```
users (1) ──< (N) user_sessions
  │
  │ (1)
  │
  ├──< (N) user_activity_logs
  │
  └──< (N) videos
```

- 한 사용자는 여러 세션을 가질 수 있음
- 한 사용자는 여러 활동 로그를 생성함
- 한 사용자는 여러 영상 정보를 저장할 수 있음

---

## 🔒 보안 관련

### 비밀번호 저장
- **PBKDF2** 알고리즘 사용 (SHA-256, 100,000회 반복)
- 각 비밀번호마다 고유한 솔트(salt) 생성
- 원본 비밀번호는 절대 저장하지 않음

### 세션 관리
- 세션 토큰은 `secrets.token_urlsafe(32)`로 생성 (안전한 랜덤 토큰)
- 세션 만료 시간 설정 (기본 24시간)
- 로그인 시 기존 세션 자동 비활성화

---

## 📈 데이터 통계 기능

`get_user_statistics()` 메서드를 통해 다음 통계를 조회할 수 있습니다:

- 전체 활성 사용자 수
- 오늘 가입한 사용자 수
- 현재 활성 세션 수
- 역할별 사용자 분포 (user, creator, admin)

---

## 💾 데이터베이스 파일

- **파일명**: `youtube_analytics.db`
- **위치**: 프로젝트 루트 디렉토리
- **인코딩**: UTF-8
- **백업**: 단일 파일로 관리되어 백업이 간단함

---

## 🚀 초기화

데이터베이스는 `init_database.py` 스크립트를 실행하거나 `UserDatabase` 클래스가 초기화될 때 자동으로 생성됩니다.

**초기화 시 생성되는 데모 데이터:**
- 데모 사용자 계정 (demo@youtubeanalytics.com)
- 관리자 계정 (admin@youtubeanalytics.com)

