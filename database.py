"""
YouTube Analytics 사용자 데이터베이스 관리 모듈
SQLite를 사용하여 사용자 정보를 저장하고 관리합니다.
"""

import sqlite3
import hashlib
import secrets
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import os

class UserDatabase:
    """사용자 데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "youtube_analytics.db"):
        """
        데이터베이스 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.init_database()
        
        # SQLite 연결 시 UTF-8 인코딩 명시
        import sqlite3
        sqlite3.register_converter("TEXT", lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    def init_database(self):
        """데이터베이스 테이블 초기화"""
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            # UTF-8 인코딩 명시
            conn.execute("PRAGMA encoding = 'UTF-8'")
            cursor = conn.cursor()
            
            # 사용자 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    profile_data TEXT
                )
            ''')
            
            # 세션 테이블 생성 (로그인 상태 관리)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # 사용자 활동 로그 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_activity_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    activity_type TEXT NOT NULL,
                    activity_data TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # 영상 정보 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    length INTEGER NOT NULL,
                    upload_time TIMESTAMP,
                    description TEXT,
                    thumbnail_image TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
    
    def _hash_password(self, password: str, salt: str = None) -> tuple:
        """
        비밀번호 해시화
        
        Args:
            password: 원본 비밀번호
            salt: 솔트 (없으면 새로 생성)
            
        Returns:
            (해시된 비밀번호, 솔트) 튜플
        """
        if salt is None:
            salt = secrets.token_hex(32)
        
        # PBKDF2를 사용한 비밀번호 해시화
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 반복 횟수
        )
        
        return password_hash.hex(), salt
    
    def _verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """
        비밀번호 검증
        
        Args:
            password: 입력된 비밀번호
            password_hash: 저장된 해시
            salt: 저장된 솔트
            
        Returns:
            비밀번호 일치 여부
        """
        computed_hash, _ = self._hash_password(password, salt)
        return computed_hash == password_hash
    
    def create_user(self, email: str, password: str, name: str, role: str = 'user', 
                   profile_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        새 사용자 생성
        
        Args:
            email: 이메일 주소
            password: 비밀번호
            name: 사용자 이름
            role: 사용자 역할
            profile_data: 추가 프로필 데이터
            
        Returns:
            생성된 사용자 정보 (비밀번호 제외)
            
        Raises:
            ValueError: 이메일이 이미 존재하는 경우
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 이메일 중복 체크
            cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
            if cursor.fetchone():
                raise ValueError('이미 존재하는 이메일입니다.')
            
            # 비밀번호 해시화
            password_hash, salt = self._hash_password(password)
            
            # 프로필 데이터 JSON 변환
            profile_json = json.dumps(profile_data) if profile_data else None
            
            # 사용자 생성
            cursor.execute('''
                INSERT INTO users (email, password_hash, salt, name, role, profile_data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (email, password_hash, salt, name, role, profile_json))
            
            user_id = cursor.lastrowid
            
            # 생성된 사용자 정보 반환 (비밀번호 제외)
            cursor.execute('''
                SELECT id, email, name, role, created_at, is_active, profile_data
                FROM users WHERE id = ?
            ''', (user_id,))
            
            user_data = cursor.fetchone()
            conn.commit()
            
            return {
                'id': user_data[0],
                'email': user_data[1],
                'name': user_data[2],
                'role': user_data[3],
                'created_at': user_data[4],
                'is_active': bool(user_data[5]),
                'profile_data': json.loads(user_data[6]) if user_data[6] else None
            }
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """
        사용자 인증
        
        Args:
            email: 이메일 주소
            password: 비밀번호
            
        Returns:
            인증 성공 시 사용자 정보, 실패 시 None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 사용자 정보 조회
            cursor.execute('''
                SELECT id, email, password_hash, salt, name, role, created_at, 
                       is_active, profile_data, last_login
                FROM users WHERE email = ? AND is_active = 1
            ''', (email,))
            
            user_data = cursor.fetchone()
            
            if not user_data:
                return None
            
            # 비밀번호 검증
            if not self._verify_password(password, user_data[2], user_data[3]):
                return None
            
            # 마지막 로그인 시간 업데이트
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_data[0],))
            
            conn.commit()
            
            return {
                'id': user_data[0],
                'email': user_data[1],
                'name': user_data[4],
                'role': user_data[5],
                'created_at': user_data[6],
                'is_active': bool(user_data[7]),
                'profile_data': json.loads(user_data[8]) if user_data[8] else None,
                'last_login': user_data[9]
            }
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        ID로 사용자 조회
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            사용자 정보 또는 None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, email, name, role, created_at, is_active, profile_data, last_login
                FROM users WHERE id = ? AND is_active = 1
            ''', (user_id,))
            
            user_data = cursor.fetchone()
            
            if not user_data:
                return None
            
            return {
                'id': user_data[0],
                'email': user_data[1],
                'name': user_data[2],
                'role': user_data[3],
                'created_at': user_data[4],
                'is_active': bool(user_data[5]),
                'profile_data': json.loads(user_data[6]) if user_data[6] else None,
                'last_login': user_data[7]
            }
    
    def create_session(self, user_id: int, expires_hours: int = 24) -> str:
        """
        사용자 세션 생성
        
        Args:
            user_id: 사용자 ID
            expires_hours: 세션 만료 시간 (시간)
            
        Returns:
            세션 토큰
        """
        session_token = secrets.token_urlsafe(32)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 기존 세션 비활성화
            cursor.execute('''
                UPDATE user_sessions SET is_active = 0 WHERE user_id = ?
            ''', (user_id,))
            
            # 새 세션 생성
            cursor.execute('''
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES (?, ?, datetime('now', '+{} hours'))
            '''.format(expires_hours), (user_id, session_token))
            
            conn.commit()
            
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """
        세션 검증
        
        Args:
            session_token: 세션 토큰
            
        Returns:
            유효한 세션의 사용자 정보 또는 None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT u.id, u.email, u.name, u.role, u.created_at, u.is_active, u.profile_data
                FROM users u
                JOIN user_sessions s ON u.id = s.user_id
                WHERE s.session_token = ? 
                AND s.is_active = 1 
                AND s.expires_at > datetime('now')
                AND u.is_active = 1
            ''', (session_token,))
            
            user_data = cursor.fetchone()
            
            if not user_data:
                return None
            
            return {
                'id': user_data[0],
                'email': user_data[1],
                'name': user_data[2],
                'role': user_data[3],
                'created_at': user_data[4],
                'is_active': bool(user_data[5]),
                'profile_data': json.loads(user_data[6]) if user_data[6] else None
            }
    
    def logout_session(self, session_token: str) -> bool:
        """
        세션 로그아웃
        
        Args:
            session_token: 세션 토큰
            
        Returns:
            로그아웃 성공 여부
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_sessions SET is_active = 0 WHERE session_token = ?
            ''', (session_token,))
            
            conn.commit()
            
            return cursor.rowcount > 0
    
    def update_user_profile(self, user_id: int, profile_data: Dict[str, Any]) -> bool:
        """
        사용자 프로필 업데이트
        
        Args:
            user_id: 사용자 ID
            profile_data: 업데이트할 프로필 데이터
            
        Returns:
            업데이트 성공 여부
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users 
                SET profile_data = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND is_active = 1
            ''', (json.dumps(profile_data), user_id))
            
            conn.commit()
            
            return cursor.rowcount > 0
    
    def log_user_activity(self, user_id: int, activity_type: str, 
                         activity_data: Dict[str, Any] = None, 
                         ip_address: str = None, user_agent: str = None):
        """
        사용자 활동 로그 기록
        
        Args:
            user_id: 사용자 ID
            activity_type: 활동 유형
            activity_data: 활동 데이터
            ip_address: IP 주소
            user_agent: User Agent
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_activity_logs 
                (user_id, activity_type, activity_data, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, activity_type, 
                  json.dumps(activity_data) if activity_data else None,
                  ip_address, user_agent))
            
            conn.commit()
    
    def get_user_statistics(self) -> Dict[str, Any]:
        """
        사용자 통계 정보 조회
        
        Returns:
            사용자 통계 데이터
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 전체 사용자 수
            cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
            total_users = cursor.fetchone()[0]
            
            # 오늘 가입한 사용자 수
            cursor.execute('''
                SELECT COUNT(*) FROM users 
                WHERE DATE(created_at) = DATE('now') AND is_active = 1
            ''')
            today_signups = cursor.fetchone()[0]
            
            # 활성 세션 수
            cursor.execute('''
                SELECT COUNT(*) FROM user_sessions 
                WHERE is_active = 1 AND expires_at > datetime('now')
            ''')
            active_sessions = cursor.fetchone()[0]
            
            # 역할별 사용자 수
            cursor.execute('''
                SELECT role, COUNT(*) FROM users 
                WHERE is_active = 1 GROUP BY role
            ''')
            role_distribution = dict(cursor.fetchall())
            
            return {
                'total_users': total_users,
                'today_signups': today_signups,
                'active_sessions': active_sessions,
                'role_distribution': role_distribution
            }
    
    def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_sessions 
                SET is_active = 0 
                WHERE expires_at <= datetime('now')
            ''')
            
            conn.commit()
    
    def create_video(self, title: str, category: str, length: int, 
                     upload_time: Optional[str] = None, description: Optional[str] = None,
                     thumbnail_image: Optional[str] = None, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        영상 정보 저장
        
        Args:
            title: 영상 제목
            category: 카테고리
            length: 영상 길이 (분)
            upload_time: 업로드 예정 시간
            description: 영상 설명
            thumbnail_image: 썸네일 이미지 (Base64)
            user_id: 사용자 ID (옵션)
            
        Returns:
            생성된 영상 정보
        """
        print(f"🔍 create_video 호출됨: title={title}, category={category}, length={length}, user_id={user_id}")
        
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        try:
            # UTF-8 인코딩 명시
            conn.execute("PRAGMA encoding = 'UTF-8'")
            cursor = conn.cursor()
            
            print(f"🔍 INSERT 쿼리 실행 전")
            cursor.execute('''
                INSERT INTO videos 
                (user_id, title, category, length, upload_time, description, thumbnail_image)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, title, category, length, upload_time, description, thumbnail_image))
            
            video_id = cursor.lastrowid
            print(f"🔍 video_id 생성됨: {video_id}")
            
            # 커밋
            conn.commit()
            print(f"🔍 커밋 완료")
            
            # 실제로 저장되었는지 확인
            cursor.execute('SELECT COUNT(*) FROM videos WHERE id = ?', (video_id,))
            count = cursor.fetchone()[0]
            print(f"🔍 저장 확인: count={count}")
            
            if count == 0:
                print(f"❌ 영상 정보가 저장되지 않았습니다!")
                raise Exception("영상 정보가 저장되지 않았습니다.")
            
            # 간단하게 딕셔너리 반환
            video = {
                'id': video_id,
                'user_id': user_id,
                'title': title,
                'category': category,
                'length': length,
                'upload_time': upload_time,
                'description': description,
                'thumbnail_image': thumbnail_image,
                'created_at': None,
                'updated_at': None
            }
            
            # 문자열 필드의 인코딩 확인 및 변환
            for key, value in video.items():
                if isinstance(value, bytes):
                    try:
                        video[key] = value.decode('utf-8')
                    except:
                        video[key] = str(value)
            
            print(f"🔍 최종 반환할 video: {video}")
            return video
            
        finally:
            conn.close()
            print(f"🔍 DB 연결 종료")
    
    def get_user_videos(self, user_id: int) -> List[Dict[str, Any]]:
        """
        사용자의 영상 목록 조회
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            영상 목록
        """
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            # UTF-8 인코딩 명시
            conn.execute("PRAGMA encoding = 'UTF-8'")
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM videos 
                WHERE user_id = ? 
                ORDER BY created_at DESC
            ''', (user_id,))
            
            rows = cursor.fetchall()
            videos = []
            for row in rows:
                video_dict = dict(row)
                # 문자열 필드의 인코딩 확인 및 변환
                for key, value in video_dict.items():
                    if isinstance(value, bytes):
                        try:
                            video_dict[key] = value.decode('utf-8')
                        except:
                            video_dict[key] = str(value)
                videos.append(video_dict)
            return videos
    
    def get_all_videos(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        모든 영상 목록 조회
        
        Args:
            limit: 최대 개수
            offset: 시작 위치
            
        Returns:
            영상 목록
        """
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            # UTF-8 인코딩 명시
            conn.execute("PRAGMA encoding = 'UTF-8'")
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM videos 
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            rows = cursor.fetchall()
            videos = []
            for row in rows:
                video_dict = dict(row)
                # 문자열 필드의 인코딩 확인 및 변환
                for key, value in video_dict.items():
                    if isinstance(value, bytes):
                        try:
                            video_dict[key] = value.decode('utf-8')
                        except:
                            video_dict[key] = str(value)
                videos.append(video_dict)
            return videos
    
    def close(self):
        """데이터베이스 연결 종료 (SQLite는 자동으로 관리되므로 빈 메서드)"""
        pass


# 전역 데이터베이스 인스턴스
db = UserDatabase()

# 데모 데이터 생성 함수
def create_demo_data():
    """데모 사용자 데이터 생성"""
    try:
        # 데모 사용자 생성
        demo_user = db.create_user(
            email='demo@youtubeanalytics.com',
            password='demo123',
            name='데모 사용자',
            role='creator',
            profile_data={
                'bio': 'YouTube Analytics 데모 계정입니다.',
                'preferences': {
                    'theme': 'light',
                    'language': 'ko'
                }
            }
        )
        print(f"데모 사용자 생성 완료: {demo_user['email']}")
        
        # 관리자 계정 생성
        admin_user = db.create_user(
            email='admin@youtubeanalytics.com',
            password='admin123',
            name='관리자',
            role='admin',
            profile_data={
                'bio': '시스템 관리자',
                'permissions': ['user_management', 'data_analysis']
            }
        )
        print(f"관리자 계정 생성 완료: {admin_user['email']}")
        
    except ValueError as e:
        print(f"데모 데이터 생성 중 오류: {e}")


if __name__ == "__main__":
    # 데이터베이스 초기화 및 데모 데이터 생성
    print("YouTube Analytics 데이터베이스 초기화 중...")
    create_demo_data()
    
    # 통계 정보 출력
    stats = db.get_user_statistics()
    print(f"\n데이터베이스 통계:")
    print(f"- 전체 사용자: {stats['total_users']}명")
    print(f"- 오늘 가입: {stats['today_signups']}명")
    print(f"- 활성 세션: {stats['active_sessions']}개")
    print(f"- 역할별 분포: {stats['role_distribution']}")
    
    print("\n데이터베이스 초기화 완료!")
