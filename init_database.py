#!/usr/bin/env python3
"""
YouTube Analytics 데이터베이스 초기화 스크립트
이 스크립트를 실행하여 데이터베이스를 초기화하고 데모 데이터를 생성합니다.
"""

import os
import sys
from database import UserDatabase, create_demo_data

def main():
    """데이터베이스 초기화 메인 함수"""
    print("=" * 50)
    print("YouTube Analytics 데이터베이스 초기화")
    print("=" * 50)
    
    # 데이터베이스 파일 경로
    db_path = "youtube_analytics.db"
    
    # 기존 데이터베이스 파일이 있는지 확인
    if os.path.exists(db_path):
        print(f"\n기존 데이터베이스 파일이 발견되었습니다: {db_path}")
        response = input("기존 데이터를 삭제하고 새로 초기화하시겠습니까? (y/N): ")
        
        if response.lower() in ['y', 'yes']:
            try:
                os.remove(db_path)
                print("기존 데이터베이스 파일이 삭제되었습니다.")
            except Exception as e:
                print(f"데이터베이스 파일 삭제 중 오류 발생: {e}")
                return False
        else:
            print("기존 데이터베이스를 유지합니다.")
    
    try:
        # 데이터베이스 초기화
        print("\n데이터베이스 초기화 중...")
        db = UserDatabase(db_path)
        print("✓ 데이터베이스 테이블 생성 완료")
        
        # 데모 데이터 생성
        print("\n데모 데이터 생성 중...")
        create_demo_data()
        print("✓ 데모 데이터 생성 완료")
        
        # 통계 정보 출력
        print("\n" + "=" * 30)
        print("데이터베이스 통계")
        print("=" * 30)
        
        stats = db.get_user_statistics()
        print(f"전체 사용자: {stats['total_users']}명")
        print(f"오늘 가입: {stats['today_signups']}명")
        print(f"활성 세션: {stats['active_sessions']}개")
        print(f"역할별 분포:")
        for role, count in stats['role_distribution'].items():
            print(f"  - {role}: {count}명")
        
        print(f"\n✓ 데이터베이스 초기화 완료!")
        print(f"데이터베이스 파일: {os.path.abspath(db_path)}")
        
        # 데모 계정 정보 출력
        print("\n" + "=" * 30)
        print("데모 계정 정보")
        print("=" * 30)
        print("이메일: demo@youtubeanalytics.com")
        print("비밀번호: demo123")
        print("\n관리자 계정:")
        print("이메일: admin@youtubeanalytics.com")
        print("비밀번호: admin123")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 데이터베이스 초기화 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 데이터베이스 초기화가 성공적으로 완료되었습니다!")
        print("이제 웹 서버를 실행할 수 있습니다.")
    else:
        print("\n💥 데이터베이스 초기화에 실패했습니다.")
        sys.exit(1)
