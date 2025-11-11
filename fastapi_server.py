#!/usr/bin/env python3
"""
YouTube Analytics FastAPI 서버
OpenAPI 스펙을 사용하여 REST API를 제공합니다.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import json
from datetime import datetime
import sys
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# 현재 스크립트와 같은 디렉토리에서 모듈 import
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from database import UserDatabase, db
from tags.tag_recommendation_model import TagRecommendationModel
from enrich_tags import run_pipeline
from openai import OpenAI

# OpenAI API 키는 .env 파일에서 자동으로 로드됩니다
# OPENAI_API_KEY 환경변수가 없으면 에러 발생
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️ 경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# FastAPI 앱 초기화
app = FastAPI(
    title="YouTube Analytics API",
    description="YouTube 영상 데이터 분석 및 태그 추천 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# JSON 인코딩 설정 (한글 지원)
import json
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse as FastAPIJSONResponse

class UTF8JSONResponse(FastAPIJSONResponse):
    """UTF-8 인코딩을 보장하는 JSON 응답 클래스"""
    def render(self, content: Any) -> bytes:
        return json.dumps(
            jsonable_encoder(content),
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

# JSONResponse의 ensure_ascii 설정을 위한 커스텀 응답 클래스
def create_json_response(data: dict, status_code: int = 200):
    """한글 인코딩을 올바르게 처리하는 JSON 응답 생성"""
    return UTF8JSONResponse(content=data, status_code=status_code)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 데이터베이스 인스턴스
user_db = db

# 태그 추천 모델 인스턴스
tag_model = None

# Pydantic 모델 정의
class UserRegister(BaseModel):
    email: str = Field(..., description="사용자 이메일")
    password: str = Field(..., min_length=6, description="비밀번호 (최소 6자)")
    name: str = Field(..., description="사용자 이름")
    role: str = Field(..., description="사용자 역할")
    profile_data: Optional[Dict[str, Any]] = Field(default={}, description="추가 프로필 데이터")

class UserLogin(BaseModel):
    email: str = Field(..., description="사용자 이메일")
    password: str = Field(..., description="비밀번호")

class UserLogout(BaseModel):
    session_token: str = Field(..., description="세션 토큰")

class ProfileUpdate(BaseModel):
    profile_data: Dict[str, Any] = Field(..., description="업데이트할 프로필 데이터")

class TagRecommendRequest(BaseModel):
    title: str = Field(..., description="유튜브 영상 제목")
    top_k: int = Field(default=20, ge=1, le=50, description="추천할 태그 개수")
    method: str = Field(default="hybrid", description="추천 방법 (hybrid, sbert, similarity)")

class TagRefineRequest(BaseModel):
    title: str = Field(..., description="유튜브 영상 제목")
    candidate_tags: Optional[List[str]] = Field(default=[], description="수정할 후보 태그들")

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class TagRecommendResponse(BaseModel):
    success: bool
    title: str
    recommended_tags: List[str]
    method: str
    similar_titles: List[Dict[str, Any]]

class TagRefineResponse(BaseModel):
    success: bool
    title: str
    original_candidate_tags: List[str]
    refined_tags: List[str]
    prompt: str

class TagEnrichRequest(BaseModel):
    title: str = Field(..., description="유튜브 영상 제목")
    description: Optional[str] = Field(default="", description="유튜브 영상 설명")
    top_k: int = Field(default=15, ge=1, le=50, description="추천할 태그 개수")
    title_sim_threshold: float = Field(default=0.30, description="제목 유사도 임계값")
    tag_abs_threshold: float = Field(default=0.30, description="태그 유사도 임계값")
    extra_k: int = Field(default=10, description="추가 태그 개수")
    api_key: Optional[str] = Field(default=None, description="OpenAI API 키 (없으면 환경변수 사용)")

class TagEnrichResponse(BaseModel):
    success: bool
    title: str
    description: str
    candidates: List[str]
    scored: List[Dict[str, Any]]
    kept: List[str]
    dropped: List[str]
    final_tags: List[str]
    extra_tags: List[str]

class TitleSuggestRequest(BaseModel):
    keyword: str = Field(..., description="제목 추천을 위한 키워드")
    imageText: Optional[str] = Field(default="", description="이미지 텍스트 (선택사항)")
    n: int = Field(default=5, ge=1, le=10, description="생성할 제목 개수")

class TitleSuggestResponse(BaseModel):
    success: bool
    titles: List[str]

class VideoCreateRequest(BaseModel):
    title: str = Field(..., description="영상 제목")
    category: str = Field(..., description="카테고리")
    length: int = Field(..., ge=1, description="영상 길이 (분)")
    upload_time: Optional[str] = Field(default=None, description="업로드 예정 시간")
    description: Optional[str] = Field(default=None, description="영상 설명")
    thumbnail_image: Optional[str] = Field(default=None, description="썸네일 이미지 (Base64)")
    caption_status: Optional[str] = Field(default=None, description="캡션 상태")
    quality: Optional[str] = Field(default=None, description="화질")

class VideoResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

def load_tag_model():
    """태그 추천 모델 로드"""
    global tag_model
    try:
        # 여러 경로에서 모델 파일 찾기
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        possible_paths = [
            os.path.join(script_dir, "tag_recommendation_model.pkl"),  # 유튜브서버/tag_recommendation_model.pkl
            os.path.join(project_root, "tags", "tag_recommendation_model.pkl"),  # 프로젝트 루트/tags/tag_recommendation_model.pkl
            os.path.join(project_root, "유튜브서버", "tag_recommendation_model.pkl"),  # 프로젝트 루트/유튜브서버/tag_recommendation_model.pkl
            "tag_recommendation_model.pkl",
            "/Users/han-yujeong/Desktop/유튜브 데이터/유튜브서버/tag_recommendation_model.pkl"  # 절대 경로
        ]
        
        model_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path) if not os.path.isabs(path) else path
            if os.path.exists(abs_path):
                model_path = abs_path
                break
        
        if model_path:
            tag_model = TagRecommendationModel()
            tag_model.load_model(model_path)
            print(f"✅ 태그 추천 모델 로드 완료: {model_path}")
        else:
            print("⚠️ 태그 추천 모델 파일이 없습니다. 먼저 모델을 학습시켜주세요.")
            print(f"   시도한 경로들: {possible_paths}")
            tag_model = None
    except Exception as e:
        print(f"❌ 태그 추천 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        tag_model = None

def get_current_user(session_token: str = None):
    """현재 사용자 인증"""
    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증이 필요합니다."
        )
    
    user = user_db.validate_session(session_token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 세션입니다."
        )
    return user

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
    load_tag_model()

@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지"""
    return """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>YouTube Analytics API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #e74c3c; text-align: center; }
            .api-list { margin: 20px 0; }
            .api-item { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #e74c3c; }
            .method { font-weight: bold; color: #e74c3c; }
            .endpoint { font-family: monospace; background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
            .description { margin-top: 5px; color: #666; }
            .docs-link { text-align: center; margin: 20px 0; }
            .docs-link a { background: #e74c3c; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 YouTube Analytics API</h1>
            <p style="text-align: center; color: #666;">사용자 인증 및 태그 추천 API 서버</p>
            
            <div class="docs-link">
                <a href="/docs" target="_blank">📚 API 문서 보기 (Swagger UI)</a>
                <a href="/redoc" target="_blank" style="margin-left: 10px;">📖 API 문서 보기 (ReDoc)</a>
            </div>
            
            <div class="api-list">
                <h2>📋 주요 API 엔드포인트</h2>
                
                <div class="api-item">
                    <div><span class="method">POST</span> <span class="endpoint">/api/auth/register</span></div>
                    <div class="description">새 사용자 회원가입</div>
                </div>
                
                <div class="api-item">
                    <div><span class="method">POST</span> <span class="endpoint">/api/auth/login</span></div>
                    <div class="description">사용자 로그인</div>
                </div>
                
                <div class="api-item">
                    <div><span class="method">POST</span> <span class="endpoint">/api/tags/recommend</span></div>
                    <div class="description">제목 기반 태그 추천</div>
                </div>
                
                <div class="api-item">
                    <div><span class="method">POST</span> <span class="endpoint">/api/tags/refine</span></div>
                    <div class="description">프롬프트 기반 태그 수정</div>
                </div>
            </div>
            
            <div style="margin-top: 30px; padding: 20px; background: #e8f5e8; border-radius: 5px;">
                <h3>🚀 서버 상태</h3>
                <p>✅ FastAPI 서버가 정상적으로 실행 중입니다.</p>
                <p>📊 데이터베이스: SQLite</p>
                <p>🔗 CORS: 활성화</p>
                <p>📚 OpenAPI: 지원</p>
            </div>
        </div>
    </body>
    </html>
    """

# 인증 API
@app.post("/api/auth/register", response_model=APIResponse)
async def register(user_data: UserRegister):
    """사용자 회원가입"""
    try:
        # 이메일 형식 검증
        email = user_data.email.strip().lower()
        if '@' not in email or '.' not in email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="올바른 이메일 형식을 입력해주세요."
            )
        
        # 사용자 생성
        user = user_db.create_user(
            email=email,
            password=user_data.password,
            name=user_data.name.strip(),
            role=user_data.role,
            profile_data=user_data.profile_data
        )
        
        # 활동 로그 기록
        user_db.log_user_activity(
            user_id=user['id'],
            activity_type='register',
            activity_data={'email': email, 'role': user_data.role},
            ip_address="127.0.0.1",  # FastAPI에서는 request.remote_addr 대신
            user_agent="FastAPI Client"
        )
        
        return UTF8JSONResponse(
            content={
                "success": True,
                "message": "회원가입이 완료되었습니다.",
                "data": {"user": user}
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서버 오류가 발생했습니다."
        )

@app.post("/api/auth/login", response_model=APIResponse)
async def login(login_data: UserLogin):
    """사용자 로그인"""
    try:
        # 사용자 인증
        user = user_db.authenticate_user(
            email=login_data.email.strip().lower(),
            password=login_data.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="이메일 또는 비밀번호가 올바르지 않습니다."
            )
        
        # 세션 생성
        session_token = user_db.create_session(user['id'])
        
        # 활동 로그 기록
        user_db.log_user_activity(
            user_id=user['id'],
            activity_type='login',
            activity_data={'email': login_data.email},
            ip_address="127.0.0.1",
            user_agent="FastAPI Client"
        )
        
        return UTF8JSONResponse(
            content={
                "success": True,
                "message": "로그인에 성공했습니다.",
                "data": {
                    "user": user,
                    "session_token": session_token
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서버 오류가 발생했습니다."
        )

@app.post("/api/auth/logout", response_model=APIResponse)
async def logout(logout_data: UserLogout):
    """사용자 로그아웃"""
    try:
        # 세션 검증
        user = user_db.validate_session(logout_data.session_token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="유효하지 않은 세션입니다."
            )
        
        # 세션 로그아웃
        user_db.logout_session(logout_data.session_token)
        
        # 활동 로그 기록
        user_db.log_user_activity(
            user_id=user['id'],
            activity_type='logout',
            ip_address="127.0.0.1",
            user_agent="FastAPI Client"
        )
        
        return UTF8JSONResponse(
            content={
                "success": True,
                "message": "로그아웃되었습니다."
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서버 오류가 발생했습니다."
        )

@app.get("/api/auth/profile", response_model=APIResponse)
async def get_profile(session_token: str = None):
    """사용자 프로필 조회"""
    try:
        user = get_current_user(session_token)
        return UTF8JSONResponse(
            content={
                "success": True,
                "data": {"user": user}
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서버 오류가 발생했습니다."
        )

@app.put("/api/auth/profile", response_model=APIResponse)
async def update_profile(profile_data: ProfileUpdate, session_token: str = None):
    """사용자 프로필 업데이트"""
    try:
        user = get_current_user(session_token)
        
        # 프로필 업데이트
        success = user_db.update_user_profile(user['id'], profile_data.profile_data)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="프로필 업데이트에 실패했습니다."
            )
        
        # 활동 로그 기록
        user_db.log_user_activity(
            user_id=user['id'],
            activity_type='profile_update',
            activity_data={'updated_fields': list(profile_data.profile_data.keys())},
            ip_address="127.0.0.1",
            user_agent="FastAPI Client"
        )
        
        return UTF8JSONResponse(
            content={
                "success": True,
                "message": "프로필이 업데이트되었습니다."
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서버 오류가 발생했습니다."
        )

@app.get("/api/stats", response_model=APIResponse)
async def get_stats():
    """시스템 통계 조회"""
    try:
        stats = user_db.get_user_statistics()
        return UTF8JSONResponse(
            content={
                "success": True,
                "data": {"stats": stats}
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서버 오류가 발생했습니다."
        )

# 태그 추천 API
@app.post("/api/tags/recommend", response_model=TagRecommendResponse)
async def recommend_tags(request: TagRecommendRequest):
    """제목 기반 태그 추천"""
    try:
        if tag_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="태그 추천 모델이 로드되지 않았습니다."
            )
        
        title = request.title.strip()
        top_k = request.top_k
        method = request.method
        
        # 태그 추천
        if method == "sbert":
            # SBERT 직접 추천
            recommended_tags = tag_model.recommend_tags_with_sbert(title, top_k=top_k)
            result_tags = [item['tag'] for item in recommended_tags]
        elif method == "similarity":
            # 유사한 제목 기반 추천
            recommended_tags = tag_model.recommend_tags(title, top_k=top_k)
            result_tags = recommended_tags
        else:  # hybrid
            # 두 방법을 결합한 하이브리드 추천
            sbert_tags = tag_model.recommend_tags_with_sbert(title, top_k=top_k//2)
            similarity_tags = tag_model.recommend_tags(title, top_k=top_k//2)
            
            # 중복 제거하면서 결합
            all_tags = []
            for item in sbert_tags:
                all_tags.append(item['tag'])
            for tag in similarity_tags:
                if tag not in all_tags:
                    all_tags.append(tag)
            
            result_tags = all_tags[:top_k]
        
        # 유사한 제목들도 함께 반환 (참고용)
        similar_titles = tag_model.find_similar_titles(title, top_k=3)
        
        return TagRecommendResponse(
            success=True,
            title=title,
            recommended_tags=result_tags,
            method=method,
            similar_titles=similar_titles
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"태그 추천 중 오류가 발생했습니다: {str(e)}"
        )

@app.post("/api/tags/refine", response_model=TagRefineResponse)
async def refine_tags(request: TagRefineRequest):
    """프롬프트 기반 태그 수정"""
    try:
        if tag_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="태그 추천 모델이 로드되지 않았습니다."
            )
        
        title = request.title.strip()
        candidate_tags = request.candidate_tags or []
        
        # 후보 태그가 없으면 먼저 추천
        if not candidate_tags:
            recommended_tags = tag_model.recommend_tags(title, top_k=15)
            candidate_tags = recommended_tags
        
        # 프롬프트 생성
        prompt = f"""
아래는 유튜브 영상 제목과 SBERT가 유사도 기반으로 추천한 태그 후보입니다.
제목: {title}
후보 태그: {', '.join(candidate_tags)}

위 제목의 문맥과 의미에 어울리도록 태그를 자연스럽게 수정하거나 보완해줘.
불필요하거나 제목과 관련 없는 건 제거하고, 관련 있는 표현은 새로 추가해도 좋아.
최종 결과는 쉼표로 구분된 형태로 작성해줘.
"""
        
        # 간단한 후처리로 태그 수정
        refined_tags = refine_tags_simple(title, candidate_tags)
        
        return TagRefineResponse(
            success=True,
            title=title,
            original_candidate_tags=candidate_tags,
            refined_tags=refined_tags,
            prompt=prompt
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"태그 수정 중 오류가 발생했습니다: {str(e)}"
        )

def refine_tags_simple(title: str, candidate_tags: List[str]) -> List[str]:
    """간단한 태그 수정 로직 (실제 LLM 대신 사용)"""
    # 제목에서 키워드 추출
    title_keywords = []
    title_lower = title.lower()
    
    # 브이로그 관련 키워드
    if any(keyword in title_lower for keyword in ['브이로그', 'vlog', '일상']):
        title_keywords.extend(['브이로그', 'vlog', '일상'])
    
    # 먹방 관련 키워드
    if any(keyword in title_lower for keyword in ['먹방', '먹', '음식', '요리']):
        title_keywords.extend(['먹방', '먹방브이로그', '음식'])
    
    # 여행 관련 키워드
    if any(keyword in title_lower for keyword in ['여행', '여행브이로그', '여행vlog']):
        title_keywords.extend(['여행', '여행브이로그', 'travel'])
    
    # 메이크업 관련 키워드
    if any(keyword in title_lower for keyword in ['메이크업', '화장', 'grwm']):
        title_keywords.extend(['메이크업', 'grwm', '화장'])
    
    # 다이어트 관련 키워드
    if any(keyword in title_lower for keyword in ['다이어트', '다이어트브이로그', '급찐급빠']):
        title_keywords.extend(['다이어트', '다이어트브이로그', '다이어트vlog'])
    
    # 기존 후보 태그와 제목 키워드 결합
    refined_tags = list(set(candidate_tags + title_keywords))
    
    # 관련성 낮은 태그 제거 (간단한 필터링)
    filtered_tags = []
    for tag in refined_tags:
        if len(tag) > 1 and not any(char.isdigit() for char in tag):
            filtered_tags.append(tag)
    
    return filtered_tags[:15]  # 최대 15개로 제한

@app.post("/api/tags/enrich", response_model=TagEnrichResponse)
async def enrich_tags(request: TagEnrichRequest):
    """제목 기반 태그 추천 및 OpenAI 보정 (enrich_tags.py 기능)"""
    try:
        title = request.title.strip()
        description = request.description.strip() if request.description else ""
        
        if not title:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="제목을 입력해주세요."
            )
        
        # 모델 경로 설정 (여러 경로 시도)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        possible_paths = [
            os.path.join(script_dir, "tag_recommendation_model.pkl"),  # 유튜브서버/tag_recommendation_model.pkl
            os.path.join(project_root, "tags", "tag_recommendation_model.pkl"),  # 프로젝트 루트/tags/tag_recommendation_model.pkl
            os.path.join(project_root, "유튜브서버", "tag_recommendation_model.pkl"),  # 프로젝트 루트/유튜브서버/tag_recommendation_model.pkl
            "tag_recommendation_model.pkl",
            "/Users/han-yujeong/Desktop/유튜브 데이터/유튜브서버/tag_recommendation_model.pkl"  # 절대 경로
        ]
        
        model_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path) if not os.path.isabs(path) else path
            if os.path.exists(abs_path):
                model_path = abs_path
                print(f"✅ 모델 파일 발견: {model_path}")
                break
        
        if not model_path:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"태그 추천 모델 파일을 찾을 수 없습니다. 시도한 경로: {possible_paths}"
            )
        
        # API 키 설정 (요청에서 받거나 환경변수 사용)
        api_key = request.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OpenAI API 키가 설정되지 않았습니다. api_key 파라미터를 제공하거나 OPENAI_API_KEY 환경변수를 설정해주세요."
            )
        
        # enrich_tags 파이프라인 실행 (제목과 설명 모두 사용)
        result = run_pipeline(
            model_path=model_path,
            title=title,
            description=description,
            top_k=request.top_k,
            title_sim_threshold=request.title_sim_threshold,
            tag_abs_threshold=request.tag_abs_threshold,
            extra_k=request.extra_k,
            api_key=api_key
        )
        
        # 응답 형식 변환
        scored_list = [{"tag": tag, "score": score} for tag, score in result["scored"]]
        
        return TagEnrichResponse(
            success=True,
            title=result["title"],
            description=result["description"],
            candidates=result["candidates"],
            scored=scored_list,
            kept=result["kept"],
            dropped=result["dropped"],
            final_tags=result["openai_result"].get("final_tags", []),
            extra_tags=result["openai_result"].get("extra_tags", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"태그 추천 중 오류가 발생했습니다: {str(e)}"
        )

@app.post("/api/titles/suggest", response_model=TitleSuggestResponse)
async def suggest_titles(request: TitleSuggestRequest):
    """제목 추천 (OpenAI GPT 사용)"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요."
            )
        
        client = OpenAI(api_key=api_key)
        
        prompt = f"""
사용자가 '{request.keyword}'라는 주제를 입력했습니다.
{request.imageText if request.imageText else ''}

아래 유튜브 제목 작성 전략을 참고하여, 실제 유튜브에서 클릭을 유도할 수 있는 흥미롭고 자극적인 제목 {request.n}개를 만들어 주세요.

유튜브 제목 최적화 전략 (Actionable Tips):
1. [필수] 일치성 및 신뢰: 제목은 콘텐츠 내용을 정확히 반영해야 합니다 (클릭베이트 금지).
2. [검색/SEO] 키워드: 제목 앞부분에 주요 키워드를 자연스럽게 배치합니다 (60자 미만 권장).
3. [클릭률] 가치/질문/긴박감: 시청자의 문제(질문)를 제기하거나, 명확한 가치("10분 만에 전문가 되기")나 긴박감("놓치지 마세요")을 제시합니다.
4. [형식] 숫자/괄호: 홀수(7, 9)를 포함한 리스트 스타일 제목과 괄호 () 사용은 클릭률을 높입니다.
5. [타깃] 언어: 시청자의 은어/전문 용어를 사용하고 시청자("당신")에게 직접 명령합니다.
6. [시너지] 썸네일: 제목은 썸네일과 일관성 있게 조화되어야 합니다.
7. [경쟁] 분석: 경쟁사 상위 동영상의 제목 구조를 연구하여 변형합니다.
8. [추가] 와우 요소: '놀라운', '충격적인', '역대급' 등의 감탄사/수식어를 활용하여 호기심을 극대화합니다.
9. [교육/리스트] '하우투(How-to)' 또는 '리스트 스타일' 형식을 적용합니다.

제목은 자연스럽고 유머/감탄사/의문형 등을 적절히 활용하여 사람들의 호기심을 자극하세요.
제목 목록만 출력해 주세요. (예: 1. ... 2. ...)
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
        )
        
        titles_text = response.choices[0].message.content.strip()
        
        # 제목을 배열로 파싱
        import re
        titles_list = []
        for line in titles_text.split('\n'):
            title = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            if title and len(title) > 0:
                titles_list.append(title)
            if len(titles_list) >= request.n:
                break
        
        return TitleSuggestResponse(
            success=True,
            titles=titles_list
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"제목 추천 중 오류가 발생했습니다: {str(e)}"
        )

@app.post("/api/videos/create", response_model=VideoResponse)
async def create_video(request: VideoCreateRequest, session_token: str = None):
    """영상 정보 저장"""
    print(f"🚀 /api/videos/create 엔드포인트 호출됨")
    print(f"🚀 요청 데이터: {request}")
    print(f"🚀 세션 토큰: {session_token}")
    
    try:
        # 사용자 ID 추출 (로그인한 경우)
        user_id = None
        if session_token:
            user = user_db.validate_session(session_token)
            if user:
                user_id = user['id']
                print(f"🚀 사용자 ID: {user_id}")
            else:
                print(f"🚀 세션 토큰이 유효하지 않음")
        else:
            print(f"🚀 세션 토큰이 없음")
        
        print(f"🚀 create_video 함수 호출 전")
        # 영상 정보 저장
        video = user_db.create_video(
            title=request.title,
            category=request.category,
            length=request.length,
            upload_time=request.upload_time,
            description=request.description,
            thumbnail_image=request.thumbnail_image,
            user_id=user_id
        )
        print(f"🚀 create_video 함수 호출 완료: {video}")
        
        response_data = {
            "success": True,
            "message": "영상 정보가 저장되었습니다.",
            "data": {"video": video}
        }
        
        # JSONResponse를 사용하여 한글 인코딩 보장
        return UTF8JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"영상 저장 중 오류가 발생했습니다: {str(e)}"
        )

@app.get("/api/videos/list", response_model=APIResponse)
async def get_videos(session_token: str = None, limit: int = 100, offset: int = 0):
    """영상 목록 조회"""
    try:
        user_id = None
        if session_token:
            user = user_db.validate_session(session_token)
            if user:
                user_id = user['id']
        
        if user_id:
            # 로그인한 사용자의 영상 목록
            videos = user_db.get_user_videos(user_id)
        else:
            # 전체 영상 목록
            videos = user_db.get_all_videos(limit=limit, offset=offset)
        
        # 응답 데이터 준비
        response_data = {
            "success": True,
            "message": "영상 목록을 조회했습니다.",
            "data": {
                "videos": videos,
                "count": len(videos)
            }
        }
        
        # JSONResponse를 사용하여 한글 인코딩 보장
        return UTF8JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"영상 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("YouTube Analytics FastAPI 서버 시작")
    print("=" * 50)
    
    # 데이터베이스 파일 존재 확인
    if not os.path.exists('youtube_analytics.db'):
        print("❌ 데이터베이스 파일이 없습니다.")
        print("먼저 'python init_database.py'를 실행하여 데이터베이스를 초기화하세요.")
        exit(1)
    
    print("✅ 데이터베이스 연결 확인")
    print("🚀 FastAPI 서버 시작 중...")
    print("\n서버 주소: http://localhost:8001")
    print("API 문서: http://localhost:8001/docs")
    print("ReDoc 문서: http://localhost:8001/redoc")
    print("\n종료하려면 Ctrl+C를 누르세요.")
    print("=" * 50)
    
    import os
    port = int(os.environ.get("PORT", 8001))
    
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=port,
        reload=False  # 프로덕션에서는 reload=False
    )
