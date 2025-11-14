#!/usr/bin/env python3
"""
1등 유튜버 되기 FastAPI 서버
OpenAPI 스펙을 사용하여 REST API를 제공합니다.
"""

from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import json
from datetime import datetime
import sys
import os
import numpy as np
import pandas as pd
import re
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Kaggle API import (환경 변수 KAGGLE_USERNAME, KAGGLE_KEY 사용)
from kaggle.api.kaggle_api_extended import KaggleApi

# YouTube API import
try:
    from googleapiclient.discovery import build
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    print("⚠️ googleapiclient가 설치되지 않았습니다. 'pip install google-api-python-client'를 실행하세요.")

# 2025년 데이터 캐시 (전역 변수)
df_2025_cache = None
cache_metadata = {
    'date_column': None,
    'category_column': None,
    'views_column': None,
    'video_id_column': None
}

# 현재 스크립트의 디렉토리와 상위 디렉토리를 sys.path에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from database import UserDatabase, db
from tags.tag_recommendation_model import TagRecommendationModel
from enrich_tags import run_pipeline
from openai import OpenAI

# ML 모델 라이브러리 import
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost가 설치되지 않았습니다.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM이 설치되지 않았습니다.")

try:
    import xgboost as xgb
    import joblib
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost가 설치되지 않았습니다.")

# OpenAI API 키 설정
# 환경 변수에서만 키를 가져옵니다 (보안상 하드코딩하지 않음)
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    print("⚠️ 경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("   환경 변수를 설정하세요: export OPENAI_API_KEY='your-api-key-here'")
else:
    print("✅ OpenAI API 키가 환경 변수에서 로드되었습니다.")

# FastAPI 앱 초기화
app = FastAPI(
    title="1등 유튜버 되기 API",
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

# 조회수 예측 모델 캐시 (카테고리별)
prediction_models = {}

# 모델 파일 경로 설정 (여러 경로 시도)
def get_model_base_path():
    """모델 디렉토리 경로 찾기 (여러 경로 시도)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    current_dir = os.getcwd()
    
    possible_paths = [
        os.path.join(project_root, "모델"),  # 프로젝트 루트/모델
        os.path.join(script_dir, "모델"),  # tags/모델
        "/app/모델",  # Railway 배포 환경 (루트)
        "/app/tags/모델",  # Railway 배포 환경 (tags 디렉토리)
        os.path.join(current_dir, "모델"),  # 현재 작업 디렉토리/모델
        os.path.join(current_dir, "tags", "모델"),  # 현재 작업 디렉토리/tags/모델
        "모델",  # 상대 경로
    ]
    
    print(f"🔍 모델 디렉토리 검색 시작...")
    print(f"   script_dir: {script_dir}")
    print(f"   project_root: {project_root}")
    print(f"   current_dir: {current_dir}")
    
    for path in possible_paths:
        abs_path = os.path.abspath(path) if not os.path.isabs(path) else path
        exists = os.path.exists(abs_path)
        is_dir = os.path.isdir(abs_path) if exists else False
        print(f"   시도: {abs_path} (존재: {exists}, 디렉토리: {is_dir})")
        
        if exists and is_dir:
            # 모델 파일이 하나라도 있는지 확인
            try:
                files = os.listdir(abs_path)
                model_files = [f for f in files if f.endswith(('.cbm', '.pkl'))]
                print(f"      파일 수: {len(files)}, 모델 파일 수: {len(model_files)}")
                if model_files:
                    print(f"      모델 파일 예시: {model_files[:3]}")
                    print(f"✅ 모델 디렉토리 발견: {abs_path}")
                    return abs_path
            except Exception as e:
                print(f"      디렉토리 읽기 실패: {e}")
    
    # 기본 경로 반환 (존재하지 않아도)
    default_path = os.path.join(project_root, "모델")
    print(f"⚠️ 모델 디렉토리를 찾을 수 없습니다. 기본 경로 사용: {default_path}")
    print(f"💡 Railway 배포 시 Git LFS 파일이 제대로 다운로드되었는지 확인하세요.")
    print(f"💡 Railway 로그에서 'git lfs pull' 명령이 성공했는지 확인하세요.")
    return default_path

MODEL_BASE_PATH = get_model_base_path()

def load_prediction_models(category: str):
    """카테고리별 분류/회귀 모델 로드"""
    if category in prediction_models:
        return prediction_models[category]
    
    category_int = int(category)
    models = {}
    
    print(f"🔍 카테고리 {category} 모델 로드 시작")
    print(f"🔍 MODEL_BASE_PATH: {MODEL_BASE_PATH}")
    print(f"🔍 MODEL_BASE_PATH 존재 여부: {os.path.exists(MODEL_BASE_PATH)}")
    
    try:
        # 카테고리별 모델 파일 경로 설정
        if category_int in [1, 15, 19]:  # CatBoost
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost가 설치되지 않았습니다.")
            cls_model_path = os.path.join(MODEL_BASE_PATH, f"catboost_model_{category_int}_class.cbm")
            reg_model_path = os.path.join(MODEL_BASE_PATH, f"catboost_model_{category_int}.cbm")
            print(f"🔍 CatBoost 모델 경로:")
            print(f"   - 분류: {cls_model_path} (존재: {os.path.exists(cls_model_path)})")
            print(f"   - 회귀: {reg_model_path} (존재: {os.path.exists(reg_model_path)})")
            
            if os.path.exists(cls_model_path) and os.path.exists(reg_model_path):
                cls_model = CatBoostClassifier()
                cls_model.load_model(cls_model_path)
                reg_model = CatBoostRegressor()
                reg_model.load_model(reg_model_path)
                models = {
                    'cls': cls_model,
                    'reg': reg_model,
                    'type': 'catboost'
                }
                
        elif category_int in [10, 22, 24, 26]:  # LightGBM
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM이 설치되지 않았습니다.")
            cls_model_path = os.path.join(MODEL_BASE_PATH, f"lgbm_model_{category_int}_class.pkl")
            reg_model_path = os.path.join(MODEL_BASE_PATH, f"lgbm_model_{category_int}.pkl")
            print(f"🔍 LightGBM 모델 경로:")
            print(f"   - 분류: {cls_model_path} (존재: {os.path.exists(cls_model_path)})")
            print(f"   - 회귀: {reg_model_path} (존재: {os.path.exists(reg_model_path)})")
            
            if os.path.exists(cls_model_path) and os.path.exists(reg_model_path):
                print(f"📦 모델 파일 로딩 시작...")
                cls_model = joblib.load(cls_model_path)
                reg_model = joblib.load(reg_model_path)
                models = {
                    'cls': cls_model,
                    'reg': reg_model,
                    'type': 'lightgbm'
                }
                print(f"📦 모델 파일 로딩 완료")
            else:
                # 디렉토리 내용 확인
                if os.path.exists(MODEL_BASE_PATH):
                    print(f"📂 모델 디렉토리 내용:")
                    for file in os.listdir(MODEL_BASE_PATH):
                        if f"_{category_int}" in file:
                            print(f"   - {file}")
                
        elif category_int in [17, 20, 23, 28]:  # XGBoost
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost가 설치되지 않았습니다.")
            cls_model_path = os.path.join(MODEL_BASE_PATH, f"xgb_model_{category_int}_class.pkl")
            reg_model_path = os.path.join(MODEL_BASE_PATH, f"xgb_model_{category_int}.pkl")
            print(f"🔍 XGBoost 모델 경로:")
            print(f"   - 분류: {cls_model_path} (존재: {os.path.exists(cls_model_path)})")
            print(f"   - 회귀: {reg_model_path} (존재: {os.path.exists(reg_model_path)})")
            
            if os.path.exists(cls_model_path) and os.path.exists(reg_model_path):
                cls_model = joblib.load(cls_model_path)
                reg_model = joblib.load(reg_model_path)
                models = {
                    'cls': cls_model,
                    'reg': reg_model,
                    'type': 'xgboost'
                }
        
        if models:
            prediction_models[category] = models
            print(f"✅ 카테고리 {category} 모델 로드 완료")
        else:
            print(f"⚠️ 카테고리 {category} 모델 파일을 찾을 수 없습니다.")
            if os.path.exists(MODEL_BASE_PATH):
                print(f"📂 모델 디렉토리 전체 내용:")
                for file in os.listdir(MODEL_BASE_PATH):
                    print(f"   - {file}")
            
    except Exception as e:
        print(f"❌ 카테고리 {category} 모델 로드 실패: {str(e)}")
        import traceback
        print(f"❌ 상세 오류:\n{traceback.format_exc()}")
    
    return models

def predict_views(category: str, input_df: pd.DataFrame) -> Dict[str, Any]:
    """카테고리별 모델을 사용하여 조회수 예측"""
    category_int = int(category)
    
    print(f"🔍 predict_views 시작 - 카테고리: {category}")
    
    # 모델 로드
    models = load_prediction_models(category)
    print(f"🔍 모델 로드 결과: {models}")
    
    if not models:
        error_msg = f"카테고리 {category} 모델을 찾을 수 없습니다. MODEL_BASE_PATH: {MODEL_BASE_PATH}"
        print(f"❌ {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_msg
        )
    
    cls_model = models['cls']
    reg_model = models['reg']
    model_type = models['type']
    
    print(f"🔍 모델 타입: {model_type}")
    print(f"🔍 분류 모델: {type(cls_model)}")
    print(f"🔍 회귀 모델: {type(reg_model)}")
    
    try:
        # 분류 모델로 pred_popular_prob 생성
        print(f"🔍 분류 모델 feature 추출 시작...")
        if model_type == 'catboost':
            cls_features = cls_model.feature_names_
        elif model_type == 'lightgbm':
            # LightGBM: feature_name_ 또는 feature_names_ 사용
            if hasattr(cls_model, 'feature_name_'):
                cls_features = cls_model.feature_name_
            elif hasattr(cls_model, 'feature_names_'):
                cls_features = cls_model.feature_names_
            else:
                raise AttributeError("LightGBM 모델에서 feature_name_ 또는 feature_names_ 속성을 찾을 수 없습니다.")
        else:  # xgboost
            # XGBoost: get_booster() 사용 시도, 없으면 직접 feature_names_ 사용
            try:
                cls_features = cls_model.get_booster().feature_names
            except:
                if hasattr(cls_model, 'feature_names_'):
                    cls_features = cls_model.feature_names_
                else:
                    raise AttributeError("XGBoost 모델에서 feature_names를 찾을 수 없습니다.")
        
        # 누락된 피처는 0으로 채우기
        print("=" * 80)
        print("🔍 [분류 모델] Feature 확인")
        print("=" * 80)
        print(f"🔍 분류 모델 feature 개수: {len(cls_features)}")
        print(f"🔍 분류 모델 features: {cls_features}")
        
        missing_features = []
        for f in cls_features:
            if f not in input_df.columns:
                input_df[f] = 0
                missing_features.append(f)
        if missing_features:
            print(f"⚠️ 누락된 feature를 0으로 채움: {missing_features}")
        
        print("\n📊 [분류 모델] 입력 데이터 준비 완료")
        print(f"📊 분류 모델 입력 컬럼 ({len(input_df[cls_features].columns)}개): {input_df[cls_features].columns.tolist()}")
        print(f"\n📊 분류 모델 입력 데이터:")
        print(input_df[cls_features])
        
        # pred_popular_prob 계산
        print("\n" + "=" * 80)
        print("🔍 [분류 모델] predict_proba 실행 시작...")
        print("=" * 80)
        try:
            if model_type == 'catboost':
                proba_result = cls_model.predict_proba(input_df[cls_features])
                print(f"📊 CatBoost predict_proba 결과 shape: {proba_result.shape}")
                print(f"📊 CatBoost predict_proba 결과: {proba_result}")
                input_df['pred_popular_prob'] = proba_result[:, 1]
            elif model_type == 'lightgbm':
                proba_result = cls_model.predict_proba(input_df[cls_features])
                print(f"📊 LightGBM predict_proba 결과 shape: {proba_result.shape}")
                print(f"📊 LightGBM predict_proba 결과: {proba_result}")
                print(f"📊 LightGBM predict_proba 결과 (상세): 클래스 0 확률={proba_result[0][0]:.6f}, 클래스 1 확률={proba_result[0][1]:.6f}")
                input_df['pred_popular_prob'] = proba_result[:, 1]
            else:  # xgboost
                proba_result = cls_model.predict_proba(input_df[cls_features])
                print(f"📊 XGBoost predict_proba 결과 shape: {proba_result.shape}")
                print(f"📊 XGBoost predict_proba 결과: {proba_result}")
                input_df['pred_popular_prob'] = proba_result[:, 1]
            
            print("\n" + "=" * 80)
            print("✅ [분류 모델] pred_popular_prob 계산 완료")
            print("=" * 80)
            pred_value = input_df['pred_popular_prob'].iloc[0]
            print(f"📊 pred_popular_prob 값: {pred_value}")
            print(f"📊 pred_popular_prob 타입: {type(pred_value)}")
            print(f"📊 pred_popular_prob (퍼센트): {pred_value * 100:.2f}%")
            print("=" * 80 + "\n")
        except Exception as e:
            print("\n" + "=" * 80)
            print("❌ [분류 모델] 예측 실패")
            print("=" * 80)
            print(f"❌ 오류 메시지: {str(e)}")
            import traceback
            print(f"❌ 상세 오류:\n{traceback.format_exc()}")
            print("=" * 80 + "\n")
            raise
        
        # 회귀 모델로 조회수 예측
        print("=" * 80)
        print("🔍 [회귀 모델] Feature 확인")
        print("=" * 80)
        if model_type == 'catboost':
            reg_features = reg_model.feature_names_
        elif model_type == 'lightgbm':
            # LightGBM: feature_name_ 또는 feature_names_ 사용
            if hasattr(reg_model, 'feature_name_'):
                reg_features = reg_model.feature_name_
            elif hasattr(reg_model, 'feature_names_'):
                reg_features = reg_model.feature_names_
            else:
                raise AttributeError("LightGBM 회귀 모델에서 feature_name_ 또는 feature_names_ 속성을 찾을 수 없습니다.")
        else:  # xgboost
            # XGBoost: get_booster() 사용 시도, 없으면 직접 feature_names_ 사용
            try:
                reg_features = reg_model.get_booster().feature_names
            except:
                if hasattr(reg_model, 'feature_names_'):
                    reg_features = reg_model.feature_names_
                else:
                    raise AttributeError("XGBoost 회귀 모델에서 feature_names를 찾을 수 없습니다.")
        
        print(f"🔍 회귀 모델 feature 개수: {len(reg_features)}")
        print(f"🔍 회귀 모델 features: {reg_features}")
        print(f"🔍 예상 features (11개): ['caption_available', 'pub_month', 'pub_day', 'pub_hour_sin', 'pub_hour_cos', 'pub_weekday_sin', 'pub_weekday_cos', 'duration_sec', 'definition', 'subscriber_count_log', 'pred_popular_prob']")
        
        # 누락된 피처는 0으로 채우기
        missing_reg_features = []
        for f in reg_features:
            if f not in input_df.columns:
                input_df[f] = 0
                missing_reg_features.append(f)
        if missing_reg_features:
            print(f"⚠️ 누락된 feature를 0으로 채움: {missing_reg_features}")
        
        print("\n📊 [회귀 모델] 입력 데이터 준비 완료")
        print(f"📊 회귀 모델 입력 컬럼 ({len(input_df[reg_features].columns)}개): {input_df[reg_features].columns.tolist()}")
        print(f"\n📊 회귀 모델 입력 데이터:")
        print(input_df[reg_features])
        
        print("\n" + "=" * 80)
        print("🔍 [회귀 모델] predict 실행 시작...")
        print("=" * 80)
        
        # XGBoost의 경우 input_df를 명시적으로 전달해야 할 수 있음
        if model_type == 'xgboost' and category_int == 23:
            # 카테고리 23의 경우 특별 처리
            input_df_for_pred = input_df[reg_features]
            print(f"📊 XGBoost (카테고리 23) - 사용할 데이터 shape: {input_df_for_pred.shape}")
            y_pred_log = reg_model.predict(input_df_for_pred)
        else:
            print(f"📊 {model_type} 회귀 모델 predict 실행...")
            print(f"📊 사용할 컬럼: {reg_features}")
            y_pred_log = reg_model.predict(input_df[reg_features])
        
        print(f"✅ [회귀 모델] 예측 완료")
        print(f"📊 예측값 (로그 스케일): {y_pred_log[0]:.6f}")
        print("=" * 80)
        
        # 카테고리별 스케일 적용
        if category_int in [10, 23]:  # 100만 단위
            y_pred = np.expm1(y_pred_log) * 1_000_000
            print(f"📊 스케일 적용 (100만 단위): {y_pred[0]:,.0f}")
        else:  # 10만 단위
            y_pred = np.expm1(y_pred_log) * 100_000
            print(f"📊 스케일 적용 (10만 단위): {y_pred[0]:,.0f}")
        
        # 실제 사용된 모델 파일명 생성
        if category_int in [1, 15, 19]:  # CatBoost
            cls_model_file = f"catboost_model_{category_int}_class.cbm"
            reg_model_file = f"catboost_model_{category_int}.cbm"
        elif category_int in [10, 22, 24, 26]:  # LightGBM
            cls_model_file = f"lgbm_model_{category_int}_class.pkl"
            reg_model_file = f"lgbm_model_{category_int}.pkl"
        else:  # XGBoost
            cls_model_file = f"xgb_model_{category_int}_class.pkl"
            reg_model_file = f"xgb_model_{category_int}.pkl"
        
        # 모델 이름 생성 (표시용)
        model_type_names = {
            'catboost': 'CatBoost',
            'lightgbm': 'LightGBM',
            'xgboost': 'XGBoost'
        }
        cls_model_name = f"{model_type_names[model_type]} Classifier"
        reg_model_name = f"{model_type_names[model_type]} Regressor"
        
        return {
            'predicted_views': int(y_pred[0]),
            'pred_popular_prob': float(input_df['pred_popular_prob'].iloc[0]),
            'confidence': float(input_df['pred_popular_prob'].iloc[0]) * 100,
            'cls_model': cls_model_name,
            'reg_model': reg_model_name,
            'cls_model_file': cls_model_file,
            'reg_model_file': reg_model_file,
            'model_type': model_type
        }
        
    except Exception as e:
        print(f"❌ 예측 중 오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"예측 중 오류가 발생했습니다: {str(e)}"
        )

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
    extra_k: int = Field(default=20, description="추가 태그 개수")
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

class TitleGenerateRequest(BaseModel):
    keyword: str = Field(..., description="주제 키워드")
    imageText: Optional[str] = Field(default="", description="이미지 내용 요약")
    n: int = Field(default=5, ge=1, le=10, description="생성할 제목 개수")

class TitleGenerateResponse(BaseModel):
    success: bool
    titles: List[str]
    message: Optional[str] = None

class VideoCreateRequest(BaseModel):
    title: str = Field(..., description="영상 제목")
    category: str = Field(..., description="카테고리")
    length: float = Field(..., ge=0.01, description="영상 길이 (분, 소수점 가능)")
    upload_time: Optional[str] = Field(default=None, description="업로드 예정 시간")
    description: Optional[str] = Field(default=None, description="영상 설명")
    thumbnail_image: Optional[str] = Field(default=None, description="썸네일 이미지 (Base64)")
    has_subtitles: Optional[str] = Field(default=None, description="자막 제공 여부 (provided/not_provided)")
    video_quality: Optional[str] = Field(default=None, description="해상도 품질 (HD/SD)")
    subscriber_count: Optional[int] = Field(default=None, description="구독자수")

def preprocess_input_data(request: VideoCreateRequest) -> pd.DataFrame:
    """사용자 입력 데이터를 모델 입력 형식으로 전처리"""
    # 업로드 예정 시간 파싱
    upload_time = request.upload_time
    if not upload_time:
        # 기본값 (현재 시간 + 1시간)
        dt = datetime.now()
        dt = dt.replace(hour=(dt.hour + 1) % 24)
    else:
        # datetime-local 형식: YYYY-MM-DDTHH:mm
        dt = datetime.fromisoformat(upload_time)
        if dt.tzinfo:
            dt = dt.astimezone().replace(tzinfo=None)
    
    month = dt.month
    day = dt.day
    hour = dt.hour
    weekday_python = dt.weekday()  # Python: 0=월요일, 6=일요일
    
    # 요일 변환: Python weekday (0=월, 6=일) -> JavaScript/일반 형식 (0=일, 6=토)
    # Python weekday + 1 -> modulo 7로 변환
    weekday = (weekday_python + 1) % 7  # 0=일요일, 1=월요일, ..., 6=토요일
    
    # 데이터프레임 생성
    # 학습 시 subscriber_count는 drop되고 subscriber_count_log만 사용됨
    # 모델이 요구하는 컬럼명: duration_sec (duration 아님!)
    df = pd.DataFrame({
        'duration_sec': [request.length * 60],  # 분을 초로 변환
        'definition': [1 if request.video_quality == 'HD' else 0],
        'caption_available': [1 if request.has_subtitles == 'provided' else 0],
        'pub_month': [month],
        'pub_day': [day],
        'pub_hour_sin': [np.sin(2 * np.pi * hour / 24)],
        'pub_hour_cos': [np.cos(2 * np.pi * hour / 24)],
        'pub_weekday_sin': [np.sin(2 * np.pi * weekday / 7)],
        'pub_weekday_cos': [np.cos(2 * np.pi * weekday / 7)],
        'subscriber_count_log': [np.log1p(request.subscriber_count) if request.subscriber_count and request.subscriber_count > 0 else 0]
    })
    
    return df

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
        current_dir = os.getcwd()
        
        possible_paths = [
            os.path.join(script_dir, "tag_recommendation_model.pkl"),  # tags/tag_recommendation_model.pkl
            os.path.join(project_root, "tags", "tag_recommendation_model.pkl"),  # 프로젝트 루트/tags/tag_recommendation_model.pkl
            "/app/tags/tag_recommendation_model.pkl",  # Railway 배포 환경 (tags 디렉토리)
            "/app/tag_recommendation_model.pkl",  # Railway 배포 환경 (루트)
            os.path.join(current_dir, "tag_recommendation_model.pkl"),  # 현재 작업 디렉토리
            os.path.join(current_dir, "tags", "tag_recommendation_model.pkl"),  # 현재 작업 디렉토리/tags
            "tag_recommendation_model.pkl",  # 상대 경로
        ]
        
        print(f"🔍 태그 추천 모델 파일 검색 시작...")
        print(f"   script_dir: {script_dir}")
        print(f"   project_root: {project_root}")
        print(f"   current_dir: {current_dir}")
        
        model_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path) if not os.path.isabs(path) else path
            print(f"   시도: {abs_path} (존재: {os.path.exists(abs_path)})")
            if os.path.exists(abs_path):
                model_path = abs_path
                print(f"   ✅ 모델 파일 발견: {abs_path}")
                break
        
        if model_path:
            print(f"📦 태그 추천 모델 로딩 시작: {model_path}")
            tag_model = TagRecommendationModel()
            tag_model.load_model(model_path)
            print(f"✅ 태그 추천 모델 로드 완료: {model_path}")
        else:
            print("⚠️ 태그 추천 모델 파일을 찾을 수 없습니다.")
            print(f"   시도한 경로들:")
            for path in possible_paths:
                abs_path = os.path.abspath(path) if not os.path.isabs(path) else path
                print(f"     - {abs_path}")
            print("   💡 Railway 배포 시 tags/tag_recommendation_model.pkl 파일이 포함되어 있는지 확인하세요.")
            tag_model = None
    except Exception as e:
        print(f"❌ 태그 추천 모델 로드 실패: {e}")
        import traceback
        print(f"❌ 상세 오류:\n{traceback.format_exc()}")
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
    # 모델 파일이 없으면 Hugging Face에서 다운로드 시도
    try:
        import subprocess
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        download_script = os.path.join(project_root, "download_models.py")
        
        # 모델 디렉토리와 태그 모델 파일 확인
        model_dir = os.path.join(project_root, "모델")
        tag_model_path = os.path.join(script_dir, "tag_recommendation_model.pkl")
        
        # 모델이 하나도 없으면 다운로드 시도
        models_exist = (
            os.path.exists(model_dir) and 
            len([f for f in os.listdir(model_dir) if f.endswith(('.pkl', '.cbm'))]) > 0
        ) or os.path.exists(tag_model_path)
        
        if not models_exist and os.path.exists(download_script):
            print("📥 모델 파일이 없습니다. Hugging Face에서 다운로드를 시도합니다...")
            try:
                result = subprocess.run(
                    [sys.executable, download_script],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5분 타임아웃
                )
                if result.returncode == 0:
                    print("✅ 모델 다운로드 완료")
                else:
                    print(f"⚠️ 모델 다운로드 실패: {result.stderr}")
            except Exception as e:
                print(f"⚠️ 모델 다운로드 중 오류: {e}")
    except Exception as e:
        print(f"⚠️ 모델 다운로드 체크 중 오류: {e}")
    
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
        <title>1등 유튜버 되기 API</title>
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
            <h1>🎬 1등 유튜버 되기 API</h1>
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
        if tag_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="태그 추천 모델이 로드되지 않았습니다."
            )
        
        title = request.title.strip()
        description = request.description.strip() if request.description else ""
        
        if not title:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="제목을 입력해주세요."
            )
        
        # 모델 경로 설정 (여러 경로 시도) - load_tag_model과 동일한 로직 사용
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        current_dir = os.getcwd()
        
        possible_paths = [
            os.path.join(script_dir, "tag_recommendation_model.pkl"),  # tags/tag_recommendation_model.pkl
            os.path.join(project_root, "tags", "tag_recommendation_model.pkl"),  # 프로젝트 루트/tags/tag_recommendation_model.pkl
            "/app/tags/tag_recommendation_model.pkl",  # Railway 배포 환경 (tags 디렉토리)
            "/app/tag_recommendation_model.pkl",  # Railway 배포 환경 (루트)
            os.path.join(current_dir, "tag_recommendation_model.pkl"),  # 현재 작업 디렉토리
            os.path.join(current_dir, "tags", "tag_recommendation_model.pkl"),  # 현재 작업 디렉토리/tags
            "tag_recommendation_model.pkl",  # 상대 경로
        ]
        
        model_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path) if not os.path.isabs(path) else path
            if os.path.exists(abs_path):
                model_path = abs_path
                print(f"✅ enrich_tags 모델 파일 발견: {model_path}")
                break
        
        if not model_path:
            error_detail = f"태그 추천 모델 파일을 찾을 수 없습니다.\n시도한 경로:\n"
            for path in possible_paths:
                abs_path = os.path.abspath(path) if not os.path.isabs(path) else path
                error_detail += f"  - {abs_path}\n"
            error_detail += "\nRailway 배포 시 tags/tag_recommendation_model.pkl 파일이 포함되어 있는지 확인하세요."
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_detail
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
            api_key=request.api_key or os.environ.get("OPENAI_API_KEY")
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

@app.post("/api/titles/generate", response_model=TitleGenerateResponse)
async def generate_titles(request: TitleGenerateRequest):
    """제목 추천 기능 (OpenAI GPT 사용)"""
    try:
        # OpenAI API 키 확인
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OPENAI_API_KEY 환경 변수가 설정되지 않았습니다."
            )
        
        # OpenAI 클라이언트 초기화 (proxies 관련 에러 방지)
        # httpx 클라이언트를 직접 설정하여 proxies 문제 회피
        try:
            import httpx
            http_client = httpx.Client(
                timeout=60.0,  # 60초 타임아웃
                follow_redirects=True
            )
            client = OpenAI(
                api_key=api_key,
                http_client=http_client,
                max_retries=2
            )
        except Exception as e:
            # httpx 클라이언트 설정 실패 시 기본 초기화
            print(f"⚠️ httpx 클라이언트 설정 실패, 기본 초기화 사용: {e}")
            client = OpenAI(api_key=api_key)
        
        prompt = f"""
            사용자가 '{request.keyword}'라는 주제를 입력했습니다.
            {f'이미지에 포함된 내용 요약: {request.imageText}' if request.imageText else ''}

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
        
        content = response.choices[0].message.content.strip()
        
        # 제목 목록 파싱 (1. 2. 3. 형식 또는 - 형식)
        titles = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 숫자나 기호로 시작하는 경우 제거 (정규식 사용)
            line = re.sub(r'^\d+[\.\)]\s*', '', line)  # "1. " 또는 "1) " 제거
            line = re.sub(r'^[-•]\s*', '', line)  # "- " 또는 "• " 제거
            line = line.strip()
            if line and len(line) > 0:
                titles.append(line)
        
        # 요청한 개수만큼만 반환
        titles = titles[:request.n]
        
        if not titles:
            return TitleGenerateResponse(
                success=False,
                titles=[],
                message="제목 생성에 실패했습니다."
            )
        
        return TitleGenerateResponse(
            success=True,
            titles=titles,
            message=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        error_detail = ""
        
        # OpenAI API 관련 에러 메시지 개선
        if "401" in error_msg or "invalid_api_key" in error_msg.lower() or "incorrect api key" in error_msg.lower():
            error_detail = "OpenAI API 키가 유효하지 않거나 설정되지 않았습니다. Railway 환경 변수에서 OPENAI_API_KEY를 확인해주세요."
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            error_detail = "OpenAI API 요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."
            status_code = status.HTTP_504_GATEWAY_TIMEOUT
        elif "rate limit" in error_msg.lower():
            error_detail = "OpenAI API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
        elif "authentication" in error_msg.lower():
            error_detail = "OpenAI API 인증에 실패했습니다. API 키를 확인해주세요."
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            error_detail = f"제목 생성 중 오류가 발생했습니다: {error_msg}"
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        raise HTTPException(
            status_code=status_code,
            detail=error_detail
        )

@app.post("/api/videos/create", response_model=VideoResponse)
async def create_video(
    request: VideoCreateRequest,
    session_token: Optional[str] = Query(None, description="세션 토큰")
):
    """영상 정보 저장 및 조회수 예측"""
    print(f"🚀 /api/videos/create 엔드포인트 호출됨")
    print(f"🚀 요청 데이터: {request}")
    print(f"🚀 세션 토큰: {session_token}")
    
    try:
        # 사용자 ID 추출 (로그인한 경우)
        user_id = None
        if session_token:
            try:
                validated_user = user_db.validate_session(session_token)
                if validated_user:
                    user_id = validated_user['id']
                    print(f"🚀 사용자 ID: {user_id}")
                else:
                    print(f"🚀 세션 토큰이 유효하지 않음")
            except Exception as e:
                print(f"⚠️ 세션 토큰 검증 오류: {str(e)}")
        else:
            print(f"🚀 세션 토큰이 없음 (비로그인 상태)")
        
        print(f"🚀 create_video 함수 호출 전")
        print(f"🚀 저장할 데이터: title={request.title}, category={request.category}, length={request.length}")
        
        # 조회수 예측 수행
        print("\n" + "!" * 80)
        print("!" * 80)
        print("🚀 [조회수 예측 수행 시작 - 코드 실행 확인]")
        print("!" * 80)
        print("!" * 80 + "\n")
        
        prediction_result = None
        print("\n" + "=" * 80)
        print("🚀 [조회수 예측 시작]")
        print("=" * 80)
        print(f"🔍 예측 시작 - 카테고리: {request.category}")
        try:
            # 데이터 전처리
            print(f"\n📊 [1단계] 전처리 시작...")
            input_df = preprocess_input_data(request)
            print(f"✅ 전처리 완료!")
            print(f"📊 입력 데이터 shape: {input_df.shape}")
            print(f"📊 입력 데이터 columns: {input_df.columns.tolist()}")
            print(f"\n📊 전처리된 입력 데이터:")
            print(input_df)
            
            # 예측 수행
            print(f"\n🔍 [2단계] predict_views 호출 시작, category: {request.category}")
            prediction_result = predict_views(request.category, input_df)
            print(f"\n" + "=" * 80)
            print("✅ [조회수 예측 완료]")
            print("=" * 80)
            print(f"✅ 조회수 예측 결과: {prediction_result}")
            print("=" * 80 + "\n")
        except HTTPException as pred_error:
            # HTTPException은 그대로 재발생 (상태 코드와 함께)
            print(f"⚠️ 조회수 예측 실패 (HTTPException): {str(pred_error)}")
            print(f"⚠️ 상태 코드: {pred_error.status_code}")
            print(f"⚠️ 상세 메시지: {pred_error.detail}")
            # HTTPException은 상위로 전파되어야 함
            raise
        except Exception as pred_error:
            print(f"⚠️ 조회수 예측 실패: {str(pred_error)}")
            import traceback
            print(f"⚠️ 예측 실패 상세 오류:\n{traceback.format_exc()}")
            # 예측 실패해도 저장은 계속 진행 (prediction_result는 None으로 유지)
        
        # 영상 정보 저장
        try:
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
        except Exception as db_error:
            print(f"❌ 데이터베이스 저장 오류: {str(db_error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"데이터베이스 저장 중 오류가 발생했습니다: {str(db_error)}"
            )
        
        # prediction이 None이어도 명시적으로 포함
        print("\n" + "=" * 80)
        print("📤 [응답 생성]")
        print("=" * 80)
        print(f"📊 prediction_result 값: {prediction_result}")
        print(f"📊 prediction_result 타입: {type(prediction_result)}")
        print(f"📊 prediction_result is None: {prediction_result is None}")
        
        data_dict = {
            "video": video
        }
        # None이어도 명시적으로 포함
        data_dict["prediction"] = prediction_result
        
        print(f"📊 data_dict에 prediction 포함 여부: {'prediction' in data_dict}")
        print(f"📊 data_dict['prediction'] 값: {data_dict['prediction']}")
        
        response_data = {
            "success": True,
            "message": "영상 정보가 저장되었습니다.",
            "data": data_dict
        }
        
        print(f"📊 response_data['data']에 prediction 포함 여부: {'prediction' in response_data['data']}")
        print(f"📊 response_data['data']['prediction'] 값: {response_data['data'].get('prediction')}")
        print(f"✅ 최종 응답 데이터 구조:")
        print(f"   - success: {response_data['success']}")
        print(f"   - data.video: {response_data['data'].get('video') is not None}")
        print(f"   - data.prediction: {response_data['data'].get('prediction')}")
        print("=" * 80 + "\n")
        
        # JSONResponse를 사용하여 한글 인코딩 보장
        response = UTF8JSONResponse(content=response_data)
        print(f"📊 응답 직렬화 후 확인: prediction이 포함되어 있는지 확인 필요")
        return response
        
    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {str(e)}")
        import traceback
        print(f"❌ 트레이스백: {traceback.format_exc()}")
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

class TrendUpdateResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

@app.get("/api/trends/test-kaggle")
async def test_kaggle_download():
    """Kaggle 데이터 다운로드 테스트"""
    try:
        # Kaggle API 인증
        kaggle_username = os.environ.get('KAGGLE_USERNAME')
        kaggle_key = os.environ.get('KAGGLE_KEY')
        
        if not kaggle_username or not kaggle_key:
            return UTF8JSONResponse(content={
                "success": False,
                "message": "KAGGLE_USERNAME 또는 KAGGLE_KEY 환경 변수가 설정되지 않았습니다.",
                "data": None
            })
        
        print(f"🔐 Kaggle API 인증 시도...")
        os.environ['KAGGLE_USERNAME'] = kaggle_username
        os.environ['KAGGLE_KEY'] = kaggle_key
        
        # .kaggle 폴더에 파일 생성
        import pathlib
        home_dir = pathlib.Path.home()
        kaggle_home_dir = home_dir / '.kaggle'
        kaggle_home_json = kaggle_home_dir / 'kaggle.json'
        
        if not kaggle_home_dir.exists():
            kaggle_home_dir.mkdir(parents=True, exist_ok=True)
        
        kaggle_creds = {'username': kaggle_username, 'key': kaggle_key}
        with open(str(kaggle_home_json), 'w') as f:
            json.dump(kaggle_creds, f)
        
        try:
            os.chmod(str(kaggle_home_json), 0o600)
        except:
            pass
        
        api = KaggleApi()
        api.authenticate()
        print(f"✅ Kaggle API 인증 성공!")
        
        # 데이터셋 다운로드 테스트
        dataset = "asaniczka/trending-youtube-videos-113-countries"
        print(f"📥 Kaggle 데이터셋 다운로드 시작: {dataset}")
        
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        print(f"   📁 임시 디렉토리 생성: {temp_dir}")
        try:
            print(f"   ⏳ 데이터셋 다운로드 중... (시간이 걸릴 수 있습니다)")
            api.dataset_download_files(dataset, path=temp_dir, unzip=True)
            print(f"   ✅ 데이터셋 다운로드 완료!")
            
            # 디렉토리 구조 확인
            print(f"   📂 다운로드된 파일 목록:")
            for root, dirs, files in os.walk(temp_dir):
                level = root.replace(temp_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f"{subindent}{file} ({file_size:,} bytes)")
            
            # CSV 파일 찾기
            csv_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        csv_files.append({
                            'path': file_path,
                            'name': file,
                            'size': os.path.getsize(file_path)
                        })
                        print(f"   ✅ CSV 파일 발견: {file} ({os.path.getsize(file_path):,} bytes)")
            
            if not csv_files:
                return UTF8JSONResponse(content={
                    "success": False,
                    "message": "CSV 파일을 찾을 수 없습니다.",
                    "data": {"temp_dir": temp_dir, "files": list(os.walk(temp_dir))}
                })
            
            # 첫 번째 CSV 파일 읽기 (샘플)
            csv_file = csv_files[0]['path']
            print(f"📖 CSV 파일 읽기: {csv_file}")
            df = pd.read_csv(csv_file, encoding='utf-8-sig', nrows=100)  # 처음 100행만
            
            # 컬럼 정보 상세 출력
            print(f"\n{'='*80}")
            print(f"📊 CSV 파일 컬럼 정보")
            print(f"{'='*80}")
            print(f"   총 컬럼 수: {len(df.columns)}")
            print(f"   컬럼 목록:")
            for i, col in enumerate(df.columns, 1):
                print(f"      {i:2d}. {col}")
            print(f"\n   컬럼 타입:")
            for col in df.columns:
                dtype = df[col].dtype
                null_count = df[col].isna().sum()
                print(f"      {col:30s} : {str(dtype):15s} (null: {null_count}/{len(df)})")
            print(f"{'='*80}\n")
            
            # NaN 값을 None으로 변환 (JSON 직렬화를 위해)
            df = df.replace({pd.NA: None, pd.NaT: None})
            df = df.where(pd.notnull(df), None)
            
            # 샘플 데이터 준비 (NaN 처리)
            sample_data = {
                "rows": len(df),
                "columns": list(df.columns),
                "first_row": None,
                "sample": []
            }
            
            if len(df) > 0:
                # 첫 번째 행 (NaN을 None으로 변환)
                first_row = df.iloc[0].to_dict()
                first_row = {k: (None if pd.isna(v) else v) for k, v in first_row.items()}
                sample_data["first_row"] = first_row
                
                # 샘플 5행 (NaN을 None으로 변환)
                sample_df = df.head(5).copy()
                sample_records = []
                for _, row in sample_df.iterrows():
                    record = {}
                    for col in sample_df.columns:
                        val = row[col]
                        if pd.isna(val):
                            record[col] = None
                        elif isinstance(val, (int, float)) and (pd.isna(val) or not np.isfinite(val)):
                            record[col] = None
                        else:
                            record[col] = val
                    sample_records.append(record)
                sample_data["sample"] = sample_records
            
            return UTF8JSONResponse(content={
                "success": True,
                "message": "Kaggle 데이터 다운로드 성공!",
                "data": {
                    "csv_files": csv_files,
                    "sample_data": sample_data
                }
            })
            
        finally:
            try:
                shutil.rmtree(temp_dir)
                print(f"🗑️ 임시 디렉토리 삭제 완료")
            except Exception as e:
                print(f"⚠️ 임시 디렉토리 삭제 실패: {e}")
                
    except Exception as e:
        import traceback
        error_detail = f"Kaggle 데이터 다운로드 테스트 실패: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_detail}")
        return UTF8JSONResponse(content={
            "success": False,
            "message": error_detail,
            "data": None
        })

@app.post("/api/trends/update-month", response_model=TrendUpdateResponse)
async def update_trends_month(month: int = Query(..., ge=1, le=12, description="월 (1-12)")):
    """Kaggle API를 사용하여 특정 월의 트렌드 분석 업데이트"""
    import tempfile
    import shutil
    import io
    import zipfile
    
    try:
        # Kaggle API 인증 - 환경 변수에서만 읽기
        kaggle_username = os.environ.get('KAGGLE_USERNAME')
        kaggle_key = os.environ.get('KAGGLE_KEY')
        
        # 환경 변수 확인
        if not kaggle_username or not kaggle_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="KAGGLE_USERNAME 또는 KAGGLE_KEY 환경 변수가 설정되지 않았습니다. .env 파일에 설정해주세요."
            )
        
        print(f"🔐 Kaggle API 인증 시도...")
        print(f"   KAGGLE_USERNAME: {kaggle_username}")
        print(f"   KAGGLE_KEY: {'*' * len(kaggle_key)}")
        
        # 환경 변수 설정 (확실히 설정)
        os.environ['KAGGLE_USERNAME'] = kaggle_username
        os.environ['KAGGLE_KEY'] = kaggle_key
        
        # .kaggle 폴더에 파일 생성 (Kaggle API가 파일도 확인하므로)
        import pathlib
        home_dir = pathlib.Path.home()
        kaggle_home_dir = home_dir / '.kaggle'
        kaggle_home_json = kaggle_home_dir / 'kaggle.json'
        
        # .kaggle 폴더가 없으면 생성
        if not kaggle_home_dir.exists():
            kaggle_home_dir.mkdir(parents=True, exist_ok=True)
            print(f"   📁 .kaggle 폴더 생성: {kaggle_home_dir}")
        
        # kaggle.json 파일 생성 (환경 변수에서 읽은 값으로)
        kaggle_creds = {
            'username': kaggle_username,
            'key': kaggle_key
        }
        with open(str(kaggle_home_json), 'w') as f:
            json.dump(kaggle_creds, f)
        print(f"   📋 kaggle.json 파일 생성: {kaggle_home_json}")
        
        # 파일 권한 설정
        try:
            os.chmod(str(kaggle_home_json), 0o600)
        except:
            pass
        
        # Kaggle API 초기화 및 인증
        api = KaggleApi()
        
        try:
            api.authenticate()
            print(f"✅ Kaggle API 인증 성공!")
        except Exception as auth_error:
            # 인증 실패 시 상세 정보 출력
            error_msg = str(auth_error)
            print(f"❌ Kaggle API 인증 실패:")
            print(f"   에러 메시지: {error_msg}")
            print(f"   KAGGLE_USERNAME: {kaggle_username if kaggle_username else '설정 안됨'}")
            print(f"   KAGGLE_KEY: {'설정됨' if kaggle_key else '설정 안됨'}")
            print(f"   kaggle.json 파일: {kaggle_home_json}")
            print(f"   파일 존재: {os.path.exists(str(kaggle_home_json))}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Kaggle API 인증 실패: {error_msg}. 환경 변수 KAGGLE_USERNAME과 KAGGLE_KEY를 확인해주세요."
            )
        
        # 전역 캐시 확인 및 데이터 로드
        global df_2025_cache, cache_metadata
        
        if df_2025_cache is None:
            # 캐시가 없으면 데이터 다운로드 및 처리
            print(f"📥 데이터 다운로드 및 2025년 데이터 필터링 시작...")
            
            dataset = "asaniczka/trending-youtube-videos-113-countries"
            print(f"📥 Kaggle 데이터셋 다운로드 시작: {dataset}")
            
            import tempfile
            import shutil
            
            # 임시 디렉토리 생성
            temp_dir = tempfile.mkdtemp()
            try:
                # 임시 디렉토리에 다운로드
                api.dataset_download_files(dataset, path=temp_dir, unzip=True)
                
                # CSV 파일 찾기
                csv_file = None
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.csv') and 'trending' in file.lower():
                            csv_file = os.path.join(root, file)
                            break
                    if csv_file:
                        break
                
                if not csv_file or not os.path.exists(csv_file):
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="CSV 파일을 찾을 수 없습니다."
                    )
                
                # CSV 파일 읽기 (메모리로)
                print(f"📖 CSV 파일 읽기: {csv_file}")
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                    print(f"   ✅ CSV 파일 읽기 성공: {len(df)}행")
                except Exception as csv_error:
                    print(f"   ❌ CSV 파일 읽기 실패: {csv_error}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"CSV 파일 읽기 실패: {str(csv_error)}"
                    )
                
            finally:
                # 임시 디렉토리 삭제
                try:
                    shutil.rmtree(temp_dir)
                    print(f"🗑️ 임시 디렉토리 삭제 완료: {temp_dir}")
                except Exception as e:
                    print(f"⚠️ 임시 디렉토리 삭제 실패: {e}")
            
            # 데이터프레임 유효성 검사
            if df is None or df.empty:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="데이터프레임이 비어있거나 None입니다."
                )
            
            # 데이터프레임 정보 출력 (디버깅)
            print(f"📊 데이터프레임 정보:")
            print(f"   행 수: {len(df)}")
            print(f"   컬럼: {list(df.columns)}")
            print(f"   컬럼 타입: {type(df.columns)}")
            
            # 1. country 컬럼에서 KR, kr인 것만 필터링
            if 'country' in df.columns:
                df = df[df['country'].str.upper().isin(['KR', 'KOREA'])]
                print(f"   ✅ KR 데이터 필터링 후 행 수: {len(df)}")
            else:
                print(f"   ⚠️ country 컬럼이 없습니다. 모든 국가 데이터 사용")
            
            # 2. published_at 또는 다른 날짜 컬럼 확인
            date_column = None
            try:
                for col in ['published_at', 'publish_date', 'publishedAt', 'publishDate', 'published_date', 'trending_date', 'snapshot_date']:
                    if col in df.columns:
                        date_column = col
                        break
            except Exception as col_error:
                print(f"   ❌ 컬럼 확인 중 오류: {col_error}")
                print(f"   df 타입: {type(df)}")
                print(f"   df.columns 타입: {type(df.columns)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"컬럼 확인 중 오류 발생: {str(col_error)}"
                )
            
            if not date_column:
                available_cols = ', '.join(df.columns.tolist())
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"날짜 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {available_cols}"
                )
            
            print(f"   ✅ 날짜 컬럼 발견: {date_column}")
            
            # 날짜 컬럼을 datetime으로 변환
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df = df.dropna(subset=[date_column])
            
            # 2025년 데이터만 필터링
            df_2025 = df[df[date_column].dt.year == 2025]
            print(f"   ✅ 2025년 데이터 행 수: {len(df_2025)}")
            
            # 3. video_id, title, published_date, views만 남기고 나머지 컬럼 삭제
            video_id_column = None
            for col in ['video_id', 'videoId', 'id']:
                if col in df_2025.columns:
                    video_id_column = col
                    break
            
            if not video_id_column:
                available_cols = ', '.join(df_2025.columns.tolist())
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"video_id 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {available_cols}"
                )
            
            title_column = None
            for col in ['title', 'Title', 'video_title']:
                if col in df_2025.columns:
                    title_column = col
                    break
            
            if not title_column:
                available_cols = ', '.join(df_2025.columns.tolist())
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"title 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {available_cols}"
                )
            
            views_column = None
            for col in ['views', 'view_count', 'viewCount', 'view_count_total']:
                if col in df_2025.columns:
                    views_column = col
                    break
            
            if not views_column:
                available_cols = ', '.join(df_2025.columns.tolist())
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"views 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {available_cols}"
                )
            
            # 필요한 컬럼만 선택 (video_id, title, published_date, views)
            df_2025 = df_2025[[video_id_column, title_column, date_column, views_column]].copy()
            # 컬럼명을 표준화
            df_2025.columns = ['video_id', 'title', 'published_date', 'views']
            print(f"   ✅ 컬럼 정리 완료: video_id, title, published_date, views (원래 컬럼: {video_id_column}, {title_column}, {date_column}, {views_column})")
            
            # 4. YouTube API를 사용해서 category 정보 가져오기
            # 환경 변수에서 가져오거나 기본값 사용
            youtube_api_key = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyC8lNQlD0nYRlophLuezpx1ihSbzQvGLv8')
            
            if not YOUTUBE_API_AVAILABLE:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="googleapiclient 패키지가 설치되지 않았습니다."
                )
            
            print(f"   📺 YouTube API로 카테고리 정보 가져오기 시작...")
            print(f"   🔑 YouTube API 키 사용: {youtube_api_key[:10]}...")
            youtube = build('youtube', 'v3', developerKey=youtube_api_key)
            
            # video_id 리스트를 50개씩 나누어 처리 (YouTube API 제한)
            video_ids = df_2025['video_id'].unique().tolist()
            categories = {}  # category만 가져오기
            
            def chunks(lst, n):
                """리스트를 n개씩 나누기"""
                for i in range(0, len(lst), n):
                    yield lst[i:i + n]
            
            import time
            for i, chunk in enumerate(chunks(video_ids, 50)):
                try:
                    videos_response = youtube.videos().list(
                        part='snippet',  # snippet만 가져오기 (category만 필요)
                        id=','.join(chunk)
                    ).execute()
                    
                    for item in videos_response.get('items', []):
                        video_id = item['id']
                        category_id = item['snippet'].get('categoryId', '')
                        categories[video_id] = category_id
                    
                    # API 호출 제한 대비 (초당 1회)
                    if i < len(list(chunks(video_ids, 50))) - 1:
                        time.sleep(1)
                    
                    if (i + 1) % 10 == 0:
                        print(f"      진행 중: {min((i + 1) * 50, len(video_ids))}/{len(video_ids)}")
                        
                except Exception as e:
                    print(f"      ⚠️ YouTube API 호출 오류 (chunk {i+1}): {e}")
                    continue
            
            # category 컬럼만 추가 (views는 CSV에서 가져온 것 사용)
            df_2025['category'] = df_2025['video_id'].map(categories)
            df_2025 = df_2025.dropna(subset=['category'])  # category가 없는 행 제거
            print(f"   ✅ 카테고리 정보 추가 완료: {len(df_2025)}개 영상")
            
            # 컬럼명 정리
            category_column = 'category'
            views_column = 'views'
            video_id_column = 'video_id'
            date_column = 'published_date'  # 컬럼명 표준화됨
            
            # 최종 데이터프레임 확인
            print(f"   📊 최종 데이터프레임 컬럼: {list(df_2025.columns)}")
            print(f"   📊 최종 데이터프레임 행 수: {len(df_2025)}")
            
            # 캐시에 저장
            df_2025_cache = df_2025.copy()
            cache_metadata = {
                'date_column': date_column,
                'category_column': category_column,
                'views_column': views_column,
                'video_id_column': video_id_column
            }
            print(f"   💾 2025년 데이터 캐시 저장 완료")
        else:
            print(f"   ♻️ 캐시된 2025년 데이터 사용")
        
        # 캐시에서 데이터 및 메타데이터 가져오기
        df_2025 = df_2025_cache.copy()
        date_column = cache_metadata['date_column']
        category_column = cache_metadata['category_column']
        views_column = cache_metadata['views_column']
        video_id_column = cache_metadata['video_id_column']
        
        print(f"   📊 캐시에서 가져온 데이터: {len(df_2025)}행, 컬럼: {list(df_2025.columns)}")
        
        # 특정 월의 트렌드 분석
        print(f"📅 {month}월 데이터 분석 시작...")
        
        # 월별 데이터 필터링 (published_date 사용)
        df_month = df_2025[df_2025[date_column].dt.month == month].copy()
        
        if len(df_month) == 0:
            print(f"   ⚠️ {month}월 데이터가 없습니다.")
            return UTF8JSONResponse(content={
                "success": True,
                "message": f"{month}월 데이터가 없습니다.",
                "data": {
                    "month": month,
                    "trends": {
                        f'{month}월': {
                            'top5': [],
                            'total_videos': 0,
                            'top30_videos': 0
                        }
                    },
                    "updated_at": datetime.now().isoformat()
                }
            })
        
        print(f"   📊 {month}월 데이터 행 수: {len(df_month)}")
        
        # 중복 제거 (video_id 기준, 조회수 높은 것만 유지)
        df_month = df_month.sort_values(by=views_column, ascending=False) \
                          .drop_duplicates(subset=[video_id_column], keep='first')
        
        # 조회수 기준 내림차순 정렬
        df_sorted = df_month.sort_values(by=views_column, ascending=False)
        
        # 상위 30% 데이터만 추출
        top_30_percent = int(len(df_sorted) * 0.3)
        df_top30 = df_sorted.head(top_30_percent)
        
        print(f"   📈 상위 30% 영상 수: {len(df_top30)}")
        
        # 카테고리별 영상 수 집계
        category_counts = df_top30[category_column].value_counts().head(5)
        
        # TOP 5 카테고리 추출
        top5_categories = []
        for idx, (cat_id, count) in enumerate(category_counts.items(), 1):
            top5_categories.append({
                'rank': idx,
                'category_id': str(int(cat_id)) if pd.notna(cat_id) else None,
                'count': int(count)
            })
        
        print(f"   ✅ {month}월 TOP 5 카테고리: {[c['category_id'] for c in top5_categories]}")
        
        trend_results = {
            f'{month}월': {
                'top5': top5_categories,
                'total_videos': len(df_month),
                'top30_videos': len(df_top30)
            }
        }
        
        # 결과 반환
        return UTF8JSONResponse(content={
            "success": True,
            "message": f"{month}월 트렌드 분석이 완료되었습니다.",
            "data": {
                "month": month,
                "trends": trend_results,
                "updated_at": datetime.now().isoformat()
            }
        })
        
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="kaggle 패키지가 설치되지 않았습니다. 'pip install kaggle'을 실행하세요."
        )
    except Exception as e:
        import traceback
        error_detail = f"트렌드 분석 중 오류가 발생했습니다: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_detail}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("1등 유튜버 되기 FastAPI 서버 시작")
    print("=" * 50)
    
    # 데이터베이스 파일 존재 확인
    if not os.path.exists('youtube_analytics.db'):
        print("❌ 데이터베이스 파일이 없습니다.")
        print("먼저 'python init_database.py'를 실행하여 데이터베이스를 초기화하세요.")
        exit(1)
    
    print("✅ 데이터베이스 연결 확인")
    print("🚀 FastAPI 서버 시작 중...")
    
    import os
    port = int(os.environ.get("PORT", 8001))  # 기본값을 8001로 변경
    
    print(f"\n서버 주소: http://localhost:{port}")
    print(f"API 문서: http://localhost:{port}/docs")
    print(f"ReDoc 문서: http://localhost:{port}/redoc")
    print("\n종료하려면 Ctrl+C를 누르세요.")
    print("=" * 50)
    
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=port,
        reload=False  # 프로덕션에서는 reload=False
    )
