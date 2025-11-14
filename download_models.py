"""
Hugging Face에서 모델 파일을 다운로드하는 스크립트
Railway 배포 시 모델 파일이 없을 경우 Hugging Face에서 자동으로 다운로드합니다.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# Hugging Face 저장소 정보 (환경 변수로 설정 가능)
HF_REPO_ID = os.environ.get("HF_REPO_ID", "yudaag/youtube-view-predict-models")  # 기본값 설정
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # Private repo인 경우 토큰 필요

# 다운로드할 모델 파일 목록
MODEL_FILES = [
    # CatBoost 모델
    "catboost_model_1_class.cbm",
    "catboost_model_1.cbm",
    "catboost_model_15_class.cbm",
    "catboost_model_15.cbm",
    "catboost_model_19_class.cbm",
    "catboost_model_19.cbm",
    # LightGBM 모델
    "lgbm_model_10_class.pkl",
    "lgbm_model_10.pkl",
    "lgbm_model_22_class.pkl",
    "lgbm_model_22.pkl",
    "lgbm_model_24_class.pkl",
    "lgbm_model_24.pkl",
    "lgbm_model_26_class.pkl",
    "lgbm_model_26.pkl",
    # XGBoost 모델
    "xgb_model_17_class.pkl",
    "xgb_model_17.pkl",
    "xgb_cat_17_class.pkl",  # 추가 파일
    "xgb_model_20_class.pkl",
    "xgb_model_20.pkl",
    "xgb_model_23_class.pkl",
    "xgb_model_23.pkl",
    "xgb_model_28_class.pkl",
    "xgb_model_28.pkl",
    # 태그 추천 모델
    "tag_recommendation_model.pkl",
]

def download_models():
    """모델 파일을 Hugging Face에서 다운로드"""
    if not HF_REPO_ID:
        print("⚠️ HF_REPO_ID 환경 변수가 설정되지 않았습니다.")
        print("💡 Railway 환경 변수에 HF_REPO_ID를 설정해주세요. (예: username/repo-name)")
        return False
    
    # 모델 디렉토리 경로 설정
    script_dir = Path(__file__).parent
    model_dir = script_dir / "모델"
    tags_dir = script_dir / "tags"
    
    # 모델 디렉토리 생성
    model_dir.mkdir(exist_ok=True)
    tags_dir.mkdir(exist_ok=True)
    
    print(f"📥 Hugging Face에서 모델 다운로드 시작...")
    print(f"   저장소: {HF_REPO_ID}")
    print(f"   모델 디렉토리: {model_dir}")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for filename in MODEL_FILES:
        try:
            # 태그 추천 모델은 tags 디렉토리에, 나머지는 모델 디렉토리에
            if filename == "tag_recommendation_model.pkl":
                save_path = tags_dir / filename
            else:
                save_path = model_dir / filename
            
            # 이미 파일이 있으면 스킵
            if save_path.exists():
                print(f"   ⏭️  {filename} 이미 존재, 스킵")
                skip_count += 1
                continue
            
            print(f"   📥 {filename} 다운로드 중...")
            
            # Hugging Face에서 다운로드
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                token=HF_TOKEN,
                local_dir=str(save_path.parent),
                local_dir_use_symlinks=False,
            )
            
            # 다운로드된 파일 경로 확인 및 이동
            downloaded_file = Path(downloaded_path)
            if downloaded_file.exists():
                # 파일이 올바른 위치에 없으면 이동
                if downloaded_file != save_path:
                    import shutil
                    if save_path.exists():
                        save_path.unlink()  # 기존 파일 삭제
                    shutil.move(str(downloaded_file), str(save_path))
                    print(f"   📁 파일 이동: {downloaded_file.name} -> {save_path}")
                else:
                    print(f"   ✅ 파일이 올바른 위치에 있습니다: {save_path}")
            else:
                raise FileNotFoundError(f"다운로드된 파일을 찾을 수 없습니다: {downloaded_path}")
            
            print(f"   ✅ {filename} 다운로드 완료")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ {filename} 다운로드 실패: {e}")
            error_count += 1
    
    print(f"\n📊 다운로드 결과:")
    print(f"   ✅ 성공: {success_count}")
    print(f"   ⏭️  스킵: {skip_count}")
    print(f"   ❌ 실패: {error_count}")
    
    return error_count == 0

if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)

