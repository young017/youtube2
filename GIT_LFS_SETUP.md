# Git LFS 설정 가이드

큰 모델 파일(.pkl, .cbm)을 GitHub에 올리기 위한 Git LFS 설정 방법입니다.

## 1. Git 저장소 초기화 (아직 안 했다면)

```bash
cd /Users/han-yujeong/Downloads/youtube-main
git init
```

## 2. Git LFS 초기화

```bash
git lfs install
```

## 3. Git LFS로 추적할 파일 타입 지정

```bash
# .pkl 파일을 LFS로 추적
git lfs track "*.pkl"

# .cbm 파일도 LFS로 추적 (선택사항)
git lfs track "*.cbm"
```

## 4. .gitattributes 파일 커밋

```bash
git add .gitattributes
git commit -m "Add Git LFS configuration for model files"
```

## 5. 모델 파일 추가 및 커밋

```bash
# 모든 파일 추가
git add .

# 커밋
git commit -m "Initial commit with model files via Git LFS"
```

## 6. GitHub 저장소에 푸시

```bash
# 원격 저장소 추가 (GitHub에서 생성한 저장소 URL)
git remote add origin https://github.com/your-username/your-repo.git

# 푸시
git push -u origin main
```

## 주의사항

- **Git LFS는 GitHub에서 무료로 제공하지만 용량 제한이 있습니다:**
  - 저장소당 1GB 저장 공간
  - 월 1GB 대역폭
  
- **137MB 모델 파일은 LFS로 관리 가능합니다.**

- **이미 일반 Git으로 커밋한 경우:**
  ```bash
  # Git 히스토리에서 파일 제거
  git rm --cached tags/tag_recommendation_model.pkl
  git rm --cached 모델/*.pkl
  
  # LFS로 다시 추가
  git add tags/tag_recommendation_model.pkl
  git add 모델/*.pkl
  
  git commit -m "Move model files to Git LFS"
  ```

## 대안: 모델 파일을 별도로 관리

모델 파일이 너무 크거나 여러 개라면:

1. **모델 파일은 .gitignore에 추가** (이미 설정됨)
2. **README.md에 모델 다운로드 링크 추가**
3. **Google Drive, Dropbox, 또는 별도 저장소에 모델 업로드**

