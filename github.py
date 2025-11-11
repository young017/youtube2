

# .pkl 파일을 LFS로 추적
git lfs track "tag_recommendation_model.pkl"

# Git이 생성한 .gitattributes 파일 포함해서 커밋
git add .gitattributes
git add tag_recommendation_model.pkl
git commit -m "Add model with Git LFS"
git push
