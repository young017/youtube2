# predict_tags.py
# 사용 전: pip install sentence-transformers scikit-learn pandas numpy

import argparse
from typing import List, Dict
from tag_recommendation_model import TagRecommendationModel

def print_similar(similar_list: List[Dict], top_k: int):
    print(f"\n[유사 제목 상위 {top_k}]")
    for item in similar_list[:top_k]:
        t = item['title']
        s = item['similarity']
        tags = item['tags'][:5] if isinstance(item['tags'], list) else item['tags']
        print(f" - {t} (유사도 {s:.3f}) | 태그 예시: {tags}")

def main():
    parser = argparse.ArgumentParser(description="저장된 모델로 제목 → 태그 추론")
    parser.add_argument("title", type=str, help="테스트할 영상 제목")
    parser.add_argument("--model_path", type=str, default="tag_recommendation_model.pkl")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--min_similarity", type=float, default=0.30,
                        help="제목 기반 빈도 추천에서 유사 제목 필터 임계값")
    parser.add_argument("--model_name", type=str,
                        default="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
                        help="훈련 때 사용한 SBERT 모델명과 동일하게 맞추세요")
    args = parser.parse_args()

    # 1) 모델 인스턴스만 만들고 (학습 X), 2) 저장된 pkl 로드
    mdl = TagRecommendationModel(model_name=args.model_name)
    mdl.load_model(args.model_path)

    # A. 제목-제목 유사도 기반(가중 빈도) 추천
    rec1 = mdl.recommend_tags(args.title, top_k=args.top_k, min_similarity=args.min_similarity)

    # B. 제목-태그 직접 유사도 기반 추천
    rec2 = mdl.recommend_tags_with_sbert(args.title, top_k=args.top_k)

    # (선택) 유사 제목 몇 개 출력
    similar = mdl.find_similar_titles(args.title, top_k=5)
    print_similar(similar, top_k=5)

    print(f"\n[추천 태그 - 제목 유사도 가중 빈도 기반] (top_k={args.top_k}, min_sim={args.min_similarity})")
    print(", ".join(rec1) if rec1 else "(추천 없음)")

    print(f"\n[추천 태그 - SBERT 직접 유사도 기반] (top_k={args.top_k})")
    if rec2:
        for item in rec2:
            print(f" - {item['tag']} (sim={item['similarity']:.3f})")
    else:
        print("(추천 없음)")

    # LLM 프롬프트 (원하시면 이걸 OpenAI API에 넘기면 됩니다)
    candidate_tags = rec1 if rec1 else [x["tag"] for x in rec2]
    prompt = f"""
아래는 유튜브 영상 제목과 SBERT가 유사도 기반으로 추천한 태그 후보입니다.
제목: {args.title}
후보 태그: {', '.join(candidate_tags)}

위 제목의 문맥과 의미에 어울리도록 태그를 자연스럽게 수정하거나 보완해줘.
불필요하거나 제목과 관련 없는 건 제거하고, 관련 있는 표현은 새로 추가해도 좋아.
최종 결과는 쉼표로 구분된 형태로 작성해줘.
""".strip()
    print("\n--- LLM 프롬프트 ---")
    print(prompt)

if __name__ == "__main__":
    main()
