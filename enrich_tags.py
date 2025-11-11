# enrich_tags.py
# pip install openai numpy scikit-learn sentence-transformers

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
from openai import OpenAI

# 당신의 후보 추천 모델 클래스를 import (파일명에 맞게 바꾸세요)
from tags.tag_recommendation_model import TagRecommendationModel


# ========= 1) 후보 태그 뽑기 (당신의 모델 재사용) =========
def get_candidate_tags(
    mdl: TagRecommendationModel,
    title: str,
    top_k: int = 15,
    min_similarity_title: float = 0.30
) -> List[str]:
    """
    A. 제목-제목 유사도 가중 빈도 기반 추천
    B. 제목-태그 직접 유사도 기반 추천
    → 합쳐서 중복 제거 후 상위 top_k 반환
    """
    rec1 = mdl.recommend_tags(title, top_k=top_k, min_similarity=min_similarity_title)  # List[str]
    rec2_items = mdl.recommend_tags_with_sbert(title, top_k=top_k)  # List[{"tag","similarity"}]
    rec2 = [x["tag"] for x in rec2_items]

    seen, merged = set(), []
    for t in rec1 + rec2:
        if t not in seen:
            merged.append(t); seen.add(t)
    return merged[:top_k]


# ========= 2) OpenAI 임베딩으로 제목–태그 유사도 계산 =========
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def compute_title_tag_similarities_openai(
    title: str,
    tags: List[str],
    api_key: str,
    embed_model: str = "text-embedding-3-large"
) -> List[Tuple[str, float]]:
    """
    OpenAI 임베딩으로 제목과 각 태그의 의미 유사도 계산
    반환: [(tag, similarity)] 내림차순
    """
    if not tags:
        return []
    client = OpenAI(api_key=api_key)
    res = client.embeddings.create(model=embed_model, input=[title] + tags)
    vecs = [np.array(d.embedding, dtype=np.float32) for d in res.data]
    title_vec, tag_vecs = vecs[0], vecs[1:]
    sims = [(t, _cosine(title_vec, v)) for t, v in zip(tags, tag_vecs)]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims


# ========= 3) 유사도 필터(임계값 0.28) =========
def filter_by_threshold(
    tag_sims: List[Tuple[str, float]],
    abs_threshold: float = 0.30,   # 기본값도 0.30으로
) -> Tuple[List[str], List[str]]:
    """
    엄격 모드: 유사도 ≤ abs_threshold 는 모두 제거.
    최소 개수 보장(keep_min) 없음.
    """
    if not tag_sims:
        return [], []
    # 내림차순 정렬은 보기 편하도록 유지
    sorted_pairs = sorted(tag_sims, key=lambda x: x[1], reverse=True)
    # "이하면 삭제" → keep 은 "초과"만
    keep = [t for t, s in sorted_pairs if s > abs_threshold]
    dropped = [t for t, s in sorted_pairs if s <= abs_threshold]
    return keep, dropped


# ========= 4) GPT로 '필요한 것만' 보정 + 추가 =========
def build_openai_prompt(title: str, kept: List[str], dropped: List[str], extra_k: int) -> str:
    return f"""
아래는 유튜브 영상 제목과 모델이 추천해 유지된 태그 목록입니다.
제목: {title}
유지된 태그: {', '.join(kept) if kept else '(없음)'}
(참고) 제거된 태그: {', '.join(dropped) if dropped else '(없음)'}

요청사항:
1) 기존 태그는 가능한 한 그대로 유지하되, 제목과 맞지 않는 것만 최소한으로 제거.
2) 필요한 경우에만 자연스럽게 일부 표현만 수정(동의어/표기 통일).
3) 제목과 직접적으로 관련 있지만 빠진 내용이 있다면 추가 태그 {extra_k}개를 제안(겹치지 않게).
4) 아래 JSON 형식으로만 출력(설명/문장 금지).

{{
  "final_tags": ["태그1", "태그2", "..."],
  "extra_tags": ["추가태그1", "추가태그2", "..."]
}}
""".strip()

def refine_with_openai_json(
    title: str,
    kept_tags: List[str],
    dropped_tags: List[str],
    extra_k: int = 10,
    chat_model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    api_key: str | None = None,
    timeout: int = 60
) -> Dict[str, List[str]]:
    """
    Chat Completions로 최종 태그 JSON 반환
    """
    # 키 확보
    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수 또는 --api_key 인자를 설정해주세요.")
    client = OpenAI(api_key=api_key)

    prompt = build_openai_prompt(title, kept_tags, dropped_tags, extra_k)
    resp = client.chat.completions.create(
        model=chat_model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "너는 유튜브 태그를 간결하고 정확하게 보정하는 도우미야."},
            {"role": "user", "content": prompt},
        ],
        timeout=timeout,
    )
    text = resp.choices[0].message.content.strip()

    # JSON 파싱(방어)
    try:
        data = json.loads(text)
        final_tags = [t.strip() for t in data.get("final_tags", []) if isinstance(t, str) and t.strip()]
        extra_tags = [t.strip() for t in data.get("extra_tags", []) if isinstance(t, str) and t.strip()]
        return {"final_tags": final_tags, "extra_tags": extra_tags}
    except Exception:
        # 백업: JSON이 아니면 유지 태그 그대로 반환
        return {"final_tags": kept_tags, "extra_tags": []}


# ========= 5) 파이프라인 실행 =========
def run_pipeline(
    model_path: str,
    title: str,
    description: str = "",
    sbert_name: str = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
    top_k: int = 15,
    title_sim_threshold: float = 0.30,   # 후보 생성 단계에만 사용
    tag_abs_threshold: float = 0.30,     # 임베딩 유사도 필터
    keep_min: int = 8,
    extra_k: int = 10,
    api_key: str | None = None,
    embed_model: str = "text-embedding-3-large",
    chat_model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    1) pkl 모델 로드 → 2) 후보 태그 생성(당신의 모델) →
    3) OpenAI 임베딩으로 제목+설명–태그 유사도 계산 →
    4) 유사도 0.28 이하 제거(+ 최소 유지 개수 보장) →
    5) GPT로 '필요한 것만' 보정 + 추가
    """
    # 제목과 설명을 결합하여 더 풍부한 컨텍스트 생성
    combined_text = title
    if description and description.strip():
        combined_text = f"{title} {description.strip()}"
    
    # 1) 후보 생성 (제목만 사용)
    mdl = TagRecommendationModel(model_name=sbert_name)
    mdl.load_model(model_path)
    candidates = get_candidate_tags(mdl, title, top_k=top_k, min_similarity_title=title_sim_threshold)

    # 2) 임베딩 유사도 계산 (제목+설명 조합 사용)
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY 환경변수 또는 --api_key 인자를 설정해주세요.")
    sims = compute_title_tag_similarities_openai(combined_text, candidates, key, embed_model=embed_model)

    # 3) 0.30 필터링
    kept, dropped = filter_by_threshold(sims, abs_threshold=tag_abs_threshold)


    # 4) GPT 보정 + 추가 (제목+설명 조합 사용)
    refined = refine_with_openai_json(
        title=combined_text,
        kept_tags=kept,
        dropped_tags=dropped,
        extra_k=extra_k,
        chat_model=chat_model,
        api_key=key
    )

    return {
        "title": title,
        "description": description,
        "candidates": candidates,
        "scored": sims,       # [(tag, score)] 내림차순
        "kept": kept,
        "dropped": dropped,
        "openai_result": refined
    }


# ========= 6) CLI =========
def main():
    p = argparse.ArgumentParser(description="모델 후보 → OpenAI 임베딩(0.28 필터) → GPT 보정/추가")
    p.add_argument("title", type=str, help="영상 제목")
    p.add_argument("--model_path", type=str, default="tag_recommendation_model.pkl")
    p.add_argument("--sbert_name", type=str,
                   default="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
    p.add_argument("--top_k", type=int, default=15)
    p.add_argument("--title_sim_threshold", type=float, default=0.30)
    p.add_argument("--tag_abs_threshold", type=float, default=0.30)  
    p.add_argument("--keep_min", type=int, default=8)
    p.add_argument("--extra_k", type=int, default=10)
    p.add_argument("--api_key", type=str, default=None, help="환경변수 대신 직접 키 전달")
    p.add_argument("--embed_model", type=str, default="text-embedding-3-large")
    p.add_argument("--chat_model", type=str, default="gpt-4o-mini")
    args = p.parse_args()

    result = run_pipeline(
        model_path=args.model_path,
        title=args.title,
        sbert_name=args.sbert_name,
        top_k=args.top_k,
        title_sim_threshold=args.title_sim_threshold,
        tag_abs_threshold=args.tag_abs_threshold,
        keep_min=args.keep_min,
        extra_k=args.extra_k,
        api_key=args.api_key,
        embed_model=args.embed_model,
        chat_model=args.chat_model
    )

    print("\n[입력 제목]")
    print(result["title"])

    print("\n[모델 후보 태그]")
    print(", ".join(result["candidates"]) if result["candidates"] else "(없음)")

    print("\n[제목↔태그 유사도(임베딩) 상위]")
    for t, s in result["scored"][:args.top_k]:
        print(f" - {t}: {s:.3f}")

    print("\n[필터 후 유지 태그]")
    print(", ".join(result["kept"]) if result["kept"] else "(없음)")

    print("\n[필터로 제외된 태그]")
    print(", ".join(result["dropped"]) if result["dropped"] else "(없음)")

    print("\n[OpenAI 결과(JSON)]")
    print(json.dumps(result["openai_result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
