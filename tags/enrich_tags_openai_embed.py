# enrich_tags_openai_embed.py
# pip install openai numpy

import os
import json
import argparse
import numpy as np
from openai import OpenAI


# ====== 유틸 ======
def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ====== OpenAI 임베딩 기반 유사도 ======
def compute_title_tag_similarities(title: str, tags: list[str], api_key: str) -> list[tuple[str, float]]:
    """text-embedding-3-large 모델로 제목과 각 태그의 의미 유사도 계산"""
    client = OpenAI(api_key=api_key)
    res = client.embeddings.create(
        model="text-embedding-3-large",
        input=[title] + tags
    )
    vecs = [d.embedding for d in res.data]
    title_vec, tag_vecs = vecs[0], vecs[1:]

    sims = [(t, cosine(title_vec, v)) for t, v in zip(tags, tag_vecs)]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims


# ====== 태그 필터링 (유사도 기준) ======
def filter_tags_by_similarity(sim_pairs: list[tuple[str, float]], threshold: float = 0.3):
    """유사도 0.3 이하 태그 자동 삭제"""
    keep = [t for t, s in sim_pairs if s > threshold]
    drop = [t for t, s in sim_pairs if s <= threshold]
    return keep, drop


# ====== OpenAI 보정 ======
def refine_with_openai(title: str, kept_tags: list[str], dropped_tags: list[str], api_key: str):
    client = OpenAI(api_key=api_key)

    prompt = f"""
제목: {title}
현재 유지 태그: {', '.join(kept_tags)}
삭제된 태그: {', '.join(dropped_tags)}

요청사항:
1) 기존 태그는 최대한 유지하되, 제목 의미와 어울리지 않는 것은 제거.
2) 문맥상 필요한 경우에만 약간 수정하거나 비슷한 태그로 교체.
3) 제목과 어울리는 새 태그 5개 제안.
4) 아래 JSON 형식만 출력.

{{
  "final_tags": ["..."],
  "extra_tags": ["..."]
}}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "너는 유튜브 태그를 자연스럽게 보정하는 도우미야."},
            {"role": "user", "content": prompt}
        ]
    )

    text = resp.choices[0].message.content.strip()
    try:
        data = json.loads(text)
        return data
    except Exception:
        return {"final_tags": kept_tags, "extra_tags": []}


# ====== 실행 파이프라인 ======
def main():
    p = argparse.ArgumentParser(description="OpenAI 임베딩 기반 제목-태그 유사도 필터링 + 보정")
    p.add_argument("title", type=str, help="유튜브 영상 제목")
    p.add_argument("--tags", type=str, required=True,
                   help="쉼표로 구분된 초기 태그 (예: '브이로그,교토,오눅,틱톡,불닭,가을')")
    p.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"))
    args = p.parse_args()

    if not args.api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수 또는 --api_key 인자를 설정하세요.")

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    print(f"입력 제목: {args.title}")
    print(f"입력 태그: {tags}")

    # 1) 유사도 계산
    sims = compute_title_tag_similarities(args.title, tags, args.api_key)
    print("\n[유사도 결과]")
    for t, s in sims:
        print(f" - {t}: {s:.3f}")

    # 2) 0.3 이하 태그 제거
    kept, dropped = filter_tags_by_similarity(sims, threshold=0.3)
    print(f"\n[유지 태그] {kept}")
    print(f"[삭제된 태그] {dropped}")

    # 3) OpenAI로 보정 및 추가
    refined = refine_with_openai(args.title, kept, dropped, args.api_key)
    print("\n[OpenAI 최종 보정 결과]")
    print(json.dumps(refined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
