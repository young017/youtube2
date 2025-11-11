import pandas as pd
import numpy as np
import pickle
import json
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import os

class TagRecommendationModel:
    def __init__(self, model_name='sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'):
        """
        한국어 제목-태그 연관성 학습 모델
        
        Args:
            model_name: SBERT 모델명 (한국어 지원)
        """
        self.model = SentenceTransformer(model_name)
        self.title_embeddings = None
        self.tag_embeddings = None
        self.title_to_tags = {}
        self.all_tags = set()
        self.tag_to_embedding = {}
        
    def load_data(self, csv_path):
        """CSV 데이터 로드 및 전처리"""
        print("데이터 로딩 중...")
        df = pd.read_csv(csv_path)
        
        # 태그 문자열을 리스트로 변환
        df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # 제목-태그 매핑 생성
        for idx, row in df.iterrows():
            title = row['title']
            tags = row['tags']
            
            # 제목 정리 (특수문자 제거, 공백 정리)
            clean_title = re.sub(r'[^\w\s가-힣]', ' ', title).strip()
            clean_title = re.sub(r'\s+', ' ', clean_title)
            
            self.title_to_tags[clean_title] = tags
            self.all_tags.update(tags)
        
        print(f"총 {len(df)}개의 제목-태그 쌍 로드 완료")
        print(f"고유 태그 수: {len(self.all_tags)}")
        
        return df
    
    def prepare_embeddings(self, df):
        """제목과 태그의 임베딩 생성"""
        print("임베딩 생성 중...")
        
        # 제목 임베딩 생성
        titles = list(self.title_to_tags.keys())
        self.title_embeddings = self.model.encode(titles, show_progress_bar=True)
        
        # 태그별 임베딩 생성
        unique_tags = list(self.all_tags)
        self.tag_embeddings = self.model.encode(unique_tags, show_progress_bar=True)
        
        # 태그-임베딩 매핑
        for tag, embedding in zip(unique_tags, self.tag_embeddings):
            self.tag_to_embedding[tag] = embedding
        
        print("임베딩 생성 완료")
    
    def find_similar_titles(self, query_title, top_k=5):
        """유사한 제목 찾기"""
        if self.title_embeddings is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 실행하세요.")
        
        # 쿼리 제목 정리
        clean_query = re.sub(r'[^\w\s가-힣]', ' ', query_title).strip()
        clean_query = re.sub(r'\s+', ' ', clean_query)
        
        # 쿼리 임베딩 생성
        query_embedding = self.model.encode([clean_query])
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_embedding, self.title_embeddings)[0]
        
        # 상위 k개 인덱스 찾기
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        titles = list(self.title_to_tags.keys())
        for idx in top_indices:
            results.append({
                'title': titles[idx],
                'similarity': similarities[idx],
                'tags': self.title_to_tags[titles[idx]]
            })
        
        return results
    
    def recommend_tags(self, query_title, top_k=10, min_similarity=0.3):
        """제목 기반 태그 추천"""
        similar_titles = self.find_similar_titles(query_title, top_k=20)
        
        # 유사도가 임계값 이상인 제목들만 필터링
        filtered_titles = [item for item in similar_titles if item['similarity'] >= min_similarity]
        
        if not filtered_titles:
            return []
        
        # 태그 빈도 계산 (유사도 가중치 적용)
        tag_scores = Counter()
        for item in filtered_titles:
            similarity = item['similarity']
            for tag in item['tags']:
                tag_scores[tag] += similarity
        
        # 상위 태그 반환
        recommended_tags = [tag for tag, score in tag_scores.most_common(top_k)]
        
        return recommended_tags
    
    def recommend_tags_with_sbert(self, query_title, top_k=10):
        """SBERT 기반 직접 태그 추천"""
        if self.tag_embeddings is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 실행하세요.")
        
        # 쿼리 제목 정리
        clean_query = re.sub(r'[^\w\s가-힣]', ' ', query_title).strip()
        clean_query = re.sub(r'\s+', ' ', clean_query)
        
        # 쿼리 임베딩 생성
        query_embedding = self.model.encode([clean_query])
        
        # 모든 태그와의 유사도 계산
        similarities = cosine_similarity(query_embedding, self.tag_embeddings)[0]
        
        # 상위 k개 태그 선택
        top_indices = np.argsort(similarities)[::-1][:top_k]
        unique_tags = list(self.all_tags)
        
        recommended_tags = []
        for idx in top_indices:
            recommended_tags.append({
                'tag': unique_tags[idx],
                'similarity': similarities[idx]
            })
        
        return recommended_tags
    
    def fit(self, csv_path):
        """모델 학습"""
        df = self.load_data(csv_path)
        self.prepare_embeddings(df)
        print("모델 학습 완료!")
    
    def save_model(self, model_path):
        """모델 저장"""
        model_data = {
            'title_embeddings': self.title_embeddings,
            'tag_embeddings': self.tag_embeddings,
            'title_to_tags': self.title_to_tags,
            'all_tags': list(self.all_tags),
            'tag_to_embedding': {tag: emb.tolist() for tag, emb in self.tag_to_embedding.items()}
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"모델이 {model_path}에 저장되었습니다.")
    
    def load_model(self, model_path):
        """모델 로드"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.title_embeddings = model_data['title_embeddings']
        self.tag_embeddings = model_data['tag_embeddings']
        self.title_to_tags = model_data['title_to_tags']
        self.all_tags = set(model_data['all_tags'])
        
        # 태그 임베딩 복원
        self.tag_to_embedding = {}
        for tag, emb_list in model_data['tag_to_embedding'].items():
            self.tag_to_embedding[tag] = np.array(emb_list)
        
        print(f"모델이 {model_path}에서 로드되었습니다.")

def main():
    """모델 학습 및 저장"""
    # 모델 초기화
    model = TagRecommendationModel()
    
    # 데이터 경로
    csv_path = "제목,태그/korean_youtube_tags.csv"
    model_path = "tag_recommendation_model.pkl"
    
    # 모델 학습
    model.fit(csv_path)
    
    # 모델 저장
    model.save_model(model_path)
    
    # 테스트
    test_title = "10월의 교토 브이로그 ❀"
    print(f"\n테스트 제목: {test_title}")
    
    # 유사한 제목 찾기
    similar_titles = model.find_similar_titles(test_title, top_k=3)
    print("\n유사한 제목들:")
    for item in similar_titles:
        print(f"- {item['title']} (유사도: {item['similarity']:.3f})")
        print(f"  태그: {item['tags'][:5]}...")
    
    # 태그 추천
    recommended_tags = model.recommend_tags(test_title, top_k=10)
    print(f"\n추천 태그: {recommended_tags}")
    
    # SBERT 직접 추천
    sbert_tags = model.recommend_tags_with_sbert(test_title, top_k=10)
    print(f"\nSBERT 직접 추천 태그:")
    for item in sbert_tags:
        print(f"- {item['tag']} (유사도: {item['similarity']:.3f})")

if __name__ == "__main__":
    main()
