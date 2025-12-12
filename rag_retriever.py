import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# --- 설정 상수 (1. data_indexer.py와 동일하게 설정해야 합니다) ---
EMBEDDING_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS" # 사용한 KoELECTRA 기반 모델
COLLECTION_NAME = 'koelectra_pet_knowledge_index'
DB_PATH = "./chroma_db_koelectra" 

# --- 필수 초기화 (2. 파일이 로드될 때 한 번 실행) ---

# 1. 임베딩 모델 로드: 질문 벡터화에 필요
try:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ 임베딩 모델 로드 완료.")
except Exception as e:
    print(f"❌ 임베딩 모델 로드 실패: {e}")
    # 모델 로드 실패 시 검색 불가하므로 프로그램 종료 또는 예외 처리 필요

# 2. 벡터 DB 및 컬렉션 연결: 검색에 필요
try:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME) # DB에서 기존 컬렉션 연결
    print("✅ Chroma DB 컬렉션 연결 완료.")
except Exception as e:
    print(f"❌ DB 연결 실패: {e}")
    # DB 연결 실패 시 검색 불가

# -------------------------------------------------------------

def retrieve_knowledge(query: str, filters: dict = None, top_k: int = 3) -> List[str]:
    """
    사용자 질문을 벡터화하고, 벡터 DB에서 관련 전문 지식을 검색합니다.
    """
    
    # 1. 질문 벡터화 (전역 변수 'model' 사용)
    query_vector = model.encode(query).tolist()
    
    # 2. 벡터 DB 검색 (전역 변수 'collection' 사용)
    results = collection.query(
        query_embeddings=[query_vector],
        where=filters, # 메타데이터 필터 적용 (예: {"department_meta": "내과"})
        n_results=top_k
    )
    
    # 3. 검색된 컨텍스트 추출 및 반환
    # results['documents']는 리스트의 리스트 형태이므로 [0]을 사용하여 첫 번째 결과(질문)의 문서를 가져옵니다.
    retrieved_contexts = results['documents'][0]
    return retrieved_contexts

# -------------------------------------------------------------

# (선택 사항: 테스트 코드)
if __name__ == "__main__":
    test_query = "우리 강아지가 눈물 자국이 많고 눈을 자주 비벼요. 안과 관련 지식을 찾아줘."
    
    # 1. 필터 없이 검색
    contexts_no_filter = retrieve_knowledge(test_query, top_k=2)
    print(f"\n--- [필터 없음] 검색 결과 ({len(contexts_no_filter)}개) ---")
    for ctx in contexts_no_filter:
        print(f"- {ctx[:50]}...")
        
    # 2. 안과 필터 적용하여 검색
    contexts_filtered = retrieve_knowledge(test_query, filters={"department_meta": "안과"}, top_k=2)
    print(f"\n--- [안과 필터 적용] 검색 결과 ({len(contexts_filtered)}개) ---")
    for ctx in contexts_filtered:
        print(f"- {ctx[:50]}...")