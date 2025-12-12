from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import os

# 🚨 KoELECTRA 기반 모델로 변경 🚨
# 'monologg/koelectra-base-v3-discriminator' 기반으로 훈련된 SBERT 모델을 사용합니다.
# 이는 KoELECTRA의 강력한 성능을 문장 임베딩에 활용한 모델입니다.
EMBEDDING_MODEL_NAME = 'snunlp/KR-SBERT-V40K-klueNLI-AS' # 기존 모델
# 다른 강력한 한국어 임베딩 모델 예시: 'BM-K/KoSimCSE-RoBERTa-base'

# 1. 모델 및 파일 설정
CSV_FILE = 'final_rag_data_combined_raw.csv'
COLLECTION_NAME = 'pet_veterinary_knowledge_electra'

# 2. 임베딩 모델 로드
# KoELECTRA 기반 SBERT 모델 로드
print(f"✅ 임베딩 모델 로드: {EMBEDDING_MODEL_NAME}")
try:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME) 
except Exception as e:
    print(f"❌ 오류: 모델 로드 실패. {EMBEDDING_MODEL_NAME} 모델을 Hugging Face에서 찾을 수 없습니다.")
    print("KoELECTRA 기반 모델 중 SentenceTransformer 호환 모델을 확인해 주세요.")
    exit()

# 3. 데이터 로드
try:
    df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
except FileNotFoundError:
    print(f"❌ 오류: '{CSV_FILE}' 파일을 찾을 수 없습니다. 전처리 파일을 확인하세요.")
    exit()

# 4. Chroma DB 초기화 및 컬렉션 생성
# 데이터 영구 저장 경로 설정
DB_PATH = "./chroma_db_electra"
client = chromadb.PersistentClient(path=DB_PATH) 
collection = client.get_or_create_collection(COLLECTION_NAME)

# 기존 데이터가 있으면 삭제하고 새로 시작 (선택 사항)
# collection.delete(ids=collection.get()['ids']) 

# 5. 데이터 임베딩 및 저장 (인덱싱)
chunks = df['RAG_Chunk'].tolist()
# 메타데이터 추출 (필터링에 사용할 컬럼)
metadata_list = df[['disease', 'department_meta', 'lifeCycle']].to_dict('records')
ids_list = [f"doc_{i}" for i in range(len(chunks))]

print(f"총 {len(chunks)}개의 청크를 임베딩하고 DB에 저장합니다...")

# 데이터가 많을 경우 배치(Batch) 처리 권장
# 32개 단위로 배치 처리 예시
batch_size = 32
for i in range(0, len(chunks), batch_size):
    batch_chunks = chunks[i:i + batch_size]
    batch_metadata = metadata_list[i:i + batch_size]
    batch_ids = ids_list[i:i + batch_size]
    
    # 임베딩 생성
    batch_vectors = model.encode(batch_chunks, convert_to_numpy=False) # list of lists로 변환
    
    # DB에 저장
    collection.add(
        embeddings=batch_vectors.tolist(),
        documents=batch_chunks,
        metadatas=batch_metadata,
        ids=batch_ids
    )

print("✅ 지식 기반(벡터 DB) 구축 완료.")
print(f"저장된 컬렉션 이름: {COLLECTION_NAME}, DB 경로: {DB_PATH}")