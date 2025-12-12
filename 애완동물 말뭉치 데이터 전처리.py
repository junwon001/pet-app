import json
import pandas as pd
import re
import glob
import os
from typing import List, Dict, Any

# 🚨 최상위 폴더 경로 설정 🚨
# '내과', '안과' 등의 폴더를 포함하고 있는 'train_말뭉치' 폴더의 상위 경로를 지정합니다.
DATA_ROOT = r'C:\Users\jwm02\OneDrive\바탕 화면\애완동물 관리' 
# 최종적으로 탐색할 경로는 'C:\Users\jwm02\OneDrive\바탕 화면\애완동물 관리\train_말뭉치'가 됩니다.


# --- 함수 정의 ---

def preprocess_and_combine_data_no_cleaning(root_dir: str) -> pd.DataFrame:
    """
    지정된 디렉토리 내의 모든 하위 폴더에서 JSON 파일을 찾아 통합하며, 
    BOILERPLATE 제거 없이 원본 텍스트를 사용합니다.
    """
    
    # root_dir 아래의 모든 json 파일을 찾습니다.
    all_json_files = glob.glob(os.path.join(root_dir, '**', '*.json'), recursive=True)
    
    if not all_json_files:
        print(f"경로 ({root_dir})에서 JSON 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return pd.DataFrame()

    print(f"총 {len(all_json_files)}개의 JSON 파일을 발견했습니다. 전처리를 시작합니다...")
    
    processed_list = []
    
    # 부서명 추출을 위한 정규식 패턴 (내과, 안과, 치과, 외과, 피부과 등)
    department_folder_pattern = re.compile(r'[\\/](내과|안과|치과|외과|피부과)[\\/]', re.IGNORECASE)

    for file_path in all_json_files:
        try:
            # 파일 경로에서 부서 정보 추출
            department_match = department_folder_pattern.search(file_path)
            folder_department = department_match.group(1) if department_match else 'Unknown'

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON 데이터 처리 (리스트 또는 단일 객체)
            data_list = data if isinstance(data, list) else [data]
            
            for item in data_list:
                meta = item.get('meta', {})
                qa = item.get('qa', {})
                
                question = qa.get('input', '')
                raw_answer = qa.get('output', '')
                
                # 🚨 BOILERPLATE 제거 로직 생략! 🚨
                # raw_answer를 그대로 사용합니다.
                
                # RAG 청크 생성
                rag_chunk = f"질문: {question}\n\n답변: {raw_answer}"

                # 데이터 구조화
                processed_list.append({
                    'FilePath': file_path,
                    'Folder_Department': folder_department,
                    'lifeCycle': meta.get('lifeCycle'),
                    'department_meta': meta.get('department'),
                    'disease': meta.get('disease'),
                    'Question': question,
                    'Original_Answer': raw_answer, # 컬럼명을 Original_Answer로 변경
                    'RAG_Chunk': rag_chunk
                })

        except Exception as e:
            print(f"파일 처리 중 오류 발생: {file_path} - {e}")
            continue

    print("전처리 및 통합 완료.")
    return pd.DataFrame(processed_list)

# --- 실행 ---

# 최종적으로 탐색을 시작할 'train_말뭉치' 폴더 경로를 구성합니다.
FINAL_DATA_ROOT = os.path.join(DATA_ROOT, 'train_말뭉치')

# 함수 실행
final_df_raw = preprocess_and_combine_data_no_cleaning(FINAL_DATA_ROOT)

# 결과 확인 및 저장
if not final_df_raw.empty:
    print("\n--- 통합된 데이터프레임 구조 ---")
    print(f"총 {len(final_df_raw)}개의 데이터 항목이 통합되었습니다.")
    
    # 원본 답변이 그대로 포함되었는지 확인
    print("\n--- 전처리된 데이터 예시 (Original_Answer 확인) ---")
    df_preview = final_df_raw[['Folder_Department', 'lifeCycle', 'Question', 'Original_Answer']].head()
    for index, row in df_preview.iterrows():
        print(f"[{row['Folder_Department']} - {row['lifeCycle']}] Q: {row['Question'][:30]}... A: {row['Original_Answer'][:100]}...")
    
    # CSV 파일로 저장
    output_path = 'final_rag_data_combined_raw.csv'
    final_df_raw.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n성공적으로 '{output_path}' 파일로 저장 완료. (BOILERPLATE 포함됨)")