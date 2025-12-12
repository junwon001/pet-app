import os
import json
import pandas as pd
from typing import List, Dict, Any

# JSON 파일들이 저장된 루트 디렉토리 경로를 지정하세요.
JSON_DIR = 'C:/Users/jwm02/OneDrive/바탕 화면/애완동물 관리/train'

def extract_metadata_with_details(json_dir: str) -> List[Dict[str, Any]]:
    """
    JSON 파일에서 정보를 추출하되,
    breed 값만 'dog_품종명', 'cat_품종명' 형태로 변경한다.
    """
    
    metadata_list = []
    
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading JSON file {filename}: {e}")
                continue
            
            try:
                # 메타데이터 주요 섹션 분리
                metadata_id = data['metadata']['id']
                physical = data['metadata']['physical']
                breeding = data['metadata']['breeding']
                
                species = metadata_id['species']
                breed = metadata_id['breed']

                # species → prefix 변환
                if species == '10':
                    breed_value = f"dog_{breed}"
                elif species == '20':
                    breed_value = f"cat_{breed}"
                else:
                    breed_value = f"unknown_{breed}"
                
                # 추출된 정보 저장
                metadata_list.append({
                    'image_id': data['annotations']['image-id'],
                    'species': species,
                    'breed': breed_value,  # ← 여기만 변경됨
                    'age': metadata_id['age'],
                    'sex': metadata_id['sex'],
                    
                    # 추가 정보
                    'class': metadata_id['class'],
                    'group': metadata_id['group'],
                    
                    # 신체 정보
                    'weight': physical['weight'],
                    'shoulder_height': physical['shoulder-height'],
                    'chest_size': physical['chest-size'],
                    'BCS': physical['BCS'],
                    'neck_size': physical['neck-size'],
                    'back_length': physical['back-length'],
                    
                    # 사육 환경
                    'exercise': breeding['exercise'],
                    'food_count': breeding['food-count'],
                    'food_amount': breeding['food-amount'],
                    'snack_amount': breeding['snack-amount'],
                })

            except KeyError as e:
                print(f"Key missing in {filename}: {e}")
                continue
            
    return metadata_list


# --- 실행 부분 ---
if __name__ == "__main__":
    if not os.path.isdir(JSON_DIR):
        print(f"Error: Directory not found at {JSON_DIR}")
        print("Please update the JSON_DIR variable with the correct path to your JSON files.")
    else:
        print(f"Parsing JSON files in: {JSON_DIR}...")
        
        extracted_metadata = extract_metadata_with_details(JSON_DIR)
        df_metadata = pd.DataFrame(extracted_metadata)
        
        print(f"\nSuccessfully extracted {len(df_metadata)} records.")
        print("\n--- 구성된 데이터프레임 (상위 5개 행) ---")
        print(df_metadata.head())

        df_metadata.to_csv('aihub_pet_data.csv', index=False, encoding='utf-8-sig')
