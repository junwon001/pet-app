import os
import json
import pandas as pd
from typing import List, Dict, Any

# 1. 환경 설정: JSON 파일들이 저장된 루트 디렉토리 경로를 지정하세요.
# 예시: './aihub_pet_data/labeling_data/json'
JSON_DIR = 'C:/Users/jwm02/OneDrive/바탕 화면/애완동물 관리/train' 

def parse_json_data(json_dir: str) -> List[Dict[str, Any]]:
    """
    지정된 디렉토리 내의 모든 JSON 파일을 읽고 필요한 정보를 추출합니다.
    
    Args:
        json_dir: JSON 파일들이 위치한 디렉토리 경로.
        
    Returns:
        추출된 데이터 항목들의 리스트.
    """
    
    parsed_data = []
    
    # 1. JSON 파일 읽기 및 반복 처리
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading JSON file {filename}: {e}")
                continue
            
            # 2. 정보 추출
            try:
                # 2-1. 이미지 파일 경로 (annotations.image-id)
                image_id = data['annotations']['image-id']
                
                # 2-2. 클래스 라벨 (예: metadata.physical.BCS 또는 annotations.label.label)
                # 모델 목적에 따라 BCS를 정답으로 사용한다고 가정
                bcs_label = data['metadata']['physical']['BCS'] 
                
                # 2-3. 바운딩 박스 좌표 (annotations.label.points)
                # [[x1, y1], [x2, y2]] 형태
                points = data['annotations']['label']['points']
                x_min, y_min = points[0][0], points[0][1]
                x_max, y_max = points[1][0], points[1][1]
                
                # 바운딩 박스 좌표 정규화는 이미지 크기 정보가 필요하므로 일단 원본 픽셀 좌표를 저장합니다.
                # (정규화는 데이터셋 로더 단계에서 이미지 로드 후 진행하는 것이 더 일반적입니다.)

                # 추출된 정보 저장
                parsed_data.append({
                    'image_id': image_id,
                    'bcs_label': bcs_label,
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max
                    # 'json_filename': filename # 디버깅을 위해 JSON 파일명도 저장 가능
                })

            except KeyError as e:
                print(f"Key missing in {filename}: {e}")
                continue
            
    return parsed_data

# --- 실행 부분 ---
if __name__ == "__main__":
    if not os.path.isdir(JSON_DIR):
        print(f"Error: Directory not found at {JSON_DIR}")
        print("Please update the JSON_DIR variable with the correct path to your JSON files.")
    else:
        print(f"Parsing JSON files in: {JSON_DIR}...")
        
        # 데이터 파싱 함수 실행
        extracted_list = parse_json_data(JSON_DIR)
        
        # 3. 데이터셋 구성: 리스트를 Pandas DataFrame으로 정리
        df = pd.DataFrame(extracted_list)
        
        print(f"\nSuccessfully extracted {len(df)} records.")
        print("\n--- 구성된 데이터프레임 (상위 5개 행) ---")
        print(df.head())
        
        # 필요하다면 CSV 등으로 저장
        df.to_csv('aihub_pet_data_summary.csv', index=False, encoding='utf-8')