import pandas as pd

# 1. CSV 파일 읽기
try:
    df = pd.read_csv('aihub_pet_data.csv')
    print("✅ CSV 파일 읽기 성공.")
except FileNotFoundError:
    print("❌ 에러: 'aihub_pet_data.csv' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()

# --- species == 20 개수 계산 ---
print(df.info())
if 'species' in df.columns:
    species_20_count = len(df[df['species'] == 20])
    print(f"\n✨ 'species'가 '20'인 데이터의 개수: {species_20_count}개")
else:
    print("\n⚠️ 'species' 열 없음.")

# --- breed 전체 고유값 + 비율 출력 ---
if 'breed' in df.columns:
    print("\n--- 📌 'breed' 전체 리스트 및 비율 (%) ---")

    # 개수 + 비율 계산
    breed_stats = df['breed'].value_counts(normalize=False)      # 개수
    breed_percent = df['breed'].value_counts(normalize=True) * 100  # 퍼센트(%)

    # 하나의 데이터프레임으로 합침
    breed_df = pd.DataFrame({
        "count": breed_stats,
        "percent": breed_percent.round(2)
    })

    # 전체 출력
    print(breed_df)

else:
    print("\n❌ 에러: 데이터프레임에 'breed' 열이 없습니다.")
