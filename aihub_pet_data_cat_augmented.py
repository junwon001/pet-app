import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# 1) 원본데이터
df = pd.read_csv("aihub_pet_data.csv")
df_cat = df[df["species"] == 20].copy()

print(len(df_cat))

# 2) 메타데이터
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_cat)

categorical_cols = ["breed", "sex", "class", "group"]
for col in categorical_cols:
    if col in df_cat.columns:
        metadata.update_column(col, sdtype="categorical")

# image_id는 절대 primary key로 두지 않는다!!!
metadata.update_column("image_id", sdtype="id")

# 3) Synthesizer
synth = GaussianCopulaSynthesizer(
    metadata=metadata,
    enforce_min_max_values=True,
    enforce_rounding=True
)

# 4) 학습
print("학습 시작...")
synth.fit(df_cat)
print("학습 완료!")

# 5) 샘플 생성
AUG_SIZE = 10000
df_cat_aug = synth.sample(AUG_SIZE)

# species 고정
df_cat_aug["species"] = 20

# id 새로 생성
df_cat_aug["image_id"] = [f"cat_aug_{i}" for i in range(len(df_cat_aug))]

# 6) 원본 + 증강 merge
df_final = pd.concat([df, df_cat_aug], ignore_index=True)

# 7) 저장
df_cat_aug.to_csv("cat_augmented_only.csv", index=False)
df_final.to_csv("pet_with_augmented_cat.csv", index=False)

print("완료!")
