import os
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# =========================================================
# ✅ 1. Groq LLaMA 기반 BCS 추정 (신규 품종 처리용)
# =========================================================

import re

def estimate_bcs_with_llm(
    weight, age, chest_size, exercise,
    shoulder_height, neck_size, back_length,
    food_amount, snack_amount, food_count
):
    try:
        from groq import Groq
        import os

        if not os.getenv("GROQ_API_KEY"):
            print("⚠️ GROQ_API_KEY 없음 → LLM 예측 불가")
            return None

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""
다음 반려동물 정보를 기반으로 BCS 점수(1~9) 중 하나의 숫자만 출력하세요.
설명 없이 반드시 숫자 하나만 출력하세요.

- 체중: {weight}kg
- 나이: {age}세
- 가슴둘레: {chest_size}cm
- 운동량: 하루 {exercise}시간
- 어깨높이: {shoulder_height}cm
- 목둘레: {neck_size}cm
- 등길이: {back_length}cm
- 하루 사료량: {food_amount}g
- 하루 간식량: {snack_amount}g
- 하루 식사 횟수: {food_count}회
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        raw_text = response.choices[0].message.content.strip()
        print("📌 LLM 원본 응답:", raw_text)

        # ✅ ✅ ✅ 핵심: 1~9 숫자 하나만 정규식으로 안전 추출
        match = re.search(r"\b[1-9]\b", raw_text)

        if match:
            bcs_score = int(match.group())
            return bcs_score
        else:
            print("⚠️ BCS 숫자 추출 실패 → LLM 응답:", raw_text)
            return None

    except Exception as e:
        print(f"❌ LLM BCS 추정 실패: {e}")
        return None



def classify_bcs_from_score(score):
    if score <= 3:
        return "Underweight"
    elif score <= 5:
        return "Normal"
    elif score <= 7:
        return "Overweight"
    else:
        return "Obese"


# =========================================================
# ✅ 2. 개인화 조언 함수
# =========================================================

def generate_advice(predicted_class, food_amount, snack_amount, exercise, food_count):
    advice_parts = []

    if predicted_class == 'Obese':
        advice_parts.append("현재 비만도가 심각합니다. 즉시 수의사 상담 후 체중 감량 프로그램을 시작하는 것이 좋습니다.")
    elif predicted_class == 'Overweight':
        advice_parts.append("체중이 과체중 범주입니다. 사료량을 조절하고 활동량을 늘려야 합니다.")
    elif predicted_class == 'Normal':
        advice_parts.append("정상 체형입니다. 현재 식단과 운동 관리를 잘 유지하고 계십니다.")
    else:
        advice_parts.append("저체중입니다. 영양 상태와 기저 질환 여부를 수의사와 상담해 보세요.")

    if snack_amount > food_amount * 0.1:
        advice_parts.append(
            f"간식 비율({snack_amount}g)이 다소 높습니다. "
            f"간식을 줄이고 정량 사료({food_amount}g)에 집중하는 것이 좋습니다."
        )

    if predicted_class in ['Overweight', 'Obese'] and exercise < 2.0:
        advice_parts.append(
            f"현재 운동량({exercise}시간)은 비만도 관리에 부족합니다. "
            f"최소 30분 이상 더 활동량을 늘려주세요."
        )

    if predicted_class in ['Overweight', 'Obese'] and food_count > 3:
        advice_parts.append(
            f"식사 횟수({food_count}회)가 잦은 편입니다. "
            f"총량을 유지하며 1~2회로 줄여 급여 간격을 늘리는 것을 고려하세요."
        )
    elif predicted_class == 'Underweight' and food_count < 2:
        advice_parts.append(
            f"식사 횟수({food_count}회)가 너무 적습니다. "
            f"하루 2~3회로 나누어 안정적인 영양 공급을 해주는 것이 좋습니다."
        )

    return " ".join(advice_parts)


# =========================================================
# ✅ 3. 저장 경로 & 전역 변수
# =========================================================

MODEL_PATH = 'catboost_obesity_model.cbm'
SCALER_PATH = 'scaler.pkl'
LE_BREED_PATH = 'le_breed.pkl'
LE_SEX_PATH = 'le_sex.pkl'
CLASSES_PATH = 'classes.pkl'
DATA_FILE = 'aihub_pet_data.csv'

model = None
scaler = None
le_breed = None
le_sex = None
target_classes = None


# =========================================================
# ✅ 4. 한글 → 코드 매핑 (신규 입력용)
# =========================================================

BREED_MAPPING = {
    "비글": "dog_BEA",
    "비숑프리제": "dog_BIC",
    "불독": "dog_BUL",
    "치와와 장모": "dog_CHL",
    "치와와 단모": "dog_CHS",
    "코카스패니얼": "dog_COC",
    "닥스훈트 장모": "dog_DAL",
    "닥스훈트 단모": "dog_DAS",
    "도베르만 핀셔": "dog_DOB",
    "골든리트리버": "Ddog_GOL",
    "시추": "dog_DRI", #########?
    "저먼셰퍼드": "dog_GER",
    "그레이트피레니즈": "dog_GRE",
    "하운드": "dog_HOU",
    "허스키": "dog_HUS",
    "진도": "dog_JIN",
    "래브라도리트리버": "dog_LAB",
    "몰티즈": "dog_MAL",
    "믹스 장모": "dog_MIL",
    "믹스 단모": "dog_MIS",
    "말라뮤트": "dog_MUT",
    "포메라니안": "dog_POM",
    "푸들": "dog_POO",
    "슈나우저": "dog_SCH",
    "쉽독": "dog_SHE",
    "테리어": "dog_TER",
    "웰시코기": "dog_WEL",
    "개_기타": "dog_ETC",

    "코리안숏헤어": "cat_KOR",
    "페르시안": "cat_PER",
    "러시안블루": "cat_RUS",
    "스코티시폴드": "cat_SCO",
    "샴": "cat_SIA",
    "터키시앙고라": "cat_TUR",
    "고양이_기타": "cat_ETC"
}

SEX_MAPPING = {
    '수컷': 'IM',
    '암컷': 'IF',
    '중성화 수컷': 'CM',
    '중성화 암컷': 'SF',
}


# =========================================================
# ✅ 5. 아티팩트 로드
# =========================================================

def load_artifacts():
    global model, scaler, le_breed, le_sex, target_classes

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(LE_BREED_PATH, 'rb') as f:
        le_breed = pickle.load(f)
    with open(LE_SEX_PATH, 'rb') as f:
        le_sex = pickle.load(f)
    with open(CLASSES_PATH, 'rb') as f:
        target_classes = pickle.load(f)

    print("✅ CatBoost 모델 및 전처리 도구 로드 완료.")


# =========================================================
# ✅ 6. 최종 예측 함수 (한글 → 코드 → CatBoost / LLM)
# =========================================================

def predict_obesity(
    weight: float,
    age: int,
    breed: str,
    sex: str,

    chest_size: float = None,
    exercise: float = None,
    shoulder_height: float = None,
    neck_size: float = None,
    back_length: float = None,
    food_amount: float = None,
    snack_amount: float = None,
    food_count: int = None
):

    global model
    if model is None:
        load_artifacts()

    # -----------------------------
    # ✅ 선택 입력 기본값 자동 보정
    # -----------------------------
    if chest_size is None:
        chest_size = 0.0
    if exercise is None:
        exercise = 1.5
    if shoulder_height is None:
        shoulder_height = 0.0
    if neck_size is None:
        neck_size = 0.0
    if back_length is None:
        back_length = 0.0
    if food_amount is None:
        food_amount = weight * 20
    if snack_amount is None:
        snack_amount = 0.0
    if food_count is None:
        food_count = 2

    # -----------------------------
    # ✅ 한글 → 코드 매핑
    # -----------------------------
    breed_code = BREED_MAPPING.get(breed)
    sex_code = SEX_MAPPING.get(sex)

    use_llm = False

    if breed_code is None or sex_code is None:
        print("⚠️ 매핑 테이블에 없는 품종/성별 → LLM 처리")
        use_llm = True
    else:
        breed_code_norm = breed_code.lower()

        try:
            breed_encoded = le_breed.transform([breed_code_norm])[0]
            sex_encoded = le_sex.transform([sex_code])[0]
            use_llm = False
        except ValueError:
            print("⚠️ 학습 데이터에 없는 코드 → LLM 처리")
            use_llm = True

    # -----------------------------
    # ✅ LLM 우회
    # -----------------------------
    if use_llm:
        bcs_score = estimate_bcs_with_llm(
            weight, age, chest_size, exercise,
            shoulder_height, neck_size, back_length,
            food_amount, snack_amount, food_count
        )

        if bcs_score is None:
            return "❌ LLM BCS 추정 실패"

        predicted_class = classify_bcs_from_score(bcs_score)
        advice = generate_advice(
            predicted_class,
            food_amount,
            snack_amount,
            exercise,
            food_count
        )

        return f"""
📌 입력 정보
- 품종: {breed}
- 나이: {age}세
- 체중: {weight}kg

✅ [LLM 기반 품종 미등록 처리]
👉 추정 BCS: {bcs_score}
👉 비만도 결과: **{predicted_class}**
👉 개인화 조언: {advice}
"""

    # -----------------------------
    # ✅ CatBoost 예측
    # -----------------------------
    X = np.array([[
        weight, age, chest_size, exercise,
        shoulder_height, neck_size, back_length,
        food_amount, snack_amount, food_count,
        breed_encoded, sex_encoded
    ]])

    X_scaled = scaler.transform(X)
    pred_idx = model.predict(X_scaled)[0]
    result_class = target_classes[int(pred_idx)]

    advice = generate_advice(
        result_class,
        food_amount,
        snack_amount,
        exercise,
        food_count
    )

    return f"""
📌 입력 정보
- 품종: {breed}
- 나이: {age}세
- 체중: {weight}kg

✅ CatBoost 모델 예측 결과: **{result_class}**
👉 개인화 조언: {advice}
"""

    # -----------------------------
    # 3. CatBoost 모델 예측
    # -----------------------------
    X = np.array([[
        weight, age, chest_size, exercise,
        shoulder_height, neck_size, back_length,
        food_amount, snack_amount, food_count,
        breed_encoded, sex_encoded
    ]])

    X_scaled = scaler.transform(X)
    pred_idx = model.predict(X_scaled)[0]
    result_class = target_classes[int(pred_idx)]  # ✅ 여기서 result_class 정의

    advice = generate_advice(
        result_class,
        food_amount,
        snack_amount,
        exercise,
        food_count
    )

    return f"""
📌 입력 정보
- 품종: {breed}
- 나이: {age}세
- 체중: {weight}kg

✅ CatBoost 모델 예측 결과: **{result_class}**
👉 개인화 조언: {advice}
"""

# =========================================================
# ✅ 7. 학습 파이프라인 (CSV는 "코드" 기준)
# =========================================================

if __name__ == "__main__":

    if not os.path.exists(DATA_FILE):
        print("❌ 데이터 파일 없음. 학습 중단.")
        exit()

    print("📌 데이터 로드 중...")
    df_raw = pd.read_csv(DATA_FILE)
    df = df_raw.dropna()

    # ✅ BCS → classification 생성
    if "classification" not in df.columns:
        if "BCS" in df.columns:
            df["BCS_score"] = df["BCS"]
        elif "BCS_score" in df.columns:
            df["BCS_score"] = df["BCS_score"]
        else:
            raise ValueError("❌ CSV에 BCS 또는 BCS_score 컬럼이 없습니다.")

        df["classification"] = df["BCS_score"].apply(classify_bcs_from_score)
        print("✅ BCS → classification 컬럼 자동 생성 완료")

    # ✅ Label Encoding (CSV는 이미 코드라고 가정)
    le_breed = LabelEncoder()
    le_sex = LabelEncoder()
    le_target = LabelEncoder()

    df["breed_encoded"] = le_breed.fit_transform(df["breed"])
    df["sex_encoded"] = le_sex.fit_transform(df["sex"])
    df["target"] = le_target.fit_transform(df["classification"])
    target_classes = le_target.classes_

    features = [
        'weight', 'age', 'chest_size', 'exercise',
        'shoulder_height', 'neck_size', 'back_length',
        'food_amount', 'snack_amount', 'food_count',
        'breed_encoded', 'sex_encoded'
    ]

    X = df[features].values
    y = df["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🚀 CatBoost 학습 시작...")

    model = CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        loss_function='MultiClass',
        verbose=False
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- ✅ CatBoost 평가 ---")
    print(classification_report(y_test, y_pred, target_names=target_classes))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # ✅ 모델 저장
    model.save_model(MODEL_PATH)

    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    with open(LE_BREED_PATH, 'wb') as f: pickle.dump(le_breed, f)
    with open(LE_SEX_PATH, 'wb') as f: pickle.dump(le_sex, f)
    with open(CLASSES_PATH, 'wb') as f: pickle.dump(target_classes, f)

    print("\n✅ 모든 아티팩트 저장 완료.")



    #예시 1
    print(predict_obesity(
    weight=12.0,
    age=5,
    breed="래브라도리트리버",
    sex="수컷",
    exercise=0.5,
    food_amount=400,
    snack_amount=120,
    food_count=4
))
    # 예시 2
    print(predict_obesity(
    weight=5.5,
    age=3,
    breed="비숑프리제",
    sex="암컷"
))
    #예시 3
    print(predict_obesity(
    weight=10.0,
    age=4,
    breed="시베리안라이카",
    sex="수컷"
))
    