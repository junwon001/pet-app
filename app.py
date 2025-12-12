from fastapi import FastAPI
from pydantic import BaseModel
from obesity_model import predict_obesity

from bcs_repository import save_bcs
from utils import extract_bcs_number   # 정규식 함수

app = FastAPI(title="Pet Obesity AI API")

class PredictRequest(BaseModel):
    weight: float
    age: int
    breed: str
    sex: str
    chest_size: float | None = None
    exercise: float | None = None
    shoulder_height: float | None = None
    neck_size: float | None = None
    back_length: float | None = None
    food_amount: float | None = None
    snack_amount: float | None = None
    food_count: int | None = None

@app.post("/predict")
def predict(req: PredictRequest):
    # 1️⃣ BCS 예측
    result = predict_obesity(
        weight=req.weight,
        age=req.age,
        breed=req.breed,
        sex=req.sex,
        chest_size=req.chest_size,
        exercise=req.exercise,
        shoulder_height=req.shoulder_height,
        neck_size=req.neck_size,
        back_length=req.back_length,
        food_amount=req.food_amount,
        snack_amount=req.snack_amount,
        food_count=req.food_count
    )

    # 2️⃣ BCS 숫자만 추출
    bcs_value = extract_bcs_number(result)

    # 3️⃣ DB 저장 (일단 pet_id는 1로)
    save_bcs(pet_id=1, bcs_value=bcs_value)

    # 4️⃣ 응답
    return {
        "bcs": bcs_value,
        "raw_result": result
    }

from recommend.recommend_service import recommend_feed_by_bcs

@app.get("/recommend/{pet_id}")
def recommend(pet_id: int):
    return recommend_feed_by_bcs(pet_id)


