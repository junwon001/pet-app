from fastapi import FastAPI
from pydantic import BaseModel
from obesity_model import predict_obesity  # 너가 만든 함수 그대로 import!

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
    return {"result": result}
