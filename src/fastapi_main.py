from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

class IrisInput(BaseModel):
    Sepal_Length: float
    Sepal_Width: float
    Petal_Length: float
    Petal_Width: float

@app.get("/health")
def health():
    return {"status": "ok", "from": "FastAPI"}

@app.post("/predict")
def predict(input: IrisInput):
    # 轉成 R API 所需格式（欄位名稱需對應）
    r_payload = {
        "Sepal.Length": input.Sepal_Length,
        "Sepal.Width": input.Sepal_Width,
        "Petal.Length": input.Petal_Length,
        "Petal.Width": input.Petal_Width,
    }

    try:
        r_response = requests.post("http://localhost:8000/predict", json=r_payload)
        r_response.raise_for_status()
        return {"source": "R plumber", **r_response.json()}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


#curl -X POST http://localhost:8080/predict   -H "Content-Type: application/json"   -d '{"Sepal_Length": 5.1, "Sepal_Width": 3.5, "Petal_Length": 1.4, "Petal_Width": 0.2}'
#swagger: http://localhost:8080/docs