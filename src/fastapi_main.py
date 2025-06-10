from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import mlflow

app = FastAPI()

# 🔹 給 Plumber 的預測資料格式
class IrisInput(BaseModel):
    Sepal_Length: float
    Sepal_Width: float
    Petal_Length: float
    Petal_Width: float

# 🔹 給 MLflow 註冊模型用的格式
class RegisterRequest(BaseModel):
    run_id: str
    model_name: str
    artifact_path: str = "iris_rf_model.rds"  # 預設為你目前用的路徑

@app.get("/health")
def health():
    return {"status": "ok", "from": "FastAPI"}

@app.post("/predict")
def predict(input: IrisInput):
    # 轉為 Plumber 所需的欄位命名格式
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

@app.post("/register")
def register_model(req: RegisterRequest):
    try:
        # ✅ 指定你正確的 tracking 路徑（請根據實際專案調整）
        mlflow.set_tracking_uri("file:///Users/charleschuang/Desktop/atgenomix/ToolDev/mlops-demo-iris-classifier/mlruns")

        model_uri = f"runs:/{req.run_id}/{req.artifact_path}"

        result = mlflow.register_model(
            model_uri=model_uri,
            name=req.model_name
        )

        return {
            "status": "success",
            "model_name": result.name,
            "version": result.version
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#curl -X POST http://localhost:8080/predict   -H "Content-Type: application/json"   -d '{"Sepal_Length": 5.1, "Sepal_Width": 3.5, "Petal_Length": 1.4, "Petal_Width": 0.2}'
#swagger: http://localhost:8080/docs

# curl -X POST http://localhost:8080/register \
#   -H "Content-Type: application/json" \
#   -d '{
#         "run_id": "797558b609404e0ea672fb8ecc2b5965",
#         "model_name": "testRModel_iris_example"
#       }'
