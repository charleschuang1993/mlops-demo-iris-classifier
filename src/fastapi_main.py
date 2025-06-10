from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import mlflow
from mlflow import MlflowClient

app = FastAPI()

# ====================== Input Schemas =========================

class IrisInput(BaseModel):
    Sepal_Length: float
    Sepal_Width: float
    Petal_Length: float
    Petal_Width: float

class RegisterRequest(BaseModel):
    run_id: str
    model_name: str
    artifact_path: str = "iris_rf_model.rds"

class PromoteRequest(BaseModel):
    model_name: str
    version: int
    stage: str  # e.g., "Staging", "Production"

class DeleteRequest(BaseModel):
    model_name: str
    version: int

# ======================= Initialization ==========================

# Set MLflow tracking path (can be file:// or http:// depending on usage)
mlflow.set_tracking_uri("file:///Users/charleschuang/Desktop/atgenomix/ToolDev/mlops-demo-iris-classifier/mlruns")
client = MlflowClient()

# ======================= Core APIs ==========================

@app.get("/health")
def health():
    return {"status": "ok", "from": "FastAPI"}

@app.post("/predict")
def predict(input: IrisInput):
    # Convert to R API compatible format (column names)
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

# ===================== MLflow Model Registry APIs ======================

@app.post("/register")
def register_model(req: RegisterRequest):
    try:
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

@app.post("/promote")
def promote_model(req: PromoteRequest):
    try:
        client.transition_model_version_stage(
            name=req.model_name,
            version=req.version,
            stage=req.stage,
            archive_existing_versions=False
        )
        return {"status": "success", "message": f"Model promoted to {req.stage}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def list_models():
    try:
        models = client.list_registered_models()
        result = []
        for m in models:
            versions = [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "creation_timestamp": v.creation_timestamp
                }
                for v in m.latest_versions
            ]
            result.append({"name": m.name, "versions": versions})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete")
def delete_model_version(req: DeleteRequest):
    try:
        client.delete_model_version(
            name=req.model_name,
            version=req.version
        )
        return {"status": "success", "message": f"Version {req.version} deleted from {req.model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# uvicorn src.fastapi_main:app --host 0.0.0.0 --port 8080 --reload
#curl -X POST http://localhost:8080/predict   -H "Content-Type: application/json"   -d '{"Sepal_Length": 5.1, "Sepal_Width": 3.5, "Petal_Length": 1.4, "Petal_Width": 0.2}'
#swagger: http://localhost:8080/docs

# curl -X POST http://localhost:8080/register \
#   -H "Content-Type: application/json" \
#   -d '{
#         "run_id": "797558b609404e0ea672fb8ecc2b5965",
#         "model_name": "testRModel_iris_example"
#       }'


# promote model
# curl -X POST http://localhost:8080/promote \
#   -H "Content-Type: application/json" \
#   -d '{"model_name": "testRModel_iris_example", "version": 1, "stage": "Production"}'

# list models
# curl -X GET http://localhost:8080/models

# delete model version
# curl -X DELETE http://localhost:8080/delete \
#   -H "Content-Type: application/json" \
#   -d '{"model_name": "testRModel_iris_example", "version": 1}'
