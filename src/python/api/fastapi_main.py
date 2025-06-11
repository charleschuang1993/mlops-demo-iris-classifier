# src/python/api/fastapi_main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Path
from pydantic import BaseModel
import requests
import mlflow
from mlflow import MlflowClient
from pathlib import Path
import uuid
import joblib

# 导入 Python pipeline 函数
from src.python.preprocessing.preprocess import load_and_preprocess
from src.python.modeling.run_experiment import train as py_train
from src.python.modeling.evaluate import evaluate as py_evaluate
from src.python.modeling.predict import load_model as py_load_model, predict as py_predict

app = FastAPI(title="Iris MLOps API")

# ── Shared schemas ──────────────────────────────────────────────
class IrisInput(BaseModel):
    Sepal_Length: float
    Sepal_Width: float
    Petal_Length: float
    Petal_Width: float

class TrainParams(BaseModel):
    n_estimators: int = 100
    max_depth: int = 5

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

# ── Initialization ──────────────────────────────────────────────
# MLflow Tracking URI & Client
mlflow.set_tracking_uri("http://localhost:5001")
mlflow_client = MlflowClient()

# Paths
DATA_PATH = Path(__file__).parents[2] / "data" / "iris.csv"
PY_MODEL_DIR = Path(__file__).parents[2] / "models" / "python"

# ── Health Check ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "from": "FastAPI"}

# ── R-Plumber 转发（原 /predict 接口）──────────────────────────
@app.post("/r/predict", summary="Forward to R-Plumber prediction")
def predict_r(input: IrisInput):
    payload = {
        "Sepal.Length": input.Sepal_Length,
        "Sepal.Width":  input.Sepal_Width,
        "Petal.Length": input.Petal_Length,
        "Petal.Width":  input.Petal_Width,
    }
    try:
        resp = requests.post("http://localhost:8000/predict", json=payload)
        resp.raise_for_status()
        return {"source": "R-plumber", **resp.json()}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Python Pipeline Endpoints ──────────────────────────────────
@app.post("/py/train", summary="Train Python model in background")
def train_py(params: TrainParams, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    model_path = PY_MODEL_DIR / f"{task_id}.pkl"
    background_tasks.add_task(
        py_train,
        data_path=str(DATA_PATH),
        model_output_path=str(model_path),
        n_estimators=params.n_estimators,
        max_depth=params.max_depth
    )
    return {"task_id": task_id, "model_path": str(model_path)}

@app.get("/py/evaluate/{model_filename}", summary="Evaluate Python model")
def eval_py(model_filename: str = Path(..., description="Filename of the .pkl model")):
    model_path = PY_MODEL_DIR / model_filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    metrics = py_evaluate(str(model_path), str(DATA_PATH))
    return {"source": "Python", "metrics": metrics}

@app.post("/py/predict/{model_filename}", summary="Predict with Python model")
def predict_py(model_filename: str, input: IrisInput):
    model = py_load_model(model_filename)
    preds = py_predict(model, input.dict())
    return {"source": "Python", "predictions": preds}

# ── MLflow Model Registry APIs ─────────────────────────────────
@app.post("/register", summary="Register R model to MLflow")
def register_model(req: RegisterRequest):
    try:
        uri = f"runs:/{req.run_id}/{req.artifact_path}"
        result = mlflow.register_model(model_uri=uri, name=req.model_name)
        return {"status": "success", "model_name": result.name, "version": result.version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/promote", summary="Promote model version stage")
def promote_model(req: PromoteRequest):
    try:
        mlflow_client.transition_model_version_stage(
            name=req.model_name,
            version=req.version,
            stage=req.stage,
            archive_existing_versions=False
        )
        return {"status": "success", "message": f"Promoted to {req.stage}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", summary="List registered models")
def list_models():
    try:
        regs = mlflow_client.search_registered_models()
        output = []
        for m in regs:
            versions = [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "creation_timestamp": v.creation_timestamp
                }
                for v in m.latest_versions
            ]
            output.append({"name": m.name, "versions": versions})
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete", summary="Delete a single model version")
def delete_model_version(req: DeleteRequest):
    try:
        mlflow_client.delete_model_version(name=req.model_name, version=req.version)
        return {"status": "success", "message": f"Deleted version {req.version}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/{model_name}", summary="Delete entire registered model")
def delete_registered_model(model_name: str = Path(..., description="Model to delete")):
    try:
        mlflow_client.delete_registered_model(name=model_name)
        return {"status": "success", "message": f"Deleted model '{model_name}'"}
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




# mlflow server \
#   --backend-store-uri sqlite:///$(pwd)/mlflow.db \
#   --default-artifact-root $(pwd)/mlruns \
#   --serve-artifacts \
#   --host 0.0.0.0 \
#   --port 5001
