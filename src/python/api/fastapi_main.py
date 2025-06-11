# src/python/api/fastapi_main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Path as ApiPath
from pydantic import BaseModel
import requests
import mlflow
from mlflow import MlflowClient
from pathlib import Path as P
import uuid
import joblib
import numpy as np

# -------------------------------------------------------------
# Import pipeline functions (已無 fire / load_model)
# -------------------------------------------------------------
from src.python.modeling.train import train as py_train
from src.python.modeling.evaluate import evaluate as py_evaluate
from src.python.modeling.predict import predict as py_predict

app = FastAPI(title="Iris MLOps API")

# ── Schemas ───────────────────────────────────────────────────
class IrisInput(BaseModel):
    Sepal_Length: float
    Sepal_Width: float
    Petal_Length: float
    Petal_Width: float

class TrainRequest(BaseModel):
    data: str = "data/iris.csv"              # 可改成其他 CSV
    experiment_name: str = "iris-classification"
    output_dir: str = "models/python"

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

# ── Initialization ────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5001")
mlflow_client = MlflowClient()

BASE_DIR = P(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "iris.csv"
PY_MODEL_DIR = BASE_DIR / "models" / "python"
PY_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Health Check ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "FastAPI Iris MLOps"}

# ── R‑Plumber 轉發（若仍需要）────────────────────────────────
@app.post("/r/predict", summary="Forward to R‑Plumber prediction")
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

# ── Python Pipeline Endpoints ─────────────────────────────────
@app.post("/py/train", summary="Train Python model in background")
def train_py(req: TrainRequest, background_tasks: BackgroundTasks):
    """啟動背景訓練，將最終模型存到 req.output_dir/best_model.pkl"""
    task_id = str(uuid.uuid4())
    out_dir = BASE_DIR / req.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    background_tasks.add_task(
        py_train,
        data=req.data,
        experiment_name=req.experiment_name,
        output_dir=str(out_dir)
    )
    model_path = out_dir / "best_model.pkl"
    return {"task_id": task_id, "model_path": str(model_path)}

@app.get("/py/evaluate/{model_filename}", summary="Evaluate Python model")
def eval_py(model_filename: str = ApiPath(..., description=".pkl model filename")):
    model_path = PY_MODEL_DIR / model_filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found. Train first.")
    metrics = py_evaluate(
        model_path=str(model_path),
        data_path=str(DATA_PATH),
        output_dir=str(PY_MODEL_DIR)
    )
    return {"source": "Python", "metrics": metrics}

@app.post("/py/predict/{model_filename}", summary="Predict with Python model (single sample)")
def predict_py(model_filename: str, input: IrisInput):
    model_path = PY_MODEL_DIR / model_filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found. Train first.")

    # 直接 joblib 載入模型，對單筆樣本做推論
    model = joblib.load(model_path)
    sample = np.array([[input.Sepal_Length, input.Sepal_Width, input.Petal_Length, input.Petal_Width]])
    preds = model.predict(sample).tolist()
    return {"source": "Python", "predictions": preds}

# ── MLflow Model Registry APIs ────────────────────────────────
@app.post("/register", summary="Register model to MLflow")
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
        return [
            {
                "name": m.name,
                "versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "status": v.status,
                        "creation_timestamp": v.creation_timestamp,
                    }
                    for v in m.latest_versions
                ],
            }
            for m in regs
        ]
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
def delete_registered_model(model_name: str = ApiPath(..., description="Model to delete")):
    try:
        mlflow_client.delete_registered_model(name=model_name)
        return {"status": "success", "message": f"Deleted model '{model_name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== Duplicate import/path conflicts resolved; pathlib.Path renamed to P =====
