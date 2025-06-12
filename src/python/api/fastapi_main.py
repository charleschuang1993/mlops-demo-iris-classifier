# src/python/api/fastapi_main.py

"""FastAPI application for the Iris‑classifier MLOps demo.

• Supports **training**, **evaluation**, **prediction**, and **model registry** ops.  
• Accepts either the built‑in *Iris* dataset (`data_path="iris"`) **or** any CSV
  file path that contains the same columns as the Iris dataset.
• All model artefacts are managed **exclusively** via MLflow (no local .pkl).
"""

from __future__ import annotations

from pathlib import Path as _Path
from uuid import uuid4

import mlflow
from fastapi import BackgroundTasks, FastAPI, HTTPException, Path
from mlflow import MlflowClient, sklearn as mlflow_sklearn
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris

from src.python.preprocessing.preprocess import load_and_preprocess
from src.python.modeling.train import train as py_train
from src.python.modeling.evaluate import evaluate as py_evaluate

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI + MLflow initialisation
# ──────────────────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5001")
client = MlflowClient()

app = FastAPI(
    title="Iris MLOps API",
    description=(
        "Train / evaluate / predict with models stored in the MLflow Model\n"
        "Registry. Supports the built‑in Iris dataset **and** custom CSVs."
    ),
    version="1.3.0",
)

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class TrainParams(BaseModel):
    data_path: str = Field("iris", description="'iris' or path/to/csv")
    test_size: float = 0.3
    random_state: int = 42
    n_splits: int = 10
    experiment_name: str = "iris-classification"

class EvaluateRequest(BaseModel):
    data_path: str = Field("iris", description="'iris' or path/to/csv")
    mlflow_model_name: str
    mlflow_model_stage: str = Field("Production", description="Stage or version")

# ──────────────────────────────────────────────────────────────────────────────
# Helper — load data
# ──────────────────────────────────────────────────────────────────────────────

def _load_data(path: str):
    """Return (X, y) for built‑in Iris or CSV path."""
    if path == "iris":
        iris = load_iris()
        return iris.data, iris.target
    csv_path = _Path(path)
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {path}")
    return load_and_preprocess(str(csv_path))

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/train")
async def train(params: TrainParams, background_tasks: BackgroundTasks):
    """Launch training in a background thread, returns expected model name."""
    background_tasks.add_task(
        py_train,
        data=params.data_path,
        test_size=params.test_size,
        random_state=params.random_state,
        n_splits=params.n_splits,
        experiment_name=params.experiment_name,
    )
    model_name = f"{params.experiment_name}-rf-model"
    return {"message": "training started", "model_name": model_name}


@app.post("/evaluate")
async def evaluate_ep(req: EvaluateRequest):
    """Evaluate a registered model on the given dataset (Iris or CSV)."""
    report = py_evaluate(
        data_path=req.data_path,
        mlflow_model_name=req.mlflow_model_name,
        mlflow_model_stage=req.mlflow_model_stage,
    )
    return report


@app.post("/predict")
async def predict(
    payload: IrisInput,
    model_name: str = "iris-classification-rf-model",
    model_stage: str = "Production",
):
    # Stage → attempt, fallback to latest version if stage missing
    try:
        model_uri = f"models:/{model_name}/{model_stage}"
        model = mlflow_sklearn.load_model(model_uri)
    except mlflow.exceptions.RestException:
        latest = client.get_latest_versions(model_name)[0]
        model = mlflow_sklearn.load_model(f"models:/{model_name}/{latest.version}")

    X = [[
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width,
    ]]
    pred = int(model.predict(X)[0])
    return {"prediction": pred}


# ── Model registry helpers
# -----------------------------------------------------------------------------
@app.post("/register")
async def register_run(run_id: str, model_name: str, stage: str = "Staging"):
    """Register a given run_id as a new model version and set stage."""
    try:
        mv = client.create_model_version(
            name=model_name, source=f"runs:/{run_id}/model", run_id=run_id
        )
        client.transition_model_version_stage(
            name=model_name, version=mv.version, stage=stage
        )
        return {"model_name": model_name, "version": mv.version, "stage": stage}
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))


@app.post("/promote/{model_name}/{version}/{stage}")
async def promote(model_name: str, version: int, stage: str):
    try:
        client.transition_model_version_stage(
            name=model_name, version=version, stage=stage
        )
        return {"status": "success", "model_name": model_name, "version": version, "stage": stage}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.get("/models")
async def list_registered():
    regs = client.search_registered_models()
    return {
        m.name: [{"version": v.version, "stage": v.current_stage} for v in m.latest_versions]
        for m in regs
    }


@app.delete("/models/{model_name}/{version}")
async def delete_version(model_name: str, version: int):
    try:
        client.delete_model_version(name=model_name, version=version)
        return {"status": "deleted", "model_name": model_name, "version": version}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    try:
        client.delete_registered_model(name=model_name)
        return {"status": "deleted", "model_name": model_name}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
