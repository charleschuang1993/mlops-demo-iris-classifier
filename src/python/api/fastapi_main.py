# src/python/api/fastapi_main.py

"""FastAPI application for Iris classification MLOps demo.

All model management is handled **exclusively** by MLflow.  
This API offers endpoints to train, check status, evaluate, predict, and manage
registered models.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Path
from pydantic import BaseModel, Field
import mlflow
from mlflow import sklearn as mlflow_sklearn, MlflowClient
import uuid

from src.python.modeling.train import train as py_train
from src.python.modeling.evaluate import evaluate as py_evaluate

# -----------------------------------------------------------------------------
# Initialisation
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Iris MLOps API",
    description="Endpoints for training, evaluating, predicting, and managing models via MLflow",
    version="1.0.2",
)

mlflow.set_tracking_uri("http://localhost:5001")
client = MlflowClient()

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class IrisInput(BaseModel):
    """Input features for single‑record prediction."""
    sepal_length: float = Field(..., alias="sepal_length")
    sepal_width: float = Field(..., alias="sepal_width")
    petal_length: float = Field(..., alias="petal_length")
    petal_width: float = Field(..., alias="petal_width")

class TrainParams(BaseModel):
    """Parameters accepted by the training endpoint.

    Only the built‑in Iris dataset (string literal "iris") is supported at the
    moment because `train.py` does not implement custom CSV loading.
    """

    data_path: str = Field("iris", description="Either the literal 'iris' or a future custom CSV path.")
    test_size: float = 0.3
    random_state: int = 42
    n_splits: int = 10
    experiment_name: str = "iris-classification"

class EvaluateRequest(BaseModel):
    data_path: str = Field("iris", description="Dataset identifier—only 'iris' supported for now.")
    mlflow_model_name: str
    mlflow_model_stage: str = "Production"

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def _validate_data_path(data_path: str):
    """Only allow the built‑in Iris dataset until custom loader is implemented."""
    if data_path != "iris":
        raise HTTPException(
            status_code=400,
            detail="Custom CSV loading is not implemented; use data_path='iris'",
        )

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


@app.post("/train", summary="Launch model training in the background")
def train_endpoint(params: TrainParams, background_tasks: BackgroundTasks):
    _validate_data_path(params.data_path)

    # Launch training asynchronously; we cannot capture MLflow run_id directly
    background_tasks.add_task(
        py_train,
        data=params.data_path,
        test_size=params.test_size,
        random_state=params.random_state,
        n_splits=params.n_splits,
        experiment_name=params.experiment_name,
    )

    return {
        "message": "Training started – check MLflow UI for progress.",
        "expected_model_name": f"{params.experiment_name}-rf-model",
    }


@app.get(
    "/train/status/{run_id}",
    summary="Query MLflow for the status of a specific run_id",
)
def get_training_status(run_id: str = Path(..., description="MLflow run ID")):
    try:
        run = client.get_run(run_id)
        return {"run_id": run_id, "status": run.info.status}
    except Exception as err:
        raise HTTPException(status_code=404, detail=str(err))


@app.post("/evaluate", summary="Evaluate a registered model")
def evaluate_endpoint(req: EvaluateRequest):
    _validate_data_path(req.data_path)
    report = py_evaluate(
        data_path=req.data_path,
        mlflow_model_name=req.mlflow_model_name,
        mlflow_model_stage=req.mlflow_model_stage,
    )
    return report


@app.post("/predict", summary="Predict using a registered model")
def predict_endpoint(
    input: IrisInput,
    model_name: str = "iris-classification-rf-model",
    model_stage: str = "Production",
):
    X = [[
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width,
    ]]
    # Attempt to load the requested stage, fallback to latest version if needed
    try:
        model_uri = f"models:/{model_name}/{model_stage}"
        model = mlflow_sklearn.load_model(model_uri)
    except mlflow.exceptions.RestException:
        latest = client.get_latest_versions(model_name)[0]
        model_uri = f"models:/{model_name}/{latest.version}"
        model = mlflow_sklearn.load_model(model_uri)
    except Exception as err:
        raise HTTPException(status_code=404, detail=str(err))

    try:
        preds = model.predict(X)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

    return {"prediction": int(preds[0])}


@app.post("/register", summary="Register a run as a model and set stage")
def register_endpoint(run_id: str, model_name: str, stage: str = "Staging"):
    try:
        mv = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage=stage,
        )
        return {"model_name": model_name, "version": mv.version, "stage": stage}
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))


@app.post("/promote/{model_name}/{version}/{stage}", summary="Promote a model version")
def promote_model(
    model_name: str = Path(...),
    version: int = Path(...),
    stage: str = Path(...),
):
    try:
        client.transition_model_version_stage(
            name=model_name, version=version, stage=stage
        )
        return {"status": "success", "model_name": model_name, "version": version, "stage": stage}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.get("/models", summary="List registered models")
def list_models():
    models = client.search_registered_models()
    return [
        {
            m.name: [
                {"version": v.version, "stage": v.current_stage}
                for v in m.latest_versions
            ]
        }
        for m in models
    ]


@app.delete("/models/{model_name}/{version}", summary="Delete a model version")
def delete_model_version(
    model_name: str = Path(...),
    version: int = Path(...),
):
    try:
        client.delete_model_version(name=model_name, version=version)
        return {"status": "deleted", "model_name": model_name, "version": version}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.delete("/models/{model_name}", summary="Delete a registered model")
def delete_registered_model(model_name: str = Path(...)):
    try:
        client.delete_registered_model(name=model_name)
        return {"status": "deleted", "model_name": model_name}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
