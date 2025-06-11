# src/python/api/fastapi_main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Path as FastAPIPath
from pydantic import BaseModel
import requests
import mlflow
from mlflow import MlflowClient
import uuid
import joblib
import pandas as pd
from pathlib import Path # 導入標準庫的 Path

# 修正：直接從 modeling.train 導入 train 函式
from src.python.modeling.train import train as py_train
from src.python.modeling.evaluate import evaluate as py_evaluate
# 註：預測函式 (py_predict) 的邏輯將直接在此檔案中處理，因此不再需要導入

app = FastAPI(title="Iris MLOps API (已修正)")

# --- Pydantic Schemas (資料模型定義) ---
class IrisInput(BaseModel):
    Sepal_Length: float
    Sepal_Width: float
    Petal_Length: float
    Petal_Width: float

class TrainParams(BaseModel):
    # 修正：移除 train.py 函式不接受的超參數。
    # train.py 中的 GridSearchCV 使用的是固定的 param_grid，
    # 因此從 API 端傳入 n_estimators 和 max_depth 會導致背景任務 TypeError。
    test_size: float = 0.3
    n_splits: int = 10

class RegisterRequest(BaseModel):
    run_id: str
    model_name: str
    artifact_path: str = "best_model.pkl" # R 語言模型通常是 .rds, Python 是 .pkl

class PromoteRequest(BaseModel):
    model_name: str
    version: int
    stage: str

class DeleteRequest(BaseModel):
    model_name: str
    version: int

# --- 初始化設定 ---
# MLflow Tracking URI & Client
mlflow.set_tracking_uri("http://localhost:5001")
mlflow_client = MlflowClient()

# 修正：修正路徑，使其正確指向專案根目錄
# __file__ 是 .../src/python/api/fastapi_main.py
# parents[3] 會指到專案根目錄 mlops-demo-iris-classifier
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "data" / "iris.csv"
PY_MODEL_DIR = BASE_DIR / "models" / "python"

# 確保模型儲存目錄存在
PY_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --- API Endpoints ---

@app.get("/health", summary="健康檢查")
def health():
    return {"status": "ok", "from": "FastAPI"}

@app.post("/r/predict", summary="轉發請求至 R-Plumber 進行預測")
def predict_r(input: IrisInput):
    # (此部分邏輯正確，維持原樣)
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
        raise HTTPException(status_code=503, detail=f"無法連接至 R Plumber 服務: {e}")

# --- Python Pipeline Endpoints ---

@app.post("/py/train", summary="在背景執行 Python 模型訓練")
def train_py(params: TrainParams, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    # 建立一個唯一的輸出目錄來存放這次訓練的所有產物 (模型、報告等)
    output_dir_path = PY_MODEL_DIR / task_id
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 修正：移除 train.py 不接受的參數 (n_estimators, max_depth)，
    # 避免在背景任務中發生 TypeError。
    background_tasks.add_task(
        py_train,
        data=str(DATA_PATH),
        output_dir=str(output_dir_path), # 將唯一的目錄路徑傳遞給訓練函式
        test_size=params.test_size,
        n_splits=params.n_splits,
        experiment_name="iris-classification-fastapi"
    )
    return {
        "message": "訓練任務已啟動",
        "task_id": task_id, 
        "info": f"模型與產出將儲存於 'models/python/{task_id}/' 目錄下"
    }

@app.get("/py/evaluate/{task_id}", summary="評估指定的 Python 模型")
def eval_py(task_id: str):
    # 模型現在位於以 task_id 命名的資料夾內
    model_path = PY_MODEL_DIR / task_id / "best_model.pkl"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"找不到 Task ID 為 {task_id} 的模型，請確認訓練是否已完成。")
    
    # 評估報告會儲存在 'outputs/evaluate'
    metrics = py_evaluate(model_path=str(model_path), data_path=str(DATA_PATH))
    return {"source": "Python", "task_id": task_id, "metrics": metrics}

@app.post("/py/predict/{task_id}", summary="使用指定的 Python 模型進行預測")
def predict_py(task_id: str, input: IrisInput):
    # 修正：重構整個預測邏輯
    model_path = PY_MODEL_DIR / task_id / "best_model.pkl"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"找不到 Task ID 為 {task_id} 的模型。")

    # 1. 直接載入模型
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"載入模型失敗: {e}")

    # 2. 將單筆輸入轉換為模型需要的 DataFrame 格式
    # 注意：欄位名稱必須與訓練時的特徵名稱完全一致
    feature_names = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]
    input_df = pd.DataFrame([input.dict()], columns=feature_names)
    
    # 3. 進行預測
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"模型預測失敗: {e}")
    
    # 4. 回傳更完整的結果
    return {
        "source": "Python",
        "task_id": task_id,
        "prediction": prediction.tolist()[0],
        "prediction_probability": prediction_proba.tolist()[0]
    }

# --- MLflow Model Registry APIs ---
# (此部分邏輯大致正確，維持原樣，僅微調)
@app.post("/register", summary="註冊模型至 MLflow Registry")
def register_model(req: RegisterRequest):
    try:
        # 取得最新一次執行的 run_id
        experiment = mlflow.get_experiment_by_name("iris-classification-fastapi")
        runs = mlflow.search_runs(experiment_ids=experiment.experiment_id, order_by=["start_time desc"], max_results=1)
        if runs.empty:
            raise HTTPException(status_code=404, detail="找不到任何 MLflow run 可供註冊。")
        
        run_id = runs.iloc[0].run_id
        artifact_path = "best_model.pkl" # 假設模型 artifact 名稱是固定的
        uri = f"runs:/{run_id}/{artifact_path}"
        
        result = mlflow.register_model(model_uri=uri, name=req.model_name)
        return {"status": "success", "model_name": result.name, "version": result.version, "run_id": run_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/promote", summary="提升模型版本階段")
def promote_model(req: PromoteRequest):
    try:
        mlflow_client.transition_model_version_stage(
            name=req.model_name,
            version=req.version,
            stage=req.stage,
            archive_existing_versions=True # 建議設為 True
        )
        return {"status": "success", "message": f"模型 {req.model_name} 版本 {req.version} 已提升至 {req.stage}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", summary="列出所有已註冊的模型")
def list_models():
    try:
        regs = mlflow_client.search_registered_models()
        output = [
            {
                "name": m.name,
                "latest_versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "status": v.status,
                        "run_id": v.run_id,
                        "creation_timestamp": v.creation_timestamp
                    } for v in m.latest_versions
                ]
            } for m in regs
        ]
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete", summary="刪除指定的模型版本")
def delete_model_version(req: DeleteRequest):
    try:
        mlflow_client.delete_model_version(name=req.model_name, version=req.version)
        return {"status": "success", "message": f"已刪除模型 {req.model_name} 的版本 {req.version}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/{model_name}", summary="刪除整個註冊模型")
def delete_registered_model(model_name: str = FastAPIPath(..., description="要刪除的模型名稱")):
    try:
        mlflow_client.delete_registered_model(name=model_name)
        return {"status": "success", "message": f"已刪除註冊模型 '{model_name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/py/train/status/{task_id}", summary="查詢背景訓練任務的狀態")
def get_training_status(task_id: str):
    output_dir = PY_MODEL_DIR / task_id
    model_path = output_dir / "best_model.pkl"
    # (可選) 假設失敗時會產生一個 .log 檔案
    error_log_path = output_dir / "error.log"

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="找不到指定的 Task ID。")

    if error_log_path.exists():
        with open(error_log_path, 'r') as f:
            error_message = f.read()
        return {"status": "failed", "task_id": task_id, "error": error_message}
    
    if model_path.exists():
        return {"status": "completed", "task_id": task_id, "model_path": str(model_path)}
    
    return {"status": "running", "task_id": task_id}