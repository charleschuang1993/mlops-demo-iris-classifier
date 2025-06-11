# src/python/modeling/train.py

import mlflow
import mlflow.sklearn
from pathlib import Path

# 導入 matplotlib 並在導入 pyplot 之前設定後端
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 導入預處理函式
from src.python.preprocessing.preprocess import load_and_preprocess

# 定義專案根目錄
BASE_DIR = Path(__file__).resolve().parents[3]

def train(
    data: str = "iris",
    test_size: float = 0.3,
    random_state: int = 42,
    n_splits: int = 10,
    experiment_name: str = "iris-classification",
    output_dir: str = "outputs",
    n_estimators: int = None,
    max_depth: int = None
):
    """
    訓練 RandomForestClassifier。
    如果提供了 n_estimators 和 max_depth，則訓練單一模型。
    否則，執行 GridSearchCV 尋找最佳模型。
    """
    # 在函式內部設定 Tracking URI，確保背景線程能正確連線
    mlflow.set_tracking_uri("http://localhost:5001")

    # --- 關鍵修正：採用更健壯的方式設定實驗 ---
    # 1. 嘗試按名稱獲取實驗
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        # 2. 如果實驗不存在，則創建它
        print(f"實驗 '{experiment_name}' 不存在，正在創建...")
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        # 3. 如果實驗已存在，則獲取其 ID
        experiment_id = experiment.experiment_id
    print(f"將在此實驗中記錄 Run: ID='{experiment_id}', Name='{experiment_name}'")
    # --- 修正結束 ---


    # 資料載入
    if data == "iris":
        iris = load_iris()
        X, y = iris.data, iris.target
        target_names = iris.target_names
    else:
        X, y = load_and_preprocess(data)
        target_names = sorted([str(item) for item in list(set(y))])

    # 資料切分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 建立輸出目錄
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 關鍵修正：明確指定 experiment_id 來啟動 Run ---
    with mlflow.start_run(experiment_id=experiment_id, run_name="rf-training") as run:
        # 將 autolog() 移至 run 的上下文內，確保記錄到正確的 run
        mlflow.sklearn.autolog()

        if n_estimators and max_depth:
            # 如果提供了超參數，直接訓練單一模型
            print(f"使用指定參數進行訓練: n_estimators={n_estimators}, max_depth={max_depth}")
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            model.fit(X_train, y_train)
            best = model
        else:
            # 否則，執行 GridSearchCV
            print("未提供指定參數，執行 GridSearchCV...")
            model = RandomForestClassifier(random_state=random_state)
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 3, 5]}
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            grid = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy")
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            print(f"GridSearchCV 找到的最佳參數: {grid.best_params_}")

        # 後續的報告、繪圖、儲存邏輯不變
        preds = best.predict(X_test)
        report_dict = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)

        # 儲存分類報告
        report_path = out_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, preds))
        mlflow.log_artifact(str(report_path))

        # 繪製並儲存混淆矩陣
        fig, ax = plt.subplots(figsize=(6, 4))
        cax = ax.matshow(cm)
        fig.colorbar(cax)
        ax.set_xticks(range(len(target_names)))
        ax.set_yticks(range(len(target_names)))
        ax.set_xticklabels(target_names, rotation=90)
        ax.set_yticklabels(target_names)
        ax.set_ylabel("True")
        ax.set_xlabel("Pred")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        cm_path = out_dir / "confusion_matrix.png"
        fig.savefig(cm_path)
        plt.close(fig) # 關閉圖像以釋放記憶體
        mlflow.log_artifact(str(cm_path))

        # 儲存最佳模型
        best_model_path = out_dir / "best_model.pkl"
        joblib.dump(best, best_model_path)
        # 將模型也記錄到 MLflow artifacts
        mlflow.log_artifact(str(best_model_path))

    print(f"訓練完成。 產出已儲存至 {out_dir}")
    return report_dict
