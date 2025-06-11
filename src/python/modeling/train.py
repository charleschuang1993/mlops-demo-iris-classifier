# src/python/modeling/train.py

import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
from pathlib import Path

# 設定 matplotlib 於無 GUI 環境仍能產生圖檔
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from src.python.preprocessing.preprocess import load_and_preprocess

# 專案根目錄
BASE_DIR = Path(__file__).resolve().parents[3]


def train(
    data: str = "iris",
    test_size: float = 0.3,
    random_state: int = 42,
    n_splits: int = 10,
    experiment_name: str = "iris-classification",
    output_dir: str = "outputs",
    # 允許從 FastAPI 傳入單一模型的超參數
    n_estimators: int | None = None,
    max_depth: int | None = None,
):
    """訓練 RandomForestClassifier。

    - 若 `n_estimators` 與 `max_depth` 皆提供，直接訓練單一模型。
    - 否則使用 GridSearchCV 尋找最佳參數。
    """

    # ---- Ensure MLflow experiment exists ----
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = client.create_experiment(experiment_name)
        print(f"Created new MLflow experiment '{experiment_name}' (id={exp_id})")
    # 將當前 run 綁定到該 experiment
    
    mlflow.sklearn.autolog()

    # 資料載入
    if data == "iris":
        iris = load_iris()
        X, y = iris.data, iris.target
        target_names = iris.target_names
    else:
        X, y = load_and_preprocess(data)
        target_names = sorted(map(str, set(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="rf-training"):
        if n_estimators and max_depth:
            # 直接訓練單一模型
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
            model.fit(X_train, y_train)
            best = model
        else:
            # 使用 GridSearchCV
            model = RandomForestClassifier(random_state=random_state)
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 3, 5],
            }
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            grid = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy")
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            print("GridSearch best params:", grid.best_params_)

        preds = best.predict(X_test)
        report_dict = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)

        # 儲存分類報告
        report_path = out_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, preds))
        mlflow.log_artifact(str(report_path))

        # 混淆矩陣
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
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))

        # 儲存模型
        best_model_path = out_dir / "best_model.pkl"
        joblib.dump(best, best_model_path)
        mlflow.log_artifact(str(best_model_path))

    print(f"Training finished. Artifacts saved to {out_dir}")
    return report_dict
