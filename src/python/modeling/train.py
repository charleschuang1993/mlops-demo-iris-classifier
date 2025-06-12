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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np

# 導入預處理函式 (假設存在)
# from src.python.preprocessing.preprocess import load_and_preprocess

def train(
    data: str = "iris",
    test_size: float = 0.3,
    random_state: int = 42,
    n_splits: int = 10,
    experiment_name: str = "iris-classification",
    n_estimators: int = None,
    max_depth: int = None
):
    """
    訓練 RandomForestClassifier，並使用 MLflow 的最佳實踐來記錄所有產出。
    此版本不再手動建立或儲存檔案到本地的 'outputs' 資料夾。
    """
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(experiment_name)

    # 資料載入
    if data == "iris":
        iris = load_iris()
        X, y = iris.data, iris.target
        target_names = iris.target_names
    else:
        # 假設您的 load_and_preprocess 返回 X, y
        # X, y = load_and_preprocess(data) 
        # target_names = sorted([str(item) for item in list(set(y))])
        pass # 根據您的需求實現

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    with mlflow.start_run(run_name="rf-training-optimized") as run:
        # 記錄參數
        mlflow.log_params({
            "data": data,
            "test_size": test_size,
            "random_state": random_state,
            "n_splits_cv": n_splits
        })
        
        if n_estimators and max_depth:
            # 如果提供了超參數，直接訓練單一模型
            print(f"使用指定參數進行訓練: n_estimators={n_estimators}, max_depth={max_depth}")
            mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            model.fit(X_train, y_train)
            best_model = model
        else:
            # 否則，執行 GridSearchCV
            print("未提供指定參數，執行 GridSearchCV...")
            model = RandomForestClassifier(random_state=random_state)
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 3, 5]}
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            grid = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy")
            grid.fit(X_train, y_train)
            
            best_model = grid.best_estimator_
            print(f"GridSearchCV 找到的最佳參數: {grid.best_params_}")
            mlflow.log_params(grid.best_params_)

        # 評估模型並記錄指標 (metrics)
        preds = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", accuracy)

        # *** 關鍵修改 1: 直接記錄文字 artifact ***
        # 不再需要先寫入檔案
        report_text = classification_report(y_test, preds, target_names=target_names)
        mlflow.log_text(report_text, "classification_report.txt")

        # *** 關鍵修改 2: 直接記錄 matplotlib 圖表 ***
        # 不再需要呼叫 fig.savefig()
        fig, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, preds)
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        ax.set_xticks(range(len(target_names)))
        ax.set_yticks(range(len(target_names)))
        ax.set_xticklabels(target_names, rotation=90)
        ax.set_yticklabels(target_names)
        ax.set_ylabel("True")
        ax.set_xlabel("Pred")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # *** 關鍵修改 3: 使用 mlflow.sklearn.log_model() 記錄模型 ***
        # 這會自動處理模型的序列化、儲存，並提供額外的 metadata
        # 不再需要手動呼叫 joblib.dump()
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model", # 在 mlflow artifact store 中的子目錄名稱
            registered_model_name=f"{experiment_name}-rf-model" # 可選：直接註冊模型
        )

    print("訓練完成。所有產出已記錄至 MLflow。")
