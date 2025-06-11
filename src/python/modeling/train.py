import mlflow
import mlflow.sklearn
from pathlib import Path
import matplotlib.pyplot as plt
import fire

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 定位專案根目錄
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def train(
    data: str = "iris",                  # 使用 'iris' 或 指定 CSV 路徑
    test_size: float = 0.3,
    random_state: int = 42,
    n_splits: int = 10,
    experiment_name: str = "iris-classification",
    output_dir: str = "outputs"
):
    """
    訓練 RandomForestClassifier 並使用 MLflow 自動記錄。

    Args:
        data: 'iris' 使用 sklearn 內建資料，或是相對/絕對路徑至 CSV 檔案。
        output_dir: 輸出 artifacts 的資料夾，相對於專案根目錄。
    Returns:
        report_dict: 分類報告字典格式
    """
    # 設定實驗
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()

    # 資料載入
    if data == "iris":
        iris = load_iris()
        X, y = iris.data, iris.target
        target_names = iris.target_names
    else:
        from src.python.preprocessing.preprocess import load_and_preprocess
        X, y = load_and_preprocess(data)
        target_names = sorted(set(y))

    # 切分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 模型與參數
    model = RandomForestClassifier(random_state=random_state)
    param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 3, 5]}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy")

    with mlflow.start_run(run_name="rf-gridsearch"):
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        preds = best.predict(X_test)
        report_dict = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)

        # 建立輸出目錄
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = BASE_DIR / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # 儲存分類報告
        report_path = out_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, preds))
        mlflow.log_artifact(str(report_path))

        # 繪製混淆矩陣
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
        mlflow.log_artifact(str(cm_path))

    print(f"Training complete. Artifacts saved to {out_dir}")
    return report_dict

if __name__ == "__main__":
    fire.Fire(train)
