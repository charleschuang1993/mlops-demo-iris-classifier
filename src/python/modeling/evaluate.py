"""Model evaluation utility.

Loads a model from MLflow Model Registry, evaluates it on either the built-in
Iris dataset **or** a custom CSV (via `load_and_preprocess`), and stores the
classification report & confusion matrix under `outputs/evaluate`.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix

from src.python.preprocessing.preprocess import load_and_preprocess


def _load_data(data_path: str):
    """Return (X, y) for either the built-in Iris dataset or a CSV path."""

    if data_path == "iris":  # built-in dataset keyword
        iris = load_iris()
        X, y = iris.data, iris.target
    else:  # assume CSV path (relative or absolute)
        X, y = load_and_preprocess(data_path)
    return X, y


def evaluate(
    data_path: str,
    mlflow_model_name: str,
    mlflow_model_stage: str = "Production",
    output_dir: str = "outputs/evaluate",
) -> dict:
    """Evaluate an MLflow-registered model and save artifacts.

    Args:
        data_path: Either the literal string ``"iris"`` (use scikit-learn's
            built-in Iris dataset) **or** a CSV file path.
        mlflow_model_name: Name of the registered model in MLflow Model
            Registry.
        mlflow_model_stage: Desired stage (e.g. ``"Production"``, ``"Staging"``)
            or a specific version number.
        output_dir: Directory to save evaluation artifacts.

    Returns:
        dict: Sklearn classification report (``output_dict=True``).
    """

    # ------------------------------------------------------------------
    # 1. Load model from MLflow Model Registry
    # ------------------------------------------------------------------
    model_uri = f"models:/{mlflow_model_name}/{mlflow_model_stage}"
    model = mlflow.sklearn.load_model(model_uri)

    # ------------------------------------------------------------------
    # 2. Load evaluation data
    # ------------------------------------------------------------------
    X, y = _load_data(data_path)

    # ------------------------------------------------------------------
    # 3. Predict & compute metrics
    # ------------------------------------------------------------------
    preds = model.predict(X)
    report_dict = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds)

    # ------------------------------------------------------------------
    # 4. Persist artifacts
    # ------------------------------------------------------------------
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4a. Classification report (JSON)
    report_path = out_dir / "classification_report.json"
    with open(report_path, "w") as fp:
        json.dump(report_dict, fp, indent=4)
    print(f"Saved classification report → {report_path}")

    # 4b. Confusion matrix (PNG)
    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    labels = sorted(map(str, set(y)))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path)
    print(f"Saved confusion matrix → {cm_path}")

    return report_dict