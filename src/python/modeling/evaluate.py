"""Model evaluation utility (built‑in Iris or custom CSV).

Fixes for runtime NameError:
• include local ``_load_data`` helper
• import ``mlflow_sklearn`` before use
• robust fallback to latest model version
"""

from pathlib import Path
import json
from typing import Tuple

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from mlflow import sklearn as mlflow_sklearn, MlflowClient

from src.python.preprocessing.preprocess import load_and_preprocess

client = MlflowClient()

# -----------------------------------------------------------------------------
# Helper: load dataset (built‑in or CSV)
# -----------------------------------------------------------------------------

def _load_data(path: str):
    """Return X, y from *path*.

    * ``"iris"`` → built‑in scikit‑learn dataset
    * *relative CSV* → resolve relative to CWD; if not found, also try relative
      to project root (three levels up from this file).
    * *absolute CSV* → use as‑is.
    """

    # Built‑in dataset ---------------------------------------------------------
    if path == "iris":
        iris = load_iris()
        return iris.data, iris.target

    csv_path = Path(path)

    # Relative path not found?  Try under project root
    if not csv_path.exists() and not csv_path.is_absolute():
        project_root = Path(__file__).resolve().parents[3]
        alt_path = project_root / csv_path
        if alt_path.exists():
            csv_path = alt_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # Use preprocessing utility to turn CSV → X, y
    return load_and_preprocess(str(csv_path))

# -----------------------------------------------------------------------------
# Main evaluate function (used by FastAPI)
# -----------------------------------------------------------------------------

def evaluate(
    data_path: str,
    mlflow_model_name: str,
    mlflow_model_stage: str = "Production",
    output_dir: str = "outputs/evaluate",
):
    """Evaluate a registered model on *data_path* and save artifacts.

    Returns the classification report as a dictionary.
    """

    # 1) Load model with stage→latest fallback
    try:
        model_uri = f"models:/{mlflow_model_name}/{mlflow_model_stage}"
        model = mlflow_sklearn.load_model(model_uri)
    except Exception:
        latest = client.get_latest_versions(mlflow_model_name)[0]
        model = mlflow_sklearn.load_model(f"models:/{mlflow_model_name}/{latest.version}")

    # 2) Load data
    X, y = _load_data(data_path)

    # 3) Predict & metrics
    preds = model.predict(X)
    report = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds)

    # 4) Persist artifacts
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "classification_report.json", "w") as fp:
        json.dump(report, fp, indent=4)

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
    fig.savefig(out / "confusion_matrix.png")

    return report
