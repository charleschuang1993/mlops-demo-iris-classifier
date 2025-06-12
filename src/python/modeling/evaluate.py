import mlflow.sklearn
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from src.python.preprocessing.preprocess import load_and_preprocess

def evaluate(
    data_path: str,
    mlflow_model_name: str,
    mlflow_model_stage: str = "Production",
    output_dir: str = "outputs/evaluate"
) -> dict:
    """
    Evaluate a trained model from the MLflow Model Registry on a dataset,
    then save the classification report and confusion matrix as artifacts.
    """
    model_uri = f"models:/{mlflow_model_name}/{mlflow_model_stage}"
    model = mlflow.sklearn.load_model(model_uri)
    X, y = load_and_preprocess(data_path)
    preds = model.predict(X)
    report_dict = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Saved classification report to {report_path}")
    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    labels = sorted(map(str, sorted(set(y))))
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
    print(f"Saved confusion matrix to {cm_path}")
    return report_dict
