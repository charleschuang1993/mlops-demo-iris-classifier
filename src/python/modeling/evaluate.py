import joblib
from pathlib import Path
import json
import fire
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from src.python.preprocessing.preprocess import load_and_preprocess

# 定位專案根目錄
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def evaluate(
    model_path: str,
    data_path: str,
    output_dir: str = "outputs/evaluate"
) -> dict:
    """
    Evaluate a trained model on a dataset and save metrics and confusion matrix.

    Args:
        model_path: 絕對或相對於專案根的模型檔案路徑 (.pkl/.joblib)
        data_path: 絕對或相對於專案根的 CSV 檔案路徑
        output_dir: 儲存報告及圖表的資料夾，預設 'outputs/evaluate'
    Returns:
        report_dict: 分類報告字典格式
    """
    # Resolve paths
    m_path = Path(model_path)
    if not m_path.is_absolute():
        m_path = BASE_DIR / m_path
    d_path = Path(data_path)
    if not d_path.is_absolute():
        d_path = BASE_DIR / d_path

    # Load model and data
    model = joblib.load(m_path)
    X, y = load_and_preprocess(str(d_path))

    # Predict and compute metrics
    preds = model.predict(X)
    report_dict = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds)

    # Prepare output directory
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save classification report as JSON
    report_path = out_dir / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Saved classification report to {report_path}")

    # Plot and save confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    labels = sorted(map(str, sorted(set(y))))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Pred")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    return report_dict

if __name__ == "__main__":
    fire.Fire(evaluate)
