import joblib
from pathlib import Path
import fire
import pandas as pd

from src.python.preprocessing.preprocess import load_and_preprocess

# 定位專案根目錄
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def predict(
    model_path: str,
    data_path: str,
    output_path: str = None
) -> list:
    """
    Load a trained model and dataset, then output predictions.

    Args:
        model_path: 絕對或相對於專案根的模型檔案路徑 (.pkl/.joblib)
        data_path: 絕對或相對於專案根的 CSV 檔案路徑
        output_path: 若提供，將 predictions 存為 CSV 至此路徑，否則直接印出列表

    Returns:
        List of predictions
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
    X, _ = load_and_preprocess(str(d_path))  # 忽略真實標籤

    # Predict
    preds = model.predict(X)

    # Output
    if output_path:
        out = Path(output_path)
        if not out.is_absolute():
            out = BASE_DIR / out
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(preds, columns=["prediction"])
        df.to_csv(out, index=False)
        print(f"Predictions saved to {out}")
    else:
        print(preds.tolist())

    return preds.tolist()

if __name__ == "__main__":
    fire.Fire(predict)