import pandas as pd
from pathlib import Path
import pickle
import fire

# Set the project root directory based on the current file's location
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


def load_and_preprocess(data_path: str):
    """
    載入 CSV 檔案，將其分割為特徵和標籤。
    標籤會被轉換為整數編碼。
    """
    path = Path(data_path)
    # ... (處理相對路徑的邏輯)
    if not path.is_absolute():
        # 確保 BASE_DIR 指向正確的專案根目錄
        BASE_DIR = Path(__file__).resolve().parents[3] 
        path = BASE_DIR / path

    df = pd.read_csv(path)
    X = df.drop(columns=["species"])
    y_str = df["species"]

    # 使用 factorize 將字串標籤轉換為整數 (e.g., "setosa" -> 0)
    # sort=True 確保映射順序與 load_iris() 一致 (按字母順序)
    y_int, _ = pd.factorize(y_str, sort=True)

    return X.values, y_int

def save_processed(
    data_path: str = "data/iris.csv",
    output_path: str = "data/iris_processed.pkl"
):
    """
    Loads data, preprocesses it, and saves the (X, y) tuple as a pickle file.
    Args:
        data_path (str): Path to the input CSV file.
        output_path (str): Path to save the processed pickle file.
    """
    X, y = load_and_preprocess(data_path)
    out = Path(output_path)
    # Convert to absolute path if needed
    if not out.is_absolute():
        out = BASE_DIR / out
    # Ensure the output directory exists
    out.parent.mkdir(parents=True, exist_ok=True)
    # Save processed data as a pickle file
    with open(out, "wb") as f:
        pickle.dump((X, y), f)
    print(f"Processed data saved to {out}")

if __name__ == "__main__":
    # Allows the function to be called from the command line
    fire.Fire(save_processed)
