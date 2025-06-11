import pandas as pd
from pathlib import Path
import pickle
import fire

# Define project root based on this file's location
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def load_and_preprocess(data_path: str):
    """
    Load a CSV dataset and split into features and labels.
    data_path can be an absolute path or relative to the project root.
    Returns:
        X: numpy.ndarray of feature values
        y: numpy.ndarray of labels
    """
    path = Path(data_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    df = pd.read_csv(path)
    X = df.drop(columns=["species"])
    y = df["species"]
    return X.values, y.values


def save_processed(
    data_path: str = "data/iris.csv",
    output_path: str = "data/iris_processed.pkl"
):
    """
    Load and preprocess data, then save (X, y) tuples to a pickle file.
    Both data_path and output_path can be absolute or relative to project root.
    """
    X, y = load_and_preprocess(data_path)
    out = Path(output_path)
    if not out.is_absolute():
        out = BASE_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump((X, y), f)
    print(f"Processed data saved to {out}")


if __name__ == "__main__":
    fire.Fire(save_processed)
