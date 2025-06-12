import pandas as pd
from pathlib import Path
import pickle
import fire

# Set the project root directory based on the current file's location
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def load_and_preprocess(data_path: str):
    """
    Loads a CSV file, splits it into features and labels.
    Args:
        data_path (str): Path to the CSV file (absolute or relative).
    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label array.
    """
    path = Path(data_path)
    # Convert to absolute path if needed
    if not path.is_absolute():
        path = BASE_DIR / path
    # Read CSV file
    df = pd.read_csv(path)
    # Separate features and label
    X = df.drop(columns=["species"])
    y = df["species"]
    return X.values, y.values

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
