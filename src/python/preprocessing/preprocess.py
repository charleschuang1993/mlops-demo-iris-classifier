# preprocess.py
import pandas as pd

def load_and_preprocess(data_path: str):
    df = pd.read_csv(data_path)
    # 举例：只做简单分割
    X = df.drop(columns=["species"])
    y = df["species"]
    return X.values, y.values

if __name__ == "__main__":
    import fire
    import pickle
    X, y = load_and_preprocess("data/iris.csv")
    with open("data/iris_processed.pkl", "wb") as f:
        pickle.dump((X,y), f)
