# evaluate.py
import joblib
from sklearn.metrics import classification_report
from ..preprocessing.preprocess import load_and_preprocess

def evaluate(model_path: str, data_path: str):
    # 加载模型和数据
    model = joblib.load(model_path)
    X, y = load_and_preprocess(data_path)
    
    # 评估
    preds = model.predict(X)
    report = classification_report(y, preds, output_dict=True)
    return report

if __name__ == "__main__":
    import fire
    fire.Fire(evaluate)
