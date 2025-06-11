import fire
from pathlib import Path

from src.python.modeling.train import train
from src.python.modeling.evaluate import evaluate

# 定位專案根目錄
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def run_experiment(
    data: str = "iris",
    test_size: float = 0.3,
    random_state: int = 42,
    n_splits: int = 10,
    experiment_name: str = "iris-classification",
    train_output_dir: str = "outputs/train",
    eval_output_dir: str = "outputs/evaluate"
) -> dict:
    """
    一鍵執行完整實驗：訓練 + 評估

    Args:
        data: 'iris' 使用內建資料集，或指定 CSV 檔案路徑（相對/絕對）
        test_size: 測試集比例
        random_state: 隨機種子
        n_splits: CV 分折數
        experiment_name: MLflow 實驗名稱
        train_output_dir: 訓練 artifacts 輸出資料夾
        eval_output_dir: 評估報告輸出資料夾

    Returns:
        dict: 包含 'train_report' 與 'eval_report' 兩份報告
    """
    # 1. 訓練並取得報告
    train_report = train(
        data=data,
        test_size=test_size,
        random_state=random_state,
        n_splits=n_splits,
        experiment_name=experiment_name,
        output_dir=train_output_dir
    )

    # 2. 構造模型路徑 (假設 train 已保存 best_model.pkl)
    model_dir = Path(train_output_dir)
    if not model_dir.is_absolute():
        model_dir = BASE_DIR / model_dir
    model_path = model_dir / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型檔案: {model_path}")

    # 3. 評估並取得報告
    eval_report = evaluate(
        model_path=str(model_path),
        data_path=data,
        output_dir=eval_output_dir
    )

    return {"train_report": train_report, "eval_report": eval_report}

if __name__ == "__main__":
    fire.Fire(run_experiment)
