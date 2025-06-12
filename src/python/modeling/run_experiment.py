from src.python.modeling.train import train
from src.python.modeling.evaluate import evaluate

def run_experiment(
    data: str = "iris",
    test_size: float = 0.3,
    random_state: int = 42,
    n_splits: int = 10,
    experiment_name: str = "iris-classification"
) -> dict:
    """
    Run a full experiment: train and then evaluate the model.
    All model management is handled by MLflow, not by local disk files.

    Args:
        data (str): "iris" for built-in dataset or a path to CSV.
        test_size (float): Proportion for the test set.
        random_state (int): Random seed.
        n_splits (int): Number of folds for cross-validation.
        experiment_name (str): MLflow experiment name.

    Returns:
        dict: Contains "train_report" and "eval_report".
    """
    # Train the model; train() should log the model to MLflow and return the run_id or relevant metadata.
    train_result = train(
        data=data,
        test_size=test_size,
        random_state=random_state,
        n_splits=n_splits,
        experiment_name=experiment_name
    )

    # Optionally, train_result could be a dict with MLflow run_id and registered model name.
    run_id = train_result.get("run_id")
    registered_model_name = train_result.get("registered_model_name")

    # Evaluate using model reference in MLflow (run_id or registered_model_name)
    eval_report = evaluate(
        data=data,
        mlflow_run_id=run_id,
        mlflow_model_name=registered_model_name,
        experiment_name=experiment_name
    )

    return {
        "train_report": train_result,
        "eval_report": eval_report
    }

# No CLI block
