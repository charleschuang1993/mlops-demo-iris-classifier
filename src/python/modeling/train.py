import mlflow
import mlflow.sklearn
from pathlib import Path

# Set the matplotlib backend to avoid issues on headless servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np

# Import your data preprocessing function if needed
# from src.python.preprocessing.preprocess import load_and_preprocess

def train(
    data: str = "iris",
    test_size: float = 0.3,
    random_state: int = 42,
    n_splits: int = 10,
    experiment_name: str = "iris-classification",
    n_estimators: int = None,
    max_depth: int = None
):
    """
    Trains a RandomForestClassifier and logs outputs to MLflow.
    Parameters can be customized, and model selection uses GridSearchCV unless parameters are explicitly provided.
    """
    # Set up MLflow tracking server
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(experiment_name)

    # Load data
    if data == "iris":
        iris = load_iris()
        X, y = iris.data, iris.target
        target_names = iris.target_names
    else:
        # You must implement your custom data loading function here
        # Example:
        # X, y = load_and_preprocess(data) 
        # target_names = sorted([str(item) for item in set(y)])
        raise NotImplementedError("Custom data loading is not implemented.")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    with mlflow.start_run(run_name="rf-training-optimized") as run:
        # Log experiment parameters
        mlflow.log_params({
            "data": data,
            "test_size": test_size,
            "random_state": random_state,
            "n_splits_cv": n_splits
        })
        
        # Model selection and training
        if n_estimators and max_depth:
            print(f"Training with user-specified parameters: n_estimators={n_estimators}, max_depth={max_depth}")
            mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            model.fit(X_train, y_train)
            best_model = model
        else:
            print("No specific parameters given, using GridSearchCV for tuning...")
            model = RandomForestClassifier(random_state=random_state)
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 3, 5]}
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            grid = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy")
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            print(f"Best parameters found by GridSearchCV: {grid.best_params_}")
            mlflow.log_params(grid.best_params_)

        # Evaluate and log metrics
        preds = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", accuracy)

        # Log the classification report as a text artifact
        report_text = classification_report(y_test, preds, target_names=target_names)
        mlflow.log_text(report_text, "classification_report.txt")

        # Log confusion matrix as a PNG image
        fig, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, preds)
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        ax.set_xticks(range(len(target_names)))
        ax.set_yticks(range(len(target_names)))
        ax.set_xticklabels(target_names, rotation=90)
        ax.set_yticklabels(target_names)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Save the trained model using MLflow's sklearn API
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=f"{experiment_name}-rf-model"
        )

    print("Training finished. All outputs have been logged to MLflow.")

