# run_experiment.py
import click
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import joblib
from ..preprocessing.preprocess import load_and_preprocess

def train(data_path: str, model_output_path: str, n_estimators=100, max_depth=5):
    # 1. 载入并预处理数据
    X, y = load_and_preprocess(data_path)

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # 3. GridSearch + CV
    param_grid = {"n_estimators":[n_estimators], "max_depth":[max_depth]}
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid = GridSearchCV(RandomForestClassifier(random_state=42),
                        param_grid, cv=cv, scoring="accuracy")
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    # 4. 保存模型
    joblib.dump(best, model_output_path)
    return {"best_params": grid.best_params_, "score": grid.best_score_}

# 如果你还想保留 cli 调用
if __name__ == "__main__":
    import fire
    fire.Fire(train)
