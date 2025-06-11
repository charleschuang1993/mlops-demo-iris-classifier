import mlflow

mlflow.set_tracking_uri("http://localhost:5001")

run_id = "797558b609404e0ea672fb8ecc2b5965"
model_uri = f"runs:/{run_id}/iris_rf_model.rds"

result = mlflow.register_model(
    model_uri=model_uri,
    name="testRModel_iris_example"
)

print(f"Model registered to {result.name}, version {result.version}")
