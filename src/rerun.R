library(mlflow)
Sys.setenv(MLFLOW_BIN = "/Users/charleschuang/.local/bin/mlflow")
mlflow_set_tracking_uri("http://localhost:5001")
old_run <- mlflow_get_run("797558b609404e0ea672fb8ecc2b5965")

run_id <- old_run$run_uuid
model_uri <- paste0("runs:/", run_id, "/model")

