# src/train.R
library(mlflow)
library(datasets)
library(randomForest)
Sys.setenv(MLFLOW_BIN = "/Users/charleschuang/.local/bin/mlflow")
mlflow_set_tracking_uri("http://localhost:5001")


# start MLflow experiment
mlflow_start_run()

# read the iris dataset
data(iris)

# train a random forest model
model <- randomForest(Species ~ ., data = iris, ntree = 100)

# predict and calculate accuracy
acc <- sum(predict(model, iris) == iris$Species) / nrow(iris)

# save the model
saveRDS(model, "./mlops-demo-iris-classifier/models/iris_rf_model.rds")

# record parameters, metrics, and artifacts in MLflow
mlflow_log_param("ntree", 100)
mlflow_log_metric("accuracy", acc)
mlflow_log_artifact("./mlops-demo-iris-classifier/models/iris_rf_model.rds")

# end the MLflow run
mlflow_end_run()

