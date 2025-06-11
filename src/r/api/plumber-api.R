# src/plumber-api.R
library(plumber)
library(jsonlite)
library(randomForest)
library(here)

# Load the model using a robust path
model_path <- here::here("/Users/charleschuang/Desktop/atgenomix/ToolDev/mlops-demo-iris-classifier/models/", "iris_rf_model.rds")
if (!file.exists(model_path)) {
  stop("Model file not found! Make sure you have run 'src/train.R'. Path: ", model_path)
}
model <- readRDS(model_path)

#* @apiTitle Iris Prediction API
#* @apiDescription An API to classify Iris species.

#* Health check endpoint
#* @get /health
function() {
  list(status = "ok", message = "API is running")
}

#* Predict the species of an Iris flower
#* @param req The request object
#* @param res The response object
#* @post /predict
function(req, res) {
  # Parse the JSON body from the request
  input <- tryCatch({
    fromJSON(req$postBody)
  }, error = function(e) {
    res$status <- 400
    return(list(error = "Invalid JSON format."))
  })

  # Return if JSON parsing failed
  if ("error" %in% names(input)) {
    return(input)
  }

  # Check for required variables
  required_vars <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")
  if (!all(required_vars %in% names(input))) {
    missing_vars <- paste(setdiff(required_vars, names(input)), collapse = ", ")
    res$status <- 400
    return(list(error = paste("Missing required variables:", missing_vars)))
  }

  # Perform the prediction
  input_df <- as.data.frame(input)
  preds <- predict(model, newdata = input_df)

  # Return the successful prediction
  list(predictions = as.character(preds))
}

