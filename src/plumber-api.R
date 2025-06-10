# src/plumber-api.R
library(plumber)
library(jsonlite)
library(randomForest)
library(here)

# ... (載入模型的程式碼不變) ...
model_path <- here::here("models", "iris_rf_model.rds")
if (!file.exists(model_path)) {
  stop("Model file not found at: ", model_path)
}
model <- readRDS(model_path)

#* 健康檢查 endpoint
#* @get /health
function() {
  list(status = "ok", message = "API is running")
}

#* 預測分類結果
#* @param req The request object
#* @param res The response object
#* @post /predict
function(req, res) { # <-- function簽名不變
  # ... (函式內部的程式碼完全不變) ...
  # 解析 JSON body
  input <- tryCatch({
    fromJSON(req$postBody)
  }, error = function(e) {
    res$status <- 400 
    return(list(error = "Invalid JSON format."))
  })
  
  if ("error" %in% names(input)) {
    return(input)
  }
  
  # 檢查資料格式
  required_vars <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")
  if (!all(required_vars %in% names(input))) {
    missing_vars <- paste(setdiff(required_vars, names(input)), collapse=", ")
    res$status <- 400
    return(list(error = paste("Missing required variables:", missing_vars)))
  }

  # 預測
  input_df <- as.data.frame(input)
  preds <- predict(model, newdata = input_df)
  
  list(predictions = as.character(preds))
}
