# src/plumber-api.R
library(plumber)
library(jsonlite)
library(randomForest)

# 載入模型
model <- readRDS("models/iris_rf_model.rds")

#* 健康檢查 endpoint
#* @get /health
function() {
  list(status = "ok", message = "API is running")
}

#* 預測分類結果
#* @post /predict
#* @param req:req
function(req) {
  # 解析 JSON body
  input <- tryCatch({
    fromJSON(req$postBody)
  }, error = function(e) {
    return(list(error = "Invalid JSON format"))
  })
  
  # 檢查資料格式
  required_vars <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")
  if (!all(required_vars %in% names(input))) {
    return(list(error = paste("Missing required variables:", paste(setdiff(required_vars, names(input)), collapse=", "))))
  }

  # 預測
  input_df <- as.data.frame(input)
  preds <- predict(model, newdata = input_df)
  list(predictions = as.character(preds))
}
