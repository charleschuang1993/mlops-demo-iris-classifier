# src/plumber-api.R
library(plumber)
library(jsonlite)
library(randomForest)
library(here) # 建議加入 here 套件

# 使用 here 建立穩健的路徑
model_path <- here::here("./models", "iris_rf_model.rds")
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
#* @post /predict
#* @param req:req
#* @param res:res
function(req, res) { # <-- 加入 res 參數
  # 解析 JSON body
  input <- tryCatch({
    fromJSON(req$postBody)
  }, error = function(e) {
    # 設定 HTTP 狀態碼為 400，並回傳錯誤訊息
    res$status <- 400 
    return(list(error = "Invalid JSON format."))
  })
  
  # 如果 tryCatch 發生錯誤，input 會是帶有 error 的 list，直接回傳
  if ("error" %in% names(input)) {
    return(input)
  }
  
  # 檢查資料格式
  required_vars <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")
  if (!all(required_vars %in% names(input))) {
    missing_vars <- paste(setdiff(required_vars, names(input)), collapse=", ")
    # 設定 HTTP 狀態碼為 400
    res$status <- 400
    return(list(error = paste("Missing required variables:", missing_vars)))
  }

  # 預測
  input_df <- as.data.frame(input)
  preds <- predict(model, newdata = input_df)
  
  # 成功，回傳 200 OK (預設)
  list(predictions = as.character(preds))
}
