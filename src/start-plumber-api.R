pr <- plumber::plumb('/Users/charleschuang/Desktop/atgenomix/ToolDev/mlops-demo-iris-classifier/src/plumber-api.R'); pr$run(host='0.0.0.0', port=8000)


# bash: 
# Running plumber API at http://0.0.0.0:8000
# Running swagger Docs at http://127.0.0.1:8000/__docs__/
# Error reporting has been turned off by default. See `pr_set_debug()` for more details.
# To disable this message, set a debug value.
# This message is displayed once per session.

# charleschuang@zhuangchengxundeMacBook-Pro mlops-demo-iris-classifier % curl -X POST http://localhost:8000/predict \
#   -H "Content-Type: application/json" \
#   -d '{"Sepal.Length": 5.1, "Sepal.Width": 3.5, "Petal.Length": 1.4, "Petal.Width": 0.2}'

# {"predictions":["setosa"]}%   ## successful prediction

# Swagger UI:
# http://127.0.0.1:8000/__docs__/
