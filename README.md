# mlops-demo-iris-classifier

> A multi-language (R & Python) MLOps demonstration project illustrating data preprocessing, model training, hyperparameter tuning, evaluation, MLflow experiment management, API deployment (Plumber & FastAPI), containerization, and CI/CD.

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Prerequisites](#prerequisites)
3. [Data Preparation](#data-preparation)
4. [MLflow Experiments](#mlflow-experiments)
5. [Pipelines](#pipelines)

   * [Python Pipeline](#python-pipeline)
   * [R Pipeline](#r-pipeline)
6. [API Deployment](#api-deployment)
7. [Containerization](#containerization)
8. [Contributing](#contributing)
9. [License](#license)

---

## Directory Structure

```text
mlops-demo-iris-classifier/
├── mlops-demo-iris-classifier-py-venv/   # Python virtual environment (gitignored)
├── data/                                  # Raw or processed data
│   └── iris.csv
├── mlflow-experiments/                    # MLflow project definitions
│   ├── r_iris/                            # R experiments
│   │   ├── MLproject
│   │   └── renv.lock
│   └── python_iris/                       # Python experiments
│       ├── MLproject
│       └── conda.yaml
├── src/
│   ├── common/                            # Shared configs and utilities
│   │   ├── config.yaml
│   │   └── utils/
│   │       ├── io.R
│   │       └── io.py
│   ├── r/                                 # R pipelines and API
│   │   ├── preprocessing/
│   │   │   └── preprocess.R
│   │   ├── modeling/
│   │   │   ├── train.R
│   │   │   └── evaluate.R
│   │   └── api/
│   │       ├── plumber-api.R
│   │       └── start-plumber.R
│   └── python/                            # Python pipelines and API
│       ├── preprocessing/
│       │   └── preprocess.py
│       ├── modeling/
│       │   ├── run_experiment.py
│       │   └── evaluate.py
│       └── api/
│           └── fastapi_main.py
├── models/                                # Trained model artifacts
│   ├── r/
│   │   └── iris_rf_model.rds
│   └── python/
│       └── iris_rf_model.pkl
├── docker/                                # Dockerfiles for each service
│   ├── Dockerfile.r                       # R + Plumber
│   └── Dockerfile.py                      # Python + FastAPI
├── deploy/                                # Deployment scripts and CI/CD templates
├── .gitignore                             # Exclude py-venv, MLflow artifacts, etc.
└── README.md                              # Project overview
```

---

## Prerequisites

### Python

```bash
# Activate the existing virtual environment
source mlops-demo-iris-classifier-py-venv/bin/activate

# Install Python dependencies
pip install -r src/python/modeling/requirements.txt
```

### R

```r
# In R console
install.packages("renv")
renv::restore()
```

---

## Data Preparation

1. Place the raw `iris.csv` file in the `data/` directory.
2. Update `src/common/config.yaml` to configure paths, MLflow experiment names, and default hyperparameters.

---

## MLflow Experiments

### Python Experiment

```bash
cd mlflow-experiments/python_iris
mlflow run .
# Override parameters:
mlflow run . -P n_estimators=200 -P max_depth=3
```

### R Experiment

```bash
cd mlflow-experiments/r_iris
mlflow run .
```

Launch the Tracking UI:

```bash
mlflow ui
```

Access at [http://localhost:5000](http://localhost:5000) to compare runs, view metrics, parameters, and artifacts (e.g., confusion matrix, ROC curves).

---

## Pipelines

### Python Pipeline

```bash
# 1. Data preprocessing
python src/python/preprocessing/preprocess.py \
  --input data/iris.csv \
  --output data/iris_processed.pkl

# 2. Model training and logging
python src/python/modeling/run_experiment.py \
  --data data/iris_processed.pkl

# 3. Model evaluation
python src/python/modeling/evaluate.py \
  --model models/python/iris_rf_model.pkl \
  --data data/iris_processed.pkl
```

### R Pipeline

```bash
# 1. Data preprocessing
Rscript src/r/preprocessing/preprocess.R \
  data/iris.csv data/iris_processed.rds

# 2. Model training and logging
Rscript src/r/modeling/train.R \
  data/iris_processed.rds models/r/iris_rf_model.rds

# 3. Model evaluation
Rscript src/r/modeling/evaluate.R \
  models/r/iris_rf_model.rds data/iris_processed.rds
```

---

## API Deployment

### FastAPI (Python)

```bash
uvicorn src/python/api/fastapi_main:app \
  --host 0.0.0.0 --port 8000
```

### Plumber (R)

```bash
Rscript src/r/api/start-plumber.R
```

---

## Containerization

### Python + FastAPI

```bash
docker build -f docker/Dockerfile.py -t iris-fastapi:latest .
```

### R + Plumber

```bash
docker build -f docker/Dockerfile.r -t iris-plumber:latest .
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. Ensure all scripts pass local tests and follow the project coding standards.

---

## License

This project is licensed under the [MIT License](LICENSE).
