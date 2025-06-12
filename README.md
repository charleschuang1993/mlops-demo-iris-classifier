# mlops-demo-iris-classifier

> **A modern, multi-language MLOps teaching demo.**
> This project demonstrates end-to-end machine learning workflow and best practices—including data preparation, training, hyperparameter tuning, experiment tracking (MLflow), evaluation, API deployment (R & Python), containerization, and CI/CD.
> **Model management is handled exclusively by MLflow.** All predictions and evaluations are performed using the models tracked and versioned within MLflow.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Prerequisites](#prerequisites)
4. [Data Preparation](#data-preparation)
5. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
6. [End-to-End Pipelines](#end-to-end-pipelines)
7. [API Deployment](#api-deployment)
8. [Containerization](#containerization)
9. [Contributing](#contributing)
10. [License](#license)

---

## Project Overview

This repository provides a hands-on MLOps learning platform.
You'll learn how to:

* Organize an ML project for both R and Python.
* Preprocess data and track data versions.
* Train and tune models with MLflow experiments.
* Register, deploy, and query models via MLflow—**no local model files required**.
* Serve predictions with FastAPI (Python) and Plumber (R).
* Build and deploy with Docker, ready for CI/CD.

---

## Directory Structure

```text
mlops-demo-iris-classifier/
├── mlops-demo-iris-classifier-py-venv/    # Python virtual environment (gitignored)
├── data/                                  # Raw and processed data (e.g. iris.csv)
├── mlflow-experiments/                    # MLflow project definitions for both languages
│   ├── r_iris/
│   └── python_iris/
├── src/
│   ├── common/                            # Shared configs/utilities
│   ├── r/                                 # R pipelines and APIs
│   └── python/                            # Python pipelines and APIs
├── models/                                # (Legacy) model outputs (not used in MLflow-only workflow)
├── docker/                                # Dockerfiles for deployment
├── deploy/                                # CI/CD templates and scripts
├── .gitignore
└── README.md
```

---

## Prerequisites

### Python

```bash
# Activate the virtual environment
source mlops-demo-iris-classifier-py-venv/bin/activate

# Install Python dependencies
pip install -r src/python/modeling/requirements.txt
```

### R

```r
# In the R console
install.packages("renv")
renv::restore()
```

---

## Data Preparation

1. Download or copy `iris.csv` to the `data/` directory.
2. Edit `src/common/config.yaml` as needed (set data paths, MLflow experiment name, and hyperparameters).

---

## MLflow Experiment Tracking

### What is `mlflow-experiments/`?

* The `mlflow-experiments/` folder is **not generated automatically**.
* You create it manually to organize your MLflow Project definitions (for both Python and R pipelines).
* Each subfolder (like `python_iris/` or `r_iris/`) holds an `MLproject` file and all the code/environment definitions needed to launch a full experiment with `mlflow run`.

### Launch the MLflow Tracking Server

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlruns --host 0.0.0.0 --port 5001
```

Visit [http://localhost:5001](http://localhost:5001) to explore experiments, runs, parameters, metrics, and all model artifacts.

---

### Running Experiments

#### Python

```bash
cd mlflow-experiments/python_iris
mlflow run .
# You can override parameters, e.g.:
mlflow run . -P n_estimators=200 -P max_depth=3
```

#### R

```bash
cd mlflow-experiments/r_iris
mlflow run .
```

---

## End-to-End Pipelines (Python Example)

### 1. Data Preprocessing

```bash
python src/python/preprocessing/preprocess.py \
  --input data/iris.csv \
  --output data/iris_processed.pkl
```

### 2. Model Training (MLflow only, no local model saved)

```bash
python src/python/modeling/run_experiment.py \
  --data data/iris_processed.pkl \
  --experiment_name iris-classification
```

* The trained model will be tracked, versioned, and stored by MLflow (in `mlruns/`), not as a `.pkl` file.

### 3. Model Evaluation

```bash
python src/python/modeling/evaluate.py \
  --data data/iris_processed.pkl \
  --mlflow_model_name iris-classification-rf-model \
  --mlflow_model_stage Production
```

* The model is loaded from the MLflow Model Registry.
* All metrics and plots are saved to the evaluation output folder.

### 4. Model Prediction

```bash
python src/python/modeling/predict.py \
  --data_path data/iris_predict.csv \
  --mlflow_model_name iris-classification-rf-model \
  --mlflow_model_stage Production
```

---

## API Deployment

### FastAPI (Python)

```bash
uvicorn src.python.api.fastapi_main:app --host 0.0.0.0 --port 8080 --reload
```

* The API loads models directly from MLflow, **never from disk**.

### Plumber (R)

```bash
Rscript src/r/api/start-plumber.R
```

---

## Containerization

### Build FastAPI Docker Image

```bash
docker build -f docker/Dockerfile.py -t iris-fastapi:latest .
```

### Build Plumber Docker Image

```bash
docker build -f docker/Dockerfile.r -t iris-plumber:latest .
```

---

## Contributing

Contributions, issues, and feature requests are welcome!
Please open an issue or submit a pull request.
Follow code style guidelines and test your changes before submission.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

**Key teaching note:**

> In this repo, *all model management is handled by MLflow*—models are only ever loaded, versioned, and used via the MLflow tracking and model registry system. This ensures full experiment reproducibility, eliminates model version confusion, and prepares your pipeline for modern MLOps and production workflows.
