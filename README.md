

# Titanic Survival Prediction - End-to-End MLOps Project

Predicting passenger survival on the Titanic using machine learning with a complete MLOps pipeline including data validation, experiment tracking with MLflow, model monitoring using Evidently AI, and FastAPI deployment.

---

# Quick Start

## Clone Repository

```bash
git clone <repo_url>
cd Mlops_unitendproject
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Complete Training Pipeline

```bash
python src/pipeline/train_pipeline.py
```

This will:

- Ingest Data
- Validate Dataset Schema
- Transform Features
- Train Multiple Models
- Compare Model Performance
- Select the Best Model
- Save Model Artifacts
- Log Experiments to MLflow
- Generate Evidently Reports

---

# Launch MLflow Dashboard

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

Use MLflow to:

- Compare model runs
- View training metrics
- Track parameters
- Review saved artifacts
- Analyze experiment history

---

# Run Monitoring

The Evidently monitoring module is maintained as a separate component to support future scalability and better project architecture.

```bash
python -m src.Monitoring.Run_evidently
```

Generated reports are saved in:

```text
artifacts/reports/
```

Available Reports:

- Classification Performance Report
- Data Drift Report
- Data Summary Report

---

# Run FastAPI Application

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Use the application to:

- Submit passenger details
- Generate survival predictions
- View prediction interface

---

# API Documentation

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Use Swagger to:

- Test prediction endpoints
- Review API schema
- Validate request and response formats

---

# MLOps Monitoring Dashboard

Open:

```text
http://127.0.0.1:8000
```

Use the dashboard to:

- View prediction metrics
- Access generated reports
- Review model performance summaries

---

# Project Structure

## artifacts/

Stores all generated project artifacts and outputs.

### reports/

- classification_report.html → Classification performance report
- data_drift_report.html → Data drift monitoring report
- data_summary_report.html → Dataset summary report

### Files

- data.csv → Raw dataset copy
- train.csv → Training dataset
- test.csv → Testing dataset
- preprocessor.pkl → Saved preprocessing pipeline
- model.pkl → Best performing trained model
- validation_status.txt → Dataset validation results

---

## logs/

Stores pipeline execution logs and debugging information.

---

## mlruns/

Stores MLflow experiment runs, metrics, parameters, and artifacts.

---

## Notebook/

Contains exploratory analysis, preprocessing experiments, and model development notebooks.

### Data/

- titanic.csv → Original Titanic dataset
- titanic_cleaned.csv → Cleaned version used during EDA
- titanic_model_ready.csv → Modeling-ready dataset
- eda_findings.csv → Summary of insights discovered during EDA

### Notebooks

- Titanic_database_EDA.ipynb → Exploratory Data Analysis
- Data_preprocessed.ipynb → Data cleaning and feature engineering
- Model Training.ipynb → Model experimentation and evaluation
- Reference_guide.ipynb → Project notes and references
- Untitled.ipynb → Temporary development notebook

---

## src/components/

Core machine learning pipeline components.

- data_ingestion.py → Loads and splits the dataset
- data_validation.py → Validates required schema and columns
- data_transformation.py → Applies preprocessing and feature engineering
- model_trainer.py → Trains models and selects the best performer

---

## src/pipeline/

Orchestrates end-to-end workflows.

- train_pipeline.py → Runs the complete training pipeline
- predict_pipeline.py → Handles prediction workflow

---

## src/Monitoring/

Monitoring and model evaluation modules.

- Evidently_report.py → Generates Evidently reports
- Run_evidently.py → Executes monitoring workflow

---

## Utility Modules

- logger.py → Centralized logging configuration
- exception.py → Custom exception handling
- utils.py → Shared helper functions and model utilities

---

# Pipeline Workflow

```text
Titanic Dataset
       │
       ▼
Data Ingestion
       │
       ▼
Data Validation
       │
       ▼
Data Transformation
       │
       ▼
Model Training
       │
       ▼
Model Comparison
       │
       ▼
MLflow Tracking
       │
       ▼
Model Saving
       │
       ▼
Evidently Monitoring
       │
       ▼
FastAPI Deployment
```

---

# Models Evaluated

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- AdaBoost
- Extra Trees
- Gradient Boosting

The best model is automatically selected using the F1 Score.

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- MLflow
- Evidently AI
- FastAPI
- Uvicorn
- Dill

---

# Future Improvements

- Docker Containerization
- AWS Deployment
- CI/CD Pipeline
- MLflow Model Registry
- Automated Retraining Pipeline
- Cloud Monitoring Integration

---

# Author

**Gargi Mitra**