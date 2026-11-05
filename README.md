
# 📊 Titanic Survival Prediction MLOps Project

## Project Summary

This project is an end-to-end Machine Learning Operations (MLOps) application built using the Titanic Survival Prediction dataset. The solution covers the complete machine learning lifecycle including data ingestion, data transformation, model training, model inference, API deployment, monitoring, and reporting.

The application predicts whether a Titanic passenger would survive based on passenger information such as:

* Passenger Class
* Gender
* Age
* Number of Siblings/Spouses
* Number of Parents/Children
* Fare
* Port of Embarkation

The project is deployed using FastAPI and includes a monitoring dashboard for model evaluation and data quality analysis.

---

# 🚀 Key Features

### Machine Learning Pipeline

* Automated Data Ingestion
* Data Cleaning and Preprocessing
* Feature Engineering
* Model Training
* Model Selection
* Model Serialization using Dill

### Prediction System

* FastAPI REST API
* Interactive Web Interface
* Real-Time Predictions
* Probability-Based Outputs

### Monitoring & Reporting

* Classification Performance Report
* Data Drift Monitoring
* Data Summary Reporting
* MLOps Monitoring Dashboard

---

# 🏗️ System Architecture

                    ┌────────────────────┐

                    │   Raw Titanic CSV  │

                    └─────────┬──────────┘

                              │

                              ▼

                    ┌────────────────────┐

                    │ Data Ingestion     │

                    └─────────┬──────────┘

                              │

                              ▼

                    ┌────────────────────┐

                    │ Data Transformation│

                    │ Encoding           │

                    │ Missing Values     │

                    │ Scaling            │

                    └─────────┬──────────┘

                              │

                              ▼

                    ┌────────────────────┐

                    │ Model Training     │

                    │ Multiple Models    │

                    │ F1 Evaluation      │

                    └─────────┬──────────┘

                              │

                              ▼

                    ┌────────────────────┐

                    │ Best Model Saved   │

                    │ model.pkl          │

                    └─────────┬──────────┘

                              │

                              ▼

                    ┌────────────────────┐

                    │ Predict Pipeline   │

                    └─────────┬──────────┘

                              │

                   ┌──────────┴──────────┐

                   ▼                     ▼

          ┌────────────────┐   ┌─────────────────┐

          │ FastAPI API    │   │ Monitoring      │

          │ Prediction UI  │   │ Dashboard       │

          └────────────────┘   └─────────────────┘

                                        │

                                        ▼

                       ┌────────────────────────────┐

                       │ Classification Report      │

                       │ Data Summary Report        │

                       │ Data Drift Report          │

                       └────────────────────────────┘

---

# 📁 Project Structure

Titanic_MLOps_Project

│

├── artifacts

│   ├── model.pkl

│   ├── preprocessor.pkl

│   ├── train.csv

│   ├── test.csv

│   └── reports

│       ├── classification_report.html

│       ├── data_summary_report.html

│       └── data_drift_report.html

│

├── notebooks

│

├── src

│   │

│   ├── components

│   │   ├── data_ingestion.py

│   │   ├── data_transformation.py

│   │   └── model_trainer.py

│   │

│   ├── pipeline

│   │   └── predict_pipeline.py

│   │

│   ├── Monitoring

│   │   ├── Evidently_report.py

│   │   └── Run_evidently.py

│   │

│   ├── logger.py

│   ├── exception.py

│   └── utils.py

│

├── templates

│   ├── index.html

│   └── Evidently_Dashboard.html

│

├── app.py

├── requirements.txt

└── README.md

---

# 🤖 Models Evaluated

The following machine learning models were evaluated:

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* AdaBoost
* Extra Trees Classifier
* Gradient Boosting Classifier

The best-performing model is automatically selected based on F1 Score.

---

# 📈 Monitoring Dashboard

The project includes an MLOps monitoring system that generates:

### Classification Report

Tracks:

* Accuracy
* Precision
* Recall
* F1 Score

### Data Summary Report

Provides:

* Dataset Statistics
* Missing Value Analysis
* Feature Distribution Analysis

### Data Drift Report

Compares:

* Training Dataset
* Testing Dataset

To identify:

* Distribution Changes
* Feature Drift
* Data Quality Issues

---

# 🛠️ Technology Stack

### Backend

* Python
* FastAPI
* Uvicorn

### Machine Learning

* Scikit-Learn
* Pandas
* NumPy

### Monitoring

* Evidently AI

### Frontend

* HTML
* CSS
* JavaScript

### Model Serialization

* Dill

---

# 🎯 Business Value

This project demonstrates:

* End-to-End Machine Learning Pipeline Development
* MLOps Best Practices
* API Deployment
* Model Monitoring
* Data Drift Detection
* Real-Time Prediction Systems
* Production-Oriented Project Structure

This solution showcases practical ML engineering skills required for Data Scientist, Machine Learning Engineer, and MLOps Engineer roles.
