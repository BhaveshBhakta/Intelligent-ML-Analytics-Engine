# AutoML - An Intelligent ML Analytics Engine

This system ingests a CSV dataset, performs automated preprocessing, exploratory data analysis, model training, evaluation, and generates a polished analytical report. It produces performance metrics and visual insights — including EDA plots and model evaluation charts — through a simple web interface. Designed for students, analysts, and data teams who want fast, structured insights without manual setup.

---

## Key Features

### Automated Analysis Suite

* **Data Preprocessing** :  Missing-value handling, Numeric + categorical feature processing, Date column detection & conversion, Automatic dataset cleaning, Outputs a clean `processed.csv`

* **Exploratory Data Analysis (EDA)**: Summary statistics (`describe`), Missing-value profiling, Histogram visualizations for numeric features, Correlation heatmap, Safe filename handling, All plots saved under `runs/<run_id>/eda_plots/`

* **AutoML Model Training** : Automatically detects problem type (regression/classification), Trains multiple baseline models, Selects best model automatically, Saves model as `best_model.pkl`

* **Model Evaluation & Reporting** : Regression Metrics -  R² Score, RMSE, MAE, MSE | Classification Metrics - Accuracy, Confusion Matrix | Evaluation Plots - Actual vs Predicted, Residual Plot, Error Distribution | PDF final report generation | Results stored in structured folders per run

---

### User-Friendly Interface

* Web UI built with Flask + vanilla HTML/JS
* Upload CSV & select target column
* Click once to run the full pipeline
* Live status updates at each step
* View metrics, charts & download report
* Runs stored safely by unique `run_id`

---

## Technology Stack

* **Backend** : Python, Flask (REST API), scikit-learn (ML models & metrics), pandas / numpy (data processing), seaborn / matplotlib (visualization), joblib (model persistence)

* **Frontend** : HTML, CSS, JavaScript (Fetch API)

* **Architecture & Infrastructure** : Thread-based async training, Structured run directories, CORS enabled frontend-backend communication, Headless plotting via `matplotlib.use("Agg")`

---

## Website Overview
<img width="3068" height="1614" alt="AutoML Intelligent Analytics 1" src="https://github.com/user-attachments/assets/e38a5fc7-e7ff-4a2a-bf8d-d3de65a378ab" />
<img width="3066" height="1558" alt="AutoML Intelligent Analytics 2" src="https://github.com/user-attachments/assets/fe37fca3-ed05-4ee2-9bbb-e43639e1b7d2" />

---

## Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/BhaveshBhakta/Intelligent-ML-Analytics-Engine.git
cd Intelligent-ML-Analytics-Engine
pip install -r requirements.txt
```

Run the backend:

```bash
python -m backend.app
```

Open the UI:

```
http://localhost:5000
```

All outputs are saved automatically under:

```
runs/<RUN_ID>/
```

---

## High-Level Architecture

```
User (Browser, CSV Upload)
        ↓
     Flask API
        ↓
 ┌────────────── Pipeline ───────────────┐
 │ Upload & Run Creation                 │
 │ Data Preprocessing                    │
 │ Exploratory Data Analysis (EDA)       │
 │ AutoML Model Training                 │
 │ Model Evaluation + Charts             │
 │ PDF Report Generation                 │
 └───────────────────────────────────────┘
        ↓
  UI Dashboard + Exportable Results
```

---

## Roadmap & Future Enhancements

* Automated hyperparameter tuning
* Explainability using SHAP / LIME
* Outlier detection & handling
* Time-series forecasting support
* Model comparison dashboard
* Authentication & multi-user runs
* Cloud deployment template
