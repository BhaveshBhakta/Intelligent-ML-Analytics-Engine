# AutoML - An Intelligent ML Analytics Engine

**AutoML** is a high-performance analytics engine designed to automate the end-to-end machine learning lifecycle. By ingesting raw CSV data, the system orchestrates a sophisticated pipeline—from automated preprocessing and meta-learning-driven model selection to hyperparameter optimization and explainable AI (XAI). 

The platform bridges the gap between raw data and actionable intelligence, delivering a polished web interface for real-time monitoring, visual insights, and comprehensive PDF reporting.

---

## Key Features

### Automated Analysis Suite
* **Intelligent Data Preprocessing**: Automated handling of missing values, encoding of categorical variables, and date-time feature engineering. Outputs a standardized `processed.csv` ready for modeling.
* **Deep Exploratory Data Analysis (EDA)**: Generates summary statistics, missing-value profiles, correlation heatmaps, and distribution plots. All assets are version-controlled under `runs/<run_id>/`.
* **Meta-Learning Engine**: Extracts high-level dataset characteristics (meta-features) to recommend optimal model architectures based on historical experiment performance.
* **AutoML Model Training**: Automatically detects problem types (Regression/Classification), trains a diverse model zoo, and persists the champion model as `best_model.pkl`.
* **Hyperparameter Optimization**: Leverages **Optuna** for Bayesian optimization, fine-tuning models like XGBoost and Random Forest beyond default configurations.
* **Comprehensive Evaluation**: 
    * **Regression**: $R^2$, $RMSE$, $MAE$, $MSE$.
    * **Classification**: Accuracy, Confusion Matrices, and Precision-Recall curves.
* **Explainable AI (XAI)**: Integrated **SHAP** and Feature Importance analysis to provide transparency into model decision-making.
* **Experiment Tracking**: A RAG-inspired memory system that stores meta-features and performance metrics to improve future model recommendations.

### 💻 User-Friendly Interface
* **Modern Web UI**: Responsive dashboard built with Flask and vanilla HTML/JS/Tailwind.
* **Seamless Workflow**: One-click pipeline execution with live status updates.
* **Centralized Results**: Interactive leaderboard, downloadable PDF reports, and localized storage indexed by unique `run_id`.

---

## Website Overview

<img width="1849" height="1080" alt="automl" src="https://github.com/user-attachments/assets/ef4c1a21-de7d-4c4c-963f-e5620c73596b" />


---

## Technology Stack

| Component | Technologies |
| :--- | :--- |
| **Backend** | Python, Flask (REST API), Scikit-learn, XGBoost |
| **Optimization/XAI** | Optuna, SHAP |
| **Data & Viz** | Pandas, Numpy, Seaborn, Matplotlib (Agg backend) |
| **Frontend** | HTML5, Tailwind CSS, JavaScript (Fetch API) |
| **Architecture** | Thread-based Async Training, Modular ML Pipelines, CORS-enabled Communication |

---

## High-Level Architecture

<img width="1536" height="1024" alt="automl arch" src="https://github.com/user-attachments/assets/951991b3-b7b1-437b-a304-9e19fccc31a5" />

---

## Quick Start

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/BhaveshBhakta/Intelligent-ML-Analytics-Engine.git
cd Intelligent-ML-Analytics-Engine
pip install -r requirements.txt
```

**2. Launch the backend:**
```bash
python -m backend.app
```

**3. Access the UI:**
Navigate to `http://localhost:5000` in your browser. All outputs are automatically persisted in:
`runs/<RUN_ID>/`

---

## Roadmap & Future Enhancements
* [ ] **Advanced Meta-Learning**: Implementing transformer-based recommendation models.
* [ ] **Deep Learning**: Integration of PyTorch/TensorFlow for neural architecture search (NAS).
* [ ] **Enterprise Readiness**: Multi-user authentication and Dockerized cloud deployment (AWS/GCP).
