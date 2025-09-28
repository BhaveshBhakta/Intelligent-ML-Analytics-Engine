import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import warnings

warnings.filterwarnings("ignore")

# Candidate models for quick baseline 
CLASSIFICATION_MODELS = {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "svc": SVC(probability=True),
    "knn": KNeighborsClassifier(),
}

REGRESSION_MODELS = {
    "linear": LinearRegression(),
    "ridge": Ridge(),
    "lasso": Lasso(),
    "random_forest": RandomForestRegressor(n_estimators=100),
    "xgboost": XGBRegressor(),
    "svr": SVR(),
    "knn": KNeighborsRegressor(),
}


def detect_problem_type(df, target_col):
    """Detect whether the task is classification or regression."""
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col} not in dataframe columns")
    if pd.api.types.is_integer_dtype(df[target_col]) or pd.api.types.is_float_dtype(df[target_col]):
        # Distinguish classification if few unique values and integers
        n_unique = df[target_col].nunique()
        if n_unique <= 20 and pd.api.types.is_integer_dtype(df[target_col]):
            return "classification"
        else:
            return "regression"
    else:
        # Non-numeric target -> classification (string labels)
        return "classification"


def scoring_for_task(task):
    """Return scoring dict for cross_validate based on task."""
    if task == "classification":
        return {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "roc_auc": "roc_auc" 
        }
    else:
        return {
            "neg_mean_squared_error": "neg_mean_squared_error",
            "neg_mean_absolute_error": "neg_mean_absolute_error",
            "r2": "r2"
        }


def safe_cross_validate(estimator, X, y, task, cv=5):
    """
    Cross-validate estimator with appropriate folds and scoring.
    Returns dict of mean scores per metric.
    """
    if task == "classification":
        # Use StratifiedKFold if labels have enough samples per class
        try:
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        except Exception:
            cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
        scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
        # Try to add roc_auc for binary classification or multilabel if possible
        try:
            if len(np.unique(y)) == 2:
                scoring["roc_auc"] = "roc_auc"
        except Exception:
            pass
    else:
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
        scoring = {"neg_mean_squared_error": "neg_mean_squared_error",
                   "neg_mean_absolute_error": "neg_mean_absolute_error",
                   "r2": "r2"}

    results = cross_validate(estimator, X, y, cv=cv_strategy, scoring=scoring, n_jobs=1, return_train_score=False)
    # convert to mean scores
    mean_scores = {k: float(np.mean(v)) for k, v in results.items() if k.startswith("test_")}
    # clean keys
    clean_scores = {k.replace("test_", ""): v for k, v in mean_scores.items()}
    return clean_scores


def evaluate_model_job(model_name, estimator, X, y, task, cv=5):
    """Train/evaluate single candidate model (no final refit here)."""
    try:
        scores = safe_cross_validate(estimator, X, y, task, cv=cv)
    except Exception as e:
        scores = {"error": str(e)}
    return {"model_name": model_name, "scores": scores}


def train_and_select_best(run_id, run_path, target_col, cv=5, max_workers=3):
    """
    Main entrypoint: load processed.csv for run, train candidate models in parallel,
    select best model (based on task-specific primary metric), and save artifacts.
    """
    processed_file = os.path.join(run_path, "processed.csv")
    if not os.path.exists(processed_file):
        raise FileNotFoundError("Processed file not found. Run preprocessing first.")

    df = pd.read_csv(processed_file)
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in processed data.")

    # Prepare X, y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    task = detect_problem_type(df, target_col)

    # Choose models
    if task == "classification":
        candidates = CLASSIFICATION_MODELS
    else:
        candidates = REGRESSION_MODELS

    # Run evaluations in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_model_job, name, model, X, y, task, cv): name for name, model in candidates.items()}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)

    # Aggregate and rank
    # Determine primary metric
    if task == "classification":
        # prefer f1_macro if present, else accuracy
        def score_key(r):
            s = r["scores"]
            return s.get("f1_macro", s.get("accuracy", -999))
        results_sorted = sorted(results, key=score_key, reverse=True)
    else:
        # for regression prefer neg_mean_squared_error (higher is better because it's negative)
        def score_key(r):
            s = r["scores"]
            return s.get("neg_mean_squared_error", -999)
        results_sorted = sorted(results, key=score_key, reverse=True)

    # Refit the best model on full data and save artifact
    best_model_name = results_sorted[0]["model_name"]
    best_estimator = candidates[best_model_name]
    try:
        best_estimator.fit(X, y)
    except Exception:
        # If fails, try a simple fallback (e.g., Linear/Logistic)
        if task == "classification":
            fallback = LogisticRegression(max_iter=1000)
        else:
            fallback = LinearRegression()
        fallback.fit(X, y)
        best_estimator = fallback
        best_model_name = "fallback_model"

    # Save models directory
    models_dir = os.path.join(run_path, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"{best_model_name}.joblib")
    joblib.dump(best_estimator, model_path)

    # Save results summary
    summary = {
        "run_id": run_id,
        "task": task,
        "target_col": target_col,
        "best_model": best_model_name,
        "best_model_path": model_path,
        "all_models": results_sorted
    }

    with open(os.path.join(models_dir, "models_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return summary
