import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier

def normalize_columns(df):
    """
    Clean column names to avoid hidden characters.
    """
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)   # remove BOM
        .str.strip()                              # trim spaces
    )
    return df

def normalize_string(s):
    return (
        str(s)
        .replace("\ufeff", "")
        .strip()
        .lower()
    )

def detect_problem_type(y):
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= 15:
            return "classification"
        return "regression"
    return "classification"

def load_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8", errors="ignore")

def train_models(processed_path, models_dir, target):

    df = load_csv_safely(processed_path)
    df = normalize_columns(df)

    # TARGET MATCHING
    target_norm = normalize_string(target)

    matches = [c for c in df.columns if normalize_string(c) == target_norm]

    if not matches:
        raise Exception(
            f"Target column '{target}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    target_col = matches[0]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    problem_type = detect_problem_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if problem_type == "regression":
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "SVR": SVR(),
            "KNNRegressor": KNeighborsRegressor(),
            "XGBRegressor": XGBRegressor(objective="reg:squarederror")
        }
        scoring = "r2"

    else:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=500),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "SVC": SVC(),
            "KNNClassifier": KNeighborsClassifier(),
            "XGBClassifier": XGBClassifier(eval_metric="logloss")
        }
        scoring = "accuracy"

    best_score = -999
    best_model = None
    best_model_name = None
    results = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            if problem_type == "regression":
                score = r2_score(y_test, y_pred)
            else:
                score = accuracy_score(y_test, y_pred)

            cv_score = cross_val_score(model, X, y, cv=5, scoring=scoring).mean()

            model_path = os.path.join(models_dir, f"{name}.pkl")
            joblib.dump(model, model_path)

            results[name] = {
                "test_score": float(score),
                "cv_score": float(cv_score),
                "model_path": model_path
            }

            if cv_score > best_score:
                best_score = cv_score
                best_model = model
                best_model_name = name

        except Exception as e:
            results[name] = {"error": str(e)}

    best_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, best_path)

    summary = {
        "problem_type": problem_type,
        "target_column_original_arg": target,
        "target_column_matched": target_col,
        "best_model": best_model_name,
        "best_score": float(best_score),
        "best_model_path": best_path,
        "all_models": results
    }

    with open(os.path.join(models_dir, "models_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return summary
