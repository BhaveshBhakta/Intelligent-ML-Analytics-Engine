import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    classification_report,
    confusion_matrix
)

def load_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8", errors="ignore")

def detect_problem_type(y):
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= 15:
            return "classification"
        return "regression"
    return "classification"

def evaluate_model(processed_path, models_dir, target, eval_dir):

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    df = load_csv_safely(processed_path)

    if target not in df.columns:
        raise Exception(f"Target column '{target}' not found in processed dataset")

    X = df.drop(columns=[target])
    y = df[target]

    problem_type = detect_problem_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_path = os.path.join(models_dir, "best_model.pkl")
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    results = {"problem_type": problem_type}

    # REGRESSION METRICS
    if problem_type == "regression":
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        results.update({
            "r2_score": float(r2),
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse)
        })

        # Actual vs Predicted
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        path1 = os.path.join(eval_dir, "actual_vs_predicted.png")
        plt.savefig(path1, bbox_inches="tight")
        plt.close()

        # Plot: Residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(0, color="red")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        path2 = os.path.join(eval_dir, "residuals.png")
        plt.savefig(path2, bbox_inches="tight")
        plt.close()

        # Plot: Error Distribution
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True)
        plt.title("Error Distribution")
        path3 = os.path.join(eval_dir, "error_distribution.png")
        plt.savefig(path3, bbox_inches="tight")
        plt.close()

        results["plots"] = {
            "actual_vs_predicted": path1,
            "residuals": path2,
            "error_distribution": path3
        }

    # CLASSIFICATION METRICS
    else:
        accuracy = accuracy_score(y_test, y_pred)
        cls_report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results.update({
            "accuracy": float(accuracy),
            "classification_report": cls_report,
            "confusion_matrix": cm
        })

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")
        plt.title("Confusion Matrix")
        path1 = os.path.join(eval_dir, "confusion_matrix.png")
        plt.savefig(path1, bbox_inches="tight")
        plt.close()

        results["plots"] = {"confusion_matrix": path1}

    with open(os.path.join(eval_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    return results
