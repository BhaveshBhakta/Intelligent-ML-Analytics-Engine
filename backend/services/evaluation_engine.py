import os
import json
import joblib
import pandas as pd
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, f1_score, roc_auc_score)
import warnings
warnings.filterwarnings("ignore")


def evaluate_model(run_id, run_path, target_col):
    """
    Load best model and processed data, evaluate on full data, return metrics.
    Compatible with older sklearn versions.
    """
    models_dir = os.path.join(run_path, "models")
    summary_path = os.path.join(models_dir, "models_summary.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError("No trained model found. Run /train first.")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    best_model_name = summary["best_model"]
    model_path = summary["best_model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError("Best model file not found.")

    model = joblib.load(model_path)
    processed_file = os.path.join(run_path, "processed.csv")
    df = pd.read_csv(processed_file)

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not in processed data.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Detect task type
    task = summary["task"]

    eval_metrics = {}
    y_pred = model.predict(X)

    if task == "regression":
        eval_metrics["MAE"] = float(mean_absolute_error(y, y_pred))
        eval_metrics["RMSE"] = float(mean_squared_error(y, y_pred) ** 0.5)
        eval_metrics["R2"] = float(r2_score(y, y_pred))
    else:
        eval_metrics["Accuracy"] = float(accuracy_score(y, y_pred))
        eval_metrics["F1_macro"] = float(f1_score(y, y_pred, average="macro"))
        if len(y.unique()) == 2:
            eval_metrics["ROC_AUC"] = float(roc_auc_score(y, y_pred))

    # Save evaluation results
    eval_path = os.path.join(models_dir, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_metrics, f, indent=4)

    return {"run_id": run_id, "best_model": best_model_name, "evaluation": eval_metrics}
