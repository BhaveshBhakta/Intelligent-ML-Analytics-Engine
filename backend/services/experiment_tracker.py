import os
import json
import pandas as pd
from datetime import datetime


EXPERIMENT_DB = os.path.join(
    os.path.dirname(__file__),
    "../experiments/experiment_history.csv"
)


def log_experiment(run_id, meta_features, model_summary, evaluation_results):

    row = {}

    row["run_id"] = run_id
    row["timestamp"] = datetime.now().isoformat()

    if meta_features:
        row.update(meta_features)

    if model_summary:
        row["best_model"] = model_summary.get("best_model")

    if evaluation_results:

        if evaluation_results["problem_type"] == "classification":
            row["score"] = evaluation_results.get("accuracy")

        else:
            row["score"] = evaluation_results.get("r2_score")

    df = pd.DataFrame([row])

    if os.path.exists(EXPERIMENT_DB):
        df_existing = pd.read_csv(EXPERIMENT_DB)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_csv(EXPERIMENT_DB, index=False)