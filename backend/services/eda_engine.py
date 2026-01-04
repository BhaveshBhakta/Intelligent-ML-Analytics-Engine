import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def load_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8", errors="ignore")


def sanitize_filename(name):
    """
    Replace any unsafe filename character with underscore
    """
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def generate_eda(df, save_dir, summary_path):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    eda_summary = {}

    eda_summary["missing_values"] = df.isnull().sum().to_dict() # Missing Values
    describe_df = df.describe(include="all").transpose() # Describe Stats
    eda_summary["describe"] = describe_df.fillna("").to_dict()

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns) # Column Types
    categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)

    eda_summary["numeric_columns"] = numeric_cols
    eda_summary["categorical_columns"] = categorical_cols

    # Histograms
    hist_paths = [] 

    for col in numeric_cols:
        try:
            plt.figure(figsize=(7, 5))
            sns.histplot(df[col], kde=True)
            plt.title(f"Histogram - {col}")

            safe_name = sanitize_filename(col)
            path = os.path.join(save_dir, f"hist_{safe_name}.png")

            plt.savefig(path, bbox_inches="tight")
            plt.close()

            hist_paths.append(path)
        except Exception as e:
            print(f"Histogram failed for {col}: {e}")

    eda_summary["histograms"] = hist_paths

    # Correlation Heatmap
    if len(numeric_cols) > 1:
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm")
            heatmap_path = os.path.join(save_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_path, bbox_inches="tight")
            plt.close()
            eda_summary["correlation_heatmap"] = heatmap_path
        except Exception as e:
            eda_summary["correlation_heatmap"] = None
            print("Heatmap failed:", e)
    else:
        eda_summary["correlation_heatmap"] = None

    with open(summary_path, "w") as f:
        json.dump(eda_summary, f, indent=4)

    return eda_summary
