import json
import os
import pandas as pd
import numpy as np


def compute_class_imbalance(y):
    """
    Measures imbalance for classification problems.
    """
    try:
        counts = y.value_counts(normalize=True)
        if len(counts) <= 1:
            return 0.0
        return float(counts.max() - counts.min())
    except:
        return 0.0


def compute_avg_correlation(df):
    """
    Average correlation between numeric features
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return 0.0

    corr = numeric_df.corr().abs()

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    values = upper.stack()

    if len(values) == 0:
        return 0.0

    return float(values.mean())


def compute_feature_variance(df):
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] == 0:
        return 0.0

    return float(numeric_df.var().mean())


def compute_sparsity(df):
    """
    Percentage of zero values
    """
    total = df.size
    zeros = (df == 0).sum().sum()

    if total == 0:
        return 0

    return float(zeros / total)


def extract_meta_features(df, target=None, save_path=None):
    """
    Extract dataset meta-features
    """

    meta = {}

    meta["n_samples"] = int(df.shape[0])
    meta["n_features"] = int(df.shape[1])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    meta["num_numeric"] = int(len(numeric_cols))
    meta["num_categorical"] = int(len(categorical_cols))

    meta["missing_ratio"] = float(df.isnull().sum().sum() / df.size)

    meta["avg_correlation"] = compute_avg_correlation(df)

    meta["feature_variance"] = compute_feature_variance(df)

    meta["sparsity"] = compute_sparsity(df)

    # Class imbalance (if target exists)
    if target and target in df.columns:
        meta["class_imbalance"] = compute_class_imbalance(df[target])
    else:
        meta["class_imbalance"] = 0.0

    if save_path:
        with open(save_path, "w") as f:
            json.dump(meta, f, indent=4)

    return meta