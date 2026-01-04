import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8", errors="ignore")


def detect_datetime_columns(df, target_name):
    datetime_cols = []

    for col in df.columns:
        if col == target_name:
            continue

        if df[col].dtype == "object":
            try:
                pd.to_datetime(df[col], errors="raise")
                datetime_cols.append(col)
            except Exception:
                continue

    return datetime_cols

def extract_datetime_features(df, datetime_cols):
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        df[f"{col}_hour"] = df[col].dt.hour

        df.drop(columns=[col], inplace=True)

    return df

def preprocess_dataset(input_path, output_path, summary_path=None, target_name=None):
    df = load_csv_safely(input_path)

    if target_name and target_name not in df.columns:
        raise Exception(f"Target column '{target_name}' not found in dataset")

    original_columns = list(df.columns)

    # Missing Value Handling
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())


    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Datetime Feature Handling
    datetime_cols = detect_datetime_columns(df, target_name)
    df = extract_datetime_features(df, datetime_cols)

    # Encode Categorical Columns
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns and col != target_name:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = list(le.classes_)

    # Feature Scaling (EXCLUDE TARGET)
    scaler = StandardScaler()

    scaled_cols = [
        c for c in df.select_dtypes(include=["int64", "float64"]).columns
        if c != target_name
    ]

    df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

    # Save Processed Data
    df.to_csv(output_path, index=False)

    # Save Summary
    if summary_path:
        summary = {
            "original_columns": original_columns,
            "numeric_columns": list(numeric_cols),
            "categorical_columns": list(categorical_cols),
            "datetime_columns": list(datetime_cols),
            "scaled_columns": list(scaled_cols),
            "target_column": target_name
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

    return True
