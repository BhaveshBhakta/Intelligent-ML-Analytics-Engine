import pandas as pd
import numpy as np
import os
import json
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from datetime import datetime

def load_dataset(file_path):
    """ Load csv files with encoding fallback """
    import pandas as pd
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")
    return df



def detect_column_types(df):
    """Detect column data types (numeric, categorical, datetime)."""
    col_types = {}
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            col_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_types[col] = "datetime"
        elif any(re.search(r"\d{4}-\d{2}-\d{2}", str(x)) for x in df[col].astype(str).head(10)):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            col_types[col] = "datetime"
        else:
            col_types[col] = "categorical"
    return df, col_types


def handle_missing_values(df):
    """Automatically handle missing values."""
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "O":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    return df


def detect_and_treat_outliers(df):
    """Treat numeric outliers using IQR method."""
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)
    return df


def scale_features(df, method="standard"):
    """Scale numeric columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def encode_categoricals(df):
    """Encode categorical columns using LabelEncoder."""
    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders


def extract_datetime_features(df, col_types):
    """Extract year, month, day, weekday, etc. from datetime columns."""
    for col, typ in col_types.items():
        if typ == "datetime":
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_weekday"] = df[col].dt.weekday
            df[f"{col}_is_weekend"] = df[col].dt.weekday >= 5
    return df


def auto_preprocess(file_path, output_dir):
    """Run the full preprocessing pipeline."""
    df = load_dataset(file_path)
    df, col_types = detect_column_types(df)
    df = handle_missing_values(df)
    df = detect_and_treat_outliers(df)
    df = extract_datetime_features(df, col_types)
    df, encoders = encode_categoricals(df)
    df = scale_features(df)

    processed_path = os.path.join(output_dir, "processed.csv")
    df.to_csv(processed_path, index=False)

    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "column_types": col_types,
        "output_file": processed_path
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return summary
