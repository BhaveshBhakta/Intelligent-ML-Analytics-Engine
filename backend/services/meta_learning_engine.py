import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


META_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "../experiments/meta_dataset.csv"
)


def load_meta_dataset():
    if not os.path.exists(META_DATA_PATH):
        return None

    return pd.read_csv(META_DATA_PATH)


def train_meta_model(meta_df):

    if meta_df is None or len(meta_df) < 5:
        return None, None

    X = meta_df.drop(columns=["best_model"])
    y = meta_df["best_model"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y_encoded)

    return model, encoder


def recommend_models(meta_features, top_k=3):

    meta_df = load_meta_dataset()

    if meta_df is None or len(meta_df) < 5:

        # fallback recommendation
        return [
            "RandomForestClassifier",
            "XGBClassifier",
            "LogisticRegression"
        ]

    model, encoder = train_meta_model(meta_df)

    if model is None:
        return [
            "RandomForestClassifier",
            "XGBClassifier"
        ]

    X_new = pd.DataFrame([meta_features])

    probs = model.predict_proba(X_new)[0]

    top_indices = np.argsort(probs)[::-1][:top_k]

    recommended = encoder.inverse_transform(top_indices)

    return list(recommended)