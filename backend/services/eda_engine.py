import pandas as pd
import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

sns.set(style="whitegrid")

def generate_base64_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_dynamic_eda(file_path, output_dir, target_col=None, top_n=10):
    df = pd.read_csv(file_path)

    # Basic Dataset Info
    info_summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "describe": df.describe(include="all").fillna("").to_dict()
    }

    # Column Type Detection
    numeric_df = df.select_dtypes(include=np.number)
    cat_cols = df.select_dtypes(include="object").columns
    datetime_cols = df.select_dtypes(include="datetime").columns

    # Numeric Visualizations
    dist_plots, box_plots, scatter_target_plots = [], [], []
    for col in numeric_df.columns:
        # Distribution
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        plt.title(f"Distribution of {col}")
        dist_plots.append({"column": col, "plot": generate_base64_plot(fig)})
        plt.close(fig)

        # Boxplot
        fig, ax = plt.subplots()
        sns.boxplot(y=df[col], ax=ax)
        plt.title(f"Boxplot of {col}")
        box_plots.append({"column": col, "plot": generate_base64_plot(fig)})
        plt.close(fig)

        # Scatter with target if available
        if target_col and target_col in df.columns and np.issubdtype(df[target_col].dtype, np.number):
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[col], y=df[target_col], ax=ax)
            plt.title(f"{col} vs {target_col}")
            scatter_target_plots.append({"column": col, "plot": generate_base64_plot(fig)})
            plt.close(fig)

    # Pairplots (<=10 numeric columns)
    if 1 <= len(numeric_df.columns) <= 10:
        fig = sns.pairplot(numeric_df)
        plt.suptitle("Pairplot of Numeric Features", y=1.02)
        pairplot_img = generate_base64_plot(fig)
        plt.close(fig)
    else:
        pairplot_img = None

    # Categorical Visualizations
    cat_plots, cat_target_plots = [], []
    for col in cat_cols:
        top_values = df[col].value_counts().nlargest(top_n)
        fig, ax = plt.subplots(figsize=(6, 4))
        top_values.plot(kind="bar", ax=ax)
        plt.title(f"Top {top_n} counts of {col}")
        cat_plots.append({"column": col, "plot": generate_base64_plot(fig)})
        plt.close(fig)

        # Target mean per category
        if target_col and target_col in df.columns and np.issubdtype(df[target_col].dtype, np.number):
            target_means = df.groupby(col)[target_col].mean().nlargest(top_n)
            fig, ax = plt.subplots(figsize=(6, 4))
            target_means.plot(kind="bar", ax=ax)
            plt.title(f"Mean {target_col} per {col} (Top {top_n})")
            cat_target_plots.append({"column": col, "plot": generate_base64_plot(fig)})
            plt.close(fig)

    # Datetime Visualizations
    datetime_plots = []
    for col in datetime_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        df.set_index(col).resample("M").size().plot(ax=ax)
        plt.title(f"Monthly Trend of {col}")
        datetime_plots.append({"column": col, "plot": generate_base64_plot(fig)})
        plt.close(fig)

    # Missing Values Heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Value Heatmap")
    missing_heatmap_img = generate_base64_plot(fig)
    plt.close(fig)

    # Compile JSON 
    eda_summary = {
        "info_summary": info_summary,
        "heatmap": missing_heatmap_img,
        "distribution_plots": dist_plots,
        "box_plots": box_plots,
        "scatter_target_plots": scatter_target_plots,
        "pairplot": pairplot_img,
        "categorical_plots": cat_plots,
        "cat_target_plots": cat_target_plots,
        "datetime_plots": datetime_plots
    }

    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eda_summary.json"), "w") as f:
        json.dump(eda_summary, f, indent=4)

    return eda_summary
