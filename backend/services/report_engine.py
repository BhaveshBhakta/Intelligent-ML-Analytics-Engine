import os
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch

def safe_load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def add_heading(story, text, styles):
    story.append(Paragraph(f"<b>{text}</b>", styles["Heading2"]))
    story.append(Spacer(1, 10))


def add_paragraph(story, text, styles):
    story.append(Paragraph(text, styles["BodyText"]))
    story.append(Spacer(1, 8))

def add_image_if_exists(story, path, width=400):
    if path and os.path.exists(path):
        story.append(Image(path, width=width, height=width * 0.6))
        story.append(Spacer(1, 10))

def generate_report(run_id, run_path, output_file):
    styles = getSampleStyleSheet()
    story = []

    preprocessing = safe_load_json(os.path.join(run_path, "preprocessing_summary.json"))
    eda = safe_load_json(os.path.join(run_path, "eda_summary.json"))
    models = safe_load_json(os.path.join(run_path, "models", "models_summary.json"))
    evaluation = safe_load_json(os.path.join(run_path, "evaluation", "evaluation_results.json"))

    # TITLE
    story.append(Paragraph("<b>Intelligent ML Analytics Report</b>", styles["Title"]))
    add_paragraph(story, f"Run ID: {run_id}", styles)

    # Preprocessing
    add_heading(story, "Dataset & Preprocessing", styles)

    if preprocessing:
        add_paragraph(story, f"Total Columns: {len(preprocessing.get('original_columns', []))}", styles)
        add_paragraph(story, f"Numeric Columns: {len(preprocessing.get('numeric_columns', []))}", styles)
        add_paragraph(story, f"Categorical Columns: {len(preprocessing.get('categorical_columns', []))}", styles)
        add_paragraph(story, f"Datetime Columns: {len(preprocessing.get('datetime_columns', []))}", styles)
    else:
        add_paragraph(story, "Preprocessing summary not available.", styles)

    # EDA
    add_heading(story, "Exploratory Data Analysis", styles)

    if eda:
        add_paragraph(story, "Sample Histogram:", styles)
        if eda.get("histograms"):
            add_image_if_exists(story, eda["histograms"][0])

        add_paragraph(story, "Correlation Heatmap:", styles)
        add_image_if_exists(story, eda.get("correlation_heatmap"))
    else:
        add_paragraph(story, "EDA summary not available.", styles)

    # Model Training
    add_heading(story, "Model Training Summary", styles)

    if models:
        add_paragraph(story, f"Problem Type: {models.get('problem_type', 'N/A')}", styles)
        add_paragraph(story, f"Target Column: {models.get('target_column', 'N/A')}", styles)
        add_paragraph(story, f"Best Model: {models.get('best_model', 'N/A')}", styles)
        add_paragraph(story, f"Best CV Score: {models.get('best_score', 'N/A')}", styles)
    else:
        add_paragraph(story, "Model summary not available.", styles)

    # Evaluation
    add_heading(story, "Model Evaluation", styles)

    if evaluation:
        if evaluation.get("problem_type") == "regression":
            add_paragraph(story, f"R² Score: {evaluation.get('r2_score', 'N/A')}", styles)
            add_paragraph(story, f"MAE: {evaluation.get('mae', 'N/A')}", styles)
            add_paragraph(story, f"MSE: {evaluation.get('mse', 'N/A')}", styles)
            add_paragraph(story, f"RMSE: {evaluation.get('rmse', 'N/A')}", styles)

            plots = evaluation.get("plots", {})
            add_paragraph(story, "Actual vs Predicted:", styles)
            add_image_if_exists(story, plots.get("actual_vs_predicted"))

            add_paragraph(story, "Residuals Plot:", styles)
            add_image_if_exists(story, plots.get("residuals"))

            add_paragraph(story, "Error Distribution:", styles)
            add_image_if_exists(story, plots.get("error_distribution"))

        else:
            add_paragraph(story, f"Accuracy: {evaluation.get('accuracy', 'N/A')}", styles)

            plots = evaluation.get("plots", {})
            add_paragraph(story, "Confusion Matrix:", styles)
            add_image_if_exists(story, plots.get("confusion_matrix"))
    else:
        add_paragraph(story, "Evaluation results not available.", styles)

    # BUILD PDF
    doc = SimpleDocTemplate(output_file, pagesize=A4)
    doc.build(story)

    return output_file
