from flask import Flask, request, jsonify
import os
import json
import uuid
from werkzeug.utils import secure_filename

# Import services
from services.preprocessing import auto_preprocess
from services.eda_engine_dynamic import generate_dynamic_eda

# CONFIG
UPLOAD_FOLDER = "runs"
ALLOWED_EXTENSIONS = {"csv", "xls", "xlsx"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# HELPERS
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ROUTES
@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload CSV/Excel file and assign run_id
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        run_id = str(uuid.uuid4())
        run_path = os.path.join(UPLOAD_FOLDER, run_id)
        os.makedirs(run_path, exist_ok=True)
        file_path = os.path.join(run_path, filename)
        file.save(file_path)

        status = {"stage": "uploaded", "progress": 10}
        with open(os.path.join(run_path, "status.json"), "w") as f:
            json.dump(status, f)

        return jsonify({"run_id": run_id, "file_name": filename})

    return jsonify({"error": "File type not allowed"}), 400


@app.route("/preprocess", methods=["POST"])
def preprocess_data():
    """
    Run preprocessing pipeline for uploaded dataset
    """
    data = request.get_json()
    run_id = data.get("run_id")

    run_path = os.path.join(UPLOAD_FOLDER, run_id)
    if not os.path.exists(run_path):
        return jsonify({"error": "Invalid run_id"}), 404

    files = os.listdir(run_path)
    csv_files = [f for f in files if f.endswith(".csv")]

    if not csv_files:
        return jsonify({"error": "No CSV found for this run"}), 400

    file_path = os.path.join(run_path, csv_files[0])

    summary = auto_preprocess(file_path, run_path)

    status = {"stage": "preprocessed", "progress": 30}
    with open(os.path.join(run_path, "status.json"), "w") as f:
        json.dump(status, f)

    return jsonify(summary)


@app.route("/eda", methods=["POST"])
def run_eda():
    """
    Run dynamic EDA for a processed dataset.
    Expects JSON: {"run_id": "abc123", "target_col": "target"} (target_col optional)
    """
    data = request.get_json()
    run_id = data.get("run_id")
    target_col = data.get("target_col")

    run_path = os.path.join(UPLOAD_FOLDER, run_id)
    if not os.path.exists(run_path):
        return jsonify({"error": "Invalid run_id"}), 404

    processed_file = os.path.join(run_path, "processed.csv")
    if not os.path.exists(processed_file):
        return jsonify({"error": "Processed file not found. Run preprocessing first."}), 400

    # Run Dynamic EDA
    eda_summary = generate_dynamic_eda(processed_file, run_path, target_col=target_col)

    # Update progress
    status = {"stage": "eda_complete", "progress": 50}
    with open(os.path.join(run_path, "status.json"), "w") as f:
        json.dump(status, f)

    return jsonify(eda_summary)


# MAIN
if __name__ == "__main__":
    app.run(debug=True)
