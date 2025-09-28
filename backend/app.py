import os
import json
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

# ==== Import Service Modules ====
from services.preprocessing import auto_preprocess as preprocess_data
from services.eda_engine import generate_eda_summary
from services.models_engine import train_and_select_best

# ==== App Configuration ====
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "runs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=2)

#  ROUTE 1: UPLOAD DATA
@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload CSV file and create new run directory"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    run_id = str(uuid.uuid4())
    run_path = os.path.join(UPLOAD_FOLDER, run_id)
    os.makedirs(run_path, exist_ok=True)

    filepath = os.path.join(run_path, file.filename)
    file.save(filepath)

    # create initial status.json
    with open(os.path.join(run_path, "status.json"), "w") as f:
        json.dump({"stage": "uploaded", "progress": 10}, f)

    return jsonify({
        "message": "File uploaded successfully",
        "run_id": run_id,
        "filename": file.filename
    })

#  ROUTE 2: PREPROCESS DATA
@app.route("/preprocess", methods=["POST"])
def preprocess():
    """Run preprocessing on uploaded data"""
    data = request.get_json()
    run_id = data.get("run_id")

    run_path = os.path.join(UPLOAD_FOLDER, run_id)
    if not os.path.exists(run_path):
        return jsonify({"error": "Invalid run_id"}), 404

    files = [f for f in os.listdir(run_path) if f.endswith(".csv")]
    if not files:
        return jsonify({"error": "No CSV file found for this run"}), 404

    input_path = os.path.join(run_path, files[0])
    output_path = os.path.join(run_path, "processed.csv")

    with open(os.path.join(run_path, "status.json"), "w") as f:
        json.dump({"stage": "preprocessing_started", "progress": 20}, f)

    try:
        preprocess_data(input_path, run_path)

        with open(os.path.join(run_path, "status.json"), "w") as f:
            json.dump({"stage": "preprocessing_complete", "progress": 40}, f)

        return jsonify({
            "message": "Preprocessing complete",
            "run_id": run_id,
            "output": "processed.csv"
        })

    except Exception as e:
        with open(os.path.join(run_path, "status.json"), "w") as f:
            json.dump({"stage": "preprocessing_failed", "progress": 0, "error": str(e)}, f)
        return jsonify({"error": str(e)}), 500

#  ROUTE 3: EDA SUMMARY
@app.route("/eda", methods=["POST"])
def eda():
    """Generate EDA summary and basic visuals"""
    data = request.get_json()
    run_id = data.get("run_id")
    target_col = data.get("target_col")

    run_path = os.path.join(UPLOAD_FOLDER, run_id)
    processed_path = os.path.join(run_path, "processed.csv")

    if not os.path.exists(processed_path):
        return jsonify({"error": "Processed file not found"}), 404

    with open(os.path.join(run_path, "status.json"), "w") as f:
        json.dump({"stage": "eda_started", "progress": 45}, f)

    try:
        summary = generate_eda_summary(processed_path, run_path, target_col=target_col)

        with open(os.path.join(run_path, "status.json"), "w") as f:
            json.dump({"stage": "eda_complete", "progress": 50}, f)

        return jsonify({
            "message": "EDA complete",
            "run_id": run_id,
            "summary": summary
        })

    except Exception as e:
        with open(os.path.join(run_path, "status.json"), "w") as f:
            json.dump({"stage": "eda_failed", "progress": 0, "error": str(e)}, f)
        return jsonify({"error": str(e)}), 500

#  ROUTE 4: START TRAINING (ASYNC)
@app.route("/train", methods=["POST"])
def start_training():
    """
    Start model training asynchronously.
    Expects JSON: {"run_id": "<id>", "target_col": "<target>", "cv": 5}
    """
    data = request.get_json()
    run_id = data.get("run_id")
    target_col = data.get("target_col")
    cv = int(data.get("cv", 5))

    run_path = os.path.join(UPLOAD_FOLDER, run_id)
    if not os.path.exists(run_path):
        return jsonify({"error": "Invalid run_id"}), 404

    # update status
    status_file = os.path.join(run_path, "status.json")
    with open(status_file, "w") as f:
        json.dump({"stage": "training_started", "progress": 55}, f)

    def job():
        try:
            summary = train_and_select_best(run_id, run_path, target_col, cv=cv, max_workers=2)
            with open(status_file, "w") as f:
                json.dump({"stage": "training_complete", "progress": 80}, f)
            return summary
        except Exception as e:
            with open(status_file, "w") as f:
                json.dump({"stage": "training_failed", "progress": 0, "error": str(e)}, f)
            return None

    executor.submit(job)
    return jsonify({"message": "Training started", "run_id": run_id})

#  ROUTE 5: FETCH MODEL RESULTS
@app.route("/results/<run_id>", methods=["GET"])
def get_results(run_id):
    """Retrieve AutoML model summary results"""
    run_path = os.path.join(UPLOAD_FOLDER, run_id)
    summary_path = os.path.join(run_path, "models", "models_summary.json")

    if not os.path.exists(summary_path):
        return jsonify({"error": "Models summary not found. Run /train first"}), 404

    with open(summary_path, "r") as f:
        summary = json.load(f)

    return jsonify(summary)

#  ROUTE 6: CHECK STATUS
@app.route("/status/<run_id>", methods=["GET"])
def get_status(run_id):
    """Check current progress stage for given run"""
    status_path = os.path.join(UPLOAD_FOLDER, run_id, "status.json")
    if not os.path.exists(status_path):
        return jsonify({"error": "Invalid run_id"}), 404

    with open(status_path, "r") as f:
        status = json.load(f)
    return jsonify(status)

#  MAIN ENTRY
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
