import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from services.utils import create_run_folder, update_status, read_status
from services.preprocessing import preprocess_dataset
from services.eda_engine import generate_eda, load_csv_safely
from concurrent.futures import ThreadPoolExecutor
from services.models_engine import train_models
from services.evaluation_engine import evaluate_model
from services.report_engine import generate_report

# Flask App Initialization
app = Flask(__name__)
CORS(app)

if not os.path.exists("runs"):
    os.makedirs("runs")
    
executor = ThreadPoolExecutor(max_workers=2)


# Health Check Route 
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Intelligent ML Analytics Engine API is running!"})

# Upload Route
@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Only CSV files are allowed"}), 400

        run_id = str(uuid.uuid4())
        run_path = create_run_folder(run_id)

        file_path = os.path.join(run_path, "raw.csv")
        file.save(file_path)

        update_status(run_id, "uploaded")

        return jsonify({
            "message": "File uploaded successfully",
            "run_id": run_id,
            "file_path": file_path
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
# Preprocess
@app.route("/preprocess", methods=["POST"])
def preprocess_route():
    try:
        data = request.get_json()

        if not data or "run_id" not in data or "target" not in data:
            return jsonify({"error": "run_id and target are required"}), 400

        run_id = data["run_id"]
        target = data["target"]

        run_path = os.path.join("runs", run_id)

        raw_path = os.path.join(run_path, "raw.csv")
        processed_path = os.path.join(run_path, "processed.csv")
        summary_path = os.path.join(run_path, "preprocessing_summary.json")

        if not os.path.exists(raw_path):
            return jsonify({"error": "Raw dataset not found"}), 404

        update_status(run_id, "preprocessing_started")

        preprocess_dataset(
            input_path=raw_path,
            output_path=processed_path,
            summary_path=summary_path,
            target_name=target
        )

        update_status(run_id, "preprocessing_complete")

        return jsonify({
            "message": "Preprocessing complete",
            "run_id": run_id,
            "processed_file": processed_path
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
# EDA Route
@app.route("/eda", methods=["POST"])
def eda_route():
    try:
        data = request.get_json()

        if not data or "run_id" not in data:
            return jsonify({"error": "run_id is required"}), 400

        run_id = data["run_id"]
        run_path = os.path.join("runs", run_id)

        processed_path = os.path.join(run_path, "processed.csv")
        eda_dir = os.path.join(run_path, "eda_plots")
        summary_path = os.path.join(run_path, "eda_summary.json")

        if not os.path.exists(processed_path):
            return jsonify({"error": "processed.csv not found"}), 404

        update_status(run_id, "eda_started")

        df = load_csv_safely(processed_path)

        eda_summary = generate_eda(
            df=df,
            save_dir=eda_dir,
            summary_path=summary_path
        )

        update_status(run_id, "eda_complete")

        return jsonify({
            "message": "EDA complete",
            "run_id": run_id,
            "summary_file": summary_path,
            "plots_folder": eda_dir,
            "eda_overview": eda_summary
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Model Training Route
@app.route("/train", methods=["POST"])
def train_route():
    try:
        data = request.get_json()

        if not data or "run_id" not in data or "target" not in data:
            return jsonify({"error": "run_id and target are required"}), 400

        run_id = data["run_id"]
        target = data["target"]

        run_path = os.path.join("runs", run_id)
        processed_path = os.path.join(run_path, "processed.csv")
        models_dir = os.path.join(run_path, "models")

        if not os.path.exists(processed_path):
            return jsonify({"error": "processed.csv not found"}), 404

        update_status(run_id, "training_started")

        executor.submit(
            run_training_job,
            processed_path,
            models_dir,
            target,
            run_id
        )

        return jsonify({
            "message": "Training started",
            "run_id": run_id
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_training_job(processed_path, models_dir, target, run_id):
    try:
        summary = train_models(processed_path, models_dir, target)
        update_status(run_id, "training_complete")
    except Exception as e:
        update_status(run_id, f"training_failed: {str(e)}")
        
# Evaluation Route
@app.route("/evaluate", methods=["POST"])
def evaluate_route():
    try:
        data = request.get_json()

        if not data or "run_id" not in data or "target" not in data:
            return jsonify({"error": "run_id and target are required"}), 400

        run_id = data["run_id"]
        target = data["target"]

        run_path = os.path.join("runs", run_id)
        processed_path = os.path.join(run_path, "processed.csv")
        models_dir = os.path.join(run_path, "models")
        eval_dir = os.path.join(run_path, "evaluation")

        if not os.path.exists(processed_path):
            return jsonify({"error": "processed.csv not found"}), 404

        if not os.path.exists(os.path.join(models_dir, "best_model.pkl")):
            return jsonify({"error": "best_model.pkl not found"}), 404

        update_status(run_id, "evaluation_started")

        results = evaluate_model(
            processed_path=processed_path,
            models_dir=models_dir,
            target=target,
            eval_dir=eval_dir
        )

        update_status(run_id, "evaluation_complete")

        return jsonify({
            "message": "Evaluation complete",
            "run_id": run_id,
            "results": results,
            "evaluation_folder": eval_dir
        }), 200

    except Exception as e:
        update_status(run_id, f"evaluation_failed: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
# Report Generation Route
@app.route("/report", methods=["POST"])
def report_route():
    try:
        data = request.get_json()

        if not data or "run_id" not in data:
            return jsonify({"error": "run_id is required"}), 400

        run_id = data["run_id"]
        run_path = os.path.join("runs", run_id)
        output_file = os.path.join(run_path, "final_report.pdf")

        update_status(run_id, "report_generating")

        generate_report(run_id, run_path, output_file)

        update_status(run_id, "report_ready")

        return jsonify({
            "message": "Report generated",
            "run_id": run_id,
            "report_path": output_file
        }), 200

    except Exception as e:
        update_status(run_id, f"report_failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Main
if __name__ == "__main__":
    app.run(debug=True)
