import os
import json

def create_run_folder(run_id):
    run_path = os.path.join("runs", run_id)

    if not os.path.exists(run_path):
        os.makedirs(run_path)

    return run_path

def update_status(run_id, status):
    run_path = os.path.join("runs", run_id)
    status_path = os.path.join(run_path, "status.json")

    data = {"status": status}

    with open(status_path, "w") as f:
        json.dump(data, f, indent=4)

def read_status(run_id):
    run_path = os.path.join("runs", run_id)
    status_path = os.path.join(run_path, "status.json")

    if not os.path.exists(status_path):
        return None

    with open(status_path, "r") as f:
        return json.load(f)
