from flask import Flask, request, jsonify, render_template
from inference_sdk import InferenceHTTPClient
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key = os.getenv("API_KEY")
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        result = client.run_workflow(
            workspace_name="amin-ddcti",
            workflow_id="maintest",
            images={"image": filepath},
            use_cache=True
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
