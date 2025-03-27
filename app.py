import os
import requests
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Roboflow API Details
API_URL = "https://classify.roboflow.com/deepfake-detection-using-yolo/1"
API_KEY = "kMpECAgHOql5zwn7jPiJ"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Ensure it's an image file
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({"error": "Invalid file type. Please upload an image."}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Send image to Roboflow API
    with open(file_path, "rb") as image_file:
        response = requests.post(
            f"{API_URL}?api_key={API_KEY}",
            files={"file": image_file}
        )

    # Handle response
    if response.status_code == 200:
        result = response.json()
        if "predictions" in result and len(result["predictions"]) > 0:
            prediction = result["predictions"][0]
            label = prediction.get("class", "Unknown")
            confidence = round(prediction.get("confidence", 0), 3)
        else:
            label = "No clear result"
            confidence = 0.0

        return jsonify({"filename": file.filename, "prediction": label, "confidence": confidence})
    else:
        return jsonify({"error": "Failed to get prediction from API"}), 500

if __name__ == "__main__":
    app.run(debug=True)
