from flask import Flask, request, jsonify, send_from_directory
from detector import DeepShield
import os

# 🔥 Serve frontend folder
app = Flask(__name__, static_folder="SOURCE-V/frontend")

# Load model once
shield = DeepShield(device="cpu")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# 🔥 FRONTEND ROUTES (ONLY ONE "/")
# =========================
@app.route("/")
def serve_frontend():
    return send_from_directory("SOURCE-V/frontend", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("SOURCE-V/frontend", path)

# =========================
# 🔥 API ROUTE
# =========================
@app.route("/analyze", methods=["POST"])
def analyze_video():

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"})

    file = request.files["video"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        result = shield.analyze(filepath)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)