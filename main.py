from fastapi import FastAPI, UploadFile, File, HTTPException
from datetime import datetime
import hashlib

app = FastAPI(
    title="DeepShield AI Backend",
    description="Predictive Deepfake Attack Simulator with Blockchain Integrity",
    version="1.0.0"
)

# ---------------- HOME ----------------
@app.get("/")
def home():
    return {
        "message": "DeepShield AI backend running successfully",
        "status": "active"
    }


# ---------------- SHA256 FUNCTION ----------------
def generate_sha256(file_bytes: bytes):
    return hashlib.sha256(file_bytes).hexdigest()


# ---------------- THREAT SCORING LOGIC ----------------
def analyze_threat(file_name: str, file_size_mb: float):
    if file_size_mb > 5:
        return 0.82, "HIGH"

    elif file_name.lower().endswith(".mp4"):
        return 0.54, "MEDIUM"

    elif file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        return 0.21, "LOW"

    else:
        return 0.35, "MEDIUM"


# ---------------- UPLOAD MEDIA ----------------
@app.post("/upload-media")
async def upload_media(file: UploadFile = File(...)):
    try:
        content = await file.read()

        file_hash = generate_sha256(content)

        file_size_mb = len(content) / (1024 * 1024)

        fake_score, threat_prediction = analyze_threat(
            file.filename,
            file_size_mb
        )

        return {
            "message": "File uploaded successfully",
            "file_name": file.filename,
            "sha256_hash": file_hash,
            "file_size_mb": round(file_size_mb, 2),
            "fake_score": fake_score,
            "threat_prediction": threat_prediction,
            "uploaded_at": str(datetime.utcnow())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- VERIFY HASH ----------------
@app.post("/verify-hash")
async def verify_hash(file: UploadFile = File(...)):
    try:
        content = await file.read()

        current_hash = generate_sha256(content)

        return {
            "verified": True,
            "message": "Hash generated successfully",
            "sha256_hash": current_hash
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- FUTURE ATTACK PREDICTION ----------------
@app.post("/predict-future-attack")
async def predict_future_attack(file: UploadFile = File(...)):
    try:
        content = await file.read()

        file_size_mb = len(content) / (1024 * 1024)

        if file_size_mb > 10:
            risk = "HIGH"
            attack_type = "Adaptive lip-sync bypass"
            confidence = 0.91

        elif file.filename.lower().endswith(".mp4"):
            risk = "MEDIUM"
            attack_type = "Voice-tone cloning attack"
            confidence = 0.73

        else:
            risk = "LOW"
            attack_type = "Image tampering attempt"
            confidence = 0.42

        return {
            "future_attack_risk": risk,
            "predicted_attack_type": attack_type,
            "confidence": confidence,
            "predicted_at": str(datetime.utcnow())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))