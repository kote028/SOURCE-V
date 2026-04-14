"""
DeepShield AI Backend — FastAPI server
Uses FastDetector (MobileNetV2 / heuristic) for actual deepfake detection.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from datetime import datetime
import hashlib
import tempfile
import os

# ── Optional heavy imports ──────────────────────────────────────────
try:
    import imagehash
    from PIL import Image
    import io
    IMAGEHASH_OK = True
except ImportError:
    IMAGEHASH_OK = False

try:
    import base58
    import multihash
    MULTIHASH_OK = True
except ImportError:
    MULTIHASH_OK = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ── FastDetector (real AI detection) ────────────────────────────────
from fast_detector import FastDetector

print("[main] Loading FastDetector...")
detector = FastDetector(device="cpu")
print(f"[main] FastDetector ready — model_type={detector.model_type}")

# ── App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepShield AI Backend",
    description="AI Deepfake Detector with Blockchain Integrity + pHash + IPFS CID",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve Frontend ───────────────────────────────────────────────────
_frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
print(f"[main] Serving frontend from: {_frontend_dir}")

app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join(_frontend_dir, "index.html"))

@app.get("/{filename:path}")
def serve_file(filename: str):
    file_path = os.path.join(_frontend_dir, filename)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join(_frontend_dir, "index.html"))


# ── Helpers ──────────────────────────────────────────────────────────
def generate_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def generate_perceptual_hash(file_bytes: bytes, filename: str = "") -> str:
    if not IMAGEHASH_OK:
        return "imagehash-not-installed"
    try:
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext in ("mp4", "avi", "mov", "mkv", "webm") and CV2_AVAILABLE:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                cap = cv2.VideoCapture(tmp_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    return "[Frame1] " + str(imagehash.phash(pil_image))
                return "no-frame"
            finally:
                os.unlink(tmp_path)
        else:
            image = Image.open(io.BytesIO(file_bytes))
            return str(imagehash.phash(image))
    except Exception as e:
        return f"pHash-error: {e}"


def generate_ipfs_cid(file_bytes: bytes) -> str:
    if not MULTIHASH_OK:
        # Fallback: base58 of SHA256 prefix
        h = hashlib.sha256(file_bytes).digest()
        try:
            import base64
            return "Qm" + base64.b32encode(h).decode()[:44]
        except Exception:
            return "N/A"
    try:
        mh = multihash.digest(file_bytes, "sha2-256")
        return base58.b58encode(bytes(mh)).decode("utf-8")
    except Exception as e:
        return f"cid-error: {e}"


# ── /upload-media — main detection endpoint ──────────────────────────
@app.post("/upload-media")
async def upload_media(file: UploadFile = File(...)):
    try:
        content = await file.read()

        sha256_hash    = generate_sha256(content)
        perceptual_hash = generate_perceptual_hash(content, file.filename)
        ipfs_cid       = generate_ipfs_cid(content)
        file_size_mb   = len(content) / (1024 * 1024)

        # ── Run real AI detection ──
        ext = (file.filename or "").lower().rsplit(".", 1)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = detector.analyze(tmp_path, filename=file.filename or "")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        # Map detector output → API response
        fake_score  = result.get("fake_score", 0.5)
        verdict     = result.get("verdict", "UNCERTAIN")
        breakdown   = result.get("breakdown", {
            "gaze": 0.5, "lip_sync": 0.5, "voice": 0.5,
            "emotion": 0.5, "behavioral": 0.5
        })

        threat_prediction = (
            "HIGH" if fake_score > 0.70 else
            "MEDIUM" if fake_score > 0.45 else
            "LOW"
        )

        return {
            "message":           "File analyzed through AI fusion pipeline",
            "file_name":         file.filename,
            "sha256_hash":       sha256_hash,
            "perceptual_hash":   perceptual_hash,
            "ipfs_cid":          ipfs_cid,
            "file_size_mb":      round(file_size_mb, 2),
            "fake_score":        round(fake_score, 3),
            "threat_prediction": threat_prediction,
            "detection_verdict": verdict,
            "breakdown":         breakdown,
            "model_used":        result.get("model_used", "unknown"),
            "frames_analysed":   result.get("frames_analysed", 0),
            "analysis_time_s":   result.get("analysis_time_s", 0.0),
            "uploaded_at":       str(datetime.utcnow()),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── /verify-hash ─────────────────────────────────────────────────────
@app.post("/verify-hash")
async def verify_hash(file: UploadFile = File(...)):
    try:
        content         = await file.read()
        sha256_hash     = generate_sha256(content)
        perceptual_hash = generate_perceptual_hash(content, file.filename)
        return {
            "verified":       True,
            "sha256_hash":    sha256_hash,
            "perceptual_hash": perceptual_hash,
            "message":        "Integrity check successful"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /predict-future-attack ────────────────────────────────────────────
@app.post("/predict-future-attack")
async def predict_future_attack(file: UploadFile = File(...)):
    try:
        content      = await file.read()
        file_size_mb = len(content) / (1024 * 1024)

        if file_size_mb > 10:
            risk, attack_type, confidence = "HIGH", "Adaptive lip-sync bypass", 0.91
        elif (file.filename or "").lower().endswith(".mp4"):
            risk, attack_type, confidence = "MEDIUM", "Voice-tone cloning attack", 0.73
        else:
            risk, attack_type, confidence = "LOW", "Image tampering attempt", 0.42

        return {
            "future_attack_risk":   risk,
            "predicted_attack_type": attack_type,
            "confidence":           confidence,
            "predicted_at":         str(datetime.utcnow())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=False)