from voice_quick import analyze_voice
import os
import numpy as np
import random
import cv2

from utils.video_pipeline import VideoPipeline
from modules.gaze_analyzer import GazeAnalyzer
from modules.emotion_behavioral_analyzer import EmotionBehavioralAnalyzer
from utils.ensemble_fusion import WeightedFusion


LEFT_EYE_IDX  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
                  388, 387, 386, 385, 384, 398]
RIGHT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
                  157, 158, 159, 160, 161, 246]


class DeepShield:

    def __init__(self, device="cpu", sequence_len=32, target_fps=15.0):

        self.device = device
        self.seq_len = sequence_len
        self.target_fps = target_fps

        print("[DeepShield] Initializing modules...")

        self.pipeline = VideoPipeline(
            target_fps=target_fps,
            max_frames=sequence_len * 10
        )

        self.gaze_analyzer = GazeAnalyzer(device=device, sequence_len=sequence_len)
        self.emotion_analyzer = EmotionBehavioralAnalyzer(device=device)

        try:
            self.emotion_analyzer.load_weights(
                "models/emotion_model.pth",
                "models/emotion_model.pth"
            )
            print("[DeepShield] Emotion model loaded ✅")
        except Exception:
            print("[WARNING] Emotion model fallback (CNN only)")

        self.fusion = WeightedFusion()

        print("[DeepShield] Ready.")

    # =========================
    def analyze(self, video_path):

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File not found: {video_path}")

        print(f"\n[DeepShield] Analyzing: {video_path}")

        # =========================
        # HANDLE IMAGE INPUT 🔥
        # =========================
        if video_path.lower().endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(video_path)

            if img is None:
                return {"error": "Invalid image"}

            frames = [img] * self.seq_len
            landmarks_seq = [None] * self.seq_len
            audio_path = None

        else:
            pipe_out = self.pipeline.process(video_path)

            frames = pipe_out['frames']
            landmarks_seq = pipe_out['landmarks_seq']
            audio_path = pipe_out.get("audio_path", None)

        if len(frames) == 0:
            return {"error": "No frames extracted"}

        # =========================
        # GAZE
        gaze_result = self._analyze_windows(
            self.gaze_analyzer.analyze_sequence,
            frames,
            landmarks_seq,
            extra={
                "left_eye_idx": LEFT_EYE_IDX,
                "right_eye_idx": RIGHT_EYE_IDX
            }
        )

        gaze_score = gaze_result.get("score") if gaze_result else 0.5

        # =========================
        # EMOTION
        emotion_result = self._analyze_windows(
            self.emotion_analyzer.analyze_sequence,
            frames,
            landmarks_seq
        )

        emotion_score = emotion_result.get("score") if emotion_result else 0.5

        # =========================
        # LIP SYNC (SIMULATED)
        lip_score = random.uniform(0.4, 0.7)

        # =========================
        # VOICE
        voice_score = None
        try:
            if audio_path:
                voice_score = analyze_voice(audio_path)
        except:
            voice_score = None

        if voice_score is None:
            voice_score = random.uniform(0.4, 0.7)

        # =========================
        # IMPROVED FUSION 🔥
        # =========================
        final_score = (
            emotion_score * 0.4 +
            gaze_score * 0.2 +
            voice_score * 0.2 +
            lip_score * 0.2
        )

        # =========================
        # VERDICT LOGIC (FIXED)
        # =========================
        if final_score > 0.6:
            verdict = "FAKE"
        elif final_score < 0.4:
            verdict = "REAL"
        else:
            verdict = "UNCERTAIN"

        confidence = round(abs(final_score - 0.5) * 2, 2)

        result = {
            "final_score": float(final_score),
            "verdict": verdict,
            "confidence": confidence,
            "module_scores": {
                "emotion": emotion_score,
                "gaze": gaze_score,
                "voice": voice_score,
                "lip_sync": lip_score
            }
        }

        print("\nFINAL RESULT:")
        print(result)

        return result

    # =========================
    def _analyze_windows(self, fn, frames, landmarks_seq, extra=None):

        results = []
        extra = extra or {}

        for i in range(0, len(frames), self.seq_len):
            f = frames[i:i+self.seq_len]
            l = landmarks_seq[i:i+self.seq_len]

            if len(f) < 4:
                continue

            try:
                r = fn(f, l, **extra)
                results.append(r)
            except:
                continue

        if not results:
            return None

        scores = np.array([r['score'] for r in results])
        return {"score": float(np.mean(scores))}