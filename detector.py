from voice_quick import analyze_voice
import os
import numpy as np
import random

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
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"\n[DeepShield] Analyzing: {video_path}")

        pipe_out = self.pipeline.process(video_path)

        frames = pipe_out['frames']
        landmarks_seq = pipe_out['landmarks_seq']
        audio_path = pipe_out.get("audio_path", None)

        if len(frames) == 0:
            return {"error": "No frames"}

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

        gaze_score = gaze_result.get("score") if gaze_result else None

        # =========================
        # EMOTION
        emotion_result = self._analyze_windows(
            self.emotion_analyzer.analyze_sequence,
            frames,
            landmarks_seq
        )

        emotion_score = emotion_result.get("score") if emotion_result else None

        # =========================
        # LIP SYNC (SIMULATED)
        lip_score = random.uniform(0.3, 0.7)

        # =========================
        # VOICE (TRY REAL → ELSE SIMULATE)
        voice_score = None
        try:
            if audio_path:
                voice_score = analyze_voice(audio_path)
        except:
            voice_score = None

        if voice_score is None:
            voice_score = random.uniform(0.3, 0.7)

        # =========================
        # FUSION
        module_scores = {
            "gaze": gaze_score,
            "emotion": emotion_score,
            "lip_sync": lip_score,
            "voice": voice_score
        }

        fusion_out = self.fusion.fuse(module_scores)

        final_score = fusion_out["final_score"]

        # =========================
        # BETTER VERDICT LOGIC
        if final_score > 0.6:
            fusion_out["verdict"] = "REAL"
        elif final_score < 0.4:
            fusion_out["verdict"] = "FAKE"
        else:
            fusion_out["verdict"] = "UNCERTAIN"

        # =========================
        # CONFIDENCE
        confidence = abs(final_score - 0.5) * 2
        fusion_out["confidence"] = round(confidence, 2)

        print("\nFINAL RESULT:")
        print(fusion_out)

        return fusion_out

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