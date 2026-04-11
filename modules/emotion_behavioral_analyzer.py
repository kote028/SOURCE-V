"""
Emotion & Behavioral Analyzer

Two signals:
1. Emotion consistency — real faces show coherent, temporally smooth
   emotional transitions. Deepfakes often freeze emotion or jump abruptly.
2. Behavioral cues — blink rate, head pose micro-movements, eye saccades.
   Fake generators either over-smooth or introduce noise in these signals.

Architecture: ResNet-based emotion CNN + head-pose LSTM.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────
# Emotion CNN (ResNet-style lightweight)
# ─────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class EmotionCNN(nn.Module):
    """
    Lightweight ResNet that classifies 7 basic emotions from a face crop.
    Emotions: neutral, happy, sad, angry, surprised, fearful, disgusted.
    Input: (3, 48, 48) face crop.
    """

    EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted']

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),   # 24x24
        )
        self.body = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64),
            nn.MaxPool2d(2),   # 12x12
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(self.EMOTIONS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x)))


# ─────────────────────────────────────────────────────────────────
# Behavioral LSTM — head pose + blink + saccade signals
# ─────────────────────────────────────────────────────────────────

class BehavioralLSTM(nn.Module):
    """
    Temporal model over behavioral signals:
    [pitch, yaw, roll, blink_l, blink_r, saccade_x, saccade_y] → 7 dims per frame.
    """

    def __init__(self, input_dim: int = 7, hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.3,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out.mean(dim=1))


# ─────────────────────────────────────────────────────────────────
# Head pose estimation from landmarks
# ─────────────────────────────────────────────────────────────────

# 3D model points for solvePnP (canonical face model)
MODEL_POINTS_3D = np.array([
    [0.0,    0.0,    0.0],     # nose tip
    [0.0,   -330.0, -65.0],    # chin
    [-225.0, 170.0, -135.0],   # left eye corner
    [225.0,  170.0, -135.0],   # right eye corner
    [-150.0, -150.0, -125.0],  # left mouth corner
    [150.0,  -150.0, -125.0],  # right mouth corner
], dtype=np.float64)

# MediaPipe landmark indices for the 6 points above
POSE_LM_IDX = [1, 152, 263, 33, 287, 57]


def estimate_head_pose(landmarks: np.ndarray,
                       frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
    """
    Returns (pitch, yaw, roll) in degrees.
    Uses solvePnP with a canonical 3D face model.
    """
    if landmarks is None or len(landmarks) < 300:
        return 0.0, 0.0, 0.0

    h, w = frame_shape
    focal = w
    cam   = np.array([[focal, 0, w / 2],
                      [0, focal, h / 2],
                      [0, 0, 1]], dtype=np.float64)
    dist  = np.zeros((4, 1))

    img_pts = np.array(
        [[landmarks[i][0] * w, landmarks[i][1] * h] for i in POSE_LM_IDX],
        dtype=np.float64,
    )

    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D, img_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    proj    = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
    pitch, yaw, roll = euler.flatten()[:3]
    return float(pitch), float(yaw), float(roll)


# ─────────────────────────────────────────────────────────────────
# Blink detection
# ─────────────────────────────────────────────────────────────────

LEFT_EYE_LM  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LM = [33,  160, 158, 133, 153, 144]


def eye_aspect_ratio(landmarks: np.ndarray,
                     eye_indices: List[int],
                     frame_shape: Tuple[int, int]) -> float:
    """
    Eye Aspect Ratio (EAR) — drops near 0 when the eye is closed.
    """
    if landmarks is None or len(landmarks) < 400:
        return 0.3
    h, w = frame_shape
    pts = np.array([[landmarks[i][0] * w, landmarks[i][1] * h]
                    for i in eye_indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return float((A + B) / (2.0 * C + 1e-8))


# ─────────────────────────────────────────────────────────────────
# EmotionBehavioralAnalyzer
# ─────────────────────────────────────────────────────────────────

class EmotionBehavioralAnalyzer:
    """
    Combines emotion consistency and behavioral signal analysis.
    """

    BLINK_THRESHOLD = 0.21  # EAR below this → blink

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.emotion_cnn    = EmotionCNN().to(self.device).eval()
        self.behavior_lstm  = BehavioralLSTM(input_dim=7, hidden=64).to(self.device).eval()

    def load_weights(self, emotion_path: str, behavior_path: str):
        self.emotion_cnn.load_state_dict(
            torch.load(emotion_path, map_location=self.device))
        self.behavior_lstm.load_state_dict(
            torch.load(behavior_path, map_location=self.device))

    def _crop_face(self, frame: np.ndarray, landmarks: Optional[np.ndarray],
                   pad: int = 10) -> np.ndarray:
        if landmarks is None:
            return np.zeros((48, 48, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        xs = landmarks[:, 0] * w
        ys = landmarks[:, 1] * h
        x1 = max(0,  int(xs.min()) - pad)
        y1 = max(0,  int(ys.min()) - pad)
        x2 = min(w,  int(xs.max()) + pad)
        y2 = min(h,  int(ys.max()) + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((48, 48, 3), dtype=np.uint8)
        return cv2.resize(crop, (48, 48))

    def _face_to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def analyze_sequence(self,
                         frames: List[np.ndarray],
                         landmarks_seq: List[Optional[np.ndarray]]) -> dict:
        """
        Args:
            frames:         BGR frames.
            landmarks_seq:  MediaPipe normalized landmarks or None.

        Returns:
            dict with score, emotion_consistency_score, blink_rate,
                  behavioral_lstm_fake_prob, head_pose_stats.
        """
        emotion_probs_seq: List[np.ndarray] = []
        behavioral_seq:    List[np.ndarray] = []
        blink_frames: int = 0

        for frame, lm in zip(frames, landmarks_seq):
            h, w = frame.shape[:2]
            frame_shape = (h, w)

            # Emotion
            crop = self._crop_face(frame, lm)
            with torch.no_grad():
                logits = self.emotion_cnn(self._face_to_tensor(crop))
                ep     = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            emotion_probs_seq.append(ep)

            # Head pose
            pitch, yaw, roll = estimate_head_pose(lm, frame_shape)

            # Blink (EAR)
            ear_l = eye_aspect_ratio(lm, LEFT_EYE_LM, frame_shape)
            ear_r = eye_aspect_ratio(lm, RIGHT_EYE_LM, frame_shape)
            blink_l = 1.0 if ear_l < self.BLINK_THRESHOLD else 0.0
            blink_r = 1.0 if ear_r < self.BLINK_THRESHOLD else 0.0
            if blink_l or blink_r:
                blink_frames += 1

            # Saccade proxy: change in EAR
            prev_ear = behavioral_seq[-1][3] if behavioral_seq else ear_l
            saccade_x = ear_l - prev_ear
            saccade_y = ear_r - (behavioral_seq[-1][4] if behavioral_seq else ear_r)

            behavioral_seq.append(
                [pitch / 90.0, yaw / 90.0, roll / 90.0,
                 blink_l, blink_r, saccade_x, saccade_y]
            )

        # LSTM over behavioral features
        beh_arr = np.array(behavioral_seq, dtype=np.float32)  # (T, 7)
        beh_t   = torch.from_numpy(beh_arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.behavior_lstm(beh_t)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Emotion consistency: real faces have smooth emotion transitions
        em_arr = np.stack(emotion_probs_seq)          # (T, 7)
        dominant = em_arr.argmax(axis=1)
        # Frame-to-frame dominant emotion changes = inconsistency
        changes = int(np.sum(np.diff(dominant) != 0))
        inconsistency = changes / max(1, len(dominant) - 1)

        # High inconsistency → fake (emotion jumping without motivation)
        # Very low inconsistency (frozen emotion) → also suspicious
        emotion_fake_score = float(
            np.clip(inconsistency * 2.0, 0, 1) if inconsistency > 0.5
            else np.clip(0.4 - inconsistency, 0, 1)
        )

        # Blink rate heuristic — real: ~15 blinks/min = ~0.5/sec
        # Fake generators often blink rarely (< 5/min) or not at all
        fps_estimate = 30.0
        blink_rate_per_min = (blink_frames / len(frames)) * fps_estimate * 60
        blink_score = float(np.clip(1.0 - blink_rate_per_min / 15.0, 0, 1))

        # Head pose micro-movement consistency
        # Fake: either too smooth (no jitter) or noisy
        pose_arr = beh_arr[:, :3]  # pitch, yaw, roll
        pose_jitter = float(np.mean(np.std(np.diff(pose_arr, axis=0), axis=0)))
        # Very low jitter = unnaturally frozen
        pose_score = float(np.clip(0.3 - pose_jitter, 0, 1)) if pose_jitter < 0.3 else 0.0

        fake_prob = float(
            0.4 * probs[1] +
            0.2 * emotion_fake_score +
            0.2 * blink_score +
            0.2 * pose_score
        )

        return {
            'score': fake_prob,
            'confidence': float(max(probs)),
            'behavioral_lstm_fake_prob': float(probs[1]),
            'emotion_inconsistency': inconsistency,
            'emotion_fake_score': emotion_fake_score,
            'blink_rate_per_min': blink_rate_per_min,
            'blink_fake_score': blink_score,
            'head_pose_jitter': pose_jitter,
            'dominant_emotions': dominant.tolist(),
        }
