"""
Lip Sync Analyzer — Phoneme-Viseme Mismatch Detection

Detects temporal misalignment between audio phonemes and mouth shape (viseme).
Fake videos generated from audio-driven synthesis often have subtle sync errors.
Architecture: CNN extracts mouth shape features; LSTM aligns with audio features.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional


# ---------------------------------------------------------------------------
# Mouth shape CNN (visual stream)
# ---------------------------------------------------------------------------

class MouthCNN(nn.Module):
    """Extracts viseme (mouth shape) features from a 64x32 crop."""

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x8
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),                                   # 2x2
            nn.Flatten(),
            nn.Linear(64 * 4, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Audio-visual sync LSTM
# ---------------------------------------------------------------------------

class LipSyncLSTM(nn.Module):
    """
    Takes concatenated [viseme_feat | audio_feat] per frame and classifies
    whether the sequence is in sync (real) or out of sync (fake).
    """

    def __init__(self, visual_dim: int = 32, audio_dim: int = 32,
                 hidden: int = 128, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            visual_dim + audio_dim, hidden,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.3,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out.mean(dim=1))


# ---------------------------------------------------------------------------
# Mouth landmark geometry helpers
# ---------------------------------------------------------------------------

# MediaPipe mouth landmark indices (subset)
MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321,
               405, 314, 17, 84, 181, 91, 146]
MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318,
               402, 317, 14, 87, 178, 88, 95]


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """
    Mouth Aspect Ratio (MAR): ratio of mouth height to width.
    Sensitive to open/close transitions that should match audio.
    """
    if landmarks is None or len(landmarks) < 420:
        return 0.0
    outer = landmarks[MOUTH_OUTER]
    h = np.linalg.norm(outer[np.argmax(outer[:, 1])] - outer[np.argmin(outer[:, 1])])
    w = np.linalg.norm(outer[np.argmax(outer[:, 0])] - outer[np.argmin(outer[:, 0])])
    return float(h / (w + 1e-8))


def mouth_openness(landmarks: np.ndarray) -> float:
    """
    Vertical distance between upper and lower inner lip — proxy for
    phoneme openness class (open vowel vs. closed consonant).
    """
    if landmarks is None or len(landmarks) < 420:
        return 0.0
    inner = landmarks[MOUTH_INNER]
    top = inner[np.argmin(inner[:, 1])]
    bot = inner[np.argmax(inner[:, 1])]
    return float(np.linalg.norm(top - bot))


def crop_mouth(frame: np.ndarray,
               landmarks: np.ndarray,
               pad: int = 8) -> np.ndarray:
    """Crop the mouth region from a frame using facial landmarks."""
    if landmarks is None or len(landmarks) < 420:
        return np.zeros((32, 64, 3), dtype=np.uint8)
    pts = landmarks[MOUTH_OUTER]
    x1 = max(0, int(pts[:, 0].min()) - pad)
    y1 = max(0, int(pts[:, 1].min()) - pad)
    x2 = min(frame.shape[1], int(pts[:, 0].max()) + pad)
    y2 = min(frame.shape[0], int(pts[:, 1].max()) + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((32, 64, 3), dtype=np.uint8)
    return cv2.resize(crop, (64, 32))


# ---------------------------------------------------------------------------
# Dummy audio feature extractor (replace with real MFCC in voice_analyzer.py)
# ---------------------------------------------------------------------------

def placeholder_audio_features(num_frames: int, audio_dim: int = 32) -> np.ndarray:
    """
    Returns zero audio features. Replace by calling VoiceAnalyzer and
    resampling the MFCC features to the video frame rate.
    """
    return np.zeros((num_frames, audio_dim), dtype=np.float32)


# ---------------------------------------------------------------------------
# LipSyncAnalyzer
# ---------------------------------------------------------------------------

class LipSyncAnalyzer:
    """
    Detects phoneme-viseme mismatch across a video sequence.
    Pass both visual landmarks and (optionally) audio features per frame.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.mouth_cnn  = MouthCNN(out_dim=32).to(self.device).eval()
        self.sync_lstm  = LipSyncLSTM(visual_dim=32, audio_dim=32).to(self.device).eval()

    def load_weights(self, cnn_path: str, lstm_path: str):
        self.mouth_cnn.load_state_dict(torch.load(cnn_path, map_location=self.device))
        self.sync_lstm.load_state_dict(torch.load(lstm_path, map_location=self.device))

    def _to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def analyze_sequence(self,
                         frames: List[np.ndarray],
                         landmarks_seq: List[Optional[np.ndarray]],
                         audio_feats: Optional[np.ndarray] = None) -> dict:
        """
        Args:
            frames:         BGR frames.
            landmarks_seq:  Mediapipe face mesh landmarks per frame.
            audio_feats:    Optional (T, 32) MFCC features per frame.
                            If None, geometric heuristics are used only.
        Returns:
            dict with score, confidence, sync_shift (estimated lag in frames).
        """
        visual_seq: List[np.ndarray] = []
        mar_signal: List[float]       = []
        open_signal: List[float]      = []

        for frame, lm in zip(frames, landmarks_seq):
            crop  = crop_mouth(frame, lm)
            mar   = mouth_aspect_ratio(lm)
            opn   = mouth_openness(lm)
            with torch.no_grad():
                vf = self.mouth_cnn(self._to_tensor(crop)).cpu().numpy()[0]
            visual_seq.append(vf)
            mar_signal.append(mar)
            open_signal.append(opn)

        # Audio features (zero-padded if not provided)
        if audio_feats is None:
            audio_feats = placeholder_audio_features(len(frames))

        # Concatenate visual + audio for LSTM
        vis_arr = np.stack(visual_seq, axis=0)    # (T, 32)
        combined = np.concatenate([vis_arr, audio_feats[:len(vis_arr)]], axis=1)  # (T, 64)
        seq_t = torch.from_numpy(combined).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.sync_lstm(seq_t)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Geometric heuristic: compute temporal correlation between
        # MAR signal and audio energy. Low correlation → suspicious.
        audio_energy = np.sum(audio_feats ** 2, axis=1)[:len(mar_signal)]
        if np.std(audio_energy) > 1e-6 and np.std(mar_signal) > 1e-6:
            corr = np.corrcoef(mar_signal[:len(audio_energy)], audio_energy)[0, 1]
            sync_score = float(np.clip((1.0 - corr) / 2.0, 0, 1))
        else:
            sync_score = 0.5   # insufficient signal

        # Temporal jitter in mouth openness (fake generators smooth over transitions)
        mar_arr = np.array(mar_signal)
        jitter_score = float(np.clip(1.0 - np.std(np.diff(mar_arr)) * 20, 0, 1))

        fake_prob = float(0.5 * probs[1] + 0.3 * sync_score + 0.2 * jitter_score)

        # Cross-correlate MAR with energy to find temporal lag
        if len(audio_energy) > 1:
            cross = np.correlate(
                (mar_arr - mar_arr.mean()) / (mar_arr.std() + 1e-8),
                (audio_energy - audio_energy.mean()) / (audio_energy.std() + 1e-8),
                mode='full',
            )
            lag = int(np.argmax(cross)) - len(audio_energy) + 1
        else:
            lag = 0

        return {
            'score': fake_prob,
            'confidence': float(max(probs)),
            'lstm_fake_prob': float(probs[1]),
            'sync_score': sync_score,
            'mouth_jitter_score': jitter_score,
            'estimated_lag_frames': lag,
            'mar_mean': float(np.mean(mar_arr)),
            'mar_std': float(np.std(mar_arr)),
        }
