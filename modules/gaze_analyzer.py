"""
Gaze Analyzer — Based on "Where Do Deep Fakes Look?" (Demir & Ciftci, 2021)

Extracts geometric, visual, temporal, spectral, and metric eye/gaze features.
Uses CNN for per-frame feature extraction + LSTM for temporal consistency.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import welch
from scipy.signal import correlate
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Per-frame eye feature extractor (CNN backbone)
# ---------------------------------------------------------------------------

class GazeCNN(nn.Module):
    """
    Lightweight CNN that ingests a 64x32 eye-region crop and outputs
    a 64-dim feature vector covering color, shape, and texture of the
    iris/pupil region.
    """

    def __init__(self, out_features: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 32x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 16x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                           # 4x4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# Temporal consistency model (LSTM over a sequence of frames)
# ---------------------------------------------------------------------------

class GazeLSTM(nn.Module):
    """
    Bidirectional LSTM that captures temporal consistency of gaze signals
    over an ω-length window. Fake videos break temporal smoothness —
    this module catches over-smoothing, noise bursts, and missing saccades.
    """

    def __init__(self, input_size: int = 64, hidden: int = 128,
                 layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64),   # ×2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)         # mean-pool across time
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Feature extraction helpers (geometric & spectral)
# ---------------------------------------------------------------------------

def extract_iris_features(eye_crop: np.ndarray) -> dict:
    """
    Extract color (CIELab) and area features from a single eye crop.
    Returns dict with keys: color_l, color_a, color_b, area_iris, area_pupil.
    """
    if eye_crop is None or eye_crop.size == 0:
        return {k: 0.0 for k in ['color_l', 'color_a', 'color_b',
                                   'area_iris', 'area_pupil']}

    lab = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Estimate iris via circular Hough on grayscale
    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1,
        minDist=10, param1=50, param2=20,
        minRadius=3, maxRadius=min(eye_crop.shape[:2]) // 2,
    )

    area_iris = area_pupil = 0.0
    color_l = color_a = color_b = 0.0

    if circles is not None:
        cx, cy, r = circles[0, 0].astype(int)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        # Pupil is roughly half the iris radius
        inner_mask = np.zeros_like(mask)
        cv2.circle(inner_mask, (cx, cy), max(1, r // 2), 255, -1)
        iris_mask = cv2.subtract(mask, inner_mask)

        area_iris = float(np.sum(iris_mask > 0))
        area_pupil = float(np.sum(inner_mask > 0))
        if area_iris > 0:
            color_l = float(np.mean(lab[:, :, 0][iris_mask > 0])) / 255.0
            color_a = float(np.mean(lab[:, :, 1][iris_mask > 0])) / 255.0
            color_b = float(np.mean(lab[:, :, 2][iris_mask > 0])) / 255.0

    h, w = eye_crop.shape[:2]
    norm = float(h * w) + 1e-6
    return {
        'color_l': color_l,
        'color_a': color_a,
        'color_b': color_b,
        'area_iris': area_iris / norm,
        'area_pupil': area_pupil / norm,
    }


def compute_gaze_vector(landmarks: np.ndarray,
                        left_eye_idx: List[int],
                        right_eye_idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate left/right gaze vectors from eye landmark centroids.
    Returns two unit vectors (3D) representing gaze direction.
    """
    def centroid(pts):
        return np.mean(pts, axis=0)

    if landmarks is None or len(landmarks) == 0:
        return np.zeros(3), np.zeros(3)

    left_pts  = landmarks[left_eye_idx]  if len(left_eye_idx)  else landmarks[:3]
    right_pts = landmarks[right_eye_idx] if len(right_eye_idx) else landmarks[3:6]

    lc = centroid(left_pts)
    rc = centroid(right_pts)

    # Simplified: gaze direction approximated from iris-centre offset
    # In production, replace with OpenFace or 3D eye model (Wood et al., 2015)
    diff = rc - lc
    mag = np.linalg.norm(diff) + 1e-8
    gaze_l = np.array([diff[0] / mag, diff[1] / mag, 0.0])
    gaze_r = np.array([-diff[0] / mag, -diff[1] / mag, 0.0])
    return gaze_l, gaze_r


def vergence_point_error(gaze_l: np.ndarray, gaze_r: np.ndarray,
                          origin_l: np.ndarray, origin_r: np.ndarray) -> float:
    """
    Compute the distance between the closest points of two 3D gaze rays
    (least-squares approximation). Real eyes → ~0; fakes → large error.
    Based on the geometric attestation in Section 4.2 of the paper.
    """
    d = gaze_l - gaze_r
    w0 = origin_l - origin_r
    a = np.dot(gaze_l, gaze_l)
    b = np.dot(gaze_l, gaze_r)
    c = np.dot(gaze_r, gaze_r)
    d_ = np.dot(gaze_l, w0)
    e  = np.dot(gaze_r, w0)
    denom = a * c - b * b + 1e-8
    sc = (b * e - c * d_) / denom
    tc = (a * e - b * d_) / denom
    p1 = origin_l + sc * gaze_l
    p2 = origin_r + tc * gaze_r
    return float(np.linalg.norm(p1 - p2))


def spectral_features(signal: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """
    Power spectral density of a 1-D temporal signal.
    Fake features → richer (noisier) spectra than real ones (Section 4.4).
    Returns 32-bin PSD vector.
    """
    if len(signal) < 4:
        return np.zeros(32)
    nperseg = min(len(signal), 64)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    # Resize to fixed 32 bins via interpolation
    indices = np.linspace(0, len(psd) - 1, 32).astype(int)
    return psd[indices].astype(np.float32)


def cross_correlation_features(sig_l: np.ndarray, sig_r: np.ndarray) -> np.ndarray:
    """
    Normalised cross-correlation between left/right eye signals.
    Symmetric real eyes → near-ideal correlation; fake eyes → degraded.
    Metric features M from Section 3.4.
    Returns 16-bin correlation feature.
    """
    if len(sig_l) < 2 or len(sig_r) < 2:
        return np.zeros(16)
    corr = correlate(sig_l, sig_r, mode='full')
    norm = np.max(np.abs(corr)) + 1e-8
    corr = corr / norm
    # Take center ±8 lags
    mid = len(corr) // 2
    window = corr[max(0, mid - 8): mid + 8]
    if len(window) < 16:
        window = np.pad(window, (0, 16 - len(window)))
    return window[:16].astype(np.float32)


# ---------------------------------------------------------------------------
# High-level GazeAnalyzer — wraps everything above
# ---------------------------------------------------------------------------

class GazeAnalyzer:
    """
    End-to-end gaze-based fake detector.
    Call analyze_sequence(frames, landmarks_seq) to get a fakeness score.
    """

    def __init__(self, device: str = 'cpu', sequence_len: int = 32):
        self.device = torch.device(device)
        self.seq_len = sequence_len
        self.cnn   = GazeCNN(out_features=64).to(self.device).eval()
        self.lstm  = GazeLSTM(input_size=64, hidden=128).to(self.device).eval()

    def load_weights(self, cnn_path: str, lstm_path: str):
        self.cnn.load_state_dict(torch.load(cnn_path, map_location=self.device))
        self.lstm.load_state_dict(torch.load(lstm_path, map_location=self.device))
        self.cnn.eval()
        self.lstm.eval()

    def _crop_eye(self, frame: np.ndarray,
                  landmarks: np.ndarray,
                  eye_indices: List[int],
                  pad: int = 6) -> np.ndarray:
        """Crop a tight eye region from the frame using landmark indices."""
        pts = landmarks[eye_indices]
        x1, y1 = pts.min(axis=0).astype(int) - pad
        x2, y2 = pts.max(axis=0).astype(int) + pad
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((32, 64, 3), dtype=np.uint8)
        return cv2.resize(crop, (64, 32))

    def _frame_to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)   # (1, 3, H, W)

    def analyze_sequence(self,
                         frames: List[np.ndarray],
                         landmarks_seq: List[Optional[np.ndarray]],
                         left_eye_idx: List[int],
                         right_eye_idx: List[int]) -> dict:
        """
        Main analysis entry point.

        Args:
            frames:          List of BGR frames (length ω).
            landmarks_seq:   Corresponding facial landmark arrays (or None).
            left_eye_idx:    Indices into landmark array for left eye.
            right_eye_idx:   Indices for right eye.

        Returns:
            dict with keys: score (0=real, 1=fake), confidence, features.
        """
        cnn_features: List[np.ndarray] = []
        iris_feats_l: List[dict] = []
        iris_feats_r: List[dict] = []
        gaze_errors:  List[float] = []

        for frame, lm in zip(frames, landmarks_seq):
            # Eye crops + CNN features
            if lm is not None:
                crop_l = self._crop_eye(frame, lm, left_eye_idx)
                crop_r = self._crop_eye(frame, lm, right_eye_idx)
                fl = extract_iris_features(crop_l)
                fr = extract_iris_features(crop_r)

                # 3D vergence error
                origin_l = np.array([lm[left_eye_idx[0]][0],  lm[left_eye_idx[0]][1],  0.0])
                origin_r = np.array([lm[right_eye_idx[0]][0], lm[right_eye_idx[0]][1], 0.0])
                gl, gr = compute_gaze_vector(lm, left_eye_idx, right_eye_idx)
                err = vergence_point_error(gl, gr, origin_l, origin_r)

                with torch.no_grad():
                    feat = self.cnn(self._frame_to_tensor(crop_l)).cpu().numpy()
            else:
                fl = fr = {k: 0.0 for k in ['color_l', 'color_a', 'color_b',
                                              'area_iris', 'area_pupil']}
                err = 0.0
                feat = np.zeros((1, 64), dtype=np.float32)

            iris_feats_l.append(fl)
            iris_feats_r.append(fr)
            gaze_errors.append(err)
            cnn_features.append(feat[0])

        # Temporal signals
        iris_area_l = np.array([f['area_iris'] for f in iris_feats_l])
        iris_area_r = np.array([f['area_iris'] for f in iris_feats_r])
        gaze_err_sig = np.array(gaze_errors)

        # Spectral features (Section 3.5)
        psd_area_l = spectral_features(iris_area_l)
        psd_area_r = spectral_features(iris_area_r)
        psd_gaze   = spectral_features(gaze_err_sig)

        # Metric features — cross-correlation (Section 3.4)
        xcorr = cross_correlation_features(iris_area_l, iris_area_r)

        # LSTM temporal classification
        seq = np.stack(cnn_features, axis=0)          # (T, 64)
        seq_t = torch.from_numpy(seq).unsqueeze(0).float().to(self.device)  # (1, T, 64)

        with torch.no_grad():
            logits = self.lstm(seq_t)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Geometric heuristic: high average vergence error → fake signal
        geo_score = float(np.clip(np.mean(gaze_err_sig) / 50.0, 0, 1))

        # Spectral heuristic: higher PSD variance → fake
        psd_var = float(np.var(psd_area_l) + np.var(psd_area_r))
        spec_score = float(np.clip(psd_var * 100, 0, 1))

        # Combine LSTM prob + geometric + spectral
        fake_prob = float(0.6 * probs[1] + 0.2 * geo_score + 0.2 * spec_score)

        return {
            'score': fake_prob,
            'confidence': float(max(probs)),
            'lstm_fake_prob': float(probs[1]),
            'geo_vergence_score': geo_score,
            'spectral_score': spec_score,
            'xcorr_mean': float(np.mean(np.abs(xcorr))),
            'psd_area_l': psd_area_l.tolist(),
            'psd_area_r': psd_area_r.tolist(),
        }
