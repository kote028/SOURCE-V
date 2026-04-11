"""
Voice Analyzer — Audio Authenticity via MFCC + Spectrogram CNN + LSTM

Synthetic voices (TTS, voice conversion) leave characteristic artifacts
in spectral features. This module detects them using:
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - Mel-spectrogram image → CNN
  - Temporal LSTM over MFCC sequences
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
import librosa.display
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Spectrogram CNN
# ---------------------------------------------------------------------------

class SpectrogramCNN(nn.Module):
    """
    Processes a mel-spectrogram image (1, H, W) and extracts audio texture
    features. Fake/synthetic audio often shows unnaturally smooth or
    repetitive spectral patterns.
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            # Input: (1, 64, T) — single-channel mel-spectrogram
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),                                # 32 x T/2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),  nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),                                # 16 x T/4
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# MFCC temporal LSTM
# ---------------------------------------------------------------------------

class AudioLSTM(nn.Module):
    """
    Bidirectional LSTM over a sequence of MFCC frames.
    Real voices show natural temporal evolution; synthetic voices can
    exhibit monotony, unnatural transitions, or periodicity artifacts.
    """

    def __init__(self, mfcc_dim: int = 40, hidden: int = 128,
                 num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            mfcc_dim, hidden,
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
# Audio feature extraction
# ---------------------------------------------------------------------------

def load_audio(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, sr_out = librosa.load(audio_path, sr=sr, mono=True)
    return y, sr_out


def extract_mfcc(y: np.ndarray, sr: int = 16000,
                 n_mfcc: int = 40, hop_length: int = 512) -> np.ndarray:
    """
    Extract MFCC features.
    Returns shape (T, n_mfcc) — one row per hop frame.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    # Add delta and delta-delta for richer representation
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # Stack: (3*n_mfcc, T) → transpose → (T, 3*n_mfcc) — use first n_mfcc only
    return mfcc.T.astype(np.float32)   # (T, n_mfcc)


def extract_mel_spectrogram(y: np.ndarray, sr: int = 16000,
                             n_mels: int = 64,
                             duration_sec: float = 3.0) -> np.ndarray:
    """
    Extract log-mel spectrogram as a 2D image (1, n_mels, T).
    Fixed-length: truncate or pad to duration_sec.
    """
    target_samples = int(sr * duration_sec)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    # Normalise to [0, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel[np.newaxis, :, :]   # (1, n_mels, T)


def spectral_flatness_score(y: np.ndarray, sr: int = 16000) -> float:
    """
    Spectral flatness (Wiener entropy). Synthetic audio often has unnaturally
    smooth spectra → higher flatness than natural voice.
    Returns a value in [0, 1] where higher = flatter = more suspicious.
    """
    flatness = librosa.feature.spectral_flatness(y=y)
    return float(np.mean(flatness))


def zero_crossing_rate_variance(y: np.ndarray) -> float:
    """
    Variance of zero-crossing rate over time.
    Natural speech → high variance; TTS / VC audio → lower variance.
    """
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return float(np.var(zcr))


def pitch_consistency_score(y: np.ndarray, sr: int = 16000) -> float:
    """
    Measure how unnaturally consistent the pitch is.
    Real speakers have natural pitch variation; synthetic voices can be
    monotonous or have phase-discontinuity artifacts.
    Score: higher = more suspicious.
    """
    try:
        f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        voiced_f0 = f0[voiced]
        if len(voiced_f0) < 10:
            return 0.5
        # Very low standard deviation in pitch → unnaturally monotone
        std = float(np.std(voiced_f0))
        # Typical std for real speech: 20-80 Hz; TTS can be < 5 Hz
        score = float(np.clip(1.0 - std / 40.0, 0, 1))
        return score
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# VoiceAnalyzer
# ---------------------------------------------------------------------------

class VoiceAnalyzer:
    """
    Detects synthetic/converted voices from an audio waveform or file path.
    """

    def __init__(self, device: str = 'cpu', sr: int = 16000, n_mfcc: int = 40):
        self.device = torch.device(device)
        self.sr     = sr
        self.n_mfcc = n_mfcc

        self.spec_cnn   = SpectrogramCNN(out_dim=64).to(self.device).eval()
        self.audio_lstm = AudioLSTM(mfcc_dim=n_mfcc, hidden=128).to(self.device).eval()

    def load_weights(self, cnn_path: str, lstm_path: str):
        self.spec_cnn.load_state_dict(torch.load(cnn_path, map_location=self.device))
        self.audio_lstm.load_state_dict(torch.load(lstm_path, map_location=self.device))

    def analyze_audio(self, audio_input) -> dict:
        """
        Analyze an audio clip.

        Args:
            audio_input: str (file path) or np.ndarray (waveform at self.sr).

        Returns:
            dict with score, confidence, and per-feature scores.
        """
        if isinstance(audio_input, str):
            y, _ = load_audio(audio_input, sr=self.sr)
        else:
            y = audio_input.astype(np.float32)

        # --- Spectrogram CNN stream ---
        mel_img = extract_mel_spectrogram(y, sr=self.sr)
        mel_t   = torch.from_numpy(mel_img).unsqueeze(0).to(self.device)   # (1, 1, H, W)
        with torch.no_grad():
            _ = self.spec_cnn(mel_t)                                        # (1, 64) features

        # --- MFCC LSTM stream ---
        mfcc = extract_mfcc(y, sr=self.sr, n_mfcc=self.n_mfcc)             # (T, 40)
        mfcc_t = torch.from_numpy(mfcc).unsqueeze(0).to(self.device)       # (1, T, 40)
        with torch.no_grad():
            logits = self.audio_lstm(mfcc_t)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # --- Hand-crafted signal-level heuristics ---
        flatness_score = spectral_flatness_score(y, self.sr)
        zcr_var        = zero_crossing_rate_variance(y)
        pitch_score    = pitch_consistency_score(y, self.sr)

        # Low ZCR variance → unnaturally flat → suspicious
        zcr_fake_score = float(np.clip(1.0 - zcr_var * 500, 0, 1))

        fake_prob = float(
            0.4 * probs[1] +
            0.2 * flatness_score +
            0.2 * pitch_score +
            0.2 * zcr_fake_score
        )

        # Also export per-frame MFCC for use in LipSyncAnalyzer
        # Resample mfcc to video frame rate (30 fps) from audio frame rate
        # Caller can use mfcc_for_lip_sync for that purpose
        return {
            'score': fake_prob,
            'confidence': float(max(probs)),
            'lstm_fake_prob': float(probs[1]),
            'spectral_flatness': flatness_score,
            'pitch_consistency_score': pitch_score,
            'zcr_fake_score': zcr_fake_score,
            'mfcc_for_lip_sync': mfcc,   # (T_audio, 40) — resample before use
        }
