import librosa
import numpy as np

def analyze_voice(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Simple stats
        mean = np.mean(mfcc)
        std = np.std(mfcc)

        # Heuristic scoring
        score = float(np.clip((std / (abs(mean) + 1e-5)), 0, 1))

        return score

    except Exception:
        return None