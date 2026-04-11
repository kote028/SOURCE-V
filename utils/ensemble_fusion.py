"""
Ensemble Score Fusion — Weighted Confidence Aggregation

MY DESIGN ADDITION: Instead of picking a single detector, fuse all five
module scores using a learned or heuristic weighting scheme.

Design rationale from the paper (Section 7): "failure cases are mutually
exclusive" — gaze fails on occluded faces; voice fails under noise; lip
sync fails on silent clips. By combining modules, coverage is near-complete.

Two fusion strategies available:
  1. WeightedFusion — fixed weights (default, fast, no training needed)
  2. LearnedFusion  — small MLP trained on score vectors (better accuracy)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional


# Module names (must match keys in result dicts from each analyzer)
MODULE_NAMES = ['gaze', 'lip_sync', 'voice', 'emotion_behavioral']

# Default weights — higher = more trusted module
# Based on accuracy numbers from the paper and common real-world reliability:
#   Gaze is most paper-validated; voice is often missing; lip is noisy.
DEFAULT_WEIGHTS = {
    'gaze':                0.35,   # most validated by paper
    'lip_sync':            0.20,   # good but requires audio
    'voice':               0.20,   # good but missing when video has no speech
    'emotion_behavioral':  0.25,   # behavioral cues are robust
}


# ─────────────────────────────────────────────────────────────────
# Heuristic weighted fusion
# ─────────────────────────────────────────────────────────────────

class WeightedFusion:
    """
    Combines per-module fake scores with fixed weights.
    Missing modules (score = None) are excluded and weights renormalized.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS

    def fuse(self, scores: Dict[str, Optional[float]]) -> dict:
        """
        Args:
            scores: dict mapping module_name → fake_probability (0-1), or None.

        Returns:
            dict with final_score, verdict, confidence, module_weights_used.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        used: Dict[str, float] = {}

        for mod, score in scores.items():
            if score is None:
                continue
            w = self.weights.get(mod, 0.1)
            weighted_sum += w * float(score)
            total_weight  += w
            used[mod] = float(score)

        if total_weight < 1e-8:
            return {
                'final_score': 0.5,
                'verdict': 'UNCERTAIN',
                'confidence': 0.0,
                'module_scores': scores,
                'module_weights': {},
            }

        final = weighted_sum / total_weight
        verdict, confidence = _score_to_verdict(final)

        return {
            'final_score': float(final),
            'verdict': verdict,
            'confidence': float(confidence),
            'module_scores': used,
            'module_weights': {k: self.weights.get(k, 0.1) for k in used},
        }


# ─────────────────────────────────────────────────────────────────
# Learned MLP fusion
# ─────────────────────────────────────────────────────────────────

class FusionMLP(nn.Module):
    """
    Small MLP that takes a vector of module scores and predicts fake probability.
    Train on a labeled dataset of [gaze_score, lip_score, voice_score, em_score]
    → real/fake labels.

    Input: (B, num_modules) where each value is in [0, 1].
    Output: (B, 2) logits.
    """

    def __init__(self, num_modules: int = 4, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_modules, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedFusion:
    """
    MLP-based score fusion. Falls back to WeightedFusion if model not trained.
    """

    def __init__(self, device: str = 'cpu',
                 modules: List[str] = None):
        self.device   = torch.device(device)
        self.modules  = modules or MODULE_NAMES
        self.mlp      = FusionMLP(num_modules=len(self.modules)).to(self.device)
        self._trained = False
        self._fallback = WeightedFusion()

    def load_weights(self, path: str):
        self.mlp.load_state_dict(torch.load(path, map_location=self.device))
        self.mlp.eval()
        self._trained = True

    def fuse(self, scores: Dict[str, Optional[float]]) -> dict:
        """Same interface as WeightedFusion.fuse()."""
        if not self._trained:
            return self._fallback.fuse(scores)

        # Build input vector; use 0.5 for missing modules
        vec = np.array(
            [scores.get(m, 0.5) or 0.5 for m in self.modules],
            dtype=np.float32,
        )
        x_t = torch.from_numpy(vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.mlp(x_t)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        final   = float(probs[1])
        verdict, confidence = _score_to_verdict(final)

        return {
            'final_score': final,
            'verdict': verdict,
            'confidence': float(confidence),
            'module_scores': {k: v for k, v in scores.items() if v is not None},
            'module_weights': {m: 1.0 / len(self.modules) for m in self.modules},
        }

    def train_step(self,
                   score_batch: np.ndarray,
                   labels: np.ndarray,
                   optimizer: torch.optim.Optimizer) -> float:
        """
        One training step. score_batch: (B, num_modules), labels: (B,) int.
        """
        self.mlp.train()
        x = torch.from_numpy(score_batch).float().to(self.device)
        y = torch.from_numpy(labels).long().to(self.device)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(self.mlp(x), y)
        loss.backward()
        optimizer.step()
        self.mlp.eval()
        self._trained = True
        return float(loss.item())


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _score_to_verdict(score: float,
                      fake_threshold: float   = 0.60,
                      real_threshold: float   = 0.40) -> tuple:
    """
    Convert continuous fake probability to a categorical verdict.

    Returns: (verdict_str, confidence_float)
      - verdict:    'FAKE' | 'REAL' | 'UNCERTAIN'
      - confidence: how far the score is from the decision boundary
    """
    if score >= fake_threshold:
        return 'FAKE', min(1.0, (score - fake_threshold) / (1.0 - fake_threshold + 1e-8))
    elif score <= real_threshold:
        return 'REAL', min(1.0, (real_threshold - score) / (real_threshold + 1e-8))
    else:
        # Grey zone
        return 'UNCERTAIN', 0.0


def format_result(fusion_result: dict,
                  module_details: Optional[Dict[str, dict]] = None) -> str:
    """Pretty-print a fusion result with module breakdown."""
    lines = [
        "=" * 50,
        f"  VERDICT:    {fusion_result['verdict']}",
        f"  Score:      {fusion_result['final_score']:.3f}  (0=real, 1=fake)",
        f"  Confidence: {fusion_result['confidence']:.1%}",
        "",
        "  Module scores:",
    ]
    for mod, score in fusion_result.get('module_scores', {}).items():
        bar   = '█' * int(score * 20) + '░' * (20 - int(score * 20))
        lines.append(f"    {mod:25s} [{bar}] {score:.3f}")

    if module_details:
        lines.append("\n  Key signals:")
        for mod, detail in module_details.items():
            for key in ['geo_vergence_score', 'spectral_score', 'estimated_lag_frames',
                        'pitch_consistency_score', 'blink_rate_per_min']:
                if key in detail:
                    lines.append(f"    {mod}.{key}: {detail[key]}")
    lines.append("=" * 50)
    return "\n".join(lines)
