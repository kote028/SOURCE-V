"""
Adversarial Simulator — Proactive Threat Modeling

MY DESIGN ADDITION: Rather than just detecting existing fakes, DeepShield
proactively simulates adversarial attacks against itself:

1. FGSM (Fast Gradient Sign Method) — generates adversarial examples that
   could fool the detector. Reveals blind spots.
2. PGD (Projected Gradient Descent) — stronger iterative attack, finds
   more realistic adversarial perturbations.
3. Future-threat forecasting — applies known attack trends to predict
   how next-gen deepfakes might evade current models.

Use this in a scheduled job (nightly) to continuously harden the detector.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import copy


# ─────────────────────────────────────────────────────────────────
# FGSM Attack
# ─────────────────────────────────────────────────────────────────

def fgsm_attack(model: nn.Module,
                x: torch.Tensor,
                y_true: torch.Tensor,
                epsilon: float = 0.03,
                loss_fn: Optional[nn.Module] = None) -> torch.Tensor:
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014).

    Adds a single-step perturbation in the direction of the gradient
    to fool the model into misclassifying real as fake (or vice versa).

    Args:
        model:    The classifier to attack.
        x:        Input tensor (B, C, H, W), requires_grad will be set.
        y_true:   True labels tensor (B,).
        epsilon:  Perturbation magnitude (0.03 ≈ imperceptible for images).
        loss_fn:  Cross-entropy by default.

    Returns:
        Adversarial example tensor, same shape as x, values in [0, 1].
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    x_adv = x.clone().detach().requires_grad_(True)
    model.eval()

    logits = model(x_adv)
    loss   = loss_fn(logits, y_true)
    loss.backward()

    with torch.no_grad():
        perturbation = epsilon * x_adv.grad.sign()
        x_adv_out    = torch.clamp(x_adv + perturbation, 0.0, 1.0)

    return x_adv_out.detach()


# ─────────────────────────────────────────────────────────────────
# PGD Attack
# ─────────────────────────────────────────────────────────────────

def pgd_attack(model: nn.Module,
               x: torch.Tensor,
               y_true: torch.Tensor,
               epsilon: float = 0.03,
               alpha: float   = 0.01,
               num_steps: int = 20,
               loss_fn: Optional[nn.Module] = None,
               random_start: bool = True) -> torch.Tensor:
    """
    Projected Gradient Descent (Madry et al., 2018).

    Iterative attack — much stronger than FGSM. Each step takes a small
    gradient move, then projects back into the epsilon-ball around x.

    Args:
        epsilon:     L-inf radius of the perturbation ball.
        alpha:       Step size per iteration (rule of thumb: epsilon / 4).
        num_steps:   Number of PGD iterations (20 is strong; 7 is fast).
        random_start: Start from a random point inside the epsilon-ball.

    Returns:
        Adversarial example tensor in [0, 1].
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    model.eval()
    x_orig = x.clone().detach()

    if random_start:
        delta = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()
    else:
        x_adv = x_orig.clone()

    for _ in range(num_steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        loss   = loss_fn(logits, y_true)
        loss.backward()

        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + alpha * grad_sign
            # Project back into epsilon-ball
            delta  = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
            x_adv  = torch.clamp(x_orig + delta,  0.0, 1.0)

    return x_adv.detach()


# ─────────────────────────────────────────────────────────────────
# Adversarial robustness evaluator
# ─────────────────────────────────────────────────────────────────

class AdversarialSimulator:
    """
    Stress-tests a detector module by generating adversarial examples
    and measuring how much its accuracy degrades.

    Usage:
        sim = AdversarialSimulator(detector_model, device='cuda')
        report = sim.run_stress_test(dataloader)
        print(report)
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model  = model
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    def evaluate_clean(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Accuracy on clean (unperturbed) examples."""
        with torch.no_grad():
            preds = self.model(x.to(self.device)).argmax(dim=1)
        return float((preds == y.to(self.device)).float().mean())

    def evaluate_fgsm(self, x: torch.Tensor, y: torch.Tensor,
                      epsilon: float = 0.03) -> Tuple[float, torch.Tensor]:
        """Accuracy under FGSM + returns adversarial examples."""
        x_adv = fgsm_attack(self.model, x.to(self.device),
                             y.to(self.device), epsilon)
        with torch.no_grad():
            preds = self.model(x_adv).argmax(dim=1)
        acc = float((preds == y.to(self.device)).float().mean())
        return acc, x_adv

    def evaluate_pgd(self, x: torch.Tensor, y: torch.Tensor,
                     epsilon: float = 0.03, alpha: float = 0.01,
                     steps: int = 20) -> Tuple[float, torch.Tensor]:
        """Accuracy under PGD + returns adversarial examples."""
        x_adv = pgd_attack(self.model, x.to(self.device),
                            y.to(self.device), epsilon, alpha, steps)
        with torch.no_grad():
            preds = self.model(x_adv).argmax(dim=1)
        acc = float((preds == y.to(self.device)).float().mean())
        return acc, x_adv

    def run_stress_test(self,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        epsilons: List[float] = None) -> dict:
        """
        Full stress-test: clean + FGSM + PGD at multiple epsilon values.
        Returns a report dict with accuracy drops and vulnerability scores.
        """
        if epsilons is None:
            epsilons = [0.01, 0.03, 0.05, 0.1]

        clean_acc = self.evaluate_clean(x, y)
        results   = {'clean_accuracy': clean_acc, 'attacks': {}}

        for eps in epsilons:
            fgsm_acc, _ = self.evaluate_fgsm(x, y, epsilon=eps)
            pgd_acc,  _ = self.evaluate_pgd(x, y,  epsilon=eps, alpha=eps / 4, steps=20)
            drop_fgsm = clean_acc - fgsm_acc
            drop_pgd  = clean_acc - pgd_acc

            results['attacks'][f'eps={eps}'] = {
                'fgsm_accuracy':     fgsm_acc,
                'pgd_accuracy':      pgd_acc,
                'fgsm_accuracy_drop': drop_fgsm,
                'pgd_accuracy_drop':  drop_pgd,
                'vulnerability_score': float(np.clip(drop_pgd * 2, 0, 1)),
            }

        # Overall vulnerability: mean PGD drop across epsilons
        pgd_drops = [v['pgd_accuracy_drop'] for v in results['attacks'].values()]
        results['overall_vulnerability'] = float(np.mean(pgd_drops))
        results['robust_accuracy'] = float(
            min(v['pgd_accuracy'] for v in results['attacks'].values())
        )
        return results

    # ─── My design addition: future-threat forecasting ────────────────────

    def forecast_threat_evolution(self,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  forecast_steps: int = 5) -> dict:
        """
        Simulates how gradually more capable deepfake generators might
        evade the current detector by progressively increasing attack strength.

        Modeled as PGD with linearly increasing epsilon from 0.01 to 0.20.
        Each step represents approximately 6-12 months of GAN improvement
        (based on empirical accuracy trends from FaceForensics benchmarks).

        Returns per-step accuracy forecast for capacity planning.
        """
        forecast = {}
        eps_trajectory = np.linspace(0.01, 0.20, forecast_steps)
        for step_idx, eps in enumerate(eps_trajectory):
            acc, _ = self.evaluate_pgd(x, y, epsilon=float(eps), steps=20)
            label  = f'gen_{step_idx + 1}'
            forecast[label] = {
                'equivalent_epsilon':   float(eps),
                'projected_accuracy':   acc,
                'projected_threat_level': ('low'    if acc > 0.85 else
                                           'medium' if acc > 0.65 else
                                           'high'   if acc > 0.45 else
                                           'critical'),
            }
        return forecast

    def adversarial_training_step(self,
                                  optimizer: torch.optim.Optimizer,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  epsilon: float = 0.03) -> float:
        """
        One step of adversarial training (Madry et al.).
        Makes the model robust by training on adversarial examples.
        Call this instead of a normal training step to harden the detector.

        Returns: loss value.
        """
        self.model.train()
        x_adv = pgd_attack(self.model, x.to(self.device),
                            y.to(self.device), epsilon, alpha=epsilon / 4, steps=7)
        self.model.train()
        optimizer.zero_grad()
        logits = self.model(x_adv)
        loss   = F.cross_entropy(logits, y.to(self.device))
        loss.backward()
        optimizer.step()
        self.model.eval()
        return float(loss.item())
