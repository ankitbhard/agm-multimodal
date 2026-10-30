"""
Adaptive Gradient Modulation (AGM) — dual-gated gradient intervention for
multimodal learning.

Wraps any standard PyTorch optimizer. When `imbalance_threshold > 0`, AGM
applies conditional, proportional gradient suppression via two gates:

    Gate 1 (imbalance):  Δ = c_dominant − c_weaker  >  θ
    Gate 2 (trend):      weaker modality NOT self-correcting over τ steps

When both gates pass, the dominant encoder's gradients are scaled by:

    ρ = (Δ − θ) / (1 − θ)
    k = 1 − tanh(α · ρ)

with optional Gaussian noise injection (GE) for generalisation enhancement.

Setting `imbalance_threshold = 0.0` disables both gates and replicates the
original always-on OGM-GE behaviour of Pan et al. (CVPR 2022).

Reference:
    Bhardwaj & Nabavi, "Adaptive Gradient Modulation for Multimodal Cancer
    Prognosis Prediction from Histopathology and Genomics", ML4BMI @ BIBM 2026.

Usage:
    from optimizers.agm import AGMWrapper

    base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # AGM (adaptive, dual-gated — paper default)
    agm = AGMWrapper(base_opt, model, modality_names=["image", "gene"],
                     imbalance_threshold=0.10, trend_window=10)

    # Always-on OGM-GE baseline (no gating)
    ogm = AGMWrapper(base_opt, model, modality_names=["image", "gene"],
                     imbalance_threshold=0.0)

    # Training loop
    for inputs, labels in loader:
        out  = model(inputs)
        loss = criterion(out, labels)
        agm.step(loss, inputs, labels)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class AGMWrapper:
    """
    Wraps any standard PyTorch optimizer with Adaptive Gradient Modulation.

    Args:
        optimizer           : base optimizer (e.g. SGD, Adam, Adadelta)
        model               : nn.Module with named `image_encoder` / `gene_encoder`
        modality_names      : list of modality name strings, e.g. ["image", "gene"]
        alpha               : suppression sensitivity (default 0.5)
        use_ge              : inject Gaussian noise for generalisation (default True)
        compute_every       : apply modulation every N steps (default 1)
        imbalance_threshold : minimum contribution gap θ to trigger suppression.
                              0.0 → always-on OGM-GE; >0 → adaptive AGM.
        trend_window        : look-back window τ for weaker-modality trend gate
    """

    def __init__(self,
                 optimizer,
                 model:               nn.Module,
                 modality_names:      List[str],
                 alpha:               float = 0.5,
                 use_ge:              bool  = True,
                 compute_every:       int   = 1,
                 imbalance_threshold: float = 0.0,
                 trend_window:        int   = 10):
        self.optimizer            = optimizer
        self.model                = model
        self.modality_names       = modality_names
        self.alpha                = alpha
        self.use_ge               = use_ge
        self.compute_every        = compute_every
        self.imbalance_threshold  = imbalance_threshold
        self.trend_window         = trend_window
        self._step_count          = 0
        self._agm_applied         = 0
        self._agm_skipped         = 0

        self.contribution_history: Dict[str, List[float]] = {
            m: [] for m in modality_names
        }

    # ── Per-modality contribution estimation ─────────────────────────────────

    @torch.no_grad()
    def _compute_contributions(self,
                               inputs: Dict[str, torch.Tensor],
                               labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Estimate each modality's contribution via unimodal forward passes
        (all other modality inputs zeroed).  Returns normalised contributions
        summing to 1.
        """
        self.model.eval()
        contributions: Dict[str, float] = {}

        surv_labels = labels.get("survival") if isinstance(labels, dict) else labels

        for modality in self.modality_names:
            uni_input = {
                k: v if k == modality else torch.zeros_like(v)
                for k, v in inputs.items()
            }
            out    = self.model(uni_input)
            logits = out["survival"] if isinstance(out, dict) else out
            preds  = logits.argmax(dim=1)
            acc    = (preds == surv_labels).float().mean().item()
            contributions[modality] = acc

        self.model.train()

        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}
        return contributions

    # ── Dual-gated gradient modulation ───────────────────────────────────────

    @torch.no_grad()
    def _modulate_gradients(self, contributions: Dict[str, float]) -> None:
        """
        Apply adaptive gradient suppression when both gates pass.

        Gate 1 — imbalance threshold:
            Δ = c_dominant − c_weaker  ≤  θ  →  skip (balanced).

        Gate 2 — trend gate (only when θ > 0):
            If weaker modality's contribution is trending upward over the
            last τ steps → skip (self-correcting).

        Suppression is proportional to excess imbalance:
            ρ = (Δ − θ) / (1 − θ)
            k = 1 − tanh(α · ρ)
        """
        dominant  = max(contributions, key=contributions.get)
        weaker    = min(contributions, key=contributions.get)
        imbalance = contributions[dominant] - contributions[weaker]

        # Gate 1: imbalance threshold
        if imbalance <= self.imbalance_threshold:
            self._agm_skipped += 1
            return

        # Gate 2: trend gate (adaptive mode only)
        if self.imbalance_threshold > 0:
            weaker_history = self.contribution_history.get(weaker, [])
            if len(weaker_history) >= 2 * self.trend_window:
                recent = sum(weaker_history[-self.trend_window:]) / self.trend_window
                older  = sum(weaker_history[-2 * self.trend_window:-self.trend_window]) / self.trend_window
                if recent > older:
                    self._agm_skipped += 1
                    return

        # Both gates passed — proportional suppression
        effective_ratio = (imbalance - self.imbalance_threshold) / (1.0 - self.imbalance_threshold)
        self._agm_applied += 1

        for modality in contributions:
            if modality == dominant:
                k = 1.0 - float(torch.tanh(torch.tensor(self.alpha * effective_ratio)))
            else:
                k = 1.0

            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                if f"{modality}_encoder" in name:
                    param.grad.mul_(k)
                    # Generalisation Enhancement (GE) noise
                    if self.use_ge and modality == dominant and k < 1.0:
                        noise_std = param.grad.std() * 0.1
                        param.grad.add_(torch.randn_like(param.grad) * noise_std)

    # ── Training step ────────────────────────────────────────────────────────

    def step(self,
             loss:   torch.Tensor,
             inputs: Optional[Dict[str, torch.Tensor]] = None,
             labels: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """
        Backward pass + gradient modulation + optimizer step.

        Args:
            loss   : scalar loss tensor
            inputs : modality input dict (needed for contribution estimation)
            labels : label dict with key "survival"
        """
        self.optimizer.zero_grad()
        loss.backward()

        self._step_count += 1
        apply_agm = (
            inputs is not None and
            labels is not None and
            self._step_count % self.compute_every == 0
        )

        if apply_agm:
            contributions = self._compute_contributions(inputs, labels)
            self._modulate_gradients(contributions)
            for m in self.modality_names:
                self.contribution_history[m].append(contributions.get(m, 0.0))

        self.optimizer.step()

    # ── Accessors ────────────────────────────────────────────────────────────

    def get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def get_contribution_history(self) -> Dict[str, List[float]]:
        """Returns per-modality contribution history (normalised, [0,1])."""
        return self.contribution_history

    def get_agm_stats(self) -> Dict[str, int]:
        """Returns count of steps where AGM was applied vs skipped."""
        return {"applied": self._agm_applied, "skipped": self._agm_skipped}
