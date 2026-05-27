"""
Equation-informed xLSTM pre-training loss with Ornstein–Uhlenbeck (OU) SDE penalty.

Total objective (Phase 2)
-------------------------
.. math::

    \\mathcal{L}_{total} = \\mathcal{L}_{MSE} + \\gamma \\mathcal{L}_{OU}

where :math:`\\mathcal{L}_{MSE}` is next-step feature prediction error and
:math:`\\mathcal{L}_{OU}` penalizes state trajectories whose variance scaling
is inconsistent with a stationary mean-reverting OU process.

Continuous OU (mean-reverting SDE)
------------------------------------
.. math::

    dX_t = \\theta (\\mu - X_t)\\, dt + \\sigma\\, dW_t

Stationary variance (continuous time):

.. math::

    \\mathrm{Var}(X) = \\frac{\\sigma^2}{2\\theta}, \\quad \\theta > 0

Discrete-time AR(1) (Euler–Maruyama with unit step)
-------------------------------------------------
With :math:`\\phi = 1 - \\theta\\,\\Delta t`:

.. math::

    X_{t+1} - \\mu = \\phi (X_t - \\mu) + \\varepsilon_t,
    \\quad |\\phi| < 1

Stationary variance:

.. math::

    \\mathrm{Var}(X) = \\frac{\\sigma_\\varepsilon^2}{1 - \\phi^2}

:math:`\\mathcal{L}_{OU}` combines (i) variance-ratio mismatch and (ii) a soft
penalty when :math:`|\\phi|` approaches non-stationarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OULossConfig:
    """Hyper-parameters for the OU SDE penalty."""

    gamma: float = 0.1
    dt: float = 1.0
    phi_stationarity_cap: float = 0.99
    min_points: int = 4
    eps: float = 1e-6
    use_hidden_states: bool = True


@dataclass
class TotalLossBreakdown:
    """Scalar components returned alongside the graph-connected total loss."""

    total: torch.Tensor
    mse: torch.Tensor
    ou: torch.Tensor
    phi_mean: torch.Tensor
    var_empirical: torch.Tensor
    var_ou_theory: torch.Tensor


def _ou_penalty_from_sequence(
    sequence: torch.Tensor,
    *,
    dt: float,
    phi_cap: float,
    min_points: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute :math:`\\mathcal{L}_{OU}` on a state sequence.

    Parameters
    ----------
    sequence
        Shape ``(batch, time, dim)`` — latent states or embeddings along time.

    Returns
    -------
    loss_ou, phi_mean, var_empirical, var_ou_theory
    """
    if sequence.dim() != 3:
        raise ValueError("sequence must have shape (batch, time, dim)")

    batch, time, _dim = sequence.shape
    if time < min_points:
        zero = sequence.new_zeros(())
        return zero, zero, zero, zero

    # Demean per batch element and time axis → estimate μ̂
    mu = sequence.mean(dim=1, keepdim=True)
    z = sequence - mu

    z_lag = z[:, :-1, :]
    z_cur = z[:, 1:, :]

    # AR(1) coefficient φ per (batch, dim): Cov(z_t, z_{t-1}) / Var(z_{t-1})
    cross = (z_lag * z_cur).sum(dim=1)
    var_lag = z_lag.pow(2).sum(dim=1) + eps
    phi = cross / var_lag
    phi = torch.clamp(phi, min=-phi_cap, max=phi_cap)

    # Innovation residuals ε_t = (z_t - μ) - φ (z_{t-1} - μ)
    residual = z_cur - phi.unsqueeze(1) * z_lag
    sigma_eps_sq = residual.pow(2).mean(dim=1) + eps

    # Empirical vs theoretical stationary variance (discrete OU / AR(1))
    var_emp = z.var(dim=1, unbiased=False) + eps
    var_ou = sigma_eps_sq / (1.0 - phi.pow(2) + eps)

    # Variance scaling violation
    var_mismatch = (var_emp - var_ou).pow(2).mean()

    # Soft stationarity guard: penalize |φ| too close to 1
    stationarity_violation = F.relu(phi.abs() - phi_cap).pow(2).mean()

    # Continuous-time θ = (1 - φ) / dt; penalize non-positive θ (non mean-reverting)
    theta = (1.0 - phi) / dt
    theta_violation = F.relu(-theta + eps).pow(2).mean()

    loss_ou = var_mismatch + stationarity_violation + theta_violation

    return (
        loss_ou,
        phi.mean(),
        var_emp.mean(),
        var_ou.mean(),
    )


def ou_sde_penalty(
    embedding: torch.Tensor,
    *,
    hidden_states: Optional[torch.Tensor] = None,
    config: Optional[OULossConfig] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Public OU penalty entry point.

    Uses ``hidden_states`` (pre-head dynamics) when provided and
    ``config.use_hidden_states`` is True; otherwise builds a degenerate
    sequence by repeating the embedding across time (weaker constraint).
    """
    cfg = config or OULossConfig()

    if cfg.use_hidden_states and hidden_states is not None and hidden_states.dim() == 3:
        sequence = hidden_states
    else:
        # Fallback: treat batch as a 2-step sequence [z_{t-1}, z_t] for minimal OU check
        sequence = torch.stack([embedding, embedding], dim=1)

    loss_ou, phi_mean, var_emp, var_ou = _ou_penalty_from_sequence(
        sequence,
        dt=cfg.dt,
        phi_cap=cfg.phi_stationarity_cap,
        min_points=cfg.min_points,
        eps=cfg.eps,
    )

    aux = {
        "phi_mean": phi_mean.detach(),
        "var_empirical": var_emp.detach(),
        "var_ou_theory": var_ou.detach(),
    }
    return loss_ou, aux


def total_pretraining_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    embedding: torch.Tensor,
    *,
    hidden_states: Optional[torch.Tensor] = None,
    config: Optional[OULossConfig] = None,
    mse_reduction: str = "mean",
) -> TotalLossBreakdown:
    """
    Compute :math:`\\mathcal{L}_{total} = \\mathcal{L}_{MSE} + \\gamma \\mathcal{L}_{OU}`.

    Parameters
    ----------
    prediction
        Next-step feature prediction ``(batch, input_dim)``.
    target
        Ground-truth features at ``t+1``, same shape as ``prediction``.
    embedding
        State embedding ``(batch, embedding_dim)`` from the encoder head.
    hidden_states
        Optional ``(batch, time, hidden_dim)`` sequence for stronger OU constraints
        on latent dynamics (recommended during xLSTM training).
    """
    cfg = config or OULossConfig()

    loss_mse = F.mse_loss(prediction, target, reduction=mse_reduction)
    loss_ou, aux = ou_sde_penalty(
        embedding,
        hidden_states=hidden_states,
        config=cfg,
    )

    total = loss_mse + cfg.gamma * loss_ou

    return TotalLossBreakdown(
        total=total,
        mse=loss_mse,
        ou=loss_ou,
        phi_mean=aux["phi_mean"],
        var_empirical=aux["var_empirical"],
        var_ou_theory=aux["var_ou_theory"],
    )


class XLSTMPretrainingLoss(nn.Module):
    """
    ``nn.Module`` wrapper for use inside the xLSTM training loop.

    Example
    -------
    >>> criterion = XLSTMPretrainingLoss(gamma=0.1)
    >>> breakdown = criterion(pred, target, embedding, hidden_states=h_seq)
    >>> breakdown.total.backward()
    """

    def __init__(
        self,
        gamma: float = 0.1,
        dt: float = 1.0,
        use_hidden_states: bool = True,
    ) -> None:
        super().__init__()
        self.config = OULossConfig(gamma=gamma, dt=dt, use_hidden_states=use_hidden_states)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> TotalLossBreakdown:
        return total_pretraining_loss(
            prediction,
            target,
            embedding,
            hidden_states=hidden_states,
            config=self.config,
        )
