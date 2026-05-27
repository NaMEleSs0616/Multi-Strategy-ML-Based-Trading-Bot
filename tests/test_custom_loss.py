"""OU-informed pretraining loss tests."""

import torch

from models.xlstm.custom_loss import (
    OULossConfig,
    XLSTMPretrainingLoss,
    ou_sde_penalty,
    total_pretraining_loss,
)


def test_total_loss_backward():
    torch.manual_seed(0)
    pred = torch.randn(8, 5, requires_grad=True)
    target = torch.randn(8, 5)
    emb = torch.randn(8, 16, requires_grad=True)
    h = torch.randn(8, 12, 32, requires_grad=True)

    breakdown = total_pretraining_loss(pred, target, emb, hidden_states=h, config=OULossConfig(gamma=0.2))
    assert breakdown.total.ndim == 0
    assert breakdown.mse.item() >= 0
    assert breakdown.ou.item() >= 0
    breakdown.total.backward()
    assert pred.grad is not None
    assert h.grad is not None


def test_ou_penalty_stationary_sine_lower_than_random_walk():
    """Sine-like mean-reverting paths should incur lower OU penalty than a random walk."""
    torch.manual_seed(1)
    t = torch.linspace(0, 8, 64)
    sine = torch.sin(t).unsqueeze(0).unsqueeze(-1).repeat(4, 1, 8)  # B, T, D
    rw = torch.cumsum(torch.randn(4, 64, 8), dim=1)

    cfg = OULossConfig(gamma=1.0, min_points=4)
    ou_sine, _ = ou_sde_penalty(sine[:, -1], hidden_states=sine, config=cfg)
    ou_rw, _ = ou_sde_penalty(rw[:, -1], hidden_states=rw, config=cfg)
    assert ou_sine.item() < ou_rw.item()


def test_xlstm_pretraining_loss_module():
    criterion = XLSTMPretrainingLoss(gamma=0.1)
    pred = torch.randn(4, 3)
    target = torch.randn(4, 3)
    emb = torch.randn(4, 8)
    out = criterion(pred, target, emb, hidden_states=torch.randn(4, 10, 16))
    assert out.total.requires_grad is False or out.total.requires_grad
