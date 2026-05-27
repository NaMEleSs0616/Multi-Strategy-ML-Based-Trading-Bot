"""
xLSTM memory cells (sLSTM + mLSTM) with exponential gating.

Reference: Beck et al., "xLSTM: Extended Long Short-Term Memory" (NeurIPS 2024).
This module is a compact PyTorch port for sequential financial features — no CUDA/Triton deps.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

State = Tuple[torch.Tensor, ...]


def _exp_gate(x: torch.Tensor, max_value: float = 0.0) -> torch.Tensor:
    """Stabilized exponential gate (values clipped before exp)."""
    return torch.exp(torch.clamp(x, max=max_value))


class SLSTMCell(nn.Module):
    """
    Scalar-memory LSTM with exponential gating and memory mixing (hidden-to-hidden).

    State: (h, c, n, m) where n, m stabilize exponential accumulators.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.weight_hr = nn.Linear(hidden_size, hidden_size, bias=False)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> State:
        h = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        n = torch.ones(batch, self.hidden_size, device=device, dtype=dtype)
        m = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        return h, c, n, m

    def forward(self, x: torch.Tensor, state: Optional[State] = None) -> Tuple[torch.Tensor, State]:
        batch = x.shape[0]
        if state is None:
            state = self.init_state(batch, x.device, x.dtype)

        h_prev, c_prev, n_prev, m_prev = state
        gates = self.weight_ih(x) + self.weight_hh(h_prev)
        i_tilde, f_tilde, o_tilde, z = gates.chunk(4, dim=-1)

        i = _exp_gate(i_tilde)
        f = _exp_gate(f_tilde)
        o = torch.sigmoid(o_tilde)

        c = f * c_prev + i * torch.tanh(z)
        n = f * n_prev + i
        m = f * m_prev + i * c
        h = o * (m / n.clamp_min(1e-6))

        # Memory mixing (recurrent hidden contribution)
        h = h + torch.tanh(self.weight_hr(h_prev))
        return h, (h, c, n, m)


class MLSTMCell(nn.Module):
    """
    Matrix-memory LSTM with covariance-style update (sequential step).

    State: (h, C) with C in R^{hidden x hidden} per batch element.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.i_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.f_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> State:
        h = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(batch, self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        return h, c

    def forward(self, x: torch.Tensor, state: Optional[State] = None) -> Tuple[torch.Tensor, State]:
        batch = x.shape[0]
        if state is None:
            state = self.init_state(batch, x.device, x.dtype)

        h_prev, c_matrix = state
        q = self.q_proj(x)
        k = self.k_proj(x) / (self.hidden_size**0.5)
        v = self.v_proj(x)

        i = _exp_gate(self.i_proj(x))
        f = _exp_gate(self.f_proj(x))
        o = torch.sigmoid(self.o_proj(x))

        outer = torch.bmm(v.unsqueeze(-1), k.unsqueeze(1))
        c_matrix = f.unsqueeze(-1) * c_matrix + i.unsqueeze(-1) * outer
        h = o * torch.bmm(c_matrix, q.unsqueeze(-1)).squeeze(-1)
        h = self.out_proj(h)
        return h, (h, c_matrix)
