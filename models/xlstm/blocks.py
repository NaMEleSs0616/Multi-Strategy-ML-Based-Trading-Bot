"""Residual xLSTM blocks stacking sLSTM and mLSTM cells over time."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from models.xlstm.cells import MLSTMCell, SLSTMCell, State

CellKind = Literal["slstm", "mlstm"]


class xLSTMBlock(nn.Module):
    """Pre-norm residual block with one xLSTM cell unrolled over the sequence."""

    def __init__(self, dim: int, cell: CellKind = "slstm", dropout: float = 0.0) -> None:
        super().__init__()
        self.cell_kind = cell
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        if cell == "slstm":
            self.cell = SLSTMCell(dim, dim)
        elif cell == "mlstm":
            self.cell = MLSTMCell(dim, dim)
        else:
            raise ValueError(f"Unknown cell kind: {cell}")

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[State] = None,
    ) -> Tuple[torch.Tensor, State]:
        """
        Parameters
        ----------
        x
            (batch, time, dim)
        """
        residual = x
        x = self.norm(x)
        batch, time, dim = x.shape
        outputs: List[torch.Tensor] = []
        for t in range(time):
            h_t, state = self.cell(x[:, t, :], state)
            outputs.append(h_t)
        y = torch.stack(outputs, dim=1)
        y = self.dropout(y)
        return residual + y, state


class xLSTMBlockStack(nn.Module):
    """Alternating sLSTM / mLSTM blocks."""

    def __init__(
        self,
        dim: int,
        num_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        blocks: List[xLSTMBlock] = []
        for i in range(num_blocks):
            kind: CellKind = "slstm" if i % 2 == 0 else "mlstm"
            blocks.append(xLSTMBlock(dim, cell=kind, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x, _ = block(x)
        return x
