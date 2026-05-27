"""
xLSTM market-state encoder with next-step prediction head.

After pre-training converges, call :func:`freeze_encoder` to lock weights for RL routing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from models.xlstm.blocks import xLSTMBlockStack


@dataclass
class XLSTMConfig:
    input_dim: int
    hidden_dim: int = 128
    embedding_dim: int = 64
    num_blocks: int = 2
    dropout: float = 0.1
    sequence_length: int = 60

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "XLSTMConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class XLSTMStateEncoder(nn.Module):
    """
    Compresses a PiT feature window into a fixed embedding; optional next-step prediction.

    Forward
    -------
    x : (batch, seq_len, input_dim)
    returns embedding (batch, embedding_dim), prediction (batch, input_dim)
    """

    def __init__(self, config: XLSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.stack = xLSTMBlockStack(
            dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
        )
        self.post_norm = nn.LayerNorm(config.hidden_dim)
        self.embed_head = nn.Linear(config.hidden_dim, config.embedding_dim)
        self.pred_head = nn.Linear(config.hidden_dim, config.input_dim)
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_prediction: bool = True,
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        h = self.input_proj(x)
        h_seq = self.stack(h)
        h = self.post_norm(h_seq)
        h_last = h[:, -1, :]
        embedding = self.embed_head(h_last)
        hidden_states = h_seq if return_hidden_states else None
        if not return_prediction:
            return embedding, None, hidden_states
        prediction = self.pred_head(h_last)
        return embedding, prediction, hidden_states

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic embedding for inference / RL observations."""
        was_training = self.training
        self.eval()
        try:
            embedding, _, _ = self.forward(x, return_prediction=False)
        finally:
            if was_training:
                self.train()
        return embedding

    @torch.no_grad()
    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode rolling windows for every valid end index in a long series.

        Parameters
        ----------
        x
            (time, input_dim) or (1, time, input_dim)

        Returns
        -------
        (time - seq_len + 1, embedding_dim) aligned so row i predicts bar seq_len-1+i
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        time, _ = x.shape[1], x.shape[2]
        seq_len = self.config.sequence_length
        if time < seq_len:
            raise ValueError(f"Need at least {seq_len} timesteps, got {time}")

        outputs: list[torch.Tensor] = []
        for end in range(seq_len, time + 1):
            window = x[:, end - seq_len : end, :]
            outputs.append(self.encode(window))
        return torch.cat(outputs, dim=0)

    def save_checkpoint(self, path: Path, metadata: Optional[dict[str, Any]] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
            "frozen": self._frozen,
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        *,
        map_location: str | torch.device = "cpu",
        freeze: bool = False,
    ) -> "XLSTMStateEncoder":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        config = XLSTMConfig.from_dict(payload["config"])
        model = cls(config)
        model.load_state_dict(payload["state_dict"])
        # `map_location` only affects where `torch.load` puts raw tensors; the
        # freshly constructed `model` is still on CPU. Move it onto the target
        # device so callers don't hit cross-device runtime errors on the first
        # forward pass.
        model.to(torch.device(map_location) if isinstance(map_location, str) else map_location)
        if payload.get("frozen") or freeze:
            freeze_encoder(model)
        return model


def freeze_encoder(model: XLSTMStateEncoder) -> XLSTMStateEncoder:
    """Freeze all parameters for use as a deterministic state encoder."""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model._frozen = True
    return model


def unfreeze_encoder(model: XLSTMStateEncoder) -> XLSTMStateEncoder:
    """Re-enable gradients for continued fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    model._frozen = False
    model.train()
    return model
