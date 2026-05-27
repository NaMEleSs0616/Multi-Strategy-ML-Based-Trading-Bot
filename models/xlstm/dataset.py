"""PyTorch datasets for next-step xLSTM pre-training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceBatch:
    """Windowed inputs and one-step-ahead targets."""

    inputs: torch.Tensor
    targets: torch.Tensor


class NextStepSequenceDataset(Dataset):
    """
    Sliding windows for MSE next-step prediction.

    For window ending at index t (inclusive in the past), the target is features[t+1].
  All rows in ``features`` must already be PiT-shifted.
    """

    def __init__(
        self,
        features: np.ndarray,
        sequence_length: int,
        *,
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> None:
        if features.ndim != 2:
            raise ValueError("features must be (time, input_dim)")
        if sequence_length < 2:
            raise ValueError("sequence_length must be >= 2")

        self.features = np.asarray(features, dtype=np.float32)
        self.sequence_length = sequence_length
        self.start_index = start_index
        self.end_index = end_index if end_index is not None else len(features)

        # Last valid target index is len-1; window ends at target-1
        self._min_t = max(start_index + sequence_length - 1, sequence_length - 1)
        self._max_t = min(self.end_index - 2, len(features) - 2)
        if self._max_t < self._min_t:
            raise ValueError("Not enough data for the requested index range")

        self._indices = list(range(self._min_t, self._max_t + 1))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._indices[idx]
        window_start = t - self.sequence_length + 1
        x = self.features[window_start : t + 1]
        y = self.features[t + 1]
        return torch.from_numpy(x), torch.from_numpy(y)

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[1])


def train_val_split(
    dataset: NextStepSequenceDataset,
    val_fraction: float = 0.2,
) -> tuple[list[int], list[int]]:
    """Chronological train/val index split (no shuffle across time)."""
    n = len(dataset)
    if n < 2:
        raise ValueError("dataset too small to split")
    split = max(1, int(n * (1.0 - val_fraction)))
    train_idx = list(range(0, split))
    val_idx = list(range(split, n))
    return train_idx, val_idx
