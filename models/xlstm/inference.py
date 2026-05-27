"""Load frozen xLSTM encoder and build timeline-aligned embeddings for RL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from config.settings_store import PROJECT_ROOT, load_settings
from models.xlstm.encoder import XLSTMStateEncoder
from models.xlstm.train import prepare_features


def resolve_checkpoint_path(settings: Optional[dict[str, Any]] = None) -> Path:
    settings = settings or load_settings()
    ckpt_dir = PROJECT_ROOT / settings.get("xlstm", {}).get(
        "checkpoint_dir", "models/xlstm/checkpoints"
    )
    frozen = ckpt_dir / "xlstm_frozen.pt"
    best = ckpt_dir / "xlstm_best.pt"
    if frozen.exists():
        return frozen
    if best.exists():
        return best
    raise FileNotFoundError(
        f"No xLSTM checkpoint in {ckpt_dir}. Run: PYTHONPATH=. python scripts/train_xlstm.py"
    )


def load_frozen_encoder(
    checkpoint: Path | None = None,
    *,
    device: Optional[str] = None,
    settings: Optional[dict[str, Any]] = None,
) -> XLSTMStateEncoder:
    path = checkpoint or resolve_checkpoint_path(settings)
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return XLSTMStateEncoder.load_checkpoint(path, map_location=device, freeze=True)


def align_embeddings_to_timeline(
    embeddings: np.ndarray,
    total_length: int,
    warmup_bars: int,
) -> np.ndarray:
    """
    Place length-(T-warmup) encoded rows into a (T, D) matrix.

    Rows before ``warmup_bars`` are zero (no valid history yet).
    """
    dim = embeddings.shape[1]
    out = np.zeros((total_length, dim), dtype=np.float64)
    end = warmup_bars + embeddings.shape[0]
    if end > total_length:
        raise ValueError("embedding series longer than target timeline")
    out[warmup_bars:end] = embeddings
    return out


@torch.no_grad()
def encode_feature_matrix(
    encoder: XLSTMStateEncoder,
    features: np.ndarray,
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode full PiT feature matrix to (T, embedding_dim)."""
    encoder.eval()
    device = device or next(encoder.parameters()).device
    seq_len = encoder.config.sequence_length
    time, input_dim = features.shape
    if time < seq_len:
        raise ValueError(f"Need at least {seq_len} rows, got {time}")

    windows: list[np.ndarray] = []
    for end in range(seq_len, time + 1):
        windows.append(features[end - seq_len : end])

    chunks: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        batch = np.stack(windows[start : start + batch_size], axis=0)
        tensor = torch.from_numpy(np.asarray(batch, dtype=np.float32)).to(device)
        emb = encoder.encode(tensor).cpu().numpy()
        chunks.append(emb)

    encoded = np.concatenate(chunks, axis=0)
    return align_embeddings_to_timeline(encoded, time, warmup_bars=seq_len - 1)


def load_or_compute_embeddings(
    settings: Optional[dict[str, Any]] = None,
    *,
    use_yfinance: bool = True,
    force_recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (embeddings, features) both shaped (T, *).

    Loads ``embeddings.npy`` when present unless ``force_recompute``.
    """
    settings = settings or load_settings()
    ckpt_dir = PROJECT_ROOT / settings.get("xlstm", {}).get(
        "checkpoint_dir", "models/xlstm/checkpoints"
    )
    emb_path = ckpt_dir / "embeddings.npy"

    features = prepare_features(settings, use_yfinance=use_yfinance)

    if emb_path.exists() and not force_recompute:
        embeddings = np.load(emb_path)
        if embeddings.shape[0] == features.shape[0]:
            return embeddings.astype(np.float64), features

    encoder = load_frozen_encoder(settings=settings)
    embeddings = encode_feature_matrix(encoder, features)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    return embeddings, features
