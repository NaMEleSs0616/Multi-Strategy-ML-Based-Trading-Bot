"""
Self-supervised xLSTM pre-training (next-step MSE) with plateau-based early stop + freeze.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config.settings_store import PROJECT_ROOT, load_settings
from data.features.pit_features import build_pit_feature_frame, feature_matrix
from models.xlstm.dataset import NextStepSequenceDataset, train_val_split
from models.xlstm.custom_loss import XLSTMPretrainingLoss
from models.xlstm.encoder import XLSTMConfig, XLSTMStateEncoder, freeze_encoder


@dataclass
class TrainResult:
    best_val_loss: float
    epochs_run: int
    frozen: bool
    checkpoint_path: Path
    history_path: Path
    embeddings_path: Optional[Path]


class PlateauStopper:
    """Stop when validation loss stops improving."""

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience


def _synthetic_bars(n: int = 2000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    return np.column_stack(
        [
            close + rng.normal(0, 0.1, n),
            close + rng.uniform(0.05, 0.5, n),
            close - rng.uniform(0.05, 0.5, n),
            close,
            rng.integers(1_000_000, 5_000_000, n),
        ]
    )


def prepare_features(
    settings: dict[str, Any],
    *,
    use_yfinance: bool = True,
    data_handler: Optional[Any] = None,
) -> np.ndarray:
    feat_cfg = settings["features"]
    pit_shift = int(feat_cfg["pit_shift"])

    if use_yfinance:
        try:
            from adapters.factory import create_data_handler
            from core.interfaces import FeatureRequest

            handler = data_handler or create_data_handler(settings)
            ticker = settings["universe"]["equities"][0]
            request = FeatureRequest(
                symbols=[ticker],
                pit_shift=pit_shift,
                include_macro=bool(feat_cfg.get("include_macro", False)),
                include_sentiment=bool(feat_cfg.get("include_sentiment", False)),
            )
            frame = handler.fetch_feature_frame(request)
            return frame.to_numpy(dtype=np.float64)
        except Exception:
            try:
                from strategies.bank import build_strategy_bank

                bundle = build_strategy_bank(settings)
                matrix, _ = feature_matrix(
                    bundle.primary_bars,
                    pit_shift=pit_shift,
                    autocorr_window=int(feat_cfg["autocorr_window"]),
                    atr_window=int(feat_cfg["atr_window"]),
                    frac_diff_d=float(feat_cfg["frac_diff_d"]),
                    frac_diff_thresh=float(feat_cfg.get("frac_diff_thresh", 1e-3)),
                )
                return matrix
            except Exception:
                pass

    import pandas as pd

    n = int(settings.get("xlstm", {}).get("synthetic_bars", 2500))
    raw = _synthetic_bars(n)
    bars = pd.DataFrame(
        raw,
        columns=["open", "high", "low", "close", "volume"],
    )
    matrix, _ = feature_matrix(
        bars,
        pit_shift=pit_shift,
        autocorr_window=int(feat_cfg["autocorr_window"]),
        atr_window=int(feat_cfg["atr_window"]),
        frac_diff_d=float(feat_cfg["frac_diff_d"]),
        frac_diff_thresh=float(feat_cfg.get("frac_diff_thresh", 1e-3)),
    )
    return matrix


def train_xlstm(
    settings: Optional[dict[str, Any]] = None,
    *,
    use_yfinance: bool = True,
    device: Optional[str] = None,
) -> TrainResult:
    settings = settings or load_settings()
    xlstm_cfg = settings.get("xlstm", {})
    rl_cfg = settings["rl"]

    features = prepare_features(settings, use_yfinance=use_yfinance)
    input_dim = int(features.shape[1])
    seq_len = int(xlstm_cfg.get("sequence_length", 60))
    dataset = NextStepSequenceDataset(features, seq_len)
    train_idx, val_idx = train_val_split(dataset, val_fraction=float(xlstm_cfg.get("val_fraction", 0.2)))

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(xlstm_cfg.get("batch_size", 64)),
        shuffle=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(xlstm_cfg.get("batch_size", 64)),
        shuffle=False,
    )

    config = XLSTMConfig(
        input_dim=input_dim,
        hidden_dim=int(xlstm_cfg.get("hidden_dim", 128)),
        embedding_dim=int(rl_cfg.get("embedding_dim", 64)),
        num_blocks=int(xlstm_cfg.get("num_blocks", 2)),
        dropout=float(xlstm_cfg.get("dropout", 0.1)),
        sequence_length=seq_len,
    )
    model = XLSTMStateEncoder(config)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device_t = torch.device(device)
    model.to(device_t)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(xlstm_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(xlstm_cfg.get("weight_decay", 1e-4)),
    )
    criterion = XLSTMPretrainingLoss(
        gamma=float(xlstm_cfg.get("ou_gamma", 0.1)),
        dt=float(xlstm_cfg.get("ou_dt", 1.0)),
    )
    stopper = PlateauStopper(
        patience=int(xlstm_cfg.get("plateau_patience", 5)),
        min_delta=float(xlstm_cfg.get("plateau_min_delta", 1e-4)),
    )

    ckpt_dir = PROJECT_ROOT / xlstm_cfg.get("checkpoint_dir", "models/xlstm/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "xlstm_best.pt"
    frozen_path = ckpt_dir / "xlstm_frozen.pt"
    history_path = ckpt_dir / "train_history.json"

    history: list[dict[str, float]] = []
    best_val = float("inf")
    frozen = False
    max_epochs = int(xlstm_cfg.get("max_epochs", 100))

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        train_mse_acc: list[float] = []
        train_ou_acc: list[float] = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device_t), yb.to(device_t)
            optimizer.zero_grad()
            emb, pred, h_seq = model(xb, return_prediction=True, return_hidden_states=True)
            breakdown = criterion(pred, yb, emb, hidden_states=h_seq)
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(breakdown.total.item())
            train_mse_acc.append(breakdown.mse.item())
            train_ou_acc.append(breakdown.ou.item())

        model.eval()
        val_losses = []
        val_mse_acc: list[float] = []
        val_ou_acc: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device_t), yb.to(device_t)
                emb, pred, h_seq = model(xb, return_prediction=True, return_hidden_states=True)
                breakdown = criterion(pred, yb, emb, hidden_states=h_seq)
                val_losses.append(breakdown.total.item())
                val_mse_acc.append(breakdown.mse.item())
                val_ou_acc.append(breakdown.ou.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mse": float(np.mean(train_mse_acc)),
                "val_mse": float(np.mean(val_mse_acc)),
                "train_ou": float(np.mean(train_ou_acc)),
                "val_ou": float(np.mean(val_ou_acc)),
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            model.save_checkpoint(
                best_path,
                metadata={"epoch": epoch, "val_mse": val_loss},
            )

        if stopper.step(val_loss):
            frozen = True
            break

    model = XLSTMStateEncoder.load_checkpoint(best_path, map_location=device_t)
    if frozen:
        freeze_encoder(model)
        model.save_checkpoint(
            frozen_path,
            metadata={"best_val_mse": best_val, "frozen": True},
        )

    embeddings_path: Optional[Path] = None
    if bool(xlstm_cfg.get("export_embeddings", True)):
        from models.xlstm.inference import encode_feature_matrix

        embeddings_path = ckpt_dir / "embeddings.npy"
        with torch.no_grad():
            emb = encode_feature_matrix(model, features, device=device_t)
        np.save(embeddings_path, emb)

    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    return TrainResult(
        best_val_loss=best_val,
        epochs_run=len(history),
        frozen=frozen or model.frozen,
        checkpoint_path=frozen_path if frozen_path.exists() else best_path,
        history_path=history_path,
        embeddings_path=embeddings_path,
    )


if __name__ == "__main__":
    result = train_xlstm()
    print(f"epochs={result.epochs_run} best_val_mse={result.best_val_loss:.6f}")
    print(f"frozen={result.frozen} checkpoint={result.checkpoint_path}")
