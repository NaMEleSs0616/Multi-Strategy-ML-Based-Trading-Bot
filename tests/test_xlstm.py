"""xLSTM encoder, dataset, and training tests."""

import numpy as np
import pytest
import torch

from data.features.pit_features import build_pit_feature_frame
from models.xlstm.dataset import NextStepSequenceDataset
from models.xlstm.encoder import XLSTMConfig, XLSTMStateEncoder, freeze_encoder
from models.xlstm.train import PlateauStopper, prepare_features


def _synthetic_features(n: int = 300, dim: int = 5) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, size=(n, dim)).astype(np.float32)


def test_encoder_forward_shape():
    config = XLSTMConfig(input_dim=5, hidden_dim=32, embedding_dim=16, num_blocks=2, sequence_length=20)
    model = XLSTMStateEncoder(config)
    x = torch.randn(4, 20, 5)
    emb, pred, h = model(x, return_hidden_states=True)
    assert emb.shape == (4, 16)
    assert pred.shape == (4, 5)
    assert h is not None and h.shape == (4, 20, 32)


def test_freeze_disables_gradients():
    config = XLSTMConfig(input_dim=5, hidden_dim=32, embedding_dim=16, sequence_length=10)
    model = XLSTMStateEncoder(config)
    freeze_encoder(model)
    assert model.frozen
    assert all(not p.requires_grad for p in model.parameters())


def test_dataset_target_is_next_step():
    features = _synthetic_features(100, 4)
    ds = NextStepSequenceDataset(features, sequence_length=10)
    x, y = ds[0]
    t = ds._indices[0]
    np.testing.assert_array_equal(x.numpy(), features[t - 9 : t + 1])
    np.testing.assert_array_equal(y.numpy(), features[t + 1])


def test_pit_features_shifted():
    import pandas as pd

    n = 80
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1e6, 2e6, n),
        }
    )
    pit = build_pit_feature_frame(bars, pit_shift=1)
    assert pit.isna().sum().sum() == 0
    assert len(pit) < len(bars)


def test_plateau_stopper():
    stopper = PlateauStopper(patience=2, min_delta=1e-3)
    assert not stopper.step(1.0)
    assert not stopper.step(0.9)
    assert not stopper.step(0.899)
    assert stopper.step(0.8995)


def test_prepare_features_synthetic():
    settings = {
        "universe": {"equities": ["NVDA"]},
        "features": {
            "frac_diff_d": 0.4,
            "autocorr_window": 20,
            "atr_window": 14,
            "pit_shift": 1,
        },
        "xlstm": {"synthetic_bars": 400},
    }
    matrix = prepare_features(settings, use_yfinance=False)
    assert matrix.ndim == 2
    assert matrix.shape[0] > 100


@pytest.mark.slow
def test_train_xlstm_short_run():
    from config.settings_store import load_settings
    from models.xlstm import train as train_module

    settings = load_settings()
    settings["xlstm"].update(
        {
            "max_epochs": 3,
            "batch_size": 32,
            "sequence_length": 30,
            "hidden_dim": 32,
            "num_blocks": 1,
            "plateau_patience": 2,
            "synthetic_bars": 400,
            "export_embeddings": False,
        }
    )
    settings["rl"]["embedding_dim"] = 16
    result = train_module.train_xlstm(settings, use_yfinance=False, device="cpu")
    assert result.epochs_run >= 1
    assert result.checkpoint_path.exists()
