"""OU-integrated xLSTM training smoke test."""

import pytest

from config.settings_store import load_settings


def test_train_xlstm_logs_ou_components():
    from models.xlstm import train as train_module

    settings = load_settings()
    settings["xlstm"].update(
        {
            "max_epochs": 2,
            "batch_size": 16,
            "sequence_length": 20,
            "hidden_dim": 32,
            "num_blocks": 1,
            "plateau_patience": 5,
            "synthetic_bars": 200,
            "export_embeddings": False,
            "ou_gamma": 0.1,
            "ou_dt": 1.0,
        }
    )
    settings["rl"]["embedding_dim"] = 16
    result = train_module.train_xlstm(settings, use_yfinance=False, device="cpu")
    assert result.epochs_run >= 1
    import json

    history = json.loads(result.history_path.read_text())
    assert "train_ou" in history[0]
    assert "val_ou" in history[0]
