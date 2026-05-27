"""xLSTM inference alignment tests."""

import numpy as np
import torch

from models.xlstm.encoder import XLSTMConfig, XLSTMStateEncoder, freeze_encoder
from models.xlstm.inference import align_embeddings_to_timeline, encode_feature_matrix


def test_align_embeddings_padding():
    emb = np.ones((10, 4))
    out = align_embeddings_to_timeline(emb, total_length=15, warmup_bars=5)
    assert out.shape == (15, 4)
    assert np.allclose(out[:5], 0)
    assert np.allclose(out[5:], 1)


def test_encode_feature_matrix_full_length():
    config = XLSTMConfig(input_dim=3, hidden_dim=16, embedding_dim=8, sequence_length=10)
    model = XLSTMStateEncoder(config)
    freeze_encoder(model)
    features = np.random.randn(40, 3).astype(np.float32)
    emb = encode_feature_matrix(model, features, device=torch.device("cpu"), batch_size=8)
    assert emb.shape == (40, 8)
    assert np.allclose(emb[:9], 0)
