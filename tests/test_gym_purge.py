"""Walk-forward purging and Gymnasium env tests."""

import numpy as np

from rl.gym_trading_env import TradingRoutingEnv, run_walk_forward_smoke_test
from rl.validation.purge import PurgedWalkForwardSplitter, purge_indices, validate_no_train_test_overlap


def test_purge_removes_embargo_zone():
    n = 100
    train = np.arange(0, 80)
    test = np.arange(80, 90)
    purged = purge_indices(train, test, n_samples=n, embargo=5)
    assert purged.max() < 75
    assert 79 not in purged


def test_walk_forward_no_train_in_purge_zone():
    splitter = PurgedWalkForwardSplitter(n_splits=2, embargo=3, min_train_size=30, test_size=10)
    for fold in splitter.split(80):
        assert validate_no_train_test_overlap(fold, embargo=3)


def test_gym_walk_forward_smoke():
    summary = run_walk_forward_smoke_test(n_steps=300, n_splits=2, embargo=3)
    assert len(summary["folds"]) >= 1
    for fold in summary["folds"]:
        assert fold["steps"] == fold["test_size"]
