"""
Purged walk-forward cross-validation with embargo.

Removes training observations whose label/feature windows overlap the test
period, plus an embargo of k bars immediately before and after each test fold
boundary to reduce autocorrelation leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Sequence

import numpy as np


def purge_indices(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    n_samples: int,
    embargo: int,
) -> np.ndarray:
    """
    Drop train indices that fall inside [test_start - embargo, test_end + embargo].

    Parameters
    ----------
    train_indices
        Integer positions eligible for training.
    test_indices
        Integer positions reserved for out-of-sample evaluation.
    n_samples
        Total number of observations in the dataset.
    embargo
        Number of bars to exclude immediately before and after the test block.

    Returns
    -------
    np.ndarray
        Purged training indices (sorted, unique).
    """
    if embargo < 0:
        raise ValueError("embargo must be non-negative")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    if test.size == 0:
        return np.unique(train)

    test_start = int(test.min())
    test_end = int(test.max())
    purge_start = max(0, test_start - embargo)
    purge_end = min(n_samples - 1, test_end + embargo)

    in_purge_zone = (train >= purge_start) & (train <= purge_end)
    purged = train[~in_purge_zone]
    return np.unique(purged)


@dataclass(frozen=True)
class WalkForwardFold:
    """Single walk-forward fold with purged train and contiguous test indices."""

    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    purge_start: int
    purge_end: int


class PurgedWalkForwardSplitter:
    """
    Expanding-window walk-forward splitter with purging and embargo.

    Example
    -------
    >>> splitter = PurgedWalkForwardSplitter(n_splits=3, embargo=5)
    >>> for fold in splitter.split(n_samples=1000):
    ...     X_train, X_test = X[fold.train_indices], X[fold.test_indices]
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo: int = 5,
        min_train_size: int = 252,
        test_size: int | None = 63,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if embargo < 0:
            raise ValueError("embargo must be non-negative")
        if min_train_size < 1:
            raise ValueError("min_train_size must be >= 1")

        self.n_splits = n_splits
        self.embargo = embargo
        self.min_train_size = min_train_size
        self.test_size = test_size

    def split(
        self,
        n_samples: int,
    ) -> Generator[WalkForwardFold, None, None]:
        """
        Yield purged walk-forward folds.

        Training window expands; each test segment is disjoint and followed by
        an embargo strip removed from subsequent training sets.
        """
        if n_samples < self.min_train_size + 1:
            raise ValueError(
                f"n_samples ({n_samples}) too small for min_train_size "
                f"({self.min_train_size})"
            )

        if self.test_size is None:
            remaining = n_samples - self.min_train_size
            step = max(1, remaining // self.n_splits)
            test_size = step
        else:
            test_size = self.test_size

        test_start = self.min_train_size
        fold_id = 0

        while fold_id < self.n_splits and test_start < n_samples:
            test_end = min(test_start + test_size - 1, n_samples - 1)
            if test_end < test_start:
                break

            test_indices = np.arange(test_start, test_end + 1, dtype=np.int64)
            candidate_train = np.arange(0, test_start, dtype=np.int64)
            train_indices = purge_indices(
                candidate_train,
                test_indices,
                n_samples=n_samples,
                embargo=self.embargo,
            )

            purge_start = max(0, int(test_indices.min()) - self.embargo)
            purge_end = min(n_samples - 1, int(test_indices.max()) + self.embargo)

            yield WalkForwardFold(
                fold_id=fold_id,
                train_indices=train_indices,
                test_indices=test_indices,
                purge_start=purge_start,
                purge_end=purge_end,
            )

            fold_id += 1
            test_start = test_end + 1 + self.embargo

    def all_test_indices(self, n_samples: int) -> list[np.ndarray]:
        """Return test index arrays for each fold (convenience for env setup)."""
        return [fold.test_indices for fold in self.split(n_samples)]


def validate_no_train_test_overlap(
    fold: WalkForwardFold,
    embargo: int,
) -> bool:
    """True if no train index lies inside the purged embargo zone."""
    if fold.train_indices.size == 0:
        return True
    zone = set(range(fold.purge_start, fold.purge_end + 1))
    return not any(int(i) in zone for i in fold.train_indices)


def contiguous_segments(indices: Iterable[int]) -> list[tuple[int, int]]:
    """Convert sorted indices into inclusive (start, end) segments."""
    arr = np.unique(np.asarray(list(indices), dtype=np.int64))
    if arr.size == 0:
        return []

    segments: list[tuple[int, int]] = []
    start = end = int(arr[0])
    for value in arr[1:]:
        value = int(value)
        if value == end + 1:
            end = value
        else:
            segments.append((start, end))
            start = end = value
    segments.append((start, end))
    return segments
