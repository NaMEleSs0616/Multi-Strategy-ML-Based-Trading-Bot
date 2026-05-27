#!/usr/bin/env python3
"""CLI entrypoint for xLSTM self-supervised pre-training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings_store import load_settings  # noqa: E402
from core.paths import ArtifactPaths  # noqa: E402
from models.xlstm.train import train_xlstm  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train xLSTM next-step encoder")
    parser.add_argument(
        "--no-yfinance",
        action="store_true",
        help="Use synthetic OHLCV instead of yfinance",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override artifact directory (default: artifacts.root_dir from settings)",
    )
    args = parser.parse_args()

    settings = load_settings()
    out_dir = args.out_dir or ArtifactPaths.from_settings(settings).root
    print(f"out_dir={out_dir}")

    result = train_xlstm(
        settings=settings,
        use_yfinance=not args.no_yfinance,
        device=args.device,
        out_dir=out_dir,
    )
    print(f"epochs={result.epochs_run} best_val_mse={result.best_val_loss:.6f}")
    print(f"frozen={result.frozen}")
    print(f"checkpoint={result.checkpoint_path}")
    if result.embeddings_path:
        print(f"embeddings={result.embeddings_path}")


if __name__ == "__main__":
    main()
