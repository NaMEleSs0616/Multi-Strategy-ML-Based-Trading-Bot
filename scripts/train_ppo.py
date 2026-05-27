#!/usr/bin/env python3
"""CLI entrypoint for PPO strategy routing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ppo.train_router import train_ppo_router  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO capital router on frozen xLSTM embeddings")
    parser.add_argument("--require-encoder", action="store_true", help="Fail if xLSTM checkpoint missing")
    parser.add_argument("--fold", type=int, default=None, help="Train only this walk-forward fold id")
    args = parser.parse_args()
    result = train_ppo_router(require_encoder=args.require_encoder, fold_id=args.fold)
    print(f"model={result.model_path} folds={result.folds_trained}")


if __name__ == "__main__":
    main()
