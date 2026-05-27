#!/usr/bin/env python3
"""CLI entrypoint for PPO strategy routing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings_store import load_settings  # noqa: E402
from core.paths import ArtifactPaths  # noqa: E402
from models.ppo.train_router import train_ppo_router  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO capital router on frozen xLSTM embeddings")
    parser.add_argument("--require-encoder", action="store_true", help="Fail if xLSTM checkpoint missing")
    parser.add_argument("--fold", type=int, default=None, help="Train only this walk-forward fold id")
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

    result = train_ppo_router(
        settings=settings,
        require_encoder=args.require_encoder,
        fold_id=args.fold,
        out_dir=out_dir,
    )
    print(f"model={result.model_path} folds={result.folds_trained}")


if __name__ == "__main__":
    main()
