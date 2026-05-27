"""
Centralized artifact path resolution.

This is the single source of truth for where xLSTM checkpoints, PPO router
checkpoints, per-ticker fine-tuned policies, and the embeddings cache live.

Both the standalone CLI scripts (``scripts/train_xlstm.py``,
``scripts/train_ppo.py``) and the orchestrator (``scripts/run_orchestrator.py``)
resolve artifact locations through this module, so Path A (CLI) and
Path B (Orchestrator) always read and write the **same** files in the
**same** directory.

Defaults:

    {project_root}/models/checkpoints/
        ├── xlstm_frozen.pt
        ├── xlstm_best.pt
        ├── xlstm_train_history.json
        ├── embeddings.npy
        ├── ppo_router.zip
        ├── ppo_train_summary.json
        └── ticker_policies/
            └── {TICKER}_active_policy.pt

Override the root via ``artifacts.root_dir`` in ``config/settings.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings_store import PROJECT_ROOT


DEFAULT_ROOT = "models/checkpoints"
DEFAULT_TICKER_POLICIES_SUBDIR = "ticker_policies"


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolved artifact paths for one configured project root."""

    root: Path
    ticker_policies_subdir: str = DEFAULT_TICKER_POLICIES_SUBDIR

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "ArtifactPaths":
        """Resolve artifact paths from the merged settings dict."""
        artifacts_cfg = settings.get("artifacts", {}) or {}
        rel = artifacts_cfg.get("root_dir", DEFAULT_ROOT)
        subdir = artifacts_cfg.get(
            "ticker_policies_subdir", DEFAULT_TICKER_POLICIES_SUBDIR
        )
        root = (PROJECT_ROOT / rel) if not Path(rel).is_absolute() else Path(rel)
        return cls(root=root, ticker_policies_subdir=subdir)

    def ensure_dirs(self) -> None:
        """Create the root and ticker policies directory if missing."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.ticker_policies_dir.mkdir(parents=True, exist_ok=True)

    @property
    def xlstm_frozen(self) -> Path:
        return self.root / "xlstm_frozen.pt"

    @property
    def xlstm_best(self) -> Path:
        return self.root / "xlstm_best.pt"

    @property
    def xlstm_history(self) -> Path:
        return self.root / "xlstm_train_history.json"

    @property
    def xlstm_embeddings(self) -> Path:
        return self.root / "embeddings.npy"

    @property
    def ppo_router(self) -> Path:
        return self.root / "ppo_router.zip"

    @property
    def ppo_summary(self) -> Path:
        return self.root / "ppo_train_summary.json"

    @property
    def ticker_policies_dir(self) -> Path:
        return self.root / self.ticker_policies_subdir

    def ticker_policy(self, ticker: str) -> Path:
        return self.ticker_policies_dir / f"{ticker.upper()}_active_policy.pt"
