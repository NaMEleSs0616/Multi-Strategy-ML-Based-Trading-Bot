"""
Live inference router that fuses the global SB3 PPO base policy (.zip) with a
ticker-specific Stage-3 fine-tuned head (.pt).

Stage 2 saves an entire SB3 ``PPO`` model as ``ppo_router.zip``. Stage 3 saves
only the delta layers for a single ticker — the ``action_net`` ``state_dict``
and (optionally) the ``log_std`` parameter — in
``{TICKER}_active_policy.pt``. At inference time we need to:

    1. Load the base SB3 model exactly once.
    2. For each ticker, surgically overwrite ``model.policy.action_net``
       (and ``log_std``) with the ticker-specific weights.
    3. Hand the resulting PyTorch module back to the caller (typically the
       Kelly risk manager → execution router) for a single forward pass.

The Stage-3 payload layout produced by
``pipeline.train_orchestrator.stage_3_finetune_ticker`` is::

    {
        "ticker": str,
        "action_net_state_dict": dict[str, torch.Tensor],
        "log_std": torch.Tensor | None,
        "metadata": {...},
    }
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from config.settings_store import load_settings
from core.paths import ArtifactPaths

logger = logging.getLogger(__name__)


class LiveInferenceRouter:
    """
    Combine the base SB3 PPO router with per-ticker fine-tuned action heads.

    The base model is loaded once at construction. ``load_ticker_policy``
    swaps the action head (and ``log_std``) in place and returns the modified
    underlying PyTorch network so the caller can run a forward pass.

    The class is process-safe via an internal lock — concurrent calls from
    different threads serialize the head swap so the cached model never sees
    a half-applied state dict.
    """

    def __init__(
        self,
        base_model_path: Optional[Path] = None,
        *,
        active_policies_dir: Optional[Path] = None,
        device: Optional[str] = None,
        settings: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError(
                "Install stable-baselines3 to use LiveInferenceRouter"
            ) from exc

        cfg = settings or load_settings()
        paths = ArtifactPaths.from_settings(cfg)

        self._base_path: Path = Path(base_model_path) if base_model_path else paths.ppo_router
        if not self._base_path.exists():
            raise FileNotFoundError(
                f"Base PPO checkpoint not found at {self._base_path}. "
                "Train it via scripts/train_ppo.py or scripts/run_orchestrator.py first."
            )

        self._active_dir: Path = (
            Path(active_policies_dir) if active_policies_dir else paths.ticker_policies_dir
        )
        self._device = torch.device(device) if device else torch.device("cpu")

        self._model = PPO.load(str(self._base_path), device=str(self._device))
        self._model.policy.eval()

        self._pristine_action_state: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in self._model.policy.action_net.state_dict().items()
        }
        self._pristine_log_std: Optional[torch.Tensor] = None
        if hasattr(self._model.policy, "log_std") and isinstance(
            self._model.policy.log_std, nn.Parameter
        ):
            self._pristine_log_std = self._model.policy.log_std.detach().clone()

        self._loaded_ticker: Optional[str] = None
        self._lock = RLock()

        logger.info(
            "LiveInferenceRouter ready: base=%s active_dir=%s device=%s",
            self._base_path,
            self._active_dir,
            self._device,
        )

    @property
    def base_model(self) -> Any:
        """The underlying ``stable_baselines3.PPO`` instance (read-only access intended)."""
        return self._model

    @property
    def loaded_ticker(self) -> Optional[str]:
        """Ticker whose head is currently active, or ``None`` if base weights are in place."""
        return self._loaded_ticker

    def reset_to_base(self) -> nn.Module:
        """Restore the pristine base ``action_net`` (and ``log_std``)."""
        with self._lock:
            self._model.policy.action_net.load_state_dict(self._pristine_action_state)
            if self._pristine_log_std is not None and hasattr(self._model.policy, "log_std"):
                with torch.no_grad():
                    self._model.policy.log_std.data.copy_(
                        self._pristine_log_std.to(self._model.policy.log_std.device)
                    )
            self._loaded_ticker = None
            self._model.policy.eval()
            return self._model.policy.action_net

    def _resolve_ticker_path(self, ticker: str) -> Path:
        return self._active_dir / f"{ticker.upper()}_active_policy.pt"

    def load_ticker_policy(self, ticker: str) -> nn.Module:
        """
        Overwrite the base ``action_net`` (and ``log_std``) with the
        ticker-specific fine-tuned weights and return the modified network.

        The returned module shares parameters with ``self.base_model.policy``
        so the next ``model.predict(...)`` call uses the patched head.
        """
        path = self._resolve_ticker_path(ticker)
        if not path.exists():
            raise FileNotFoundError(
                f"No active policy for ticker={ticker!r} at {path}. "
                "Run Stage 3 fine-tune via scripts/run_orchestrator.py."
            )

        payload = torch.load(path, map_location=self._device, weights_only=False)
        action_state: dict[str, torch.Tensor] = payload["action_net_state_dict"]
        log_std_tensor: Optional[torch.Tensor] = payload.get("log_std")

        with self._lock:
            action_net: nn.Module = self._model.policy.action_net

            target_state = {k: v.to(self._device) for k, v in action_state.items()}
            missing, unexpected = action_net.load_state_dict(target_state, strict=False)
            if missing or unexpected:
                logger.warning(
                    "Ticker=%s action_net load: missing=%s unexpected=%s",
                    ticker,
                    missing,
                    unexpected,
                )

            if (
                log_std_tensor is not None
                and hasattr(self._model.policy, "log_std")
                and isinstance(self._model.policy.log_std, nn.Parameter)
            ):
                with torch.no_grad():
                    self._model.policy.log_std.data.copy_(log_std_tensor.to(self._device))

            self._model.policy.eval()
            self._loaded_ticker = ticker.upper()

        logger.info(
            "Loaded ticker policy: %s ← %s", ticker.upper(), path.name
        )
        return action_net

    def predict_weights(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Convenience helper: run a full forward pass on the currently-loaded
        policy (base or ticker-specific) and return the softmax weight vector
        used by the capital router. The caller is responsible for having
        called :meth:`load_ticker_policy` first when a per-ticker head is
        desired.
        """
        from rl.gym_trading_env import TradingRoutingEnv

        with self._lock:
            obs = np.asarray(observation, dtype=np.float64)
            action, _ = self._model.predict(obs, deterministic=deterministic)
            return TradingRoutingEnv.softmax(np.asarray(action, dtype=np.float64))
