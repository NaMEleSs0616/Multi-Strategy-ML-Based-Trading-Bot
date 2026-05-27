"""
pipeline/train_orchestrator.py

Three-stage Global-to-Local Dynamic Fine-Tuning orchestrator for the
multi-strategy RL trading bot.

    Stage 1: Global xLSTM pre-training (AdamW + OU SDE loss + plateau freeze)
    Stage 2: Base PPO routing with Walk-Forward Optimization + Purging
    Stage 3: Async per-ticker micro-fine-tune of ONLY the final routing layer

Constraints (per spec)
----------------------
- Data ingestion is fully abstracted via `Protocol` types; this file contains
  zero data loading, zero API calls, zero DataFrame manipulation.
- Broker / execution is similarly abstracted away.
- All PyTorch logic, CUDA mapping, and asyncio architecture live here.

Dimensional invariants enforced
-------------------------------
- `encoder.input_dim` == ticker features' column count (per stage).
- `encoder.embedding_dim` == PPO `observation_space.shape[0]`.
- Strategy bank size == PPO `action_space.shape[0]` == `action_net.out_features`
  == `log_std.numel()`.
- `PPO.load(..., env=ticker_env)` re-uses the global MLP extractor, so its
  `latent_dim_pi` is locked — Stage 3 NEVER re-instantiates `PPO(...)`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from models.xlstm.custom_loss import XLSTMPretrainingLoss
from models.xlstm.dataset import NextStepSequenceDataset, train_val_split
from models.xlstm.encoder import (
    XLSTMConfig,
    XLSTMStateEncoder,
    freeze_encoder,
)
from models.xlstm.inference import encode_feature_matrix
from rl.gym_trading_env import TradingRoutingEnv
from rl.validation.purge import PurgedWalkForwardSplitter, WalkForwardFold

LOG = logging.getLogger("train_orchestrator")


# ============================================================================
# 0. Device + load helpers
# ============================================================================

def select_device(prefer: Optional[str] = None) -> torch.device:
    """CUDA → MPS → CPU. Respects an explicit override."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_encoder_to_device(
    path: Path,
    device: torch.device,
    *,
    freeze: bool = True,
) -> XLSTMStateEncoder:
    """
    Load an xLSTM checkpoint and PIN it to ``device``.

    ``XLSTMStateEncoder.load_checkpoint`` honors ``map_location`` only for the
    ``torch.load`` call; the constructed model itself is left on CPU. We
    therefore call ``.to(device)`` explicitly before any forward pass — without
    this, mixed-device tensors raise ``RuntimeError`` on the first batch.
    """
    encoder = XLSTMStateEncoder.load_checkpoint(
        path, map_location=device, freeze=freeze
    )
    encoder.to(device)
    encoder.eval()  # frozen encoder: no dropout, no BN running stats
    return encoder


# ============================================================================
# 1. External abstractions (Protocols — data ingestion lives ELSEWHERE)
# ============================================================================

@runtime_checkable
class GlobalFeatureProvider(Protocol):
    """Returns PiT-safe, PRE-COMPUTED arrays for the global universe."""

    def get_global_features(self) -> np.ndarray:
        """(T, input_dim) float32 PiT-shifted feature matrix."""
        ...

    def get_global_strategy_returns(self) -> np.ndarray:
        """(T, n_strategies) per-bar strategy return matrix."""
        ...

    def get_global_benchmark_returns(self) -> np.ndarray:
        """(T,) benchmark (e.g. SPY) per-bar return series."""
        ...


@runtime_checkable
class TickerFeatureProvider(Protocol):
    """Returns PiT-safe, PRE-COMPUTED arrays for a single ticker."""

    def get_ticker_features(self, ticker: str) -> np.ndarray:
        """(T_i, input_dim) — MUST match global encoder input_dim."""
        ...

    def get_ticker_strategy_returns(self, ticker: str) -> np.ndarray:
        """(T_i, n_strategies) — MUST match Stage-2 action_space."""
        ...

    def get_ticker_benchmark_returns(self, ticker: str) -> np.ndarray:
        """(T_i,) per-bar benchmark return series."""
        ...


@runtime_checkable
class FeatureProvider(GlobalFeatureProvider, TickerFeatureProvider, Protocol):
    """Combined provider used by the full orchestrator."""
    ...


# ============================================================================
# 2. Stage 1 — Global xLSTM pre-training
# ============================================================================

@dataclass
class Stage1Config:
    input_dim: int
    hidden_dim: int = 128
    embedding_dim: int = 64
    num_blocks: int = 2
    dropout: float = 0.1
    sequence_length: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    plateau_patience: int = 5
    plateau_min_delta: float = 1e-4
    val_fraction: float = 0.2
    ou_gamma: float = 0.1
    ou_dt: float = 1.0
    grad_clip_norm: float = 1.0


def stage_1_pretrain_global_xlstm(
    features_np: np.ndarray,
    config: Stage1Config,
    out_path: Path,
    device: torch.device,
) -> Path:
    """
    Stage 1 — Train the xLSTM with AdamW on the OU SDE-informed pre-training
    loss to predict t+1 features; on validation plateau, detach all gradients
    (``requires_grad=False``) and persist the locked encoder to ``out_path``.
    """
    if features_np.ndim != 2:
        raise ValueError(f"features must be 2D (T, D); got {features_np.shape}")
    if features_np.shape[1] != config.input_dim:
        raise ValueError(
            f"feature input_dim {features_np.shape[1]} != "
            f"config.input_dim {config.input_dim}"
        )

    dataset = NextStepSequenceDataset(
        features_np.astype(np.float32), config.sequence_length
    )
    train_idx, val_idx = train_val_split(dataset, val_fraction=config.val_fraction)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=config.batch_size, shuffle=False
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=config.batch_size, shuffle=False
    )

    model_cfg = XLSTMConfig(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
        sequence_length=config.sequence_length,
    )
    model = XLSTMStateEncoder(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = XLSTMPretrainingLoss(gamma=config.ou_gamma, dt=config.ou_dt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_path.with_name(out_path.stem + "_best.pt")
    best_val = float("inf")
    plateau_wait = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            emb, pred, h_seq = model(
                xb, return_prediction=True, return_hidden_states=True
            )
            breakdown = criterion(pred, yb, emb, hidden_states=h_seq)
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.grad_clip_norm
            )
            optimizer.step()

        model.eval()
        val_acc, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                emb, pred, h_seq = model(
                    xb, return_prediction=True, return_hidden_states=True
                )
                breakdown = criterion(pred, yb, emb, hidden_states=h_seq)
                val_acc += float(breakdown.total.item()) * xb.size(0)
                val_n += xb.size(0)
        val_loss = val_acc / max(val_n, 1)
        LOG.info("[Stage1] epoch=%d val_loss=%.6f", epoch, val_loss)

        if val_loss < best_val - config.plateau_min_delta:
            best_val = val_loss
            plateau_wait = 0
            model.save_checkpoint(
                best_ckpt, metadata={"epoch": epoch, "val_loss": val_loss}
            )
        else:
            plateau_wait += 1
            if plateau_wait >= config.plateau_patience:
                LOG.info("[Stage1] plateau hit @ epoch=%d, freezing", epoch)
                break

    # Reload best, pin to device, freeze, persist final global weights.
    model = _load_encoder_to_device(best_ckpt, device=device, freeze=False)
    freeze_encoder(model)
    # Explicit per-parameter detachment (defense in depth — spec requirement).
    for param in model.parameters():
        param.requires_grad = False
    model.save_checkpoint(
        out_path, metadata={"frozen": True, "best_val_loss": best_val}
    )
    LOG.info("[Stage1] saved frozen global xLSTM → %s", out_path)
    return out_path


# ============================================================================
# 3. Stage 2 — Base PPO with Walk-Forward Optimization + Purging
# ============================================================================

@dataclass
class Stage2Config:
    n_splits: int = 5
    embargo: int = 5
    min_train_size: int = 252
    test_size: int = 63
    total_timesteps_per_fold: int = 20_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    turnover_penalty_lambda: float = 1e-4
    turnover_penalty_multiplier: float = 1.0
    sortino_weight: float = 0.5
    outperformance_weight: float = 0.5
    sortino_window: int = 30


def stage_2_train_base_ppo(
    encoder_path: Path,
    provider: GlobalFeatureProvider,
    config: Stage2Config,
    out_path: Path,
    device: torch.device,
    env_settings: dict[str, Any],
) -> Path:
    """
    Stage 2 — Train the base PPO router with Walk-Forward Optimization +
    Purging, treating the frozen xLSTM as a deterministic state encoder.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError("Install stable-baselines3 to run Stage 2") from exc

    encoder = _load_encoder_to_device(encoder_path, device=device, freeze=True)

    feats = np.asarray(provider.get_global_features(), dtype=np.float32)
    if feats.shape[1] != encoder.config.input_dim:
        raise ValueError(
            f"[Stage2] feature input_dim {feats.shape[1]} != "
            f"encoder.input_dim {encoder.config.input_dim}"
        )

    # Frozen xLSTM forward pass: materialize PiT embeddings (no grad).
    embeddings = encode_feature_matrix(encoder, feats, device=device)
    strat_ret = np.asarray(
        provider.get_global_strategy_returns(), dtype=np.float32
    )
    bench_ret = np.asarray(
        provider.get_global_benchmark_returns(), dtype=np.float32
    )

    if not (embeddings.shape[0] == strat_ret.shape[0] == bench_ret.shape[0]):
        raise ValueError(
            "[Stage2] timeline mismatch: "
            f"emb T={embeddings.shape[0]}, strat T={strat_ret.shape[0]}, "
            f"bench T={bench_ret.shape[0]}"
        )

    n_strategies = int(strat_ret.shape[1])
    emb_dim = int(embeddings.shape[1])

    splitter = PurgedWalkForwardSplitter(
        n_splits=config.n_splits,
        embargo=config.embargo,
        min_train_size=config.min_train_size,
        test_size=config.test_size,
    )
    folds = list(splitter.split(embeddings.shape[0]))
    if not folds:
        raise RuntimeError("[Stage2] walk-forward produced no folds")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model: Optional["PPO"] = None

    for fold in folds:
        def make_env(_fold: WalkForwardFold = fold) -> TradingRoutingEnv:
            return TradingRoutingEnv(
                embeddings,
                strat_ret,
                bench_ret,
                walk_forward_fold=_fold,
                turnover_penalty_lambda=config.turnover_penalty_lambda,
                turnover_penalty_multiplier=config.turnover_penalty_multiplier,
                sortino_weight=config.sortino_weight,
                outperformance_weight=config.outperformance_weight,
                sortino_window=config.sortino_window,
                purge_embargo=config.embargo,
                settings=env_settings,
            )

        vec_env = DummyVecEnv([make_env])

        if model is None:
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                device=str(device),
                verbose=0,
            )
            # Stage-2/3 handoff invariants — fail loudly NOW, not at load time.
            assert model.observation_space.shape == (emb_dim,), (
                f"obs_space {model.observation_space.shape} != emb_dim {emb_dim}"
            )
            assert model.action_space.shape == (n_strategies,), (
                f"act_space {model.action_space.shape} != n_strategies {n_strategies}"
            )
            assert model.policy.action_net.out_features == n_strategies, (
                f"action_net.out_features {model.policy.action_net.out_features} "
                f"!= n_strategies {n_strategies}"
            )
            if hasattr(model.policy, "log_std") and isinstance(
                model.policy.log_std, nn.Parameter
            ):
                assert model.policy.log_std.shape == (n_strategies,), (
                    f"log_std {tuple(model.policy.log_std.shape)} != ({n_strategies},)"
                )
        else:
            model.set_env(vec_env)

        model.learn(
            total_timesteps=config.total_timesteps_per_fold,
            reset_num_timesteps=False,
        )
        LOG.info("[Stage2] fold=%d trained", fold.fold_id)

    assert model is not None
    model.save(str(out_path))
    LOG.info("[Stage2] saved base PPO router → %s", out_path)
    return out_path


# ============================================================================
# 4. Stage 3 — Per-ticker fine-tune of ONLY the final routing layer
# ============================================================================

@dataclass
class Stage3Config:
    epochs: int = 20
    n_steps_per_epoch: int = 1024
    batch_size: int = 64
    learning_rate: float = 1e-3
    grad_clip_norm: float = 0.5
    weight_decay: float = 0.0


@dataclass
class FineTuneRequest:
    """Job descriptor pushed onto the async fine-tune queue."""

    ticker: str
    encoder_path: Path
    base_policy_path: Path
    out_dir: Path


def _freeze_policy_except_action_head(model: Any) -> list[nn.Parameter]:
    """
    Freeze every PPO parameter except the **final linear routing layer**
    (``action_net``) and — for Gaussian continuous policies — the action
    log-std parameter. Returns the trainable parameter list.
    """
    policy = model.policy
    for p in policy.parameters():
        p.requires_grad = False

    trainable: list[nn.Parameter] = []
    for p in policy.action_net.parameters():
        p.requires_grad = True
        trainable.append(p)

    if hasattr(policy, "log_std") and isinstance(policy.log_std, nn.Parameter):
        policy.log_std.requires_grad = True
        trainable.append(policy.log_std)

    return trainable


def stage_3_finetune_ticker(
    request: FineTuneRequest,
    provider: TickerFeatureProvider,
    config: Stage3Config,
    device: torch.device,
    env_settings: dict[str, Any],
) -> Path:
    """
    Stage 3 — Aggressive 20-epoch micro-fine-tune of the **final linear
    routing layer** of the base PPO on a single ticker's environment.

    xLSTM remains 100% frozen. Only ``policy.action_net`` (+ optional
    ``log_std``) is unfrozen and optimized.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError("Install stable-baselines3 to run Stage 3") from exc

    encoder = _load_encoder_to_device(
        request.encoder_path, device=device, freeze=True
    )

    feats = np.asarray(
        provider.get_ticker_features(request.ticker), dtype=np.float32
    )
    if feats.shape[1] != encoder.config.input_dim:
        raise ValueError(
            f"[Stage3:{request.ticker}] feature input_dim {feats.shape[1]} "
            f"!= global encoder input_dim {encoder.config.input_dim}"
        )

    embeddings = encode_feature_matrix(encoder, feats, device=device)
    strat_ret = np.asarray(
        provider.get_ticker_strategy_returns(request.ticker), dtype=np.float32
    )
    bench_ret = np.asarray(
        provider.get_ticker_benchmark_returns(request.ticker), dtype=np.float32
    )

    if not (embeddings.shape[0] == strat_ret.shape[0] == bench_ret.shape[0]):
        raise ValueError(
            f"[Stage3:{request.ticker}] timeline mismatch: "
            f"emb T={embeddings.shape[0]}, strat T={strat_ret.shape[0]}, "
            f"bench T={bench_ret.shape[0]}"
        )

    # Treat the whole post-warmup range as a single training fold.
    warmup = encoder.config.sequence_length - 1
    train_idx = np.arange(warmup, embeddings.shape[0], dtype=np.int64)
    fold = WalkForwardFold(
        fold_id=0,
        train_indices=train_idx,
        test_indices=np.array([], dtype=np.int64),
    )

    def make_env() -> TradingRoutingEnv:
        return TradingRoutingEnv(
            embeddings,
            strat_ret,
            bench_ret,
            walk_forward_fold=fold,
            turnover_penalty_lambda=float(
                env_settings.get("rl", {}).get("turnover_penalty_lambda", 1e-4)
            ),
            purge_embargo=int(
                env_settings.get("rl", {}).get("purge_embargo_bars", 5)
            ),
            settings=env_settings,
        )

    vec_env = DummyVecEnv([make_env])

    # Critical: PPO.load preserves the global mlp_extractor + action_net
    # shapes; never re-instantiate `PPO(...)` here or the latent_dim_pi /
    # action_net dimensions get re-rolled and the load fails.
    model = PPO.load(str(request.base_policy_path), env=vec_env, device=str(device))

    # Stage-2 ↔ Stage-3 handoff dimension assertions.
    n_strategies = int(strat_ret.shape[1])
    emb_dim = int(embeddings.shape[1])
    if model.observation_space.shape != (emb_dim,):
        raise ValueError(
            f"[Stage3:{request.ticker}] obs space "
            f"{model.observation_space.shape} != ({emb_dim},)"
        )
    if model.action_space.shape != (n_strategies,):
        raise ValueError(
            f"[Stage3:{request.ticker}] act space "
            f"{model.action_space.shape} != ({n_strategies},)"
        )
    if model.policy.action_net.out_features != n_strategies:
        raise ValueError(
            f"[Stage3:{request.ticker}] action_net.out_features "
            f"{model.policy.action_net.out_features} != n_strategies {n_strategies}"
        )
    if hasattr(model.policy, "log_std") and isinstance(
        model.policy.log_std, nn.Parameter
    ):
        if model.policy.log_std.shape != (n_strategies,):
            raise ValueError(
                f"[Stage3:{request.ticker}] log_std "
                f"{tuple(model.policy.log_std.shape)} != ({n_strategies},)"
            )

    # Surgical freeze: ONLY action_net + log_std remain trainable.
    trainable = _freeze_policy_except_action_head(model)
    if not trainable:
        raise RuntimeError(
            f"[Stage3:{request.ticker}] no trainable parameters found"
        )
    for p in trainable:
        if not p.is_leaf:
            raise RuntimeError(
                "trainable param is not a leaf tensor — cannot be optimized"
            )

    # Replace SB3's internal optimizer with a fresh AdamW restricted to the
    # unfrozen params on the correct device. Frozen params still appear in
    # `policy.parameters()` but have requires_grad=False, so PyTorch will not
    # populate their .grad fields — the optimizer step is a no-op for them.
    model.policy.optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_timesteps = config.epochs * config.n_steps_per_epoch
    LOG.info(
        "[Stage3:%s] starting fine-tune: %d epochs × %d steps = %d timesteps",
        request.ticker,
        config.epochs,
        config.n_steps_per_epoch,
        total_timesteps,
    )
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=True)

    request.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = request.out_dir / f"{request.ticker}_active_policy.pt"

    # Save ONLY the delta layers — tiny file, hot-swappable on top of the
    # global base policy at inference time.
    payload: dict[str, Any] = {
        "ticker": request.ticker,
        "action_net_state_dict": {
            k: v.detach().cpu()
            for k, v in model.policy.action_net.state_dict().items()
        },
        "log_std": (
            model.policy.log_std.detach().cpu().clone()
            if hasattr(model.policy, "log_std")
            and isinstance(model.policy.log_std, nn.Parameter)
            else None
        ),
        "metadata": {
            "epochs": config.epochs,
            "total_timesteps": total_timesteps,
            "n_strategies": n_strategies,
            "embedding_dim": emb_dim,
            "device": str(device),
            "base_policy_path": str(request.base_policy_path),
            "encoder_path": str(request.encoder_path),
        },
    }
    torch.save(payload, out_path)
    LOG.info("[Stage3:%s] saved active policy → %s", request.ticker, out_path)
    return out_path


# ============================================================================
# 5. Async background worker — asyncio.Queue + thread pool for blocking torch
# ============================================================================

class TickerFineTuneQueue:
    """
    Asynchronous, FIFO Stage-3 dispatcher.

    PyTorch / SB3 training is synchronous-blocking, so each job is offloaded
    to a ``ThreadPoolExecutor`` and concurrent GPU access is serialized via
    an ``asyncio.Semaphore`` to avoid VRAM contention.
    """

    def __init__(
        self,
        provider: TickerFeatureProvider,
        config: Stage3Config,
        device: torch.device,
        env_settings: dict[str, Any],
        *,
        max_concurrent: int = 1,
    ) -> None:
        self._provider = provider
        self._config = config
        self._device = device
        self._env_settings = env_settings
        self._queue: asyncio.Queue[FineTuneRequest] = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_concurrent),
            thread_name_prefix="ft-worker",
        )
        self._workers: list[asyncio.Task[Any]] = []
        self._stop = asyncio.Event()

    async def submit(self, request: FineTuneRequest) -> None:
        """Enqueue a fine-tune job. Non-blocking, returns immediately."""
        await self._queue.put(request)

    async def _worker_loop(self, worker_id: int) -> None:
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            try:
                request = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                async with self._semaphore:
                    LOG.info(
                        "[FT worker %d] start ticker=%s", worker_id, request.ticker
                    )
                    await loop.run_in_executor(
                        self._executor,
                        stage_3_finetune_ticker,
                        request,
                        self._provider,
                        self._config,
                        self._device,
                        self._env_settings,
                    )
                    if self._device.type == "cuda":
                        torch.cuda.empty_cache()
                    LOG.info(
                        "[FT worker %d] done  ticker=%s", worker_id, request.ticker
                    )
            except Exception:
                LOG.exception(
                    "[FT worker %d] failed on ticker=%s",
                    worker_id,
                    request.ticker,
                )
            finally:
                self._queue.task_done()

    async def start(self, n_workers: int = 1) -> None:
        for i in range(n_workers):
            self._workers.append(asyncio.create_task(self._worker_loop(i)))

    async def join(self) -> None:
        """Block until every queued job has finished."""
        await self._queue.join()

    async def stop(self) -> None:
        self._stop.set()
        for t in self._workers:
            t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._executor.shutdown(wait=True)


# ============================================================================
# 6. Top-level orchestration entry point
# ============================================================================

async def run_full_orchestration(
    provider: FeatureProvider,
    stage1_config: Stage1Config,
    stage2_config: Stage2Config,
    stage3_config: Stage3Config,
    env_settings: dict[str, Any],
    out_dir: Path,
    tickers: list[str],
    *,
    prefer_device: Optional[str] = None,
    max_concurrent_ft: int = 1,
) -> dict[str, Path]:
    """
    Run Stage 1 and Stage 2 sequentially on the calling task, then dispatch
    every requested ticker's Stage-3 fine-tune through the async queue.

    Returns a mapping ``ticker -> active policy path``.
    """
    device = select_device(prefer_device)
    LOG.info("[orchestrator] device=%s", device)

    # Canonical filenames are shared with scripts/train_xlstm.py + scripts/train_ppo.py
    # so Path A (CLI) and Path B (Orchestrator) always read/write the same artifacts.
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = out_dir / "xlstm_frozen.pt"
    base_policy_path = out_dir / "ppo_router.zip"
    ticker_dir = out_dir / "ticker_policies"
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # ===== Stage 1 =====
    feats = np.asarray(provider.get_global_features(), dtype=np.float32)
    stage_1_pretrain_global_xlstm(feats, stage1_config, encoder_path, device)

    # ===== Stage 2 =====
    stage_2_train_base_ppo(
        encoder_path=encoder_path,
        provider=provider,
        config=stage2_config,
        out_path=base_policy_path,
        device=device,
        env_settings=env_settings,
    )

    # ===== Stage 3 (async) =====
    queue = TickerFineTuneQueue(
        provider=provider,
        config=stage3_config,
        device=device,
        env_settings=env_settings,
        max_concurrent=max_concurrent_ft,
    )
    await queue.start(n_workers=max_concurrent_ft)

    results: dict[str, Path] = {}
    for ticker in tickers:
        request = FineTuneRequest(
            ticker=ticker,
            encoder_path=encoder_path,
            base_policy_path=base_policy_path,
            out_dir=ticker_dir,
        )
        await queue.submit(request)
        results[ticker] = ticker_dir / f"{ticker}_active_policy.pt"

    await queue.join()
    await queue.stop()
    return results


def main(
    provider: FeatureProvider,
    stage1_config: Stage1Config,
    stage2_config: Stage2Config,
    stage3_config: Stage3Config,
    env_settings: dict[str, Any],
    out_dir: Path,
    tickers: list[str],
    *,
    prefer_device: Optional[str] = None,
    max_concurrent_ft: int = 1,
) -> dict[str, Path]:
    """Synchronous wrapper around :func:`run_full_orchestration`."""
    return asyncio.run(
        run_full_orchestration(
            provider=provider,
            stage1_config=stage1_config,
            stage2_config=stage2_config,
            stage3_config=stage3_config,
            env_settings=env_settings,
            out_dir=out_dir,
            tickers=tickers,
            prefer_device=prefer_device,
            max_concurrent_ft=max_concurrent_ft,
        )
    )
