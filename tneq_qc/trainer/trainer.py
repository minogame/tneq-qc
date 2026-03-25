"""
Single-process Trainer (Phase 3.0).

Encapsulates the train loop that every example script previously
duplicated by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.qctn import QCTN
    from ..core.engine_common import EngineCommon
    from ..optim.base import OptimizerBase
    from ..optim.lr_scheduler import StepLRScheduler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training configuration.

    Attributes:
        max_steps: Total number of training steps.
        log_every: Print a log line every N steps (0 = silent).
        save_every: Save a checkpoint every N steps (0 = disabled).
        save_path: File path for checkpoint saving.
        tol: Early-stop when loss drops below this value (0 = disabled).
    """

    max_steps: int = 1000
    log_every: int = 10
    save_every: int = 0
    save_path: str = ""
    tol: float = 0.0


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class Callback:
    """Base class for training callbacks."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        pass

    def on_step_end(self, step: int, loss: float, trainer: "Trainer") -> None:
        pass

    def on_train_end(self, loss_history: List[float], trainer: "Trainer") -> None:
        pass


class TqdmCallback(Callback):
    """Show a tqdm progress bar during training."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        from tqdm import tqdm

        self._pbar = tqdm(total=trainer.config.max_steps, desc="Training")

    def on_step_end(self, step: int, loss: float, trainer: "Trainer") -> None:
        self._pbar.update(1)
        self._pbar.set_postfix(loss=f"{loss:.6f}")

    def on_train_end(self, loss_history: List[float], trainer: "Trainer") -> None:
        self._pbar.close()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Single-process training loop for EngineCommon.

    Supports two modes:

    1. **Static**: *target* and *loss* are fixed across all steps
       (e.g. ``train_tneq``, ``train_mnist``).
    2. **Dynamic data**: a *data_fn* is called before each step to update
       QCTN cores in-place (e.g. ``train_quadratic`` where Mx changes
       every step).

    Example (static)::

        trainer = Trainer(engine, qctn, optimizer)
        history = trainer.fit(target=1.0, loss='mse')

    Example (dynamic)::

        def data_fn(step):
            Mx_list, _ = data_gen.generate(x, K=2, ret_type='TNTensor')
            for i, name in enumerate(mx_names):
                qctn[name] = Mx_list[i]

        trainer = Trainer(engine, qctn, optimizer)
        history = trainer.fit(target=y, loss=nll_loss, data_fn=data_fn)

    Args:
        engine: :class:`EngineCommon` instance.
        qctn: QCTN model (all cores already embedded).
        optimizer: Optimizer instance (wraps ``backend.optimizer_update``).
        config: Training configuration.
        scheduler: Optional LR scheduler.
        callbacks: Optional list of :class:`Callback` instances.
    """

    def __init__(
        self,
        engine: "EngineCommon",
        qctn: "QCTN",
        optimizer: "OptimizerBase",
        config: Optional[TrainConfig] = None,
        scheduler: Optional["StepLRScheduler"] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.engine = engine
        self.qctn = qctn
        self.optimizer = optimizer
        self.config = config or TrainConfig()
        self.scheduler = scheduler
        self.callbacks: List[Callback] = callbacks or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        target=None,
        loss=None,
        data_fn: Optional[Callable[[int], None]] = None,
    ) -> List[float]:
        """Execute the training loop.

        Args:
            target: Learning target forwarded to
                :meth:`EngineCommon.contract_with_compiled_strategy_for_gradient`.
            loss: Loss specification (string name, callable, or
                :class:`BaseLoss` instance).
            data_fn: Optional callable ``(step) -> None`` invoked before
                each forward pass to update data cores in *qctn*.

        Returns:
            List of loss values, one per completed step.
        """
        cfg = self.config
        loss_history: List[float] = []

        for cb in self.callbacks:
            cb.on_train_begin(self)

        for step in range(1, cfg.max_steps + 1):
            # Optional per-step data update
            if data_fn is not None:
                data_fn(step)

            # Forward + backward
            loss_tensor, grads = self.engine.contract_for_gradient(
                self.qctn, target=target, loss=loss,
            )

            # Optimizer step
            self.optimizer.step(list(grads))

            # LR schedule
            if self.scheduler is not None:
                self.scheduler.step()

            # Record
            lv = float(loss_tensor)
            loss_history.append(lv)

            # Callbacks
            for cb in self.callbacks:
                cb.on_step_end(step, lv, self)

            # Logging
            if cfg.log_every and (step % cfg.log_every == 0 or step == 1):
                lr_str = f"  lr={self.optimizer.lr:.1e}" if self.scheduler else ""
                print(f"  Step {step:4d}/{cfg.max_steps}  loss={lv:.6f}{lr_str}")

            # Early stop
            if cfg.tol and lv < cfg.tol:
                print(f"  Converged at step {step} (loss={lv:.6f} < tol={cfg.tol})")
                break

            # Periodic checkpoint
            if cfg.save_every and cfg.save_path and step % cfg.save_every == 0:
                self.qctn.save_cores(cfg.save_path, metadata={"step": str(step)})

        for cb in self.callbacks:
            cb.on_train_end(loss_history, self)

        # Final save
        if cfg.save_path:
            self.qctn.save_cores(
                cfg.save_path,
                metadata={
                    "final_loss": f"{loss_history[-1]:.6f}" if loss_history else "N/A",
                    "total_steps": str(len(loss_history)),
                },
            )

        return loss_history
