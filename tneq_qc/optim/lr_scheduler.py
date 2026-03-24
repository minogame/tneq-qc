"""
Learning rate schedulers (Phase 3.0).
"""

from __future__ import annotations

from typing import List, Tuple

from .base import OptimizerBase


class StepLRScheduler:
    """Step-based learning rate scheduler.

    Adjusts the optimizer's learning rate according to a predefined
    ``(step, lr)`` schedule.

    Args:
        optimizer: Optimizer whose ``lr`` attribute will be modified.
        schedule: List of ``(step, lr)`` tuples in ascending step order.

    Example::

        scheduler = StepLRScheduler(optimizer, [
            (0, 1e-2), (200, 1e-3), (800, 1e-4),
        ])
        for step in range(1000):
            loss, grads = engine.contract_...(qctn, ...)
            optimizer.step(grads)
            scheduler.step()
    """

    def __init__(self, optimizer: OptimizerBase, schedule: List[Tuple[int, float]]):
        self.optimizer = optimizer
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self._step_count: int = 0

    def step(self) -> None:
        """Advance one step and update the optimizer's learning rate."""
        self._step_count += 1
        for s, lr in reversed(self.schedule):
            if self._step_count >= s:
                self.optimizer.lr = lr
                return

    def get_lr(self) -> float:
        """Return the current learning rate."""
        return self.optimizer.lr
