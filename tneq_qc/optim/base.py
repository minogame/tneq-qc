"""
Optimizer base class (Phase 3.0).

Provides an OOP wrapper around ``backend.optimizer_update()`` with a
PyTorch-like interface.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..core.tn_tensor import TNTensor


class OptimizerBase:
    """Base class for optimizers.

    Wraps ``backend.optimizer_update()`` as an object-oriented API so that
    users no longer need to manually manage ``optimizer_state`` dicts.

    Args:
        params: Trainable parameters (typically from ``qctn.parameters()``).
        backend: Compute backend instance.
        lr: Learning rate.
        **kwargs: Extra hyper-parameters forwarded to the backend.
    """

    method: str = "sgd"  # subclasses override

    def __init__(
        self,
        params: List[TNTensor],
        backend,
        lr: float = 0.01,
        **kwargs,
    ):
        self.params: List[TNTensor] = list(params)
        self.backend = backend
        self.lr = lr
        self._extra_hp: Dict[str, Any] = kwargs
        self._state: Dict[str, Any] = {}
        self._step_count: int = 0

    def step(self, grads: List) -> None:
        """Perform a single parameter update.

        Args:
            grads: Gradient list aligned with ``self.params``.
        """
        hyperparams = {
            "learning_rate": self.lr,
            "iter": self._step_count,
            **self._extra_hp,
        }
        self.params, self._state = self.backend.optimizer_update(
            self.params, grads, self._state, self.method, hyperparams
        )
        self._step_count += 1

    def zero_grad(self) -> None:
        """Zero out gradients on all parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @property
    def state(self) -> Dict[str, Any]:
        """Current optimizer state (momentum buffers, etc.)."""
        return self._state
