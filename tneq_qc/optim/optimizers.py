"""
Concrete optimizer classes (Phase 3.0).

Each class sets :attr:`method` and provides a convenience constructor
with named hyper-parameters.  The actual optimisation algorithm lives
in ``backend.optimizer_update()``; these classes are thin OOP wrappers.
"""

from __future__ import annotations

from typing import List

from ..core.tn_tensor import TNTensor
from .base import OptimizerBase


class Adam(OptimizerBase):
    """Adam optimizer.

    Args:
        params: Trainable parameters.
        backend: Compute backend.
        lr: Learning rate.
        beta1: Exponential decay rate for first moment.
        beta2: Exponential decay rate for second moment.
        epsilon: Numerical stability constant.
    """

    method = "adam"

    def __init__(
        self,
        params: List[TNTensor],
        backend,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(params, backend, lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)


class SGD(OptimizerBase):
    """Vanilla stochastic gradient descent."""

    method = "sgd"


class SGDG(OptimizerBase):
    """Stiefel Gradient Descent (SGDG).

    Preserves orthogonality constraints via Cayley transform.

    Args:
        params: Trainable parameters.
        backend: Compute backend.
        lr: Learning rate.
        momentum: Momentum factor.
        stiefel: Whether to use Stiefel manifold optimisation.
    """

    method = "sgdg"

    def __init__(
        self,
        params: List[TNTensor],
        backend,
        lr: float = 0.01,
        momentum: float = 0.9,
        stiefel: bool = True,
    ):
        super().__init__(params, backend, lr=lr, momentum=momentum, stiefel=stiefel)


class Momentum(OptimizerBase):
    """SGD with momentum."""

    method = "momentum"

    def __init__(
        self,
        params: List[TNTensor],
        backend,
        lr: float = 0.01,
        momentum: float = 0.9,
    ):
        super().__init__(params, backend, lr=lr, momentum=momentum)


class RMSProp(OptimizerBase):
    """RMSProp optimizer.

    Args:
        params: Trainable parameters.
        backend: Compute backend.
        lr: Learning rate.
        epsilon: Numerical stability constant.
    """

    method = "rmsprop"

    def __init__(
        self,
        params: List[TNTensor],
        backend,
        lr: float = 0.01,
        epsilon: float = 1e-8,
    ):
        super().__init__(params, backend, lr=lr, epsilon=epsilon)
