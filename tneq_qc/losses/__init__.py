"""tneq_qc.losses — extensible loss function library.

Quick start
-----------
Use a built-in loss by name::

    loss, grads = engine.contract_with_compiled_strategy_for_gradient(
        qctn, target=1.0, loss='mse'
    )

Register and use a custom loss::

    from tneq_qc.losses import register_loss, BaseLoss

    @register_loss('my_loss')
    class MyLoss(BaseLoss):
        def compute(self, result, target, backend):
            eff = result.tensor * result.scale
            return backend.mean(eff * eff)

    loss, grads = engine.contract_with_compiled_strategy_for_gradient(
        qctn, loss='my_loss'
    )

Pass a callable directly (no registration required)::

    def my_fn(result, target, backend):
        ...

    loss, grads = engine.contract_with_compiled_strategy_for_gradient(
        qctn, loss=my_fn
    )

Built-in losses
---------------
+------------------+-------------------------------------------------------+
| Name             | Description                                           |
+==================+=======================================================+
| ``diagonal_mse`` | Reshape → diagonal → MSE  *(default, backward-compat)*|
+------------------+-------------------------------------------------------+
| ``mse``          | Mean Squared Error                                    |
+------------------+-------------------------------------------------------+
| ``mae``          | Mean Absolute Error                                   |
+------------------+-------------------------------------------------------+
| ``nll``          | Negative Log Likelihood                               |
+------------------+-------------------------------------------------------+
| ``fidelity``     | Quantum fidelity  ``-|⟨result|target⟩|²``            |
+------------------+-------------------------------------------------------+
"""

# Base class — must be imported before registry & builtins
from .base import BaseLoss

# Registry & helpers
from .registry import LossRegistry, register_loss, FunctionalLoss

# Built-in losses (importing this module triggers @register_loss decorators)
from .builtin import (
    DiagonalMSELoss,
    MSELoss,
    MAELoss,
    NLLLoss,
    FidelityLoss,
)

# Target resolver
from .target import TargetResolver

__all__ = [
    # Base
    "BaseLoss",
    # Registry
    "LossRegistry",
    "register_loss",
    "FunctionalLoss",
    # Built-ins
    "DiagonalMSELoss",
    "MSELoss",
    "MAELoss",
    "NLLLoss",
    "FidelityLoss",
    # Target
    "TargetResolver",
]
