"""TNEQ-QC Optimizers (Phase 3.0)."""

from .base import OptimizerBase, ParamRef
from .optimizers import Adam, SGD, SGDG, Momentum, RMSProp
from .lr_scheduler import StepLRScheduler
from .ops import TensorOps, BackendTensorOps
from .registry import register_optimizer, get_registered_optimizers, create_optimizer

# Keep backward-compatible import of the legacy Optimizer.
from .optimizer import Optimizer


register_optimizer("adam", Adam)
register_optimizer("sgd", SGD)
register_optimizer("sgdg", SGDG)
register_optimizer("momentum", Momentum)
register_optimizer("rmsprop", RMSProp)

__all__ = [
    "OptimizerBase",
    "ParamRef",
    "Adam",
    "SGD",
    "SGDG",
    "Momentum",
    "RMSProp",
    "StepLRScheduler",
    "TensorOps",
    "BackendTensorOps",
    "register_optimizer",
    "get_registered_optimizers",
    "create_optimizer",
    "Optimizer",
]
