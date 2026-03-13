"""Built-in loss functions for tneq_qc.

All losses are registered in :class:`~tneq_qc.losses.registry.LossRegistry`
automatically when this module is imported.

Registered names
----------------
``'diagonal_mse'``
    Reshape → diagonal → MSE.  This is the *default* loss (backward-compatible
    with the hard-coded behaviour that existed before Phase 2.6.3).

``'mse'``
    Plain Mean Squared Error between effective result and effective target.

``'mae'``
    Mean Absolute Error.

``'nll'``
    Negative Log Likelihood for classification (result treated as a
    probability distribution).

``'fidelity'``
    Quantum fidelity loss: minimise ``-|⟨result|target⟩|²``.
"""

from __future__ import annotations
import math
from typing import Any

from .base import BaseLoss
from .registry import register_loss


# ======================================================================
# Helpers
# ======================================================================

def _effective(tn_tensor, backend):
    """Return ``tensor * scale`` (both detached from scale side)."""
    return tn_tensor.tensor * backend.detach(tn_tensor.scale)


def _real_effective(tn_tensor, backend):
    """Effective value; apply abs² if complex."""
    eff = _effective(tn_tensor, backend)
    if backend.is_complex(eff):
        eff = backend.abs_square(eff)
    return eff


# ======================================================================
# diagonal_mse  (default — backward-compatible)
# ======================================================================

@register_loss('diagonal_mse')
class DiagonalMSELoss(BaseLoss):
    """Reshape result to a square matrix, take the diagonal, then MSE.

    This reproduces exactly the hard-coded behaviour of the engine before
    Phase 2.6.3 and is therefore the **default** loss when ``loss=None``.

    Preprocessing
    ~~~~~~~~~~~~~
    Given a flat result of ``n`` elements (``n`` must be a perfect square):

    1. Reshape to ``(side, side)`` where ``side = isqrt(n)``.
    2. Extract the diagonal → vector of length ``side``.

    Loss
    ~~~~
    ``mean((abs²(diag) * scale - target_value)²)``

    where ``target_value`` comes from the resolved target (default:
    ``ones * 1/side`` broadcast).
    """

    def preprocess(self, result, backend):
        n_elem = 1
        for d in result.shape:
            n_elem *= d
        side = int(math.isqrt(n_elem))
        result = backend.reshape(result, (side, side))
        result = backend.diagonal(result)
        return result

    def compute(self, result, target, backend):
        res_tensor = result.tensor
        res_scale = backend.detach(result.scale)
        if backend.is_complex(res_tensor):
            res_tensor = backend.abs_square(res_tensor)
        effective = res_tensor * res_scale
        eff_t = _effective(target, backend)
        diff = effective - eff_t
        return backend.mean(diff * diff)


# ======================================================================
# mse
# ======================================================================

@register_loss('mse')
class MSELoss(BaseLoss):
    """Mean Squared Error: ``mean((eff_result - eff_target)²)``.

    Complex tensors are converted to real via abs² before subtraction.
    """

    def compute(self, result, target, backend):
        eff_r = _real_effective(result, backend)
        eff_t = _real_effective(target, backend)
        diff = eff_r - eff_t
        return backend.mean(diff * diff)


# ======================================================================
# mae
# ======================================================================

@register_loss('mae')
class MAELoss(BaseLoss):
    """Mean Absolute Error: ``mean(|eff_result - eff_target|)``.

    Uses ``sqrt(diff²)`` as a differentiable absolute value.
    Complex tensors are converted to real via abs².
    """

    def compute(self, result, target, backend):
        eff_r = _real_effective(result, backend)
        eff_t = _real_effective(target, backend)
        diff = eff_r - eff_t
        return backend.mean(backend.sqrt(diff * diff + 1e-12))


# ======================================================================
# nll
# ======================================================================

@register_loss('nll')
class NLLLoss(BaseLoss):
    """Negative Log Likelihood for classification.

    Assumes *result* represents (unnormalised) log-probabilities or a
    probability distribution, and *target* is a one-hot or soft label.

    ``loss = -mean(target * log(result + ε))``
    """

    EPS: float = 1e-10

    def compute(self, result: Any, target: Any, backend: Any) -> Any:
        eff_r = _real_effective(result, backend)
        eff_t = _real_effective(target, backend)
        log_p = backend.log(eff_r + self.EPS)
        return -backend.mean(eff_t * log_p)


# ======================================================================
# fidelity
# ======================================================================

@register_loss('fidelity')
class FidelityLoss(BaseLoss):
    """Quantum fidelity loss: ``-|⟨result|target⟩|²``.

    Minimising this loss maximises the fidelity (inner product) between
    the contraction result and the target state.

    Both *result* and *target* are treated as flat complex vectors.
    The effective values (``tensor * scale``) are used for the inner
    product, so the scale is taken into account.
    """

    def compute(self, result: Any, target: Any, backend: Any) -> Any:
        eff_r = _effective(result, backend)
        eff_t = _effective(target, backend)
        # Flatten both to 1-D for the inner product
        eff_r_flat = backend.reshape(eff_r, (-1,)) if hasattr(backend, 'reshape') else eff_r
        eff_t_flat = backend.reshape(eff_t, (-1,)) if hasattr(backend, 'reshape') else eff_t
        if backend.is_complex(eff_r_flat):
            inner = backend.sum(eff_r_flat.conj() * eff_t_flat)
        else:
            inner = backend.sum(eff_r_flat * eff_t_flat)
        return -backend.abs_square(inner)
