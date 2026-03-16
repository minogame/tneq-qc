"""TargetResolver — normalise any target specification into a TNTensor.

The resolved target is always detached from the computational graph so that
gradients flow only through the contraction *result*, not through the target.

Supported target types
----------------------
``None``
    Creates a constant tensor of shape ``result_shape`` filled with
    ``1 / side`` where ``side = isqrt(n_elements)``.  This reproduces the
    backward-compatible default behaviour used by ``DiagonalMSELoss``.

``float | int``
    Broadcast scalar: creates a constant tensor filled with this value,
    same shape as *result_shape*.

``list``
    Converted via ``backend.convert_to_tensor`` and detached.
    Shape must be compatible with the loss function's expectation.

``TNTensor``
    Detached in-place and returned directly.

``QCTN``  (duck-typed via ``hasattr(target, 'cores')``)
    Contracted once (without gradients) using *engine*, then detached.
    The engine instance must be provided.
"""

from __future__ import annotations
import math
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.tn_tensor import TNTensor


class TargetResolver:
    """Stateless helper: resolve a raw *target* into a detached TNTensor."""

    @staticmethod
    def resolve(
        target: Any,
        result_shape: tuple,
        backend: Any,
        engine: Optional[Any] = None,
    ) -> "TNTensor":
        """Resolve *target* into a detached TNTensor.

        Args:
            target       : Raw target specification (see module docstring).
            result_shape : Shape of the contraction result (tuple of ints).
                           Used to broadcast scalars and compute ``1/side``.
            backend      : Active backend instance (provides tensor ops).
            engine       : :class:`~tneq_qc.core.engine_common.EngineCommon`
                           instance — required only when *target* is a QCTN.

        Returns:
            TNTensor (detached) representing the resolved target.

        Raises:
            ValueError : When *target* is a QCTN but *engine* is not provided.
            TypeError  : For unsupported *target* types.
        """
        from ..core.tn_tensor import TNTensor as _TNTensor

        # ------------------------------------------------------------------
        # None → default constant (1/side per element)
        # ------------------------------------------------------------------
        if target is None:
            n_elem = 1
            for d in result_shape:
                n_elem *= d
            side = int(math.isqrt(n_elem)) if n_elem > 1 else 1
            default_val = 1.0 / side if side > 0 else 1.0
            raw = backend.ones(result_shape) * default_val
            return backend.detach(raw)

        # ------------------------------------------------------------------
        # Scalar (float / int)
        # ------------------------------------------------------------------
        if isinstance(target, (int, float)):
            raw = backend.ones(result_shape) * float(target)
            return backend.detach(raw)

        # ------------------------------------------------------------------
        # Python list → tensor
        # ------------------------------------------------------------------
        if isinstance(target, list):
            raw = backend.convert_to_tensor(target)
            return backend.detach(raw)

        # ------------------------------------------------------------------
        # TNTensor — detach and return
        # ------------------------------------------------------------------
        if isinstance(target, _TNTensor):
            return backend.detach(target)

        # ------------------------------------------------------------------
        # QCTN — contract (no grad) then detach
        # ------------------------------------------------------------------
        if hasattr(target, 'cores'):
            if engine is None:
                raise ValueError(
                    "engine must be provided when target is a QCTN. "
                    "Pass the EngineCommon instance via engine=."
                )
            contracted = engine.contract_with_compiled_strategy(target)
            return backend.detach(contracted)

        # ------------------------------------------------------------------
        # Fallback
        # ------------------------------------------------------------------
        raise TypeError(
            f"Unsupported target type: {type(target)}. "
            "Expected None, float, int, list, TNTensor, or QCTN."
        )
