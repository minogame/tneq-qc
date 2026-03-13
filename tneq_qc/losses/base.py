"""BaseLoss — abstract base class for all loss functions.

A loss is split into two composable steps:

  1. ``preprocess(result, backend) -> TNTensor``
     Optional transformation of the contraction result *before* the loss value
     is computed (e.g. reshape to a square matrix and take the diagonal, take
     the real part, apply softmax, …).  Default implementation is a no-op.

  2. ``compute(result, target, backend) -> scalar``
     Compute the actual scalar loss given the (pre-processed) result and the
     resolved target.

Calling a ``BaseLoss`` instance directly invokes both steps in order:

    loss_value = loss_obj(result, target, backend)
    # equivalent to:
    #   processed = loss_obj.preprocess(result, backend)
    #   loss_value = loss_obj.compute(processed, target, backend)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseLoss(ABC):
    """Abstract base class for all loss functions in tneq_qc.

    Subclasses must implement :meth:`compute`.  Optionally override
    :meth:`preprocess` to transform the contraction result before the loss
    is applied.
    """

    def preprocess(self, result: Any, backend: Any) -> Any:
        """Transform the contraction result before loss computation.

        Default implementation is a no-op — the result is returned unchanged.

        Args:
            result  : TNTensor returned by the contraction engine.
            backend : Active backend instance (provides tensor ops).

        Returns:
            TNTensor to pass to :meth:`compute`.
        """
        return result

    @abstractmethod
    def compute(self, result: Any, target: Any, backend: Any) -> Any:
        """Compute the scalar loss value.

        Args:
            result  : (Pre-processed) TNTensor with gradients attached.
            target  : Resolved target as a TNTensor (already detached).
            backend : Active backend instance.

        Returns:
            Scalar loss (TNTensor or raw backend tensor).
        """

    def __call__(self, result: Any, target: Any, backend: Any) -> Any:
        """Preprocess then compute the loss.

        Args:
            result  : Raw TNTensor from the contraction engine.
            target  : Resolved target (TNTensor, already detached).
            backend : Active backend instance.

        Returns:
            Scalar loss value.
        """
        processed = self.preprocess(result, backend)
        return self.compute(processed, target, backend)
