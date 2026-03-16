"""LossRegistry — global registry for loss functions.

Supports three ways to register a custom loss:

  1. Class decorator::

       @register_loss('my_loss')
       class MyLoss(BaseLoss):
           ...

  2. Manual registration::

       LossRegistry.register('my_loss', MyLoss)

  3. Inline callable (no registration needed)::

       def my_fn(result, target, backend): ...
       engine.contract_with_compiled_strategy_for_gradient(qctn, loss=my_fn)

Resolving a loss:

    loss_obj = LossRegistry.resolve(loss)
    # loss can be: None | str | BaseLoss instance | callable
"""

from __future__ import annotations
from typing import Callable, Dict, Optional, Type, Union

from .base import BaseLoss


class FunctionalLoss(BaseLoss):
    """Wraps an arbitrary callable as a :class:`BaseLoss`.

    The wrapped function must have the signature::

        fn(result: TNTensor, target: Any, backend) -> scalar

    No preprocessing is applied; ``preprocess`` is a no-op.
    """

    def __init__(self, fn: Callable) -> None:
        self.fn = fn

    def compute(self, result, target, backend):
        return self.fn(result, target, backend)

    def __repr__(self) -> str:
        return f"FunctionalLoss(fn={self.fn})"


class LossRegistry:
    """Global registry mapping loss names to :class:`BaseLoss` classes.

    All built-in losses (defined in ``builtin.py``) are registered
    automatically when this module is first imported.
    """

    _registry: Dict[str, Type[BaseLoss]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        name: str,
        loss_cls: Optional[Type[BaseLoss]] = None,
    ):
        """Register a loss class under *name*.

        Can be used as a plain call or as a class decorator::

            # plain call
            LossRegistry.register('mse', MSELoss)

            # decorator
            @LossRegistry.register('mse')
            class MSELoss(BaseLoss): ...
        """
        if loss_cls is not None:
            if not (isinstance(loss_cls, type) and issubclass(loss_cls, BaseLoss)):
                raise TypeError(
                    f"loss_cls must be a subclass of BaseLoss, got {loss_cls}"
                )
            cls._registry[name] = loss_cls
            return loss_cls

        # Used as a decorator factory: @LossRegistry.register('name')
        def decorator(klass: Type[BaseLoss]) -> Type[BaseLoss]:
            cls.register(name, klass)
            return klass

        return decorator

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, name: str) -> BaseLoss:
        """Return a fresh instance of the loss registered under *name*.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Loss '{name}' is not registered. "
                f"Available losses: {available}"
            )
        return cls._registry[name]()

    @classmethod
    def resolve(cls, loss: Union[None, str, Callable, BaseLoss]) -> BaseLoss:
        """Normalise any loss specification into a :class:`BaseLoss` instance.

        Resolution rules:

        +--------------+----------------------------------------------+
        | Input type   | Result                                       |
        +==============+==============================================+
        | ``None``     | ``DiagonalMSELoss`` (backward-compatible)    |
        +--------------+----------------------------------------------+
        | ``str``      | ``LossRegistry.get(name)``                   |
        +--------------+----------------------------------------------+
        | ``BaseLoss`` | returned as-is                               |
        +--------------+----------------------------------------------+
        | callable     | wrapped in ``FunctionalLoss``                |
        +--------------+----------------------------------------------+

        Raises:
            TypeError: If *loss* cannot be resolved.
        """
        if loss is None:
            return cls.get('diagonal_mse')
        if isinstance(loss, str):
            return cls.get(loss)
        if isinstance(loss, BaseLoss):
            return loss
        if callable(loss):
            return FunctionalLoss(loss)
        raise TypeError(
            f"Cannot resolve loss of type {type(loss)}. "
            "Expected None, str, BaseLoss instance, or callable."
        )

    @classmethod
    def available(cls):
        """Return a sorted list of all registered loss names."""
        return sorted(cls._registry.keys())

    def __repr__(cls) -> str:
        return f"LossRegistry(registered={cls.available()})"


# ------------------------------------------------------------------
# Public decorator alias
# ------------------------------------------------------------------

def register_loss(name: str):
    """Decorator that registers a :class:`BaseLoss` subclass by name.

    Usage::

        @register_loss('my_loss')
        class MyLoss(BaseLoss):
            def compute(self, result, target, backend):
                ...
    """
    return LossRegistry.register(name)
