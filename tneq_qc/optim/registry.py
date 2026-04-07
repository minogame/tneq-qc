"""Registry for built-in and user-defined optimizers."""

from __future__ import annotations

from typing import Dict, Type


_OPTIMIZER_REGISTRY: Dict[str, type] = {}


def register_optimizer(name: str, optimizer_cls: Type) -> None:
    """Register an optimizer class under a public name."""
    key = str(name).lower()
    _OPTIMIZER_REGISTRY[key] = optimizer_cls


def get_registered_optimizers() -> Dict[str, type]:
    """Return the current optimizer registry."""
    return _OPTIMIZER_REGISTRY.copy()


def create_optimizer(name: str, params, *, backend=None, ops=None, **kwargs):
    """Instantiate a registered optimizer."""
    key = str(name).lower()
    if key not in _OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. Registered: {sorted(_OPTIMIZER_REGISTRY.keys())}"
        )
    optimizer_cls = _OPTIMIZER_REGISTRY[key]
    return optimizer_cls(params, backend=backend, ops=ops, **kwargs)
