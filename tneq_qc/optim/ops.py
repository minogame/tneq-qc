"""Tensor operation adapters for backend-decoupled optimizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TensorOps(ABC):
    """Minimal tensor-ops protocol used by optimizers."""

    @abstractmethod
    def zeros_like(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def sqrt(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def conj(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def abs_square(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def copy_into(self, param: Any, new_value: Any) -> None:
        pass

    @abstractmethod
    def replace(self, param: Any, new_value: Any) -> Any:
        pass


class BackendTensorOps(TensorOps):
    """TensorOps adapter backed by an existing ComputeBackend."""

    def __init__(self, backend: Any):
        self.backend = backend

    def zeros_like(self, tensor: Any) -> Any:
        raw = tensor.tensor if hasattr(tensor, "tensor") else tensor
        lib = getattr(self.backend, "torch", None) or getattr(self.backend, "jnp", None)
        if lib is None:
            raise TypeError("BackendTensorOps requires backend.torch or backend.jnp")
        return lib.zeros_like(raw)

    def sqrt(self, tensor: Any) -> Any:
        if hasattr(self.backend, "sqrt"):
            return self.backend.sqrt(tensor)
        lib = getattr(self.backend, "torch", None) or getattr(self.backend, "jnp", None)
        return lib.sqrt(tensor)

    def conj(self, tensor: Any) -> Any:
        if hasattr(tensor, "conj"):
            return tensor.conj()
        if hasattr(self.backend, "conj"):
            return self.backend.conj(tensor)
        return tensor

    def abs_square(self, tensor: Any) -> Any:
        if hasattr(self.backend, "abs_square"):
            return self.backend.abs_square(tensor)
        return self.conj(tensor) * tensor

    def copy_into(self, param: Any, new_value: Any) -> None:
        raw = param.tensor if hasattr(param, "tensor") else param
        if hasattr(raw, "data") and hasattr(raw.data, "copy_"):
            raw.data.copy_(new_value)
            return
        raise TypeError("copy_into is only supported for mutable tensor backends")

    def replace(self, param: Any, new_value: Any) -> Any:
        if hasattr(param, "set"):
            param.set(new_value, scale=1.0)
            return param
        return new_value
