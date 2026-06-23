"""
Backend factory for creating computational backends (JAX, PyTorch, etc.).

This module provides a factory pattern for creating and managing different
computational backends for tensor operations.
"""

from __future__ import annotations
from typing import Optional, Type

from .backend_interface import ComputeBackend
from .backend_jax import BackendJAX
from .backend_pytorch import BackendPyTorch


class BackendFactory:
    """
    Factory class for creating computational backends.
    
    Usage:
        backend = BackendFactory.create_backend('jax')
        backend = BackendFactory.create_backend('pytorch', device='cuda')
    """

    _backends = {
        'jax': BackendJAX,
        'pytorch': BackendPyTorch,
    }

    _default_backend: Optional[str] = None
    _backend_instance: Optional[ComputeBackend] = None

    @classmethod
    def create_backend(cls, backend_name: str, device: Optional[str] = None,
                       tensor_type: Optional[str] = None, **kwargs) -> ComputeBackend:
        """
        Create a backend instance.
        
        Args:
            backend_name (str): Name of the backend ('jax' or 'pytorch').
            device (Optional[str]): Device specification.
            tensor_type (Optional[str]): High-level tensor wrapper type.
                Pass ``"TNTensor"`` so that :meth:`init_random_core` returns
                :class:`TNTensor` and :meth:`get_tensor_type` reports it.
            **kwargs: Additional backend-specific configuration.
        
        Returns:
            ComputeBackend: Backend instance.
        
        Raises:
            ValueError: If backend name is not supported.
        """
        backend_name = backend_name.lower()
        
        if backend_name not in cls._backends:
            raise ValueError(
                f"Unknown backend: {backend_name}. "
                f"Available backends: {list(cls._backends.keys())}"
            )

        return cls._backends[backend_name](device=device, tensor_type=tensor_type, **kwargs)

    @classmethod
    def set_default_backend(cls, backend_name: str, device: Optional[str] = None,
                            tensor_type: Optional[str] = None, **kwargs):
        """
        Set the default backend for the application.
        
        Args:
            backend_name (str): Name of the backend ('jax' or 'pytorch').
            device (Optional[str]): Device specification.
            tensor_type (Optional[str]): High-level tensor wrapper type.
            **kwargs: Additional backend-specific configuration.
        """
        cls._default_backend = backend_name.lower()
        cls._backend_instance = cls.create_backend(backend_name, device=device,
                                                   tensor_type=tensor_type, **kwargs)

    @classmethod
    def get_default_backend(cls) -> ComputeBackend:
        """
        Get the default backend instance.
        
        Returns:
            ComputeBackend: Default backend instance (creates JAX backend if not set).
        """
        if cls._backend_instance is None:
            # device=None lets the JAX backend auto-detect TPU > GPU > CPU.
            cls.set_default_backend('jax', None)
        return cls._backend_instance

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[ComputeBackend]):
        """
        Register a custom backend.
        
        Args:
            name (str): Name for the backend.
            backend_class (Type[ComputeBackend]): Backend class implementing ComputeBackend.
        """
        cls._backends[name.lower()] = backend_class
