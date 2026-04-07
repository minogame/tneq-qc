"""
Backend interface for computational backends.

This module defines the abstract base class and information structure
for computational backends.
"""

from __future__ import annotations
from typing import Optional, Any
from abc import ABC, abstractmethod
import numpy as np


class BackendInfo:
    """
    Simple information class for backend configuration.
    
    This class only records backend information, without providing
    tensor allocation or conversion methods. The actual tensor operations
    should be implemented in the caller based on this information.
    """
    
    def __init__(self, backend_type: str, device: Optional[str] = None,
                 dtype: Optional[str] = None, **kwargs):
        """
        Initialize backend information.
        
        Args:
            backend_type (str): Type of backend ('jax', 'pytorch', etc.).
            device (Optional[str]): Device specification (e.g., 'cpu', 'cuda', 'cuda:0', 'gpu').
            **kwargs: Additional backend-specific configuration.
        """
        self.backend_type = backend_type.lower()
        self.device = device
        # 逻辑上的默认 dtype（例如 'float32', 'complex64' 等），由具体 backend 解释
        self.dtype = dtype
        self.config = kwargs
    
    def __repr__(self):
        return (
            f"BackendInfo(backend_type='{self.backend_type}', "
            f"device='{self.device}', dtype='{self.dtype}', config={self.config})"
        )
    
    def __str__(self):
        return self.__repr__()

class ComputeBackend(ABC):
    """
    Abstract base class for computational backends.
    
    All backends must implement these methods for tensor operations,
    gradient computation, and JIT compilation.
    """
    
    def __init__(self, tensor_type: Optional[str] = ""):
        """Initialize backend with BackendInfo.

        Args:
            tensor_type: Optional string indicating the high-level tensor
                wrapper to use.  Currently supported: ``"TNTensor"``.
                When set, :meth:`get_tensor_type` returns the wrapper class
                and :meth:`init_random_core` automatically wraps results.
        """
        self.backend_info: Optional[BackendInfo] = None
        self._tensor_type_name: Optional[str] = tensor_type

    # ------------------------------------------------------------------
    # TNTensor helpers
    # ------------------------------------------------------------------

    @property
    def use_tn_tensor(self) -> bool:
        """Return ``True`` if the backend is configured to use TNTensor."""
        return self._tensor_type_name == "TNTensor"

    def wrap_tensor(self, tensor):
        """Wrap a raw backend tensor in :class:`TNTensor` if configured.

        If the backend's ``tensor_type`` is ``"TNTensor"`` and *tensor* is
        not already a :class:`TNTensor`, it is wrapped.  Otherwise the
        tensor is returned as-is.
        """
        if self.use_tn_tensor:
            from ..core.tn_tensor import TNTensor
            if isinstance(tensor, TNTensor):
                return tensor
            return TNTensor(tensor)
        return tensor

    def unwrap_tensor(self, tensor):
        """Extract the raw backend tensor from a :class:`TNTensor`.

        If *tensor* is a :class:`TNTensor`, returns its underlying
        ``.tensor``; otherwise returns *tensor* unchanged.
        """
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return tensor.tensor
        return tensor

    @abstractmethod
    def execute_expression(self, expression, *tensors):
        """
        Execute a contraction expression with given tensors.
        
        Args:
            expression: Optimized contraction expression (opt_einsum.ContractExpression).
            *tensors: Variable number of input tensors.
        
        Returns:
            Tensor result of the contraction.
        """
        pass

    @abstractmethod
    def compute_value_and_grad(self, loss_fn, argnums):
        """
        Create a function that computes both value and gradient.
        
        Args:
            loss_fn: Loss function to differentiate.
            argnums: Argument indices to compute gradients for.
        
        Returns:
            Function that returns (value, gradients).
        """
        pass

    @abstractmethod
    def jit_compile(self, func):
        """
        JIT compile a function for faster execution.
        
        Args:
            func: Function to compile.
        
        Returns:
            JIT-compiled function.
        """
        pass

    @abstractmethod
    def convert_to_tensor(self, array):
        """
        Convert array-like object to a TNTensor.

        Args:
            array: Array-like object (numpy array, list, TNTensor, raw backend tensor, …).

        Returns:
            TNTensor wrapping the backend tensor.  If *array* is already a
            TNTensor it is returned unchanged.
        """
        pass

    @abstractmethod
    def optimizer_update(self, params, grads, state, method: str, hyperparams: dict):
        """
        Perform a single optimization step.
        
        Args:
            params: Current parameters (Dict[key, Tensor] or List[Tensor])
            grads: Corresponding gradients
            state: Optimizer state (e.g. momentum, Adam's m/v), structure maintained by Backend
            method: Optimization method name ('adam', 'sgdg', etc.)
            hyperparams: Hyperparameters (lr, beta1, beta2, momentum, etc.)
            
        Returns:
            (new_params, new_state): Updated parameters and state
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the name of the backend."""
        pass
    
    def get_backend_info(self) -> BackendInfo:
        """
        Get the BackendInfo instance for this backend.
        
        Returns:
            BackendInfo instance.
        """
        if self.backend_info is None:
            # Create default BackendInfo
            self.backend_info = BackendInfo(self.get_backend_name())
        return self.backend_info
    
    def set_backend_info(self, backend_info: BackendInfo):
        """
        Set the BackendInfo instance for this backend.
        
        Args:
            backend_info (BackendInfo): BackendInfo instance to use.
        """
        if backend_info.backend_type != self.get_backend_name():
            raise ValueError(
                f"BackendInfo type '{backend_info.backend_type}' does not match "
                f"backend '{self.get_backend_name()}'"
            )
        self.backend_info = backend_info

    @abstractmethod
    def init_random_core(self, shape, distribution: str = "gaussian"):
        """
        Initialize a random core tensor (orthogonal initialization).
        
        Args:
            shape: Shape of the tensor.
            distribution: Random matrix distribution used before QR-based
                orthogonalization. Supported values are backend-defined;
                the framework uses ``"gaussian"`` by default and may also
                support ``"uniform"``.
            
        Returns:
            Initialized tensor.
        """
        pass

    def get_tensor_type(self):
        """
        Get the type of tensors used by this backend.

        When ``tensor_type="TNTensor"`` was passed at construction time,
        this returns :class:`TNTensor`; otherwise it delegates to
        :meth:`_get_raw_tensor_type`.

        Returns:
            Type/Class of the tensor.
        """
        if self.use_tn_tensor:
            from ..core.tn_tensor import TNTensor
            return TNTensor
        return self._get_raw_tensor_type()

    @abstractmethod
    def _get_raw_tensor_type(self):
        """
        Get the raw (unwrapped) tensor type for this backend.

        Subclasses must implement this to return the native tensor class
        (e.g. ``torch.Tensor``, ``jnp.ndarray``).

        Returns:
            Type/Class of the raw backend tensor.
        """
        pass

    @abstractmethod
    def tensor_to_numpy(self, tensor) -> np.ndarray:
        """
        Convert a backend tensor to a NumPy array.

        Args:
            tensor: Backend-specific tensor.

        Returns:
            NumPy ndarray with the same data.
        """
        pass

    @abstractmethod
    def set_random_seed(self, seed: int):
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Integer seed value.
        """
        pass

    @abstractmethod
    def reshape(self, tensor, shape):
        """
        Reshape a tensor to the given shape.
        
        Args:
            tensor: Backend-specific tensor.
            shape: New shape (list or tuple of ints).
        
        Returns:
            Reshaped tensor.
        """
        pass

    @abstractmethod
    def eye(self, n: int, dtype=None):
        """
        Create an identity matrix of size n x n.
        
        Args:
            n: Size of the matrix.
            dtype: Data type of the matrix.
            
        Returns:
            Identity matrix tensor.
        """
        pass

    @abstractmethod
    def zeros(self, shape, dtype=None):
        """
        Create a tensor filled with zeros.
        
        Args:
            shape: Shape of the tensor (tuple or list).
            dtype: Data type of the tensor.
            
        Returns:
            Zero-filled tensor.
        """
        pass

    @abstractmethod
    def ones(self, shape, dtype=None):
        """
        Create a tensor filled with ones.
        
        Args:
            shape: Shape of the tensor (tuple or list).
            dtype: Data type of the tensor.
            
        Returns:
            One-filled tensor.
        """
        pass

    @abstractmethod
    def clone(self, tensor):
        """
        Create a copy of the tensor.
        
        Args:
            tensor: Tensor to clone.
            
        Returns:
            Cloned tensor.
        """
        pass

    @abstractmethod
    def unsqueeze(self, tensor, dim):
        """
        Add a dimension of size 1 at the specified position.
        
        Args:
            tensor: Input tensor.
            dim: Position where to add the dimension.
            
        Returns:
            Tensor with added dimension.
        """
        pass

    @abstractmethod
    def expand(self, tensor, *sizes):
        """
        Expand tensor to a larger size by broadcasting.
        
        Args:
            tensor: Input tensor.
            *sizes: Target sizes for each dimension (-1 means no change).
            
        Returns:
            Expanded tensor.
        """
        pass

    @abstractmethod
    def clamp(self, tensor, min=None, max=None):
        """
        Clamp tensor values to a range.
        
        Args:
            tensor: Input tensor.
            min: Minimum value (optional).
            max: Maximum value (optional).
            
        Returns:
            Clamped tensor.
        """
        pass

    @abstractmethod
    def diagonal(self, tensor, dim1=-2, dim2=-1):
        """
        Extract diagonal from a tensor.
        
        Args:
            tensor: Input tensor.
            dim1: First dimension for diagonal.
            dim2: Second dimension for diagonal.
            
        Returns:
            Diagonal elements.
        """
        pass

    @abstractmethod
    def sum(self, tensor, dim=None, keepdim=False):
        """
        Sum tensor elements along specified dimension(s).
        
        Args:
            tensor: Input tensor.
            dim: Dimension(s) to sum over (None means all).
            keepdim: Whether to keep the reduced dimension.
            
        Returns:
            Summed tensor.
        """
        pass

    @abstractmethod
    def multinomial(self, probs, num_samples):
        """
        Sample from multinomial distribution.
        
        Args:
            probs: Probability distribution tensor (last dim is the distribution).
            num_samples: Number of samples to draw.
            
        Returns:
            Sampled indices.
        """
        pass

    @abstractmethod
    def arange(self, *args, dtype=None):
        """
        Create a 1-D tensor with evenly spaced values.
        
        Args:
            *args: start, end, step (like Python range).
            dtype: Data type of the tensor.
            
        Returns:
            1-D tensor with evenly spaced values.
        """
        pass

    @abstractmethod
    def stack(self, tensors, dim=0):
        """
        Stack tensors along a new dimension.
        
        Args:
            tensors: List of tensors to stack.
            dim: Dimension along which to stack.
            
        Returns:
            Stacked tensor.
        """
        pass

    @abstractmethod
    def log(self, tensor):
        """
        Compute natural logarithm element-wise.
        
        Args:
            tensor: Input tensor.
            
        Returns:
            Logarithm of tensor.
        """
        pass

    @abstractmethod
    def mean(self, tensor, dim=None, keepdim=False):
        """
        Compute mean of tensor elements.
        
        Args:
            tensor: Input tensor.
            dim: Dimension(s) to reduce (None means all).
            keepdim: Whether to keep the reduced dimension.
            
        Returns:
            Mean tensor.
        """
        pass

    @abstractmethod
    def squeeze(self, tensor, dim=None):
        """
        Remove dimensions of size 1.
        
        Args:
            tensor: Input tensor.
            dim: Dimension to squeeze (None means all size-1 dims).
            
        Returns:
            Squeezed tensor.
        """
        pass

    @abstractmethod
    def einsum(self, equation: str, *operands):
        """
        Perform Einstein summation convention contraction.

        Args:
            equation (str): The einsum equation string, e.g. ``'ij,jk->ik'``.
            *operands: Input tensors referenced by the equation.

        Returns:
            Tensor result of the einsum contraction.
        """
        pass

    def is_complex(self, tensor) -> bool:
        """Return True if tensor is complex dtype. Default: False (e.g. backends without complex)."""
        return False

    def abs_square(self, tensor):
        """
        Born rule: for complex tensor return |tensor|^2 (real); for real tensor return as-is.
        Default: return tensor (no-op for real-only backends).
        """
        return tensor

    # ------------------------------------------------------------------
    # TNTensor-aware convenience wrappers (Phase-1)
    #
    # These non-abstract methods accept either raw tensors or TNTensors.
    # When a TNTensor is passed, they unwrap it, call the backend op, and
    # re-wrap the result in a new TNTensor (scale is preserved or propagated).
    # ------------------------------------------------------------------

    def tn_einsum(self, equation: str, *operands) -> "Any":
        """Backend einsum that transparently handles TNTensor inputs.

        If *any* operand is a TNTensor, the result is also a TNTensor whose
        scale equals the product of all input scales.  Otherwise behaves
        identically to :meth:`einsum`.
        """
        from ..core.tn_tensor import TNTensor
        has_tn = any(isinstance(op, TNTensor) for op in operands)
        if not has_tn:
            return self.einsum(equation, *operands)

        result_scale = 1.0
        raw_ops = []
        for op in operands:
            if isinstance(op, TNTensor):
                result_scale *= op.scale
                raw_ops.append(op.tensor)
            else:
                raw_ops.append(op)

        import math
        raw_result = self.einsum(equation, *raw_ops)
        return TNTensor(raw_result, scale=result_scale)

    def tn_reshape(self, tensor, shape) -> "Any":
        """Backend reshape that transparently handles TNTensor inputs.

        Returns a TNTensor (with the same scale) when *tensor* is a TNTensor,
        otherwise delegates to :meth:`reshape`.
        """
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return tensor.reshape(shape)
        return self.reshape(tensor, shape)

    def tn_matmul(self, a, b) -> "Any":
        """Matrix multiply that handles TNTensor inputs.

        Scales are multiplied.  Falls back to ``a @ b`` for raw tensors.
        """
        from ..core.tn_tensor import TNTensor
        if isinstance(a, TNTensor) and isinstance(b, TNTensor):
            return a @ b
        if isinstance(a, TNTensor):
            raw_result = self.einsum("...ij,...jk->...ik", a.tensor, b) if False else a.tensor @ b
            return TNTensor(raw_result, scale=a.scale)
        if isinstance(b, TNTensor):
            raw_result = a @ b.tensor
            return TNTensor(raw_result, scale=b.scale)
        return a @ b
