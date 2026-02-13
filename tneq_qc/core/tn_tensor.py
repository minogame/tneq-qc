import math
from typing import Any

class TNTensor:
    """
    Tensor Network Tensor class that wraps a backend tensor and a scale factor.
    
    This class allows for high-precision scaling of tensors to avoid underflow/overflow
    issues during tensor network contractions.
    """
    
    def __init__(self, tensor: Any, scale: Any = 1.0, log_scale: float = None):
        """
        Initialize TNTensor.
        
        Args:
            tensor: The backend tensor (PyTorch tensor, JAX array, or NumPy array).
            scale: The scaling factor (float or tensor-like).
            log_scale: The logarithm of the absolute value of the scale (float).
        """
        self._tensor = tensor
        
        # Initialize scale as a tensor compatible with self._tensor if possible
        # if hasattr(self._tensor, 'new_tensor'): # Likely PyTorch
        #      if not hasattr(scale, 'shape') or len(scale.shape) == 0:
        #          # Check if scale is already a tensor
        #          if hasattr(scale, 'to') and hasattr(scale, 'device'):
        #              self.scale = scale.to(self._tensor.device)
        #          else:
        #              self.scale = self._tensor.new_tensor(scale)
        #      else:
        #          self.scale = self._tensor.new_tensor(scale)
        # else:
        #      self.scale = scale
        # import torch
        # self.scale = self.scale.to(torch.float64)

        # import numpy as np
        # self.scale = np.float64(scale)
        self.scale = float(scale)

        if log_scale is not None:
            self.log_scale = log_scale
        else:
            self.log_scale = math.log(abs(self.scale)) if self.scale != 0 else float('-inf')

    def set(self, tensor: Any, scale: Any = 1.0):
        """
        Set the tensor and scale to new values.
        
        Args:
            tensor: The new backend tensor.
            scale: The new scaling factor.
        """
        self._tensor = tensor
        self.scale = float(scale)
        self.log_scale = math.log(abs(self.scale)) if self.scale != 0 else float('-inf')

    @property
    def tensor(self) -> Any:
        """Get the underlying backend tensor."""
        return self._tensor

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the underlying tensor."""
        return self._tensor.ndim

    @property
    def shape(self) -> tuple:
        """Get the shape of the underlying tensor."""
        return self._tensor.shape

    @property
    def dtype(self) -> Any:
        """Get the dtype of the underlying tensor."""
        return self._tensor.dtype

    def auto_scale(self):
        """
        Automatically scale the tensor so that its absolute max value is 1.
        Updates self.scale accordingly.
        """
        max_val = self._tensor.abs().max()
        
        if hasattr(max_val, 'item'):
            max_val_float = max_val.item()
        else:
            max_val_float = float(max_val)
            
        if max_val_float == 0:
            return

        self._tensor /= max_val_float

        self.scale *= max_val_float
        self.log_scale += math.log(abs(max_val_float))

    def scale_to(self, new_scale: float):
        """
        Scale the tensor to a new scale value.
        The actual represented value (tensor * scale) remains unchanged.
        
        Args:
            new_scale (float): The new scale value.
        """
        new_scale = float(new_scale)
        if new_scale == 0:
             raise ValueError("Cannot scale to 0.")
             
        factor = self.scale / new_scale

        self._tensor = self._tensor * factor

        self.scale = new_scale
        self.log_scale = math.log(abs(self.scale))

    def scale_with(self, factor: float):
        """
        Multiply the scale by a factor, and divide the tensor by the same factor.
        The actual represented value remains unchanged.
        
        Args:
            factor (float): The factor to scale with.
        """
        factor = float(factor)
        if factor == 0:
            raise ValueError("Cannot scale with factor 0.")

        self._tensor = self._tensor / factor

        self.scale *= factor
        self.log_scale += math.log(abs(factor))

    def __repr__(self):
        shape = getattr(self._tensor, 'shape', 'unknown')
        return f"TNTensor(shape={shape}, scale={self.scale})"
