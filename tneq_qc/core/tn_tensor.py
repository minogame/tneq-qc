import math
from typing import Any, Optional


class TNTensor:
    """
    Tensor Network Tensor: wraps a backend tensor with a scale factor.

    Represents the value ``tensor * scale`` to avoid numeric underflow/overflow
    during tensor-network contractions.

    Phase-1 additions
    -----------------
    * ``device`` property (delegated to the underlying tensor).
    * Reference / transpose markers (``is_ref``, ``is_transposed``, ``source``)
      for zero-copy sharing inside siamese / conjugate-transpose networks.
    * Tensor-operation methods: ``reshape``, ``transpose``, ``conj``,
      ``conj_transpose``, ``clone``, ``to``.
    * Arithmetic with correct scale propagation: ``__matmul__``, ``__mul__``,
      ``__rmul__``, ``__truediv__``, ``__add__``, ``__neg__``, ``sum``, ``mean``.
    * ``einsum`` (backend-aware).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        tensor: Any,
        scale: float = 1.0,
        log_scale: Optional[float] = None,
        *,
        has_batch: bool = False,
        is_ref: bool = False,
        is_transposed: bool = False,
        source: Optional["TNTensor"] = None,
        name: Optional[str] = None,
        _source_info: Optional[dict] = None,
    ):
        """
        Args:
            tensor:        Underlying backend tensor (torch.Tensor, jnp.ndarray, …).
            scale:         Multiplicative scale factor (float).
            log_scale:     Pre-computed ``log(|scale|)``; computed if None.
            has_batch:     True when the first dimension is a batch dimension.
                           The remaining dimensions must match the edge ranks in
                           the QCTN adjacency table.  Used by ``build_graph`` to
                           inject the 'a' batch symbol into einsum expressions.
            is_ref:        True when this tensor is a view/reference of ``source``.
            is_transposed: True when this tensor is the (conjugate-)transpose of ``source``.
            source:        Original TNTensor from which this one was derived.
            name:          Optional name for debugging and visualization.
            _source_info:  Optional metadata about the tensor's origin (for debugging).
        """
        self._tensor = tensor
        self.scale = float(scale)
        self.log_scale = (
            log_scale
            if log_scale is not None
            else (math.log(abs(self.scale)) if self.scale != 0 else float("-inf"))
        )
        self.has_batch = has_batch
        self.is_ref = is_ref
        self.is_transposed = is_transposed
        self.source = source
        self.name = name
        self._source_info = _source_info

    # ------------------------------------------------------------------
    # Core mutator
    # ------------------------------------------------------------------

    def set(self, tensor: Any, scale: float = 1.0, has_batch: Optional[bool] = None):
        """Replace the underlying tensor and scale in-place.

        Args:
            tensor:    New underlying tensor.
            scale:     New scale factor.
            has_batch: If not None, overwrite the ``has_batch`` flag.
                       Passing ``True`` marks the first dimension as batch.
        """
        self._tensor = tensor
        self.scale = float(scale)
        self.log_scale = (
            math.log(abs(self.scale)) if self.scale != 0 else float("-inf")
        )
        if has_batch is not None:
            self.has_batch = has_batch

    # ------------------------------------------------------------------
    # Properties – tensor metadata
    # ------------------------------------------------------------------

    @property
    def tensor(self) -> Any:
        """The underlying raw backend tensor."""
        return self._tensor

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying tensor."""
        return self._tensor.ndim

    @property
    def shape(self) -> tuple:
        """Shape of the underlying tensor."""
        return self._tensor.shape

    @property
    def dtype(self) -> Any:
        """Data type of the underlying tensor (backend-native dtype object)."""
        return self._tensor.dtype

    @property
    def device(self) -> Any:
        """Device of the underlying tensor.

        Returns ``tensor.device`` for PyTorch tensors; for JAX arrays it
        returns ``tensor.devices()`` (a frozenset).  Falls back to ``None``
        when the backend does not expose a device attribute.
        """
        if hasattr(self._tensor, "device"):
            return self._tensor.device
        if hasattr(self._tensor, "devices"):
            return self._tensor.devices()
        return None

    # ------------------------------------------------------------------
    # Scale helpers
    # ------------------------------------------------------------------

    def _update_scale(self, new_scale: float):
        self.scale = new_scale
        self.log_scale = (
            math.log(abs(new_scale)) if new_scale != 0 else float("-inf")
        )

    def auto_scale(self):
        """Normalise so ``max(|tensor|) == 1``; absorb the factor into ``scale``."""
        max_val = self._tensor.abs().max()
        max_val_float = max_val.item() if hasattr(max_val, "item") else float(max_val)
        if max_val_float == 0:
            return
        self._tensor = self._tensor / max_val_float
        self.scale *= max_val_float
        self.log_scale += math.log(abs(max_val_float))

    def scale_to(self, new_scale: float):
        """Adjust scale to ``new_scale`` while keeping ``tensor * scale`` constant."""
        new_scale = float(new_scale)
        if new_scale == 0:
            raise ValueError("Cannot scale to 0.")
        factor = self.scale / new_scale
        self._tensor = self._tensor * factor
        self._update_scale(new_scale)

    def scale_with(self, factor: float):
        """Multiply scale by *factor*, divide tensor by *factor*; value unchanged."""
        factor = float(factor)
        if factor == 0:
            raise ValueError("Cannot scale with factor 0.")
        self._tensor = self._tensor / factor
        self._update_scale(self.scale * factor)

    # ------------------------------------------------------------------
    # Shape / layout operations – return new TNTensor (scale preserved)
    # ------------------------------------------------------------------

    def reshape(self, shape) -> "TNTensor":
        """Return a reshaped view; scale is unchanged."""
        return TNTensor(
            self._tensor.reshape(shape),
            scale=self.scale,
            log_scale=self.log_scale,
            is_ref=True,
            source=self,
        )

    def transpose(self, *dims) -> "TNTensor":
        """Return a (permuted) transpose view; scale is unchanged.

        * No *dims*: reverses all axes (equivalent to ``.T``).
        * With *dims*: passes them to ``tensor.permute(*dims)`` (PyTorch) or
          ``jnp.transpose(tensor, dims)`` (JAX).

        The returned TNTensor is marked ``is_ref=True, is_transposed=True``.
        """
        raw = self._tensor
        if dims:
            if hasattr(raw, "permute"):
                raw_t = raw.permute(*dims)
            else:
                try:
                    import jax.numpy as jnp
                    raw_t = jnp.transpose(raw, dims)
                except ImportError:
                    import numpy as np
                    raw_t = np.transpose(raw, dims)
        else:
            if hasattr(raw, "T"):
                raw_t = raw.T
            elif hasattr(raw, "permute"):
                raw_t = raw.permute(*reversed(range(raw.ndim)))
            else:
                raw_t = raw.transpose()

        return TNTensor(
            raw_t,
            scale=self.scale,
            log_scale=self.log_scale,
            is_ref=True,
            is_transposed=True,
            source=self,
        )

    def conj(self) -> "TNTensor":
        """Return element-wise complex conjugate; scale is unchanged."""
        raw = self._tensor
        if hasattr(raw, "conj"):
            raw_c = raw.conj()
        elif hasattr(raw, "conjugate"):
            raw_c = raw.conjugate()
        else:
            raw_c = raw  # real tensor – no-op
        return TNTensor(
            raw_c,
            scale=self.scale,
            log_scale=self.log_scale,
            is_ref=True,
            source=self,
        )

    def conj_transpose(self, *dims) -> "TNTensor":
        """Conjugate then transpose (dagger). Marks ``is_transposed=True``."""
        return self.conj().transpose(*dims)

    def clone(self, deep: bool = True) -> "TNTensor":
        """Return a cloned TNTensor.

        Args:
            deep: If True, creates a deep copy of the data while maintaining
                  the computation graph for gradient flow. If False, detaches
                  from the computation graph before cloning.

        Returns:
            New TNTensor with cloned data.
        """
        raw = self._tensor
        if hasattr(raw, "clone"):
            # PyTorch: clone() maintains computation graph
            if deep:
                raw_copy = raw.clone()
            else:
                raw_copy = raw.detach().clone()
        elif hasattr(raw, "copy"):
            raw_copy = raw.copy()
        else:
            import numpy as np
            raw_copy = np.array(raw)

        return TNTensor(
            raw_copy,
            scale=self.scale,
            log_scale=self.log_scale,
            name=f"{self.name}_clone" if self.name else None,
            _source_info={'type': 'clone', 'base': self.name, 'deep': deep}
        )

    def hermit(self, axes=None) -> "TNTensor":
        """Return Hermitian conjugate (conjugate transpose) of this TNTensor.

        For quantum circuit tensors, this applies:
        1. Complex conjugate
        2. Transpose (permute) specified axes

        Args:
            axes: Permutation order for transpose. If None, defaults to
                  transposing the last two dimensions (typically physical dims).

        Returns:
            New TNTensor with Hermitian conjugate data. PyTorch's autograd
            automatically handles gradient transformation during backprop.

        Example:
            For a tensor with shape (bond_left, phys_in, phys_out, bond_right),
            hermit() with default axes will swap phys_in and phys_out.
        """
        if axes is None:
            # Default: transpose last two dimensions (usually physical dims)
            axes = list(range(self.ndim))
            if len(axes) >= 2:
                axes[-2], axes[-1] = axes[-1], axes[-2]

        # Apply conjugate and permute
        raw = self._tensor
        if hasattr(raw, "conj"):
            hermit_data = raw.conj()
        elif hasattr(raw, "conjugate"):
            hermit_data = raw.conjugate()
        else:
            hermit_data = raw  # Real tensor - conjugate is no-op

        # Apply permutation
        if hasattr(hermit_data, "permute"):
            hermit_data = hermit_data.permute(*axes)
        else:
            try:
                import jax.numpy as jnp
                hermit_data = jnp.transpose(hermit_data, axes)
            except ImportError:
                import numpy as np
                hermit_data = np.transpose(hermit_data, axes)

        return TNTensor(
            hermit_data,
            scale=self.scale,
            log_scale=self.log_scale,
            is_ref=True,
            is_transposed=True,
            source=self,
            name=f"{self.name}_H" if self.name else None,
            _source_info={'type': 'hermit', 'base': self.name, 'axes': axes}
        )

    def to(self, device=None, dtype=None) -> "TNTensor":
        """Move / cast the underlying tensor; returns a new TNTensor.

        For PyTorch tensors calls ``tensor.to(...)``.
        For JAX uses ``jax.device_put`` / ``.astype``.
        """
        raw = self._tensor
        if hasattr(raw, "to"):
            # PyTorch
            kwargs = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            raw_new = raw.to(**kwargs)
        else:
            try:
                import jax
                raw_new = jax.device_put(raw, device) if device is not None else raw
                if dtype is not None:
                    raw_new = raw_new.astype(dtype)
            except ImportError:
                raw_new = raw
        return TNTensor(raw_new, scale=self.scale, log_scale=self.log_scale)

    # ------------------------------------------------------------------
    # Arithmetic – scale propagation rules
    #
    #   (a * sa) @ (b * sb)  =  (a @ b) * (sa * sb)
    #   (a * sa) * k         =  a * (sa * k)           [scalar k]
    #   sum(a * sa)          =  sum(a) * sa
    #   mean(a * sa)         =  mean(a) * sa
    # ------------------------------------------------------------------

    def __matmul__(self, other: "TNTensor") -> "TNTensor":
        """Matrix multiply two TNTensors; scales are multiplied."""
        if not isinstance(other, TNTensor):
            raise TypeError(f"Expected TNTensor, got {type(other)}")
        raw_a, raw_b = self._tensor, other._tensor
        if hasattr(raw_a, "matmul"):
            result = raw_a.matmul(raw_b)
        else:
            import numpy as np
            result = np.matmul(raw_a, raw_b)
        return TNTensor(result, scale=self.scale * other.scale)

    def __mul__(self, other) -> "TNTensor":
        """Element-wise multiply by scalar, raw tensor, or another TNTensor."""
        if isinstance(other, TNTensor):
            raw = self._tensor * other._tensor
            return TNTensor(raw, scale=self.scale * other.scale)
        # Raw backend tensor (torch.Tensor, jnp.ndarray, …)
        if hasattr(other, 'shape'):
            return TNTensor(self._tensor * other, scale=self.scale)
        # Python scalar
        return TNTensor(self._tensor, scale=self.scale * float(other))

    def __rmul__(self, other) -> "TNTensor":
        return self.__mul__(other)

    def __truediv__(self, other) -> "TNTensor":
        """Element-wise divide by scalar, raw tensor, or another TNTensor."""
        if isinstance(other, TNTensor):
            raw = self._tensor / other._tensor
            return TNTensor(raw, scale=self.scale / other.scale)
        # Raw backend tensor
        if hasattr(other, 'shape'):
            return TNTensor(self._tensor / other, scale=self.scale)
        # Python scalar
        factor = float(other)
        if factor == 0:
            raise ZeroDivisionError
        return TNTensor(self._tensor, scale=self.scale / factor)

    def __add__(self, other: "TNTensor") -> "TNTensor":
        """Element-wise add two TNTensors.

        Both are normalised to ``self.scale`` to keep the result accurate.
        """
        if not isinstance(other, TNTensor):
            raise TypeError(f"Expected TNTensor, got {type(other)}")
        if self.scale == other.scale:
            return TNTensor(self._tensor + other._tensor, scale=self.scale)
        factor = other.scale / self.scale
        return TNTensor(self._tensor + other._tensor * factor, scale=self.scale)

    def __neg__(self) -> "TNTensor":
        return TNTensor(self._tensor, scale=-self.scale)

    def sum(self, dim=None, keepdim: bool = False) -> "TNTensor":
        """Sum elements along *dim*; scale is preserved."""
        raw = self._tensor
        try:
            result = raw.sum(dim=dim, keepdim=keepdim) if dim is not None else raw.sum()
        except TypeError:
            # JAX uses axis / keepdims
            result = (
                raw.sum(axis=dim, keepdims=keepdim) if dim is not None else raw.sum()
            )
        return TNTensor(result, scale=self.scale, log_scale=self.log_scale)

    def mean(self, dim=None, keepdim: bool = False) -> "TNTensor":
        """Mean elements along *dim*; scale is preserved."""
        raw = self._tensor
        try:
            result = raw.mean(dim=dim, keepdim=keepdim) if dim is not None else raw.mean()
        except TypeError:
            result = (
                raw.mean(axis=dim, keepdims=keepdim) if dim is not None else raw.mean()
            )
        return TNTensor(result, scale=self.scale, log_scale=self.log_scale)

    def einsum(self, equation: str, *others: "TNTensor", backend=None) -> "TNTensor":
        """Einstein summation over one or more TNTensors.

        Scale rule: result scale = product of all input scales.
        If ``backend`` is provided it is used; otherwise falls back to
        ``torch.einsum`` / ``jnp.einsum``.
        """
        all_tns = (self,) + others
        raw_tensors = [t._tensor for t in all_tns]
        result_scale = math.prod(t.scale for t in all_tns)

        if backend is not None:
            raw_result = backend.einsum(equation, *raw_tensors)
        else:
            raw = self._tensor
            if hasattr(raw, "is_cuda") or hasattr(raw, "is_leaf"):
                import torch
                raw_result = torch.einsum(equation, *raw_tensors)
            else:
                try:
                    import jax.numpy as jnp
                    raw_result = jnp.einsum(equation, *raw_tensors)
                except ImportError:
                    import numpy as np
                    raw_result = np.einsum(equation, *raw_tensors)

        return TNTensor(raw_result, scale=result_scale)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        shape = getattr(self._tensor, "shape", "unknown")
        flags = []
        if self.is_ref:
            flags.append("ref")
        if self.is_transposed:
            flags.append("T")
        flag_str = f", flags=[{','.join(flags)}]" if flags else ""
        return f"TNTensor(shape={shape}, scale={self.scale:.4g}{flag_str})"
