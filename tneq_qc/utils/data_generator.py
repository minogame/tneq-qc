"""
Data generation utilities for measurement matrices (Mx).

Provides:

- :class:`DataGenerator` — Hermite-polynomial basis: φ(x) → Mx = φφ†.
- :class:`CustomDataGenerator` — user-provided generation function.

Both share the same ``.generate(x, K, ret_type)`` interface and work
seamlessly with :func:`make_data_fn` and :meth:`EngineCommon.sample`.

Typical usage::

    from tneq_qc.utils.data_generator import DataGenerator, CustomDataGenerator

    # Hermite basis (default)
    gen = DataGenerator(backend, mx_K=16)
    Mx_list, phi_x = gen.generate(x, K=8)

    # User-defined Mx (e.g. discrete distribution)
    custom_gen = CustomDataGenerator(backend, my_generate_fn, mx_K=4)
    Mx_list, _ = custom_gen.generate(x, K=4)
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional


class DataGenerator:
    """Generate Hermite-polynomial feature maps and measurement matrices.

    Args:
        backend: Compute backend instance (e.g. ``BackendPyTorch``).
        mx_K (int): Maximum polynomial order to pre-compute weights for.
            Weights are lazily extended if a higher ``K`` is requested at
            generation time.
    """

    def __init__(self, backend, mx_K: int = 100):
        self.backend = backend
        self.mx_K = mx_K
        self.mx_weights = self._init_mx_weights(mx_K)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_mx_weights(self, k_max: int):
        """Compute normalisation weights for probabilist Hermite polynomials.

        Weights are computed in NumPy (real domain) and converted to the
        configured backend tensor type.
        """
        k = np.arange(k_max + 1, dtype=np.float64)
        log_factorial = np.array(
            [math.lgamma(int(ki) + 1) for ki in k], dtype=np.float64
        )
        log_factor = -0.5 * (0.5 * math.log(2 * math.pi) + log_factorial)
        weights_np = np.exp(log_factor).astype(np.float64)

        # Cache NumPy weights for the complex-backend path.
        self._mx_weights_np = weights_np

        return self.backend.convert_to_tensor(weights_np)

    # ------------------------------------------------------------------
    # Hermite polynomial evaluation
    # ------------------------------------------------------------------

    def _eval_hermitenorm_batch(self, n_max: int, x):
        """Evaluate probabilist Hermite polynomials H_0 … H_{n_max} on *x*.

        Args:
            n_max: Highest order (inclusive).
            x: Tensor of shape ``[B, D]`` (raw backend tensor or TNTensor).

        Returns:
            Raw backend tensor of shape ``[n_max+1, B, D]``.
        """
        from ..core.tn_tensor import TNTensor as _TNT

        # Always work with the raw underlying tensor to allow index assignment.
        if isinstance(x, _TNT):
            x = x.tensor
        elif not hasattr(x, 'shape'):
            tmp = self.backend.convert_to_tensor(x)
            x = tmp.tensor if isinstance(tmp, _TNT) else tmp

        # Build output using raw tensor ops (backend-agnostic via duck-typing).
        try:
            import torch as _torch
            H = _torch.zeros((n_max + 1,) + tuple(x.shape), dtype=x.dtype, device=x.device)
        except ImportError:
            import numpy as _np
            H = _np.zeros((n_max + 1,) + tuple(x.shape), dtype=np.float64)

        # ones_like via duck-typing
        if hasattr(x, 'new_ones'):
            H[0] = x.new_ones(x.shape)      # torch
        else:
            H[0] = np.ones_like(x)

        if n_max >= 1:
            H[1] = x
            for i in range(2, n_max + 1):
                H[i] = x * H[i - 1] - (i - 1) * H[i - 2]
        return H

    def _eval_hermitenorm_batch_np(self, n_max: int, x_np: np.ndarray) -> np.ndarray:
        """NumPy version of probabilist Hermite polynomial evaluation.

        Args:
            n_max: Highest order (inclusive).
            x_np: Real NumPy array of shape ``[B, D]``.

        Returns:
            Array of shape ``[n_max+1, B, D]``.
        """
        x_np = np.asarray(x_np, dtype=np.float64)
        H = np.zeros((n_max + 1,) + x_np.shape, dtype=np.float64)
        H[0] = 1.0
        if n_max >= 1:
            H[1] = x_np
            for i in range(2, n_max + 1):
                H[i] = x_np * H[i - 1] - (i - 1) * H[i - 2]
        return H

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, x, K: Optional[int] = None, ret_type: str = 'tensor'):
        """Generate feature map φ(x) and measurement matrices Mx = φφ†.

        Args:
            x: Input batch of shape ``[B, D]`` where *D* is the number of
                qubits / features.
            K (int, optional): Number of basis functions.  Defaults to
                ``self.mx_K``.
            ret_type (str): ``'tensor'`` (default) returns plain backend
                tensors; ``'TNTensor'`` wraps each Mx in a
                :class:`~tneq_qc.core.tn_tensor.TNTensor` with auto-scaling.

        Returns:
            tuple: ``(Mx_list, phi_x)`` where

            - ``Mx_list`` is a list of *D* tensors each of shape
              ``[B, K, K]``.
            - ``phi_x`` is the feature map of shape ``[B, D, K]``.
        """
        if K is None:
            K = self.mx_K

        x = self.backend.convert_to_tensor(x)
        num_qubits = x.shape[1]

        # Lazily extend weights if a higher K is requested.
        if K > self.mx_K or K > self.mx_weights.shape[0]:
            self.mx_weights = self._init_mx_weights(K)
            self.mx_K = K

        # Detect complex backend via BackendInfo.dtype.
        backend_info = getattr(self.backend, "backend_info", None)
        backend_dtype = getattr(backend_info, "dtype", None) if backend_info is not None else None
        is_complex_backend = backend_dtype is not None and "complex" in str(backend_dtype)

        # Fallback: check the converted x directly (handles cases where
        # BackendInfo is missing but the backend runs in complex mode).
        if not is_complex_backend:
            from ..core.tn_tensor import TNTensor as _TNT
            _raw_x = x.tensor if isinstance(x, _TNT) else x
            if hasattr(_raw_x, 'is_complex') and _raw_x.is_complex():
                is_complex_backend = True

        if is_complex_backend:
            return self._generate_complex(x, K, num_qubits, ret_type)
        return self._generate_real(x, K, num_qubits, ret_type)

    # ------------------------------------------------------------------
    # Internal generation paths
    # ------------------------------------------------------------------

    def _generate_complex(self, x, K: int, num_qubits: int, ret_type: str):
        """Complex-backend path: compute in real NumPy domain, then convert."""
        from ..core.tn_tensor import TNTensor

        x_np = self.backend.tensor_to_numpy(x)
        x_np = np.asarray(x_np.real, dtype=np.float64)

        if not hasattr(self, "_mx_weights_np") or self._mx_weights_np.shape[0] < self.mx_K + 1:
            self.mx_weights = self._init_mx_weights(self.mx_K)

        weights_np = np.asarray(self._mx_weights_np[:K], dtype=np.float64)[None, None, :]
        H_np = self._eval_hermitenorm_batch_np(K - 1, x_np)
        gaussian_np = np.sqrt(np.exp(-np.square(x_np) / 2.0))[..., None]
        phi_x_np = weights_np * gaussian_np * np.transpose(H_np, (1, 2, 0))
        Mx_np = np.einsum("bdk,bdl->bdkl", phi_x_np, phi_x_np)

        phi_x_tensor = self.backend.convert_to_tensor(phi_x_np)

        Mx_list = []
        for i in range(num_qubits):
            tmp = self.backend.convert_to_tensor(Mx_np[:, i, :, :])
            if ret_type == "TNTensor":
                tmp = TNTensor(tmp, has_batch=True)
            Mx_list.append(tmp)

        return Mx_list, phi_x_tensor

    def _generate_real(self, x, K: int, num_qubits: int, ret_type: str):
        """Real-backend path: computation on raw backend tensors.

        TNTensor inputs are unwrapped at the boundary so that standard
        Python subscript / index-assignment operations work correctly.
        """
        from ..core.tn_tensor import TNTensor

        # --- unwrap TNTensor inputs ---
        mx_w = self.mx_weights.tensor if isinstance(self.mx_weights, TNTensor) else self.mx_weights
        x_raw = x.tensor if isinstance(x, TNTensor) else x

        # weights shape: [1, 1, K]
        weights = mx_w[:K].unsqueeze(0).unsqueeze(0)   # raw tensor ops

        # Hermite polynomials: [K, B, D] (raw)
        H = self._eval_hermitenorm_batch(K - 1, x_raw)

        # Permute to [B, D, K]
        if hasattr(H, 'permute'):
            out = H.permute(1, 2, 0)
        else:
            out = np.transpose(H, (1, 2, 0))

        # Gaussian factor: [B, D, 1]
        gaussian_factor = (-x_raw ** 2 / 2).exp().sqrt().unsqueeze(-1)

        out = weights * gaussian_factor * out          # [B, D, K]

        # Mx = phi * phi†: [B, D, K, K]
        if hasattr(out, 'conj'):
            Mx = self.backend.einsum("bdk,bdl->bdkl", out.conj(), out)
        else:
            Mx = self.backend.einsum("bdk,bdl->bdkl", out, out)

        # unwrap Mx if it came back as TNTensor
        Mx_raw = Mx.tensor if isinstance(Mx, TNTensor) else Mx

        Mx_list = []
        for i in range(num_qubits):
            tmp = Mx_raw[:, i, :, :]          # raw [B, K, K]
            if ret_type == "TNTensor":
                tmp = TNTensor(tmp, has_batch=True)
            Mx_list.append(tmp)

        return Mx_list, out


class CustomDataGenerator:
    """Wrap a user-provided Mx generation function.

    The wrapped function must accept ``(x_tensor, K, backend)`` and return
    ``(Mx_list, phi_x_or_None)`` where:

    - ``Mx_list`` is a list of *D* tensors, each of shape ``[B, K, K]``.
    - ``phi_x`` is an optional feature map (may be ``None``).

    The resulting object exposes the same ``.generate()`` interface as
    :class:`DataGenerator`, so it works with :func:`make_data_fn` and
    :meth:`EngineCommon.sample` without changes.

    Args:
        backend: Compute backend instance.
        generate_fn: ``(x_tensor, K, backend) -> (Mx_list, phi_x | None)``.
        mx_K (int): Default *K* when ``generate(K=None)`` is called.

    Example — discrete class labels::

        import torch

        # Pre-build a look-up table of Mx matrices (one per class).
        num_classes, phys_dim = 10, 4
        mx_table = torch.zeros(num_classes, phys_dim, phys_dim)
        for k in range(num_classes):
            mx_table[k, k % phys_dim, k % phys_dim] = 1.0

        def discrete_generate_fn(x, K, backend):
            x_np = backend.tensor_to_numpy(x).astype(int)
            B, D = x_np.shape
            Mx_list = []
            for d in range(D):
                mx = mx_table[x_np[:, d]]          # [B, K, K]
                Mx_list.append(backend.convert_to_tensor(mx))
            return Mx_list, None

        gen = CustomDataGenerator(backend, discrete_generate_fn, mx_K=phys_dim)
    """

    def __init__(self, backend, generate_fn, mx_K: int = 2):
        self.backend = backend
        self._generate_fn = generate_fn
        self.mx_K = mx_K

    def generate(self, x, K: Optional[int] = None, ret_type: str = 'tensor'):
        """Generate Mx matrices via the user-provided function.

        Args:
            x: Input batch (any type accepted by the backend).
            K (int, optional): Passed through to *generate_fn*.
                Defaults to ``self.mx_K``.
            ret_type (str): ``'tensor'`` or ``'TNTensor'``.

        Returns:
            tuple: ``(Mx_list, phi_x)`` — same contract as
            :meth:`DataGenerator.generate`.
        """
        from ..core.tn_tensor import TNTensor

        if K is None:
            K = self.mx_K

        x = self.backend.convert_to_tensor(x)
        Mx_list, phi_x = self._generate_fn(x, K, self.backend)

        if ret_type == 'TNTensor':
            wrapped = []
            for m in Mx_list:
                wrapped.append(TNTensor(m, has_batch=True))
            Mx_list = wrapped

        return Mx_list, phi_x


# ---------------------------------------------------------------------------
# Factory helper for Trainer.fit(data_fn=...) (Phase 3.0)
# ---------------------------------------------------------------------------

def make_data_fn(
    data_generator,
    qctn,
    mx_core_names=None,
    batch_size: int = 128,
    num_qubits: int = None,
    K: int = 2,
    sample_fn=None,
):
    """Create a *data_fn* callable for :meth:`Trainer.fit`.

    Each call generates a random batch, computes Mx matrices via
    *data_generator*, and writes them into *qctn* in-place.

    Args:
        data_generator: :class:`DataGenerator` or :class:`CustomDataGenerator`.
        qctn: QCTN whose mx cores will be updated.
        mx_core_names: Readable names of the mx cores to update.
            If None, auto-detected by finding cores whose readable
            name starts with ``'mx.'``.
        batch_size: Number of samples per step.
        num_qubits: Data dimensionality (D). If None, uses ``qctn.nqubits``.
        K: Polynomial order (passed through to ``data_generator.generate``).
        sample_fn: Optional callable ``(batch_size, num_qubits) -> x``
            that returns input samples. Defaults to
            ``np.random.uniform(-1, 1, (batch_size, num_qubits))``.

    Returns:
        Callable ``(step: int) -> None`` suitable for ``Trainer.fit(data_fn=...)``.
    """
    if mx_core_names is None:
        names_map = getattr(qctn, 'core_names', {})
        mx_core_names = [
            names_map[sym] for sym in qctn.cores
            if names_map.get(sym, '').startswith('mx.')
        ]

    if num_qubits is None:
        num_qubits = qctn.nqubits

    _default_sample = sample_fn is None

    def data_fn(step: int) -> None:
        if _default_sample:
            x = np.random.uniform(-1.0, 1.0, size=(batch_size, num_qubits)).astype(
                np.float32
            )
        else:
            x = sample_fn(batch_size, num_qubits)
        Mx_list, _ = data_generator.generate(x, K=K, ret_type="TNTensor")
        for i, name in enumerate(mx_core_names):
            qctn[name] = Mx_list[i]

    return data_fn
