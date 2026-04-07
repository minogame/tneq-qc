"""
Backend JAX implementation.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from .backend_interface import ComputeBackend, BackendInfo


class BackendJAX(ComputeBackend):
    """JAX computational backend."""

    def __init__(self, device: Optional[str] = None, dtype: Optional[Any] = None,
                 tensor_type: Optional[str] = None):
        """
        Initialize backend JAX.
        
        Args:
            device (Optional[str]): Device specification ('cpu', 'gpu', etc.).
                If None, automatically detects available devices.
            dtype (Optional[Any]): Default dtype for tensors. Can be a jnp.dtype
                or a string like 'float32', 'float64', 'complex64', 'complex128', 'complex'.
            tensor_type (Optional[str]): High-level tensor wrapper type.
                Pass ``"TNTensor"`` to have :meth:`init_random_core` return
                :class:`TNTensor` instances and :meth:`get_tensor_type` report
                ``TNTensor``.
        """
        super().__init__(tensor_type=tensor_type)
        try:
            import jax
            import jax.numpy as jnp
            self.jax = jax
            self.jnp = jnp

            # Auto-detect device if not specified
            if device is None:
                device = 'gpu' if jax.devices('gpu') else 'cpu'

            # Resolve and store default dtype
            self.default_dtype = self._resolve_default_dtype(dtype)
            
            # Create BackendInfo
            self.backend_info = BackendInfo(
                'jax',
                device=device,
                dtype=self._dtype_to_string(self.default_dtype),
            )
            
            # Initialize PRNG key
            self.key = self.jax.random.PRNGKey(0)
        except ImportError:
            raise ImportError("JAX is not installed. Please install it with: pip install jax jaxlib")

    def _resolve_default_dtype(self, dtype: Optional[Any]):
        """
        将用户提供的 dtype 解析为 jnp.dtype。

        支持：
        - None: 使用 jnp.float32
        - 字符串: 'float32', 'float64', 'complex64', 'complex128', 'complex'
        - 直接传入 jnp.dtype
        """
        jnp = self.jnp
        if dtype is None:
            return jnp.float32

        if isinstance(dtype, str):
            mapping = {
                'float32': jnp.float32,
                'float64': jnp.float64,
                'complex64': jnp.complex64,
                'complex128': jnp.complex128,
                # 允许简写 'complex'，默认 complex64
                'complex': jnp.complex64,
            }
            if dtype not in mapping:
                raise ValueError(
                    f"Unsupported dtype string '{dtype}' for BackendJAX. "
                    f"Supported: {list(mapping.keys())}"
                )
            return mapping[dtype]

        # 假设已经是 jnp.dtype
        return dtype

    def _dtype_to_string(self, dtype: Any) -> str:
        """将 jnp.dtype 转成逻辑上的字符串表示，便于在 BackendInfo 中记录。"""
        jnp = self.jnp
        mapping = {
            jnp.float32: 'float32',
            jnp.float64: 'float64',
            getattr(jnp, 'complex64', None): 'complex64',
            getattr(jnp, 'complex128', None): 'complex128',
        }
        for k, v in mapping.items():
            if k is not None and dtype == k:
                return v
        return str(dtype)

    def execute_expression(self, expression, *tensors):
        """Execute contraction expression using JAX."""
        return expression(*tensors)

    def compute_value_and_grad(self, loss_fn, argnums):
        """Compute value and gradient using JAX autodiff."""
        return self.jax.value_and_grad(loss_fn, argnums=argnums)

    def jit_compile(self, func):
        """JIT compile function using JAX."""
        return self.jax.jit(func)

    def convert_to_tensor(self, array):
        """Convert to TNTensor wrapping a JAX array."""
        from ..core.tn_tensor import TNTensor
        if isinstance(array, TNTensor):
            return array

        if not isinstance(array, self.jnp.ndarray):
            array = self.jnp.array(array, dtype=self.default_dtype)

        if self.backend_info.device and 'gpu' in self.backend_info.device.lower():
            devices = self.jax.devices('gpu')
            if devices:
                array = self.jax.device_put(array, devices[0])

        return TNTensor(array)

    def get_backend_name(self) -> str:
        return "jax"

    def optimizer_update(self, params: List[Any], grads: List[Any], state: Dict[str, Any], 
                        method: str, hyperparams: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Perform a single optimization step using JAX.
        """
        if method == 'adam':
            return self._adam_step(params, grads, state, hyperparams)
        elif method == 'sgd':
            return self._sgd_step(params, grads, state, hyperparams)
        else:
            raise NotImplementedError(f"Optimization method {method} not implemented for JAX backend yet.")

    def _adam_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        beta1 = hp.get('beta1', 0.9)
        beta2 = hp.get('beta2', 0.999)
        epsilon = hp.get('epsilon', 1e-8)
        iteration = hp.get('iter', 0)

        if 'm' not in state:
            state['m'] = [self.jnp.zeros_like(p) for p in params]
            state['v'] = [self.jnp.zeros_like(p) for p in params]

        new_params = []
        new_m = []
        new_v = []
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            m = state['m'][i]
            v = state['v'][i]
            
            # Update biased first moment estimate
            m_new = beta1 * m + (1 - beta1) * grad
            
            # Update biased second moment estimate
            v_new = beta2 * v + (1 - beta2) * (grad ** 2)
            
            # Compute bias-corrected first and second moment estimates
            m_hat = m_new / (1 - beta1 ** (iteration + 1))
            v_hat = v_new / (1 - beta2 ** (iteration + 1))
            
            sqrt_v_hat = self.jnp.sqrt(v_hat)
            
            # Update parameters
            update = lr * m_hat / (sqrt_v_hat + epsilon)
            new_params.append(param - update)
            
            new_m.append(m_new)
            new_v.append(v_new)
            
        new_state = state.copy()
        new_state['m'] = new_m
        new_state['v'] = new_v
            
        return new_params, new_state

    def _sgd_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        new_params = []
        for param, grad in zip(params, grads):
            new_params.append(param - lr * grad)
        return new_params, state

    def init_random_core(self, shape, distribution: str = "gaussian"):
        from ..core.tn_tensor import TNTensor
        dist = distribution.lower()
        flat_dim = int(np.prod(shape[:len(shape)//2]))

        self.key, subkey = self.jax.random.split(self.key)
        if dist in {"gaussian", "normal"}:
            random_matrix = self.jax.random.normal(subkey, (flat_dim, flat_dim))
        elif dist == "uniform":
            random_matrix = self.jax.random.uniform(
                subkey, (flat_dim, flat_dim), minval=-1.0, maxval=1.0
            )
        else:
            raise ValueError(
                f"Unsupported init distribution '{distribution}'. "
                "Supported values: 'gaussian', 'uniform'."
            )

        random_matrix = random_matrix.astype(self.default_dtype)
        Q, R = self.jnp.linalg.qr(random_matrix)
        d = self.jnp.diag(R)
        sign_correction = self.jnp.sign(d)
        Q = Q * sign_correction[None, :]
        return TNTensor(Q.reshape(shape))

    def _get_raw_tensor_type(self):
        return self.jnp.ndarray

    def set_random_seed(self, seed: int):
        self.key = self.jax.random.PRNGKey(seed)
        np.random.seed(seed)

    def tensor_to_numpy(self, tensor):
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            tensor = tensor.tensor
        return np.asarray(tensor)

    def reshape(self, tensor, shape):
        """Reshape tensor to the given shape."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return tensor.reshape(shape)
        return self.jnp.reshape(tensor, shape)

    def _gpu_put(self, tensor):
        """Move tensor to GPU if configured."""
        if self.backend_info.device and 'gpu' in self.backend_info.device.lower():
            devices = self.jax.devices('gpu')
            if devices:
                return self.jax.device_put(tensor, devices[0])
        return tensor

    def eye(self, n: int, dtype=None):
        """Create an identity matrix of size n x n."""
        from ..core.tn_tensor import TNTensor
        if dtype is None:
            dtype = self.default_dtype
        return TNTensor(self._gpu_put(self.jnp.eye(n, dtype=dtype)))

    def zeros(self, shape, dtype=None):
        """Create a tensor filled with zeros."""
        from ..core.tn_tensor import TNTensor
        if dtype is None:
            dtype = self.default_dtype
        return TNTensor(self._gpu_put(self.jnp.zeros(shape, dtype=dtype)))

    def ones(self, shape, dtype=None):
        """Create a tensor filled with ones."""
        from ..core.tn_tensor import TNTensor
        if dtype is None:
            dtype = self.default_dtype
        return TNTensor(self._gpu_put(self.jnp.ones(shape, dtype=dtype)))

    def clone(self, tensor):
        """Create a copy of the tensor."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.jnp.copy(tensor.tensor), scale=tensor.scale)
        return self.jnp.copy(tensor)

    def unsqueeze(self, tensor, dim):
        """Add a dimension of size 1 at the specified position."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.jnp.expand_dims(tensor.tensor, axis=dim), scale=tensor.scale)
        return self.jnp.expand_dims(tensor, axis=dim)

    def expand(self, tensor, *sizes):
        """Expand tensor to a larger size by broadcasting."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.jnp.broadcast_to(tensor.tensor, sizes), scale=tensor.scale)
        return self.jnp.broadcast_to(tensor, sizes)

    def clamp(self, tensor, min=None, max=None):
        """Clamp tensor values to a range."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.jnp.clip(tensor.tensor * tensor.scale, a_min=min, a_max=max))
        return self.jnp.clip(tensor, a_min=min, a_max=max)

    def diagonal(self, tensor, dim1=-2, dim2=-1):
        """Extract diagonal from a tensor."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            raw = tensor.tensor
            ndim = raw.ndim
            axis1 = dim1 if dim1 >= 0 else ndim + dim1
            axis2 = dim2 if dim2 >= 0 else ndim + dim2
            return TNTensor(self.jnp.diagonal(raw, axis1=axis1, axis2=axis2), scale=tensor.scale)
        ndim = tensor.ndim
        axis1 = dim1 if dim1 >= 0 else ndim + dim1
        axis2 = dim2 if dim2 >= 0 else ndim + dim2
        return self.jnp.diagonal(tensor, axis1=axis1, axis2=axis2)

    def sum(self, tensor, dim=None, keepdim=False):
        """Sum tensor elements along specified dimension(s)."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.jnp.sum(tensor.tensor, axis=dim, keepdims=keepdim), scale=tensor.scale)
        return self.jnp.sum(tensor, axis=dim, keepdims=keepdim)

    def multinomial(self, probs, num_samples):
        """Sample from multinomial distribution."""
        from ..core.tn_tensor import TNTensor
        if isinstance(probs, TNTensor):
            probs = probs.tensor * probs.scale
        # Split key for random sampling
        self.key, subkey = self.jax.random.split(self.key)

        batch_shape = probs.shape[:-1]
        num_categories = probs.shape[-1]
        
        # Flatten batch dimensions
        probs_flat = probs.reshape(-1, num_categories)
        batch_size = probs_flat.shape[0]
        
        # Generate random keys for each batch item
        subkeys = self.jax.random.split(subkey, batch_size)
        
        # Sample for each batch item
        samples = []
        for i in range(batch_size):
            sample = self.jax.random.categorical(subkeys[i], self.jnp.log(probs_flat[i]), shape=(num_samples,))
            samples.append(sample)
        
        samples = self.jnp.stack(samples, axis=0)  # Shape: (batch_size, num_samples)
        
        # Reshape back to original batch shape
        if batch_shape:
            samples = samples.reshape(*batch_shape, num_samples)
        
        return samples

    def arange(self, *args, dtype=None):
        """Create a 1-D tensor with evenly spaced values."""
        from ..core.tn_tensor import TNTensor
        if dtype is None:
            dtype = self.jnp.int32
        return TNTensor(self._gpu_put(self.jnp.arange(*args, dtype=dtype)))

    def stack(self, tensors, dim=0):
        """Stack tensors along a new dimension."""
        from ..core.tn_tensor import TNTensor
        if tensors and isinstance(tensors[0], TNTensor):
            scales = [t.scale for t in tensors if isinstance(t, TNTensor)]
            raw = [t.tensor if isinstance(t, TNTensor) else t for t in tensors]
            if len(set(scales)) == 1:
                return TNTensor(self.jnp.stack(raw, axis=dim), scale=scales[0])
            effective = [t.tensor * t.scale if isinstance(t, TNTensor) else t for t in tensors]
            return TNTensor(self.jnp.stack(effective, axis=dim))
        return self.jnp.stack(tensors, axis=dim)

    def log(self, tensor):
        """Compute natural logarithm element-wise."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.jnp.log(tensor.tensor * tensor.scale))
        return self.jnp.log(tensor)

    def mean(self, tensor, dim=None, keepdim=False):
        """Compute mean of tensor elements."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.jnp.mean(tensor.tensor, axis=dim, keepdims=keepdim), scale=tensor.scale)
        return self.jnp.mean(tensor, axis=dim, keepdims=keepdim)

    def squeeze(self, tensor, dim=None):
        """Remove dimensions of size 1."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            raw = self.jnp.squeeze(tensor.tensor) if dim is None else self.jnp.squeeze(tensor.tensor, axis=dim)
            return TNTensor(raw, scale=tensor.scale)
        if dim is None:
            return self.jnp.squeeze(tensor)
        return self.jnp.squeeze(tensor, axis=dim)

    def einsum(self, equation, *operands):
        """Perform Einstein summation convention contraction."""
        raw_ops = [self.unwrap_tensor(op) for op in operands]
        return self.jnp.einsum(equation, *raw_ops)
