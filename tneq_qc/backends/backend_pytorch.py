"""
PyTorch backend implementation.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import random

from .backend_interface import ComputeBackend, BackendInfo


class BackendPyTorch(ComputeBackend):
    """PyTorch computational backend."""

    def __init__(self, device: Optional[str] = None, dtype: Optional[Any] = None,
                 tensor_type: Optional[str] = None):
        """
        Initialize PyTorch backend.
        
        Args:
            device (Optional[str]): Device specification ('cpu', 'cuda', 'cuda:0', etc.).
                If None, automatically selects 'cuda' if available, otherwise 'cpu'.
            dtype (Optional[Any]): Default dtype for tensors. Can be a torch.dtype
                or a string like 'float32', 'float64', 'complex64', 'complex128', 'complex'.
            tensor_type (Optional[str]): High-level tensor wrapper type.
                Pass ``"TNTensor"`` to have :meth:`init_random_core` return
                :class:`TNTensor` instances and :meth:`get_tensor_type` report
                ``TNTensor``.
        """
        super().__init__(tensor_type=tensor_type)
        try:
            import torch
            self.torch = torch
            
            # Auto-detect device if not specified
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Resolve and store default dtype
            self.default_dtype = self._resolve_default_dtype(dtype)

            # Create BackendInfo with logical dtype string
            self.backend_info = BackendInfo(
                'pytorch',
                device=device,
                dtype=self._dtype_to_string(self.default_dtype),
            )
        except ImportError:
            raise ImportError("PyTorch is not installed. Please install it with: pip install torch")

    def _resolve_default_dtype(self, dtype: Optional[Any]):
        """
        将用户提供的 dtype 解析为 torch.dtype。

        支持：
        - None: 使用 torch.float32
        - 字符串: 'float32', 'float64', 'complex64', 'complex128', 'complex'
        - 直接传入 torch.dtype
        """
        torch = self.torch
        if dtype is None:
            return torch.float32

        if isinstance(dtype, str):
            mapping = {
                'float32': torch.float32,
                'float64': torch.float64,
                'complex64': torch.complex64,
                'complex128': torch.complex128,
                # 允许简写 'complex'，默认 complex64
                'complex': torch.complex64,
            }
            if dtype not in mapping:
                raise ValueError(
                    f"Unsupported dtype string '{dtype}' for BackendPyTorch. "
                    f"Supported: {list(mapping.keys())}"
                )
            return mapping[dtype]

        # 假设已经是 torch.dtype
        return dtype

    def _dtype_to_string(self, dtype: Any) -> str:
        """将 torch.dtype 转成逻辑上的字符串表示，便于在 BackendInfo 中记录。"""
        torch = self.torch
        mapping = {
            torch.float32: 'float32',
            torch.float64: 'float64',
            getattr(torch, 'complex64', None): 'complex64',
            getattr(torch, 'complex128', None): 'complex128',
        }
        for k, v in mapping.items():
            if k is not None and dtype == k:
                return v
        # 回退到 str 表示
        return str(dtype)

    def execute_expression(self, expression, *tensors):
        """
        Execute contraction expression using PyTorch.
        
        Note: opt_einsum's ContractExpression works with PyTorch tensors.
        """
        return expression(*tensors)

    def compute_value_and_grad(self, loss_fn, argnums):
        """
        Compute value and gradient using PyTorch autograd.
        
        Returns a function that computes loss and gradients for specified arguments.
        """
        def value_and_grad_fn(*args):
            from ..core.tn_tensor import TNTensor
            # Convert argnums to list if it's a range
            if isinstance(argnums, range):
                arg_indices = list(argnums)
            elif isinstance(argnums, (list, tuple)):
                arg_indices = argnums
            else:
                arg_indices = [argnums]

            # Compute loss (may be TNTensor)
            loss = loss_fn(*args)

            # Convert TNTensor loss to raw torch tensor (effective value)
            if isinstance(loss, TNTensor):
                loss = loss.tensor * loss.scale

            # Autograd requires a real scalar; take real part if complex
            if self.torch.is_complex(loss):
                loss = loss.real

            if loss.ndim > 0:
                loss = loss.sum()

            gradients = self.torch.autograd.grad(
                outputs=loss,
                inputs=args,
                create_graph=False,
                retain_graph=False
            )

            out_loss = loss.detach()
            if out_loss.ndim > 0:
                out_loss = out_loss.sum()
            return out_loss, gradients

        return value_and_grad_fn

    def jit_compile(self, func):
        """
        JIT compile function using PyTorch.
        
        Note: PyTorch's torch.jit.script has limitations.
        For now, we return the function as-is and rely on PyTorch's built-in optimizations.
        """
        # Could use torch.jit.trace or torch.compile in PyTorch 2.0+
        return func

    def convert_to_tensor(self, array):
        """Convert to TNTensor wrapping a PyTorch tensor."""
        from ..core.tn_tensor import TNTensor
        if isinstance(array, TNTensor):
            return array
        if isinstance(array, self.torch.Tensor):
            target = array
            if str(target.device) != self.backend_info.device:
                target = target.to(self.backend_info.device)
            if target.dtype != self.default_dtype:
                target = target.to(self.default_dtype)
            return TNTensor(target)

        if not isinstance(array, np.ndarray):
            array = np.array(array)
        tensor = self.torch.as_tensor(array, dtype=self.default_dtype)
        return TNTensor(tensor.to(self.backend_info.device))

    def get_backend_name(self) -> str:
        return "pytorch"

    def optimizer_update(self, params: List[Any], grads: List[Any], state: Dict[str, Any], 
                        method: str, hyperparams: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Perform a single optimization step using PyTorch.
        """
        with self.torch.no_grad():
            raw_params = []
            is_tntensor_info = []

            # print('params: ', len(params))
            # print('grads: ', len(grads))
            
            for p in params:
                if hasattr(p, 'tensor') and hasattr(p, 'scale') and hasattr(p, 'auto_scale'):
                    scale = p.scale

                    # p.scale_to(1.0)
                    # raw_params.append(p.tensor)
                    
                    # p.tensor /= scale
                    
                    raw_params.append(p.tensor * scale)
                    # raw_params.append(p.tensor)
                    
                    is_tntensor_info.append((True, scale, type(p)))
                else:
                    raw_params.append(p)
                    is_tntensor_info.append((False, None, None))
            
            scaled_grads = []
            for i, g in enumerate(grads):
                is_tn, scale, _ = is_tntensor_info[i]
                if is_tn:
                    scaled_grads.append(g / scale)
                else:
                    scaled_grads.append(g)
            grads = scaled_grads

            with self.torch.no_grad():
                if method == 'adam':
                    new_raw_params, new_state = self._adam_step(raw_params, grads, state, hyperparams)
                elif method == 'sgd':
                    new_raw_params, new_state = self._sgd_step(raw_params, grads, state, hyperparams)
                elif method == 'sgdg':
                    new_raw_params, new_state = self._sgdg_step(raw_params, grads, state, hyperparams)
                elif method == 'momentum':
                    new_raw_params, new_state = self._momentum_step(raw_params, grads, state, hyperparams)
                elif method == 'nesterov':
                    new_raw_params, new_state = self._nesterov_step(raw_params, grads, state, hyperparams)
                elif method == 'rmsprop':
                    new_raw_params, new_state = self._rmsprop_step(raw_params, grads, state, hyperparams)
                else:
                    raise ValueError(f"Unknown optimization method: {method}")
            
            for i, (is_tn, scale, tn_class) in enumerate(is_tntensor_info):
                if is_tn:
                    # In-place storage update: keeps the same tensor *object* so
                    # that any views derived from it (e.g. hermit conj/permute
                    # views stored in mps_h) automatically see the new values.
                    # set() would replace _tensor with a new object, breaking those views.
                    params[i].tensor.data.copy_(new_raw_params[i] / scale)
                    # params[i].tensor.requires_grad_(True)
                else:
                    params[i].data.copy_(new_raw_params[i])
                    # params[i].requires_grad_(True)
                    
            return params, new_state

    def _adam_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        beta1 = hp.get('beta1', 0.9)
        beta2 = hp.get('beta2', 0.999)
        epsilon = hp.get('epsilon', 1e-8)
        iteration = hp.get('iter', 0)

        if 'm' not in state:
            state['m'] = [self.torch.zeros_like(p) for p in params]
            state['v'] = [self.torch.zeros_like(p) for p in params]

        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update biased first moment estimate
            state['m'][i] = beta1 * state['m'][i] + (1 - beta1) * grad
            
            # Update biased second moment estimate
            state['v'][i] = beta2 * state['v'][i] + (1 - beta2) * (grad ** 2)
            
            # Compute bias-corrected first and second moment estimates
            m_hat = state['m'][i] / (1 - beta1 ** (iteration + 1))
            v_hat = state['v'][i] / (1 - beta2 ** (iteration + 1))
            
            sqrt_v_hat = self.torch.sqrt(v_hat)
            
            # Update parameters
            update = lr * m_hat / (sqrt_v_hat + epsilon)
            new_params.append(param - update)
            
        return new_params, state

    def _sgd_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        new_params = []
        for param, grad in zip(params, grads):
            new_params.append(param - lr * grad)
        return new_params, state

    def _momentum_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        momentum = hp.get('momentum', 0.0) # Default 0.0 if not provided, though usually 0.9

        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = [self.torch.zeros_like(p) for p in params]

        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            state['momentum_buffer'][i] = 0.9 * state['momentum_buffer'][i] + lr * grad
            new_params.append(param - state['momentum_buffer'][i])
            
        return new_params, state

    def _nesterov_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = [self.torch.zeros_like(p) for p in params]

        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            state['momentum_buffer'][i] = 0.9 * state['momentum_buffer'][i] + lr * grad
            new_params.append(param - (state['momentum_buffer'][i] + lr * grad))
            
        return new_params, state

    def _rmsprop_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        epsilon = hp.get('epsilon', 1e-8)

        if 'square_avg' not in state:
            state['square_avg'] = [self.torch.zeros_like(p) for p in params]

        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            state['square_avg'][i] = 0.9 * state['square_avg'][i] + 0.1 * (grad ** 2)
            new_params.append(param - lr * grad / (self.torch.sqrt(state['square_avg'][i]) + epsilon))
            
        return new_params, state

    def _sgdg_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        momentum = hp.get('momentum', 0.0)
        stiefel = hp.get('stiefel', True)
        epsilon = 1e-8

        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = [None] * len(params)

        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Check if parameter satisfies Stiefel constraint (rows <= cols after reshape)
            param_shape = param.shape
            
            # For tensors with more than 2 dimensions, reshape to matrix
            if len(param_shape) > 2:
                # Reshape similar to SGDG: flatten first dimensions
                flat_dim = int(np.prod(param_shape[:len(param_shape)//2]))
                param_2d = param.reshape(flat_dim, -1)
                grad_2d = grad.reshape(flat_dim, -1)
            else:
                param_2d = param
                grad_2d = grad

            # Detect complex for Stiefel step (conjugate and skew-Hermitian)
            is_complex = self.is_complex(param_2d)
            
            # Normalize to get orthogonal matrix
            unity, unity_norm = self._unit(param_2d)
            
            # Check if we should use Stiefel optimization
            if stiefel and unity.shape[0] <= unity.shape[1]:
                # Randomly apply QR retraction for numerical stability (1% chance)
                if random.randint(1, 101) == 1:
                    unity = self._qr_retraction(unity)
                
                # Initialize momentum buffer for this core (match dtype for complex)
                if state['momentum_buffer'][i] is None:
                    state['momentum_buffer'][i] = self.torch.zeros(
                        grad_2d.T.shape, dtype=grad_2d.dtype, device=param.device
                    )
                
                V = state['momentum_buffer'][i]
                
                # Update momentum: V = momentum * V - g^H (conjugate transpose for complex)
                g_T = self.torch.conj(grad_2d).T if is_complex else grad_2d.T
                V = momentum * V - g_T
                
                # Compute the skew-symmetric (real) or skew-Hermitian (complex) matrix W
                MX = self.torch.mm(V, unity)
                XMX = self.torch.mm(unity, MX)
                unity_H = self.torch.conj(unity).T if is_complex else unity.T
                XXMX = self.torch.mm(unity_H, XMX)
                
                W_hat = MX - 0.5 * XXMX
                W = W_hat - (self.torch.conj(W_hat).T if is_complex else W_hat.T)
                
                # Compute adaptive step size
                W_norm = self._matrix_norm_one(W)
                t = 0.5 * 2 / (W_norm + epsilon)
                alpha = min(t, lr)
                
                # Apply Cayley transform: Y(alpha) = (I - alpha/2 * W)^{-1} (I + alpha/2 * W) X
                # For complex: X = unity^H, then p_new = conj(Y)^T
                if is_complex:
                    p_new_t = self._compute_cayley_transform(alpha, W, unity_H)
                    p_new = self.torch.conj(p_new_t).T
                else:
                    p_new = self._compute_cayley_transform(alpha, W, unity.T).T
                
                # Reshape back to original shape
                if len(param_shape) > 2:
                    p_new = p_new.reshape(param_shape)
                
                new_params.append(p_new)
                
                # Update momentum buffer: V_new = W @ unity^H
                V_new = self.torch.mm(W, unity_H)
                state['momentum_buffer'][i] = V_new
                
            else:
                # Standard SGD update for non-Stiefel parameters
                new_params.append(param - lr * grad)
        
        return new_params, state

    # Helper functions for SGDG
    def _unit(self, v, dim=1, eps=1e-8):
        """Normalize a matrix to have unit norm."""
        vnorm = self.torch.norm(v, p=2, dim=dim, keepdim=True)
        return v / (vnorm + eps), vnorm
    
    def _qr_retraction(self, tan_vec):
        """QR retraction to project back onto Stiefel manifold."""
        tan_vec_T = tan_vec.T
        q, r = self.torch.linalg.qr(tan_vec_T, mode='reduced')
        d = self.torch.diag(r)
        if self.is_complex(d):
            ph = self.torch.sgn(d)
        else:
            ph = self.torch.sign(d)
        q = q * ph.unsqueeze(0)
        return q.T
    
    def _matrix_norm_one(self, W):
        """Compute matrix 1-norm (maximum absolute column sum)."""
        return self.torch.abs(W).sum(dim=0).max()
    
    def _compute_cayley_transform(self, alpha, W, X):
        """
        Compute Cayley transform: Y(alpha) = (I - alpha/2 * W)^{-1} (I + alpha/2 * W) X
        Supports both real (skew-symmetric W) and complex (skew-Hermitian W).
        """
        I = self.torch.eye(W.shape[0], dtype=W.dtype, device=W.device)
        left_matrix = I - (alpha / 2) * W
        right_matrix = I + (alpha / 2) * W
        left_inv = self.torch.inverse(left_matrix)
        Y_alpha = left_inv @ right_matrix @ X
        
        return Y_alpha

    def init_random_core(self, shape, distribution: str = "gaussian"):
        """Initialize random core using QR decomposition for orthogonality."""
        from ..core.tn_tensor import TNTensor

        dist = distribution.lower()
        flat_dim = int(np.prod(shape[:len(shape)//2]))
        if dist in {"gaussian", "normal"}:
            random_matrix = self.torch.randn(
                (flat_dim, flat_dim),
                device=self.backend_info.device,
                dtype=self.default_dtype,
            )
        elif dist == "uniform":
            random_matrix = self.torch.rand(
                (flat_dim, flat_dim),
                device=self.backend_info.device,
                dtype=self.default_dtype,
            ) * 2 - 1
        else:
            raise ValueError(
                f"Unsupported init distribution '{distribution}'. "
                "Supported values: 'gaussian', 'uniform'."
            )

        Q, R = self.torch.linalg.qr(random_matrix)
        d = self.torch.diag(R)
        if self.is_complex(d):
            phases = d / (d.abs() + 1e-12)
            sign_correction = self.torch.diag(phases.conj())
            Q = Q @ sign_correction
        else:
            sign_correction = self.torch.sign(d)
            Q = Q * sign_correction.unsqueeze(0)
            # Ensure det(Q) = +1 (proper rotation / SO(n)).
            # Without this, Q may be a reflection (det = -1).  For 2×2 real
            # matrices, SO(2) and O⁻(2) are orthogonal under the Frobenius
            # inner product, causing Tr(S·T†) = 0 when S ∈ SO(2), T ∈ O⁻(2).
            if Q.shape[0] >= 2 and self.torch.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]

        if self.is_complex(self.torch.zeros(1, dtype=self.default_dtype)):
            Q = Q.to(self.default_dtype)
        return TNTensor(Q.reshape(shape))

    def _get_raw_tensor_type(self):
        return self.torch.Tensor

    def set_random_seed(self, seed: int):
        """Set random seed for PyTorch and related libraries."""
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed(seed)
            self.torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def tensor_to_numpy(self, tensor):
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            tensor = tensor.tensor
        if not isinstance(tensor, self.torch.Tensor):
            tensor = self.torch.as_tensor(tensor)
        return tensor.detach().cpu().numpy()

    def reshape(self, tensor, shape):
        """Reshape tensor to the given shape."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return tensor.reshape(shape)
        return tensor.reshape(shape)

    def eye(self, n: int, dtype=None):
        """Create an identity matrix of size n x n."""
        from ..core.tn_tensor import TNTensor
        if dtype is None:
            dtype = self.default_dtype
        return TNTensor(self.torch.eye(n, dtype=dtype, device=self.backend_info.device))

    def zeros(self, shape, dtype=None):
        """Create a tensor filled with zeros."""
        from ..core.tn_tensor import TNTensor
        if dtype is None:
            dtype = self.default_dtype
        return TNTensor(self.torch.zeros(shape, dtype=dtype, device=self.backend_info.device))

    def ones(self, shape, dtype=None):
        """Create a tensor filled with ones."""
        from ..core.tn_tensor import TNTensor
        if dtype is None:
            dtype = self.default_dtype
        return TNTensor(self.torch.ones(shape, dtype=dtype, device=self.backend_info.device))

    def clone(self, tensor):
        """Create a copy of the tensor."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(tensor.tensor.clone(), scale=tensor.scale)
        return tensor.clone()

    def unsqueeze(self, tensor, dim):
        """Add a dimension of size 1 at the specified position."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(tensor.tensor.unsqueeze(dim), scale=tensor.scale)
        return tensor.unsqueeze(dim)

    def expand(self, tensor, *sizes):
        """Expand tensor to a larger size by broadcasting."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(tensor.tensor.expand(*sizes), scale=tensor.scale)
        return tensor.expand(*sizes)

    def clamp(self, tensor, min=None, max=None):
        """Clamp tensor values to a range.

        PyTorch 不支持对 complex 张量直接 clamp，这里只对实部做裁剪，
        虚部保持不变，满足本项目中"概率/密度在实部非负"的需求。
        """
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            raw = tensor.tensor * tensor.scale
            if self.torch.is_complex(raw):
                r = self.torch.clamp(self.torch.real(raw), min=min, max=max)
                return TNTensor(self.torch.complex(r, self.torch.imag(raw)))
            return TNTensor(self.torch.clamp(raw, min=min, max=max))
        if self.torch.is_complex(tensor):
            real = self.torch.real(tensor)
            imag = self.torch.imag(tensor)
            return self.torch.complex(self.torch.clamp(real, min=min, max=max), imag)
        return self.torch.clamp(tensor, min=min, max=max)
    def diagonal(self, tensor, dim1=-2, dim2=-1):
        """Extract diagonal from a tensor."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.torch.diagonal(tensor.tensor, dim1=dim1, dim2=dim2), scale=tensor.scale)
        return self.torch.diagonal(tensor, dim1=dim1, dim2=dim2)

    def sum(self, tensor, dim=None, keepdim=False):
        """Sum tensor elements along specified dimension(s)."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.torch.sum(tensor.tensor, dim=dim, keepdim=keepdim), scale=tensor.scale)
        return self.torch.sum(tensor, dim=dim, keepdim=keepdim)

    def multinomial(self, probs, num_samples):
        """Sample from multinomial distribution."""
        from ..core.tn_tensor import TNTensor
        if isinstance(probs, TNTensor):
            probs = probs.tensor * probs.scale
        return self.torch.multinomial(probs, num_samples=num_samples)

    def arange(self, *args, dtype=None):
        """Create a 1-D tensor with evenly spaced values."""
        from ..core.tn_tensor import TNTensor
        if dtype is None:
            dtype = self.torch.long
        return TNTensor(self.torch.arange(*args, dtype=dtype, device=self.backend_info.device))

    def stack(self, tensors, dim=0):
        """Stack tensors along a new dimension."""
        from ..core.tn_tensor import TNTensor
        if tensors and isinstance(tensors[0], TNTensor):
            scales = [t.scale for t in tensors if isinstance(t, TNTensor)]
            raw = [t.tensor if isinstance(t, TNTensor) else t for t in tensors]
            if len(set(scales)) == 1:
                return TNTensor(self.torch.stack(raw, dim=dim), scale=scales[0])
            effective = [t.tensor * t.scale if isinstance(t, TNTensor) else t for t in tensors]
            return TNTensor(self.torch.stack(effective, dim=dim))
        return self.torch.stack(tensors, dim=dim)

    def log(self, tensor):
        """Compute natural logarithm element-wise."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.torch.log(tensor.tensor * tensor.scale))
        return self.torch.log(tensor)

    def mean(self, tensor, dim=None, keepdim=False):
        """Compute mean of tensor elements."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.torch.mean(tensor.tensor, dim=dim, keepdim=keepdim), scale=tensor.scale)
        return self.torch.mean(tensor, dim=dim, keepdim=keepdim)

    def squeeze(self, tensor, dim=None):
        """Remove dimensions of size 1."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            raw = tensor.tensor.squeeze() if dim is None else tensor.tensor.squeeze(dim)
            return TNTensor(raw, scale=tensor.scale)
        if dim is None:
            return tensor.squeeze()
        return tensor.squeeze(dim)

    def detach(self, tensor):
        """Detach tensor from computation graph."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(tensor.tensor.detach(), scale=tensor.scale)
        if hasattr(tensor, 'detach'):
            return tensor.detach()
        return tensor

    def lgamma(self, tensor):
        """Compute log-gamma function element-wise."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.torch.lgamma(tensor.tensor * tensor.scale))
        return self.torch.lgamma(tensor)

    def exp(self, tensor):
        """Compute exponential function element-wise."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.torch.exp(tensor.tensor * tensor.scale))
        return self.torch.exp(tensor)

    def sqrt(self, tensor):
        """Compute square root element-wise."""
        import math
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.torch.sqrt(tensor.tensor), scale=math.sqrt(abs(tensor.scale)))
        return self.torch.sqrt(tensor)

    def square(self, tensor):
        """Compute square element-wise."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.torch.square(tensor.tensor), scale=tensor.scale ** 2)
        return self.torch.square(tensor)

    def permute(self, tensor, dims):
        """Permute tensor dimensions."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(tensor.tensor.permute(dims), scale=tensor.scale)
        return tensor.permute(dims)

    def einsum(self, equation, *operands):
        """Perform Einstein summation convention contraction."""
        raw_ops = [self.unwrap_tensor(op) for op in operands]
        return self.torch.einsum(equation, *raw_ops)

    def ones_like(self, tensor):
        """Create a tensor of ones with same shape and type as input."""
        from ..core.tn_tensor import TNTensor
        raw = self.unwrap_tensor(tensor)
        return TNTensor(self.torch.ones_like(raw))

    def linspace(self, start, end, steps, dtype=None):
        """Create a 1-D tensor of size steps with values linearly spaced in range [start, end]."""
        from ..core.tn_tensor import TNTensor
        if dtype is None:
            dtype = self.default_dtype
        return TNTensor(self.torch.linspace(start, end, steps, dtype=dtype, device=self.backend_info.device))

    def cumsum(self, tensor, dim, dtype=None):
        """Returns the cumulative sum of elements of input in the dimension dim."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.torch.cumsum(tensor.tensor, dim=dim, dtype=dtype), scale=tensor.scale)
        return self.torch.cumsum(tensor, dim=dim, dtype=dtype)

    def rand(self, size, dtype=None):
        """Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)."""
        from ..core.tn_tensor import TNTensor
        if dtype is None:
            dtype = self.default_dtype
        return TNTensor(self.torch.rand(size, dtype=dtype, device=self.backend_info.device))

    def real(self, tensor):
        """Returns the real part of the tensor."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return TNTensor(self.torch.real(tensor.tensor), scale=tensor.scale)
        return self.torch.real(tensor)

    def is_complex(self, tensor) -> bool:
        """Return True if tensor is complex dtype."""
        from ..core.tn_tensor import TNTensor
        if isinstance(tensor, TNTensor):
            return self.torch.is_complex(tensor.tensor)
        return self.torch.is_complex(tensor)

    def abs_square(self, tensor):
        """Return |tensor|^2 for complex tensors and tensor^2 for real tensors."""
        if self.is_complex(tensor):
            r, i = self.torch.real(tensor), self.torch.imag(tensor)
            return r * r + i * i
        return tensor * tensor

    def gather(self, input, dim, index):
        """Gathers values along an axis specified by dim."""
        from ..core.tn_tensor import TNTensor
        scale = 1.0
        if isinstance(input, TNTensor):
            scale = input.scale
            input = input.tensor
        if isinstance(index, TNTensor):
            index = index.tensor.long()
        return TNTensor(self.torch.gather(input, dim, index), scale=scale)
