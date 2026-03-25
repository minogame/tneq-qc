"""
Distributed SGDG Optimizer

Stiefel Gradient Descent (SGDG) optimizer for distributed training.
Maintains orthogonality constraints using Cayley transform.

Phase 4.0 update:
- step() now takes (grads) instead of (local_qctn, grads)
- Params list stored in constructor (consistent with single-process OptimizerBase)
- Removed lr /= 10 hack for TNTensor
- Uses TNTensor transparent proxy where possible

Reference:
- SGD-G: Stiefel Gradient Descent for Decorrelated Weight Matrix
"""

import random
from typing import Dict, List, Optional, Any

import torch

from ...core.tn_tensor import TNTensor


class DistributedSGDG:
    """
    Distributed Stiefel Gradient Descent (SGDG) Optimizer.

    This optimizer updates weights on the Stiefel manifold using Cayley transform,
    which preserves orthogonality constraints. Each rank independently optimizes
    its local partition weights.

    Key features:
    - Cayley transform for manifold-preserving updates
    - Momentum support
    - Adaptive step size based on matrix norm
    - Periodic QR re-orthogonalization for numerical stability

    Phase 4.0: step(grads) interface — params stored in constructor, matching
    single-process OptimizerBase convention.

    Args:
        params: List of TNTensor parameters to optimize (from qctn.parameters()
                or manually collected from local_qctn.cores_weights)
        lr: Learning rate
        momentum: Momentum factor (default: 0.0)
        stiefel: Whether to use Stiefel manifold optimization (default: True)
        epsilon: Small constant for numerical stability (default: 1e-8)
        qr_retraction_prob: Probability of QR retraction per step (default: 0.01)
    """

    def __init__(self,
                 params: Optional[List] = None,
                 lr: float = 0.01,
                 momentum: float = 0.0,
                 stiefel: bool = True,
                 epsilon: float = 1e-8,
                 qr_retraction_prob: float = 0.01):
        self.params = list(params) if params is not None else []
        self.lr = lr
        self.momentum = momentum
        self.stiefel = stiefel
        self.epsilon = epsilon
        self.qr_retraction_prob = qr_retraction_prob

        # Momentum buffers: {param_idx: velocity_tensor}
        self.momentum_buffer: Dict[int, torch.Tensor] = {}

        # Step counter
        self.step_count = 0

    def step(self, grads: List[torch.Tensor]):
        """
        Perform a single optimization step.

        Args:
            grads: List of gradients corresponding to each parameter in self.params
        """
        self.step_count += 1

        for i, param in enumerate(self.params):
            grad = grads[i]

            if param is None or grad is None:
                continue

            if isinstance(param, TNTensor):
                tensor = param.tensor
                scale = param.scale
            else:
                tensor = param
                scale = 1.0

            with torch.no_grad():
                # Undo scale on gradient, apply scale to tensor for optimization
                grad = grad / scale
                tensor = tensor * scale

                if self.stiefel:
                    updated_tensor = self._stiefel_update(tensor, grad, i)
                else:
                    updated_tensor = tensor - self.lr * grad

                # Re-apply scale division
                updated_tensor = updated_tensor / scale

            # Detach and re-enable gradients for next iteration
            updated_tensor.requires_grad_(True)
            if updated_tensor.grad is not None:
                updated_tensor.grad.detach_()
                updated_tensor.grad.zero_()

            # Update the parameter in-place
            if isinstance(param, TNTensor):
                param.set(updated_tensor, scale)
            else:
                self.params[i] = updated_tensor

    def _stiefel_update(self, param: torch.Tensor, grad: torch.Tensor,
                        param_idx: int) -> torch.Tensor:
        """
        Perform Stiefel manifold update using Cayley transform.

        The update preserves the orthogonality constraint X^T X = I.

        Args:
            param: Current parameter tensor
            grad: Gradient tensor
            param_idx: Parameter index (for momentum buffer)

        Returns:
            Updated parameter tensor
        """
        original_shape = param.shape

        # Reshape to matrix form [rows, cols] for Stiefel optimization
        if len(original_shape) > 2:
            flat_dim = 1
            for i in range(len(original_shape) // 2):
                flat_dim *= original_shape[i]
            p_reshaped = param.reshape(flat_dim, -1)
            g_reshaped = grad.reshape(flat_dim, -1)
        elif len(original_shape) == 2:
            p_reshaped = param
            g_reshaped = grad
        else:
            # Fall back to standard SGD for 1D or scalar
            return param - self.lr * grad

        # Normalize to get orthogonal matrix
        unity, unity_norm = self._unit(p_reshaped)

        # Check Stiefel condition: rows <= cols
        if unity.size(0) > unity.size(1):
            return param - self.lr * grad

        # Periodic QR retraction for numerical stability (1% chance)
        if random.randint(1, 101) == 1:
            unity = self._qr_retraction(unity)

        # Initialize or retrieve momentum buffer
        if param_idx not in self.momentum_buffer:
            self.momentum_buffer[param_idx] = torch.zeros(
                g_reshaped.t().size(),
                device=param.device,
                dtype=param.dtype
            )

        V = self.momentum_buffer[param_idx]

        # Update momentum: V = momentum * V - g^T
        V = self.momentum * V - g_reshaped.t()

        # Compute skew-symmetric matrix W
        MX = torch.mm(V, unity)
        XMX = torch.mm(unity, MX)
        XXMX = torch.mm(unity.t(), XMX)
        W_hat = MX - 0.5 * XXMX
        W = W_hat - W_hat.t()  # Ensure skew-symmetry

        # Adaptive step size based on matrix norm
        W_norm = self._matrix_norm_one(W)
        t = 0.5 * 2 / (W_norm + self.epsilon)
        alpha = min(t, self.lr)

        # Cayley transform: Y = (I - alpha/2 * W)^(-1) @ (I + alpha/2 * W) @ X
        p_new = self._compute_Y(alpha, W, unity.t()).t()

        # Update momentum buffer
        V_new = torch.mm(W, unity.t())
        self.momentum_buffer[param_idx] = V_new

        return p_new.reshape(original_shape)

    def _unit(self, v: torch.Tensor, dim: int = 1, eps: float = 1e-8):
        """Normalize a matrix to have unit norm."""
        vnorm = torch.norm(v, p=2, dim=dim, keepdim=True)
        return v / (vnorm + eps), vnorm

    def _qr_retraction(self, tan_vec: torch.Tensor) -> torch.Tensor:
        """QR retraction to project back onto Stiefel manifold."""
        tan_vec_T = tan_vec.t()
        q, r = torch.linalg.qr(tan_vec_T, mode='reduced')
        d = torch.diag(r)
        ph = torch.sign(d)
        q = q * ph.unsqueeze(0)
        return q.t()

    def _compute_Y(self, alpha: float, W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Cayley transform: Y = (I - alpha/2 * W)^(-1) @ (I + alpha/2 * W) @ X"""
        I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
        left_matrix = I - (alpha / 2) * W
        right_matrix = I + (alpha / 2) * W
        left_inv = torch.inverse(left_matrix)
        return left_inv @ right_matrix @ X

    def _matrix_norm_one(self, W: torch.Tensor) -> torch.Tensor:
        """Compute matrix 1-norm (maximum absolute column sum)."""
        return torch.abs(W).sum(dim=0).max()

    def zero_grad(self):
        """Zero out gradients on all parameters."""
        for param in self.params:
            if isinstance(param, TNTensor):
                tensor = param.tensor
            else:
                tensor = param

            if hasattr(tensor, 'grad') and tensor.grad is not None:
                tensor.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state dictionary for checkpointing."""
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'stiefel': self.stiefel,
            'step_count': self.step_count,
            'momentum_buffer': {
                k: v.cpu() for k, v in self.momentum_buffer.items()
            }
        }

    def load_state_dict(self, state_dict: Dict[str, Any], device: str = 'cpu'):
        """Load optimizer state from dictionary."""
        self.lr = state_dict.get('lr', self.lr)
        self.momentum = state_dict.get('momentum', self.momentum)
        self.stiefel = state_dict.get('stiefel', self.stiefel)
        self.step_count = state_dict.get('step_count', 0)

        if 'momentum_buffer' in state_dict:
            self.momentum_buffer = {
                k: v.to(device)
                for k, v in state_dict['momentum_buffer'].items()
            }


class LRScheduler:
    """
    Learning rate scheduler for DistributedSGDG.

    Supports step-based learning rate schedules.

    Args:
        optimizer: DistributedSGDG optimizer
        lr_schedule: List of (step, lr) tuples defining the schedule
    """

    def __init__(self, optimizer: DistributedSGDG,
                 lr_schedule: List[tuple]):
        self.optimizer = optimizer
        self.lr_schedule = sorted(lr_schedule, key=lambda x: x[0])
        self.current_step = 0

    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1

        for step, lr in reversed(self.lr_schedule):
            if self.current_step >= step:
                self.optimizer.lr = lr
                return

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr
