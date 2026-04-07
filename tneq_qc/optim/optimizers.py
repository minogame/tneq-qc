"""Concrete optimizer classes."""

from __future__ import annotations

import random
import numpy as np
from typing import List

from ..core.tn_tensor import TNTensor
from .base import OptimizerBase


class Adam(OptimizerBase):
    """Adam optimizer.

    Args:
        params: Trainable parameters.
        backend: Compute backend.
        lr: Learning rate.
        beta1: Exponential decay rate for first moment.
        beta2: Exponential decay rate for second moment.
        epsilon: Numerical stability constant.
    """

    method = "adam"

    def __init__(
        self,
        params: List[TNTensor],
        backend=None,
        ops=None,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(
            params, backend=backend, ops=ops,
            lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon
        )

    def update_raw_params(self, params, grads, state, hyperparams):
        beta1 = hyperparams.get("beta1", 0.9)
        beta2 = hyperparams.get("beta2", 0.999)
        epsilon = hyperparams.get("epsilon", 1e-8)
        lr = hyperparams.get("learning_rate", 0.01)
        iteration = hyperparams.get("iter", 0)

        if "m" not in state:
            state["m"] = [self.ops.zeros_like(p) for p in params]
            state["v"] = [self.ops.zeros_like(p) for p in params]

        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            state["m"][i] = beta1 * state["m"][i] + (1 - beta1) * grad
            state["v"][i] = beta2 * state["v"][i] + (1 - beta2) * (grad ** 2)
            m_hat = state["m"][i] / (1 - beta1 ** (iteration + 1))
            v_hat = state["v"][i] / (1 - beta2 ** (iteration + 1))
            update = lr * m_hat / (self.ops.sqrt(v_hat) + epsilon)
            new_params.append(param - update)
        return new_params, state


class SGD(OptimizerBase):
    """Vanilla stochastic gradient descent."""

    method = "sgd"

    def update_raw_params(self, params, grads, state, hyperparams):
        lr = hyperparams.get("learning_rate", 0.01)
        return [param - lr * grad for param, grad in zip(params, grads)], state


class SGDG(OptimizerBase):
    """Stiefel Gradient Descent (SGDG).

    Preserves orthogonality constraints via Cayley transform.

    Args:
        params: Trainable parameters.
        backend: Compute backend.
        lr: Learning rate.
        momentum: Momentum factor.
        stiefel: Whether to use Stiefel manifold optimisation.
    """

    method = "sgdg"

    def __init__(
        self,
        params: List[TNTensor],
        backend=None,
        ops=None,
        lr: float = 0.01,
        momentum: float = 0.9,
        stiefel: bool = True,
    ):
        super().__init__(
            params, backend=backend, ops=ops, lr=lr,
            momentum=momentum, stiefel=stiefel
        )

    def update_raw_params(self, params, grads, state, hyperparams):
        if self.backend is None:
            raise ValueError("SGDG requires `backend` for linear algebra helpers")

        lib = getattr(self.backend, "torch", None) or getattr(self.backend, "jnp", None)
        if lib is None:
            raise TypeError("SGDG requires a backend exposing `torch` or `jnp`")

        lr = hyperparams.get("learning_rate", 0.01)
        momentum = hyperparams.get("momentum", 0.0)
        stiefel = hyperparams.get("stiefel", True)
        epsilon = hyperparams.get("epsilon", 1e-8)

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = [None] * len(params)

        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            param_shape = param.shape

            if len(param_shape) > 2:
                flat_dim = int(np.prod(param_shape[:len(param_shape) // 2]))
                param_2d = param.reshape(flat_dim, -1)
                grad_2d = grad.reshape(flat_dim, -1)
            else:
                param_2d = param
                grad_2d = grad

            is_complex = self._is_complex(param_2d)
            unity, _norm = self._unit(param_2d, lib, eps=epsilon)

            if stiefel and unity.shape[0] <= unity.shape[1]:
                if random.randint(1, 101) == 1:
                    unity = self._qr_retraction(unity, lib, is_complex)

                if state["momentum_buffer"][i] is None:
                    state["momentum_buffer"][i] = self._zeros(
                        lib, grad_2d.T.shape, grad_2d.dtype, getattr(param, "device", None)
                    )

                V = state["momentum_buffer"][i]
                grad_h = self._hermitian_transpose(grad_2d, is_complex)
                V = momentum * V - grad_h

                MX = V @ unity
                XMX = unity @ MX
                unity_h = self._hermitian_transpose(unity, is_complex)
                XXMX = unity_h @ XMX

                W_hat = MX - 0.5 * XXMX
                W = W_hat - self._hermitian_transpose(W_hat, is_complex)

                W_norm = self._matrix_norm_one(W, lib)
                alpha = min(1.0 / (W_norm + epsilon), lr)

                if is_complex:
                    p_new_t = self._compute_cayley_transform(alpha, W, unity_h, lib)
                    p_new = self._hermitian_transpose(p_new_t, True)
                else:
                    p_new = self._compute_cayley_transform(alpha, W, unity.T, lib).T

                if len(param_shape) > 2:
                    p_new = p_new.reshape(param_shape)

                new_params.append(p_new)
                state["momentum_buffer"][i] = W @ unity_h
            else:
                new_params.append(param - lr * grad)

        return new_params, state

    def _is_complex(self, tensor) -> bool:
        if hasattr(self.backend, "is_complex"):
            return bool(self.backend.is_complex(tensor))
        is_complex_fn = getattr(tensor, "is_complex", None)
        if callable(is_complex_fn):
            return bool(is_complex_fn())
        return False

    def _unit(self, matrix, lib, eps=1e-8):
        norm = lib.linalg.norm(matrix, ord=2, axis=1, keepdims=True)
        return matrix / (norm + eps), norm

    def _qr_retraction(self, matrix, lib, is_complex):
        q, r = lib.linalg.qr(matrix.T, mode="reduced")
        d = lib.diag(r)
        if is_complex:
            if hasattr(lib, "sgn"):
                phases = lib.sgn(d)
            else:
                phases = d / (lib.abs(d) + 1e-12)
        else:
            phases = lib.sign(d)
        return (q * phases.reshape(1, -1)).T

    def _matrix_norm_one(self, matrix, lib):
        return lib.abs(matrix).sum(axis=0).max()

    def _compute_cayley_transform(self, alpha, W, X, lib):
        eye_kwargs = {"dtype": W.dtype}
        device = getattr(W, "device", None)
        if device is not None:
            eye_kwargs["device"] = device
        I = lib.eye(W.shape[0], **eye_kwargs)
        return lib.linalg.inv(I - (alpha / 2.0) * W) @ (I + (alpha / 2.0) * W) @ X

    def _hermitian_transpose(self, matrix, is_complex):
        return matrix.conj().T if is_complex else matrix.T

    def _zeros(self, lib, shape, dtype, device=None):
        if device is not None:
            try:
                return lib.zeros(shape, dtype=dtype, device=device)
            except TypeError:
                pass
        return lib.zeros(shape, dtype=dtype)


class Momentum(OptimizerBase):
    """SGD with momentum."""

    method = "momentum"

    def __init__(
        self,
        params: List[TNTensor],
        backend=None,
        ops=None,
        lr: float = 0.01,
        momentum: float = 0.9,
    ):
        super().__init__(params, backend=backend, ops=ops, lr=lr, momentum=momentum)

    def update_raw_params(self, params, grads, state, hyperparams):
        lr = hyperparams.get("learning_rate", 0.01)
        momentum = hyperparams.get("momentum", 0.9)

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = [self.ops.zeros_like(p) for p in params]

        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            state["momentum_buffer"][i] = (
                momentum * state["momentum_buffer"][i] + lr * grad
            )
            new_params.append(param - state["momentum_buffer"][i])
        return new_params, state


class RMSProp(OptimizerBase):
    """RMSProp optimizer.

    Args:
        params: Trainable parameters.
        backend: Compute backend.
        lr: Learning rate.
        epsilon: Numerical stability constant.
    """

    method = "rmsprop"

    def __init__(
        self,
        params: List[TNTensor],
        backend=None,
        ops=None,
        lr: float = 0.01,
        alpha: float = 0.99,
        epsilon: float = 1e-8,
    ):
        super().__init__(
            params, backend=backend, ops=ops,
            lr=lr, alpha=alpha, epsilon=epsilon
        )

    def update_raw_params(self, params, grads, state, hyperparams):
        lr = hyperparams.get("learning_rate", 0.01)
        alpha = hyperparams.get("alpha", 0.99)
        epsilon = hyperparams.get("epsilon", 1e-8)

        if "square_avg" not in state:
            state["square_avg"] = [self.ops.zeros_like(p) for p in params]

        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            state["square_avg"][i] = (
                alpha * state["square_avg"][i] + (1 - alpha) * (grad ** 2)
            )
            update = lr * grad / (self.ops.sqrt(state["square_avg"][i]) + epsilon)
            new_params.append(param - update)
        return new_params, state
