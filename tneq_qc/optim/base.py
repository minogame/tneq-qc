"""Base classes for backend-decoupled optimizers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.tn_tensor import TNTensor
from .ops import BackendTensorOps, TensorOps


@dataclass
class ParamRef:
    """Lightweight parameter reference for future optimizer extensions."""

    name: str
    value: Any
    trainable: bool = True
    metadata: Optional[Dict[str, Any]] = None


class OptimizerBase:
    """Base class for optimizers."""

    method: str = "sgd"  # subclasses override
    use_legacy_backend_update: bool = False

    def __init__(
        self,
        params: List[TNTensor],
        backend=None,
        ops: Optional[TensorOps] = None,
        lr: float = 0.01,
        **kwargs,
    ):
        self.params: List[TNTensor] = list(params)
        self.backend = backend
        self.ops: Optional[TensorOps] = ops or (
            BackendTensorOps(backend) if backend is not None else None
        )
        self.lr = lr
        self._extra_hp: Dict[str, Any] = kwargs
        self._state: Dict[str, Any] = {}
        self._step_count: int = 0

    def step(self, grads: List) -> None:
        """Perform a single parameter update.

        Args:
            grads: Gradient list aligned with ``self.params``.
        """
        if self.use_legacy_backend_update:
            if self.backend is None:
                raise ValueError(
                    f"{type(self).__name__} requires a backend for legacy optimizer_update"
                )
            hyperparams = {
                "learning_rate": self.lr,
                "iter": self._step_count,
                **self._extra_hp,
            }
            self.params, self._state = self.backend.optimizer_update(
                self.params, grads, self._state, self.method, hyperparams
            )
            self._step_count += 1
            return

        if self.ops is None:
            raise ValueError(
                f"{type(self).__name__} requires `ops` or `backend` to perform updates"
            )

        raw_params, raw_grads, update_meta = self._prepare_step_inputs(grads)
        hyperparams = {
            "learning_rate": self.lr,
            "iter": self._step_count,
            **self._extra_hp,
        }
        new_raw_params, self._state = self.update_raw_params(
            raw_params, raw_grads, self._state, hyperparams
        )
        self._assign_updated_params(new_raw_params, update_meta)
        self._step_count += 1

    def zero_grad(self) -> None:
        """Zero out gradients on all parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @property
    def state(self) -> Dict[str, Any]:
        """Current optimizer state (momentum buffers, etc.)."""
        return self._state

    def state_dict(self) -> Dict[str, Any]:
        """Return a serializable optimizer snapshot."""
        return {
            "state": deepcopy(self._state),
            "step_count": self._step_count,
            "lr": self.lr,
            "method": self.method,
            "extra_hp": deepcopy(self._extra_hp),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore optimizer state from :meth:`state_dict` output."""
        self._state = deepcopy(state_dict.get("state", {}))
        self._step_count = int(state_dict.get("step_count", 0))
        self.lr = float(state_dict.get("lr", self.lr))
        self._extra_hp = deepcopy(state_dict.get("extra_hp", self._extra_hp))

    def update_raw_params(self, params, grads, state, hyperparams):
        """Return updated raw tensors and new optimizer state."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement update_raw_params()"
        )

    def _prepare_step_inputs(self, grads):
        raw_params = []
        raw_grads = []
        update_meta = []

        for param, grad in zip(self.params, grads):
            if isinstance(param, TNTensor):
                scale = param.scale
                raw_params.append(param.tensor * scale)
                raw_grads.append(grad / scale)
                update_meta.append({
                    "is_tntensor": True,
                    "scale": scale,
                    "param": param,
                })
            else:
                raw_params.append(param)
                raw_grads.append(grad)
                update_meta.append({
                    "is_tntensor": False,
                    "scale": None,
                    "param": param,
                })

        return raw_params, raw_grads, update_meta

    def _assign_updated_params(self, new_raw_params, update_meta):
        for idx, meta in enumerate(update_meta):
            param = meta["param"]
            if meta["is_tntensor"]:
                scale = meta["scale"]
                new_unscaled = new_raw_params[idx] / scale
                raw = param.tensor
                if hasattr(raw, "data") and hasattr(raw.data, "copy_"):
                    self.ops.copy_into(param, new_unscaled)
                else:
                    self.ops.replace(param, new_unscaled)
            else:
                raw = param
                if hasattr(raw, "data") and hasattr(raw.data, "copy_"):
                    self.ops.copy_into(param, new_raw_params[idx])
                else:
                    self.params[idx] = self.ops.replace(param, new_raw_params[idx])
