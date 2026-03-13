"""
Unified engine that combines contractor (expression generation) with backend (execution).

This module provides high-level functions that use EinsumStrategy to generate
expressions and then execute them using the specified backend.

Now supports strategy-based compilation for optimized contraction paths.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Callable, Optional, Union, List, Tuple, Dict, Any

from ..contractor import EinsumStrategy, StrategyCompiler, GreedyStrategy
from ..backends.backend_factory import BackendFactory, ComputeBackend
from .tn_tensor import TNTensor
from ..losses import LossRegistry, TargetResolver
from ..losses.base import BaseLoss


class QubitOp(Enum):
    """Operation to perform on a single qubit during contraction.

    Each qubit can be independently configured to one of these modes:

    - ``TRACE``: Trace out (contract with identity matrix). The qubit
      dimension is summed over and disappears from the result.
    - ``CIRCUIT_LEFT``: Multiply the **left** (bra) side with a circuit
      state vector.
    - ``CIRCUIT_RIGHT``: Multiply the **right** (ket) side with a circuit
      state vector.
    - ``CIRCUIT_BOTH``: Multiply **both** bra and ket sides with circuit
      state vectors.
    - ``MEASURE``: Apply a measurement matrix (Mx) on this qubit.
    - ``IDENTITY``: Leave untouched – equivalent to inserting an identity
      but explicitly requested (no implicit fill).
    """
    TRACE = auto()
    CIRCUIT_LEFT = auto()
    CIRCUIT_RIGHT = auto()
    CIRCUIT_BOTH = auto()
    MEASURE = auto()
    IDENTITY = auto()



class EngineCommon:
    """
    EngineCommon that combines tensor contraction expression generation with backend execution.
    
    This class separates concerns:
    - EinsumStrategy: Generates einsum expressions using opt_einsum (legacy)
    - StrategyCompiler: Compiles optimal strategies based on network structure
    - ComputeBackend: Executes expressions using JAX, PyTorch, etc.
    """

    def __init__(self, backend: Optional[Union[str, ComputeBackend]] = None, strategy_mode: str = 'balanced', nqubits: Optional[int] = None):
        """Initialize the engine with a specific backend and strategy mode.

        Args:
            backend (str or ComputeBackend, optional): Backend to use.
                Can be ``'jax'``, ``'pytorch'``, or a
                :class:`~tneq_qc.backends.backend_factory.ComputeBackend`
                instance.  Defaults to the framework default backend.
            strategy_mode (str): Contraction strategy mode:

                - ``'fast'``: einsum only (fastest compilation)
                - ``'balanced'``: einsum + MPS chain (default)
                - ``'full'``: all available strategies

            nqubits (int, optional): Total number of qubits.  If ``None``
                (default), inferred from the QCTN at contraction time.

        Note:
            Data generation (Hermite feature maps / Mx matrices) has been
            moved to :class:`~tneq_qc.utils.data_generator.DataGenerator`.
        """
        if backend is None:
            self.backend = BackendFactory.get_default_backend()
        elif isinstance(backend, str):
            self.backend = BackendFactory.create_backend(backend, device="cpu")
        else:
            self.backend = backend

        self.contractor = EinsumStrategy()
        self.strategy_compiler = StrategyCompiler(mode=strategy_mode)
        self.strategy_mode = strategy_mode

        self._nqubits: Optional[int] = nqubits
        self._qubit_ops: Dict[int, QubitOp] = {}
        self._qubit_data: Dict[int, Any] = {}
        self._pipeline: List[Dict[str, Any]] = []

    # ================================================================
    # Qubit count helpers
    # ================================================================

    @property
    def nqubits(self) -> Optional[int]:
        """Return the configured number of qubits (may be ``None``)."""
        return self._nqubits

    @nqubits.setter
    def nqubits(self, value: Optional[int]):
        self._nqubits = value

    # ============================================================================
    # Strategy-based Compilation Methods
    # ============================================================================

    def contract_with_compiled_strategy(self, qctn) -> Any:
        """Contract using compiled strategy (auto-selected based on mode).

        Circuit states and measurement matrices must be embedded as cores
        inside *qctn* before calling this method.

        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.

        Returns:
            Backend tensor: Contraction result.
        """
        cache_key = f'_compiled_strategy_{self.strategy_mode}'

        if not hasattr(qctn, cache_key):
            compute_fn, strategy_name, cost = self.strategy_compiler.compile(
                qctn, {}, self.backend
            )
            setattr(qctn, cache_key, {
                'compute_fn': compute_fn,
                'strategy_name': strategy_name,
                'cost': cost,
            })
        else:
            compute_fn = getattr(qctn, cache_key)['compute_fn']

        cores_dict = {name: qctn.cores_weights[name] for name in qctn.cores}
        return compute_fn(cores_dict, None, None)

    def contract_with_compiled_strategy_for_gradient(
        self,
        qctn,
        target=None,
        loss: Union[None, str, Callable, BaseLoss] = None,
    ) -> Tuple:
        """Contract using compiled strategy and compute gradients.

        Circuit states and measurement matrices must be embedded as cores
        inside *qctn* before calling this method.

        Args:
            qctn   : The quantum circuit tensor network to contract.
            target : Learning target.  Accepted types:

                     * ``None``         — default ``1/side`` broadcast
                       (backward-compatible with pre-2.6.3 behaviour)
                     * ``float / int``  — broadcast scalar
                     * ``list``         — converted to tensor
                     * ``TNTensor``     — used directly (detached)
                     * ``QCTN``         — contracted first (no grad),
                       result used as target

            loss   : Loss specification.  Accepted types:

                     * ``None``         — ``'diagonal_mse'`` (default)
                     * ``str``          — registered loss name
                       (``'mse'``, ``'mae'``, ``'nll'``, ``'fidelity'``,
                       ``'diagonal_mse'``)
                     * ``callable``     — ``fn(result, target, backend)``
                     * ``BaseLoss``     — instance used directly

        Returns:
            tuple: ``(loss_value, gradients)``
        """
        cache_key = f'_compiled_strategy_{self.strategy_mode}'

        if not hasattr(qctn, cache_key):
            compute_fn, strategy_name, cost = self.strategy_compiler.compile(
                qctn, {}, self.backend
            )
            setattr(qctn, cache_key, {
                'compute_fn': compute_fn,
                'strategy_name': strategy_name,
                'cost': cost,
            })
            print(f"[EngineCommon] Compiled and cached strategy: {strategy_name}")
        else:
            compute_fn = getattr(qctn, cache_key)['compute_fn']

        # Resolve loss object once (outside the hot-path closure).
        loss_obj: BaseLoss = LossRegistry.resolve(loss)

        # Collect trainable core tensors and their scales.
        # Only collect leaf tensors (requires_grad=True AND is_leaf=True).
        # Non-leaf derived tensors (e.g. conj/permute views from hermit())
        # are intentionally skipped: autograd propagates their gradients back
        # to the originating leaf tensors automatically.
        raw_core_tensors: List[Any] = []
        core_scales: List[Any] = []
        for c_name in qctn.cores:
            c = qctn.cores_weights[c_name]
            raw = c.tensor if isinstance(c, TNTensor) else c
            if not raw.requires_grad:
                continue
            if hasattr(raw, 'is_leaf') and not raw.is_leaf:
                continue
            raw_core_tensors.append(raw)
            core_scales.append(c.scale if isinstance(c, TNTensor) else 1.0)

        # Keep a reference to self for use inside the closure.
        engine = self

        def loss_fn(*core_tensors_args):
            # Rebuild cores_dict with the current (differentiable) tensors.
            # Only leaf tensors are replaced with the incoming args; derived
            # tensors (hermit views etc.) keep their original TNTensor so that
            # the autograd graph linking them to the leaf tensors is preserved.
            offset = 0
            cores_dict: Dict[str, Any] = {}
            for c_name in qctn.cores:
                c = qctn.cores_weights[c_name]
                raw = c.tensor if isinstance(c, TNTensor) else c
                is_trainable_leaf = (
                    raw.requires_grad
                    and not (hasattr(raw, 'is_leaf') and not raw.is_leaf)
                )
                if is_trainable_leaf:
                    cores_dict[c_name] = TNTensor(
                        core_tensors_args[offset], core_scales[offset]
                    )
                    offset += 1
                else:
                    cores_dict[c_name] = c if isinstance(c, TNTensor) else TNTensor(c)

            # Contract the network.
            result = compute_fn(cores_dict, None, None)

            # Resolve target (result.shape is now known).
            resolved_target = TargetResolver.resolve(
                target, result.shape, engine.backend, engine=engine
            )

            # Delegate preprocessing + loss computation to loss_obj.
            return loss_obj(result, resolved_target, engine.backend)

        argnums = list(range(len(raw_core_tensors)))
        value_and_grad_fn = self.backend.compute_value_and_grad(loss_fn, argnums=argnums)
        return value_and_grad_fn(*raw_core_tensors)
