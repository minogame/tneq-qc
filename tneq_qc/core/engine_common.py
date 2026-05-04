"""
Unified engine that combines contractor (expression generation) with backend (execution).

This module provides high-level functions that use EinsumStrategy to generate
expressions and then execute them using the specified backend.

Now supports strategy-based compilation for optimized contraction paths.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Callable, Optional, Union, List, Tuple, Dict, Any
import warnings

from tqdm import tqdm

from ..contractor import EinsumStrategy, StrategyCompiler
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

    def __init__(
        self,
        backend: Optional[Union[str, ComputeBackend]] = None,
        strategy: Union[str, List[str], None] = None,
        strategy_mode: Optional[str] = None,
        nqubits: Optional[int] = None,
    ):
        """Initialize the engine with a specific backend and strategy mode.

        Args:
            backend (str or ComputeBackend, optional): Backend to use.
                Can be ``'jax'``, ``'pytorch'``, or a
                :class:`~tneq_qc.backends.backend_factory.ComputeBackend`
                instance.  Defaults to the framework default backend.
            strategy (str or list[str]): Strategy name, or a list of candidate
                strategy names. Defaults to ``'row_priority'``.
            strategy_mode (str): Deprecated contraction strategy mode:

                - ``'fast'``: einsum only (fastest compilation)
                - ``'balanced'``: row-priority only (default)
                - ``'full'``: row-priority only

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

        if strategy is None and strategy_mode is None:
            strategy = 'row_priority'
        elif strategy is not None and strategy_mode is not None:
            warnings.warn(
                "Both strategy and strategy_mode were provided; strategy takes precedence.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif strategy_mode is not None:
            warnings.warn(
                "strategy_mode is deprecated; pass strategy='row_priority' or "
                "strategy='einsum_default' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.contractor = EinsumStrategy()
        self.strategy_compiler = StrategyCompiler(
            strategy=strategy,
            mode=None if strategy is not None else strategy_mode,
        )
        self.strategy = self.strategy_compiler.strategy_key
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

    def contract(self, qctn) -> Any:
        """Contract using compiled strategy (auto-selected based on mode).

        Data must be embedded as cores inside *qctn* before calling.

        Args:
            qctn: The quantum circuit tensor network to contract.

        Returns:
            Contraction result.
        """
        cache_key = f'_compiled_strategy_{self.strategy}'

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

    def contract_for_gradient(
        self,
        qctn,
        target=None,
        loss: Union[None, str, Callable, BaseLoss] = None,
    ) -> Tuple:
        """Contract and compute gradients.

        Data must be embedded as cores inside *qctn* before calling.

        Args:
            qctn: The quantum circuit tensor network to contract.
            target: Learning target (None, float, list, TNTensor, or QCTN).
            loss: Loss specification (None, str, callable, or BaseLoss).

        Returns:
            tuple: ``(loss_value, gradients)``
        """
        cache_key = f'_compiled_strategy_{self.strategy}'

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

    # Aliases (backward-compatible, to be removed in Phase 5)
    contract_with_compiled_strategy = contract
    contract_with_compiled_strategy_for_gradient = contract_for_gradient

    # ============================================================================
    # Probability Calculation
    # ============================================================================

    def _mx_core_names(self, qctn) -> List[str]:
        names_map = getattr(qctn, 'core_names', {})
        return [
            names_map[sym] for sym in qctn.cores
            if names_map.get(sym, '').startswith('mx.')
        ]

    def _identity_for_core(self, qctn, name: str) -> TNTensor:
        core = qctn[name]
        raw = core.tensor if isinstance(core, TNTensor) else core
        K = raw.shape[-1]
        ident = self.backend.eye(K)
        if isinstance(ident, TNTensor):
            return ident
        return TNTensor(ident)

    def _snapshot_cores(self, qctn, names: List[str]) -> Dict[str, Any]:
        return {name: qctn[name] for name in names}

    def _restore_cores(self, qctn, snapshot: Dict[str, Any]) -> None:
        for name, tensor in snapshot.items():
            qctn[name] = tensor

    def _result_to_probability_tensor(self, result):
        """Convert a contraction result with probability semantics to a real tensor."""
        if isinstance(result, TNTensor):
            val = result.tensor * result.scale
        else:
            val = result

        # BornMachine already contracts tn† · Mx · tn, so the result is an
        # expectation value/probability. Complex dtype only reflects the tensor
        # representation; do not apply another |.|^2 here.
        if hasattr(val, 'is_complex') and val.is_complex():
            val = val.real
        return val

    def calculate_probability(
        self,
        qctn,
        mx_dict: Dict[str, Any],
        *,
        restore: bool = True,
    ) -> float:
        """Calculate probability by updating mx cores and contracting.

        Full probability: *mx_dict* covers all mx cores.
        Marginal probability: *mx_dict* covers only a subset; the rest
        stay as identity (trace-out).

        Args:
            qctn: Combined QCTN (built via ``QCTN.concat``).
            mx_dict: ``{readable_name: TNTensor}`` of mx cores to update.
                Example: ``{'mx.a': Mx_0, 'mx.b': Mx_1}``.

        Returns:
            float: Scalar probability value.
        """
        snapshot = self._snapshot_cores(qctn, list(mx_dict)) if restore else {}
        try:
            for name, tensor in mx_dict.items():
                qctn[name] = tensor

            result = self.contract(qctn)
            val = self._result_to_probability_tensor(result)
        finally:
            if restore:
                self._restore_cores(qctn, snapshot)

        return float(val.item()) if hasattr(val, 'item') else float(val)

    def full_probability(self, qctn, mx_dict: Dict[str, Any]) -> float:
        """Calculate full probability after validating all mx cores are set."""
        expected = set(self._mx_core_names(qctn))
        actual = set(mx_dict)
        missing = expected - actual
        if missing:
            raise ValueError(f"full_probability missing mx cores: {sorted(missing)}")
        return self.calculate_probability(qctn, mx_dict)

    def marginal_probability(self, qctn, mx_dict: Dict[str, Any]) -> float:
        """Calculate marginal probability; unspecified mx cores trace out."""
        mx_names = self._mx_core_names(qctn)
        snapshot = self._snapshot_cores(qctn, mx_names)
        try:
            for name in mx_names:
                if name not in mx_dict:
                    qctn[name] = self._identity_for_core(qctn, name)
            return self.calculate_probability(qctn, mx_dict, restore=False)
        finally:
            self._restore_cores(qctn, snapshot)

    # ============================================================================
    # Sampling
    # ============================================================================

    def sample(
        self,
        qctn,
        data_generator,
        sample_core_names: List[str],
        num_samples: int,
        bounds: Tuple[float, float] = (-5, 5),
        grid_size: int = 1000,
        use_marginal: bool = False,
    ):
        """Sample from the QCTN probability distribution via inverse CDF.

        Sequentially samples each core in *sample_core_names* using the
        autoregressive conditional: already-sampled cores are fixed,
        future cores keep identity (trace-out).

        Args:
            qctn: Combined QCTN (built via ``QCTN.concat``).
            data_generator: DataGenerator for Hermite Mx generation.
            sample_core_names: Readable names of mx cores to sample,
                in autoregressive order (e.g. ``['mx.a', 'mx.b']``).
            num_samples: Number of samples (S).
            bounds: Sampling range ``(x_min, x_max)``.
            grid_size: Number of CDF grid points (G).
            use_marginal: If True, sample every dimension from its marginal
                distribution instead of the autoregressive conditional.

        Returns:
            Tensor of shape ``(num_samples, len(sample_core_names))``.
        """
        initial_cores = self._snapshot_cores(qctn, sample_core_names)

        # Infer K from the first sample core.
        first_core = qctn[sample_core_names[0]]
        raw_first = first_core.tensor if isinstance(first_core, TNTensor) else first_core
        K = raw_first.shape[-1]

        x_min, x_max = bounds
        grid_x = self.backend.linspace(x_min, x_max, steps=grid_size)
        # Grid values are always real.
        if isinstance(grid_x, TNTensor):
            grid_x = grid_x.tensor
        if hasattr(grid_x, 'is_complex') and grid_x.is_complex():
            grid_x = grid_x.real

        # Identity for resetting unsampled cores.
        ident = self.backend.eye(K)
        if isinstance(ident, TNTensor):
            ident_tt = ident
        else:
            ident_tt = TNTensor(ident)

        # Reset all sample cores to identity before starting.
        for name in sample_core_names:
            qctn[name] = ident_tt

        samples = self.backend.zeros((num_samples, len(sample_core_names)))
        if isinstance(samples, TNTensor):
            samples = samples.tensor
        # Samples are always real (continuous x values).
        if hasattr(samples, 'is_complex') and samples.is_complex():
            samples = samples.real

        for i, core_name in enumerate(tqdm(
            sample_core_names, desc="Sampling"
        )):
            # A: Generate Mx on grid → (G, K, K)
            grid_x_input = self.backend.unsqueeze(grid_x, 1)  # (G, 1)
            mx_list_grid, _ = data_generator.generate(
                grid_x_input, K=K, ret_type='TNTensor'
            )
            Mx_grid = mx_list_grid[0]  # TNTensor (G, K, K)
            raw_grid = Mx_grid.tensor if isinstance(Mx_grid, TNTensor) else Mx_grid

            # B: Expand grid to (S*G, K, K) and set as current core.
            m = self.backend.unsqueeze(raw_grid, 0)               # (1, G, K, K)
            m = self.backend.expand(m, num_samples, -1, -1, -1)   # (S, G, K, K)
            m = self.backend.reshape(m, (num_samples * grid_size, K, K))
            qctn[core_name] = TNTensor(m, has_batch=True)

            # C: Expand all OTHER sample cores to match batch S*G.
            saved_cores: Dict[str, Any] = {}
            for other_name in sample_core_names:
                if other_name == core_name:
                    continue
                other = qctn[other_name]
                raw_other = other.tensor if isinstance(other, TNTensor) else other
                saved_cores[other_name] = other  # save for restore

                if raw_other.ndim == 2:
                    # Unbatched (K, K) → expand to (S*G, K, K)
                    expanded = self.backend.unsqueeze(raw_other, 0)
                    expanded = self.backend.expand(
                        expanded, num_samples * grid_size, -1, -1
                    )
                elif raw_other.shape[0] == num_samples:
                    # Batched (S, K, K) → expand to (S, G, K, K) → (S*G, K, K)
                    expanded = self.backend.unsqueeze(raw_other, 1)
                    expanded = self.backend.expand(expanded, -1, grid_size, -1, -1)
                    expanded = self.backend.reshape(
                        expanded, (num_samples * grid_size, K, K)
                    )
                else:
                    expanded = raw_other

                qctn[other_name] = TNTensor(expanded, has_batch=True)

            # D: Contract.
            result = self.contract(qctn)

            # E: Restore non-grid cores to their saved values.
            for other_name, saved in saved_cores.items():
                qctn[other_name] = saved

            # F: Extract density → CDF → inverse CDF sample.
            density = self._result_to_probability_tensor(result).reshape(
                num_samples, grid_size
            )
            density = density.clamp(min=0.0)

            cdf = density.cumsum(dim=1)
            total = cdf[:, -1].unsqueeze(1)
            cdf = cdf / (total + 1e-10)

            u = self.backend.rand(
                (num_samples, 1), dtype=self.backend.torch.float32
            )
            if isinstance(u, TNTensor):
                u = u.tensor

            mask = (cdf < u).float()
            indices = mask.sum(dim=1).long().clamp(max=grid_size - 2).unsqueeze(1)
            indices_next = indices + 1

            cdf_L = cdf.gather(1, indices)
            cdf_R = cdf.gather(1, indices_next)

            grid_raw = grid_x.tensor if isinstance(grid_x, TNTensor) else grid_x
            grid_exp = grid_raw.unsqueeze(0).expand(num_samples, -1)

            x_L = grid_exp.gather(1, indices)
            x_R = grid_exp.gather(1, indices_next)

            frac = (u - cdf_L) / (cdf_R - cdf_L + 1e-10)
            sampled_y = x_L + frac * (x_R - x_L)

            samples[:, i] = sampled_y.squeeze(1)

            # G: Update current core with sampled Mx (S, K, K).
            if use_marginal:
                qctn[core_name] = ident_tt
            else:
                mx_list_y, _ = data_generator.generate(
                    sampled_y, K=K, ret_type='TNTensor'
                )
                qctn[core_name] = mx_list_y[0]

        self._restore_cores(qctn, initial_cores)
        return samples

    def sample_discrete(
        self,
        qctn,
        data_generator,
        sample_core_names: List[str],
        num_samples: int,
        values=None,
        use_marginal: bool = False,
    ):
        """Sample discrete values by enumerating candidate measurement cores."""
        import numpy as np

        initial_cores = self._snapshot_cores(qctn, sample_core_names)

        if values is None:
            values = getattr(data_generator, "values", None)
        if values is None:
            raise ValueError("values must be provided or available on data_generator")
        values = tuple(values)

        first_core = qctn[sample_core_names[0]]
        raw_first = first_core.tensor if isinstance(first_core, TNTensor) else first_core
        K = raw_first.shape[-1]
        ident_tt = self._identity_for_core(qctn, sample_core_names[0])

        for name in sample_core_names:
            qctn[name] = ident_tt

        samples_np = np.zeros((num_samples, len(sample_core_names)), dtype=np.asarray(values).dtype)

        for i, core_name in enumerate(tqdm(sample_core_names, desc="Discrete sampling")):
            prob_columns = []
            current = qctn[core_name]
            saved_current = current.clone() if isinstance(current, TNTensor) else current

            for value in values:
                x_value = np.full((num_samples, 1), value)
                mx_list, _ = data_generator.generate(x_value, K=K, ret_type='TNTensor')
                qctn[core_name] = mx_list[0]

                saved_cores: Dict[str, Any] = {}
                for other_name in sample_core_names:
                    if other_name == core_name:
                        continue
                    other = qctn[other_name]
                    raw_other = other.tensor if isinstance(other, TNTensor) else other
                    saved_cores[other_name] = other.clone() if isinstance(other, TNTensor) else other

                    if raw_other.ndim == 2:
                        expanded = self.backend.unsqueeze(raw_other, 0)
                        expanded = self.backend.expand(expanded, num_samples, -1, -1)
                        qctn[other_name] = TNTensor(expanded, has_batch=True)

                result = self.contract(qctn)

                for other_name, saved in saved_cores.items():
                    qctn[other_name] = saved

                result_raw = self._result_to_probability_tensor(result)
                result_raw = result_raw.clamp(min=0.0)
                prob_columns.append(np.asarray(self.backend.tensor_to_numpy(result_raw)).reshape(num_samples))

            qctn[core_name] = saved_current

            probs = np.stack(prob_columns, axis=1)
            probs = np.clip(probs, 0.0, None)
            totals = probs.sum(axis=1, keepdims=True)
            probs = probs / np.maximum(totals, 1e-12)

            draws = [
                np.random.choice(len(values), p=probs[row])
                for row in range(num_samples)
            ]
            sampled_values = np.asarray([values[idx] for idx in draws])
            samples_np[:, i] = sampled_values

            if use_marginal:
                qctn[core_name] = ident_tt
            else:
                mx_list, _ = data_generator.generate(
                    sampled_values.reshape(num_samples, 1),
                    K=K,
                    ret_type='TNTensor',
                )
                qctn[core_name] = mx_list[0]

        samples = self.backend.convert_to_tensor(samples_np)
        raw_samples = samples.tensor if isinstance(samples, TNTensor) else samples
        if hasattr(raw_samples, 'is_complex') and raw_samples.is_complex():
            raw_samples = raw_samples.real
        self._restore_cores(qctn, initial_cores)
        return raw_samples
