import warnings
import re
import opt_einsum
from .tn_tensor import TNTensor
from .tn_graph import TNGraph
from ..utils.graph_generators import QCTNHelper

from ._qctn_graph import QCTNGraphMixin
from ._qctn_io import QCTNIOMixin
from ._qctn_contractor import QCTNContractorMixin, TensorSide  # TensorSide re-exported here

# Public re-export so ``from tneq_qc.core.qctn import TensorSide`` keeps working
__all__ = ["QCTN", "TensorSide"]

_FULL_CORES = tuple(opt_einsum.get_symbol(i) for i in range(10000))
_FULL_CORE_SET = set(_FULL_CORES)
_CORE2IDX = {sym: idx for idx, sym in enumerate(_FULL_CORES)}


def _preprocess_graph_string(graph: str):
    """Inject fixed identity cores for empty qubit lines."""
    if graph == "":
        return graph, {}

    raw_lines = graph.splitlines()
    if not raw_lines:
        return graph, {}

    used_symbols = {ch for ch in graph if ch in _FULL_CORE_SET}
    next_symbol_idx = 0
    injected_identity_cores = {}
    processed_lines = []

    for qubit_idx, raw_line in enumerate(raw_lines):
        stripped = raw_line.strip()
        has_core = any(ch in _FULL_CORE_SET for ch in stripped)
        if has_core:
            processed_lines.append(stripped)
            continue

        while _FULL_CORES[next_symbol_idx] in used_symbols:
            next_symbol_idx += 1
        sym = _FULL_CORES[next_symbol_idx]
        used_symbols.add(sym)
        next_symbol_idx += 1

        processed_lines.append(f"-2-{sym}-2-")
        injected_identity_cores[sym] = qubit_idx

    return "\n".join(processed_lines), injected_identity_cores


class QCTN(QCTNGraphMixin, QCTNIOMixin, QCTNContractorMixin):
    """
    Quantum Circuit Tensor Network (QCTN) class for quantum circuit simulation.

    Initialization Format:
        - A graph representing the quantum circuit, where open edges are qubits and marks are cores.
        - Each core is a tensor with a shape corresponding to the number of qubits it connects to.

    Example:
        -2-----B-5-C-3-D-----2-
        -2-A-4---------D-----2-
        -2-A-4-B-7-C-2-D-4-E-2-
        -2-A-3-B-6---------E-2-
        -2---------C-8-----E-2-

        where:
            - A, B, C, D, E are cores (tensors).
            - The numbers represent the rank of each core.

    Attributes:
        nqubits (int): Number of qubits in the quantum circuit.
        adjacency_table (list[dict]): Per-core adjacency info with in/out edges, shapes, and dims.

    """

    def __init__(self, graph=None, backend=None, *, _defer_init=False):
        """
        Initialize the QCTN with a quantum circuit graph.

        Args:
            graph (str | None): ASCII graph string defining the tensor network
                topology. Pass ``None`` to create a composite module with no
                own core tensors (submodules are registered via
                :meth:`register_module`).
            backend: Compute backend instance (e.g. BackendPyTorch). Required
                for :meth:`auto_init`; may be ``None`` for structure-only use.
            _defer_init (bool): Internal keyword-only flag. When ``True``,
                skip the automatic :meth:`_init_cores` call even when
                *backend* is provided.
        """
        # ---- Composite mode: no graph, act as a pure container ----
        if graph is None:
            self.qubits = []
            self.nqubits = 0
            self.qubit_indices = []
            self.graph = None
            self._source_graph = None
            self.tn_graph = None
            self.cores = []
            self.ncores = 0
            self.adjacency_table = []
            self.backend = backend
            self._loaded_metadata = None
            self.cores_weights: dict = {}
            self.core_names: dict = {}
            self.trace_qubits: set = set()
            self._submodules: dict = {}
            self._fixed_identity_cores: dict = {}
            self._core_batch_size = None
            return

        processed_graph, injected_identity_cores = _preprocess_graph_string(graph)

        # ---- Normal graph-based mode ----
        self.qubits = processed_graph.splitlines() if processed_graph else []
        self.nqubits = len(self.qubits)
        self.qubit_indices = list(range(self.nqubits))

        self.graph = processed_graph
        self._source_graph = graph
        self._fixed_identity_cores = injected_identity_cores
        self.tn_graph = TNGraph(processed_graph, self.nqubits)

        self.cores = sorted(
            set(c for c in processed_graph if c in _FULL_CORE_SET),
            key=lambda x: _CORE2IDX[x],
        )
        self.ncores = len(self.cores)

        self.adjacency_table = []
        self._circuit_to_adjacency()

        self.backend = backend
        self._loaded_metadata = None
        self.cores_weights: dict = {}
        self.core_names: dict = {
            s: (f"identity.q{injected_identity_cores[s]}" if s in injected_identity_cores else s)
            for s in self.cores
        }
        self.trace_qubits: set = set()
        self._submodules: dict = {}
        self._core_batch_size = None

        if not _defer_init and backend is not None:
            self._init_cores()

    def __repr__(self):
        """Concise summary showing cores with readable names and shapes."""
        if not self.cores:
            if self._submodules:
                return f"QCTN(composite, submodules={list(self._submodules.keys())})"
            return "QCTN(empty)"

        names = getattr(self, 'core_names', {})
        parts = []
        for sym in self.cores:
            name = names.get(sym, sym)
            w = self.cores_weights.get(sym)
            if w is not None:
                shape = tuple(w.shape)
                label = f"{name}{shape}" if name != sym else f"{sym}{shape}"
                if name != sym:
                    label = f"{name}{shape}"
            else:
                label = f"{name}(?)"
            parts.append(label)
        cores_str = ", ".join(parts)
        return f"QCTN(nqubits={self.nqubits}, cores=[{cores_str}])"

    def __getitem__(self, key: str):
        """Access a core tensor by readable name or symbol.

        Args:
            key: Readable name (e.g. ``'mx.a'``) or einsum symbol.

        Returns:
            Core tensor (TNTensor or raw tensor).

        Raises:
            KeyError: If no matching core is found.
        """
        # Direct symbol lookup.
        if key in self.cores_weights:
            return self.cores_weights[key]
        # Readable name lookup.
        return self.get_core_by_name(key)

    def _symbol_for_name(self, name: str) -> str:
        """Return the einsum symbol for a readable name.

        Raises:
            KeyError: If no core with that name exists.
        """
        names = getattr(self, 'core_names', {})
        for sym, n in names.items():
            if n == name:
                return sym
        raise KeyError(f"No core with name {name!r}")

    def __setitem__(self, key: str, value):
        """Set a core tensor by readable name or symbol.

        If *value* is a ``TNTensor``, uses inplace ``set()`` to preserve
        Python object identity (so hermit views remain valid). Otherwise
        replaces the entry in ``cores_weights`` directly.

        Args:
            key: Readable name (e.g. ``'mx.a'``) or einsum symbol.
            value: New tensor (TNTensor or raw tensor).
        """
        if key in self.cores_weights:
            sym = key
        else:
            sym = self._symbol_for_name(key)

        existing = self.cores_weights[sym]
        if isinstance(existing, TNTensor) and existing.is_fixed:
            warnings.warn(
                f"Core '{self.core_names.get(sym, sym)}' is fixed ({existing.fixed_kind}) and cannot be overwritten; ignoring assignment.",
                stacklevel=2,
            )
            return

        if isinstance(existing, TNTensor) and isinstance(value, TNTensor):
            existing.set(value.tensor, value.scale, has_batch=value.has_batch)
        else:
            self.cores_weights[sym] = value

    def to_graph_string(self) -> str:
        """Render the current adjacency table back into an ASCII graph string."""
        if not self.qubits:
            return f"QCTN(composite, submodules={list(self._submodules.keys())})"
        if not self.cores:
            return "QCTN(empty)"

        def is_boundary(neighbor_name):
            return neighbor_name is None or neighbor_name == ''

        core_map = {info['core_name']: info for info in self.adjacency_table}

        lines = []
        for qubit_idx in range(self.nqubits):
            cores_on_qubit = []
            for core_info in self.adjacency_table:
                has_in_edge = any(e['qubit_idx'] == qubit_idx for e in core_info['in_edge_list'])
                has_out_edge = any(e['qubit_idx'] == qubit_idx for e in core_info['out_edge_list'])
                if has_in_edge or has_out_edge:
                    cores_on_qubit.append(core_info['core_name'])

            if not cores_on_qubit:
                lines.append("-")
                continue

            start_cores = []
            for core_name in cores_on_qubit:
                core_info = core_map[core_name]
                has_in_from_boundary = any(
                    e['qubit_idx'] == qubit_idx and is_boundary(e['neighbor_name'])
                    for e in core_info['in_edge_list']
                )
                has_in_edge = any(e['qubit_idx'] == qubit_idx for e in core_info['in_edge_list'])
                has_out_edge = any(e['qubit_idx'] == qubit_idx for e in core_info['out_edge_list'])
                if has_in_from_boundary or (not has_in_edge and has_out_edge):
                    start_cores.append(core_name)

            end_cores = []
            for core_name in cores_on_qubit:
                core_info = core_map[core_name]
                has_out_to_boundary = any(
                    e['qubit_idx'] == qubit_idx and is_boundary(e['neighbor_name'])
                    for e in core_info['out_edge_list']
                )
                has_in_edge = any(e['qubit_idx'] == qubit_idx for e in core_info['in_edge_list'])
                has_out_edge = any(e['qubit_idx'] == qubit_idx for e in core_info['out_edge_list'])
                if has_out_to_boundary or (has_in_edge and not has_out_edge):
                    end_cores.append(core_name)

            if len(start_cores) == 0:
                print(f"WARNING: No start core found on qubit {qubit_idx}")
                lines.append("-")
                continue
            if len(start_cores) > 1:
                print(f"WARNING: Multiple start cores found on qubit {qubit_idx}: {start_cores}")

            if len(end_cores) == 0:
                print(f"WARNING: No end core found on qubit {qubit_idx}")
                lines.append("-")
                continue
            if len(end_cores) > 1:
                print(f"WARNING: Multiple end cores found on qubit {qubit_idx}: {end_cores}")

            start_core = start_cores[0]
            end_core = end_cores[0]

            parts = []
            current_core = start_core

            core_info = core_map[current_core]
            left_dim = None
            for in_edge in core_info['in_edge_list']:
                if in_edge['qubit_idx'] == qubit_idx and is_boundary(in_edge['neighbor_name']):
                    left_dim = in_edge['edge_rank']
                    break

            if left_dim is not None:
                parts.append(f"-{left_dim}-")
            else:
                parts.append("-")

            visited = set()
            while current_core is not None:
                if current_core in visited:
                    print(f"WARNING: Circular reference detected on qubit {qubit_idx} at core {current_core}")
                    break
                visited.add(current_core)

                parts.append(current_core)

                if current_core == end_core:
                    core_info = core_map[current_core]
                    right_dim = None
                    for out_edge in core_info['out_edge_list']:
                        if out_edge['qubit_idx'] == qubit_idx and is_boundary(out_edge['neighbor_name']):
                            right_dim = out_edge['edge_rank']
                            break

                    if right_dim is not None:
                        parts.append(f"-{right_dim}-")
                    else:
                        parts.append("-")
                    break

                core_info = core_map[current_core]
                next_core = None
                connection_dim = None
                for out_edge in core_info['out_edge_list']:
                    if out_edge['qubit_idx'] == qubit_idx and not is_boundary(out_edge['neighbor_name']):
                        next_core = out_edge['neighbor_name']
                        connection_dim = out_edge['edge_rank']
                        break

                if next_core is None:
                    print(f"WARNING: Cannot form chain on qubit {qubit_idx}: {current_core} has no next core but end_core is {end_core}")
                    break

                next_core_info = core_map[next_core]
                valid_connection = False
                for in_edge in next_core_info['in_edge_list']:
                    if in_edge['qubit_idx'] == qubit_idx and in_edge['neighbor_name'] == current_core:
                        if in_edge['edge_rank'] != connection_dim:
                            print(f"WARNING: Dimension mismatch on qubit {qubit_idx} between {current_core} and {next_core}")
                        valid_connection = True
                        break

                if not valid_connection:
                    print(f"WARNING: Invalid connection on qubit {qubit_idx}: {current_core} -> {next_core}")

                if connection_dim is not None:
                    parts.append(f"-{connection_dim}-")
                else:
                    parts.append("-")

                current_core = next_core

            lines.append(''.join(parts))

        return '\n'.join(lines)
    # ================================================================
    # Parameter collection (Phase 3.0)
    # ================================================================

    def parameters(self):
        """Return all trainable leaf TNTensor parameters.

        Collects cores where ``requires_grad=True`` and ``is_leaf=True``.
        Non-leaf derived tensors (e.g. conj/permute views from
        :meth:`hermit`) are skipped — autograd propagates their gradients
        back to the originating leaf tensors automatically.

        Returns:
            list[TNTensor]: Trainable parameters in ``self.cores`` order.
        """
        result = []
        for c_name in self.cores:
            c = self.cores_weights.get(c_name)
            if c is None:
                continue
            if not c.requires_grad:
                continue
            if not c.is_leaf:
                continue
            result.append(c)
        return result

    def named_parameters(self):
        """Return ``(name, tensor)`` pairs for all trainable leaf parameters.

        Uses the readable name from ``core_names`` when available,
        otherwise falls back to the einsum symbol.

        Returns:
            list[tuple[str, TNTensor]]: ``(name, tensor)`` pairs.
        """
        names = getattr(self, 'core_names', {})
        result = []
        for c_name in self.cores:
            c = self.cores_weights.get(c_name)
            if c is None:
                continue
            if not c.requires_grad:
                continue
            if not c.is_leaf:
                continue
            readable = names.get(c_name, c_name)
            result.append((readable, c))
        return result

    def requires_grad_(self, requires_grad=True):
        """Set requires_grad on all core tensors.

        Args:
            requires_grad: Whether to enable gradient tracking.

        Returns:
            self (for chaining).
        """
        for t in self.cores_weights.values():
            t.requires_grad_(requires_grad)
        return self

    @staticmethod
    def _batch_unsqueeze(raw):
        if hasattr(raw, 'unsqueeze'):
            return raw.unsqueeze(0)
        try:
            import jax.numpy as jnp
            return jnp.expand_dims(raw, axis=0)
        except ImportError:
            import numpy as np
            return np.expand_dims(raw, axis=0)

    @staticmethod
    def _batch_tile(raw, reps: int):
        if hasattr(raw, 'is_leaf') or hasattr(raw, 'is_cuda'):
            return raw.repeat(reps, *([1] * (raw.ndim - 1)))
        try:
            import jax.numpy as jnp
            return jnp.tile(raw, (reps,) + (1,) * (raw.ndim - 1))
        except ImportError:
            import numpy as np
            return np.tile(raw, (reps,) + (1,) * (raw.ndim - 1))

    def add_core_batch_size(self, batch_size: int):
        """Declare and, when possible, materialize a batch axis for all cores."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        self._core_batch_size = int(batch_size)

        for core_name in self.cores:
            tensor = self.cores_weights.get(core_name)
            if tensor is None:
                continue
            if not isinstance(tensor, TNTensor):
                tensor = TNTensor(tensor)

            raw = tensor.tensor.detach() if hasattr(tensor.tensor, 'detach') else tensor.tensor
            raw = raw.clone() if hasattr(raw, 'clone') else raw
            requires_grad = tensor.requires_grad and not tensor.is_fixed

            if tensor.has_batch:
                current_batch = int(raw.shape[0])
                if current_batch == batch_size:
                    new_raw = raw
                elif current_batch > batch_size:
                    new_raw = raw[:batch_size]
                else:
                    reps = (batch_size + current_batch - 1) // current_batch
                    new_raw = self._batch_tile(raw, reps)[:batch_size]
            else:
                expanded = self._batch_unsqueeze(raw)
                new_raw = self._batch_tile(expanded, batch_size)

            new_tensor = TNTensor(
                new_raw,
                scale=tensor.scale,
                has_batch=True,
                is_fixed=tensor.is_fixed,
                fixed_kind=tensor.fixed_kind,
            )
            if requires_grad:
                new_tensor.requires_grad_(True)
            self.cores_weights[core_name] = new_tensor

        for submodule in self._submodules.values():
            if hasattr(submodule, 'add_core_batch_size'):
                submodule.add_core_batch_size(batch_size)
        return self

    def set_core_batch_size(self, batch_size: int):
        """Backward-compatible alias for :meth:`add_core_batch_size`."""
        return self.add_core_batch_size(batch_size)

    def bra(self):
        """Return bra (conjugate) version of this QCTN.

        Creates a new QCTN with the bra graph topology (input-only edges)
        where each core is the complex conjugate of the original.

        Returns:
            QCTN: New QCTN suitable for closing the right boundary.
        """
        nqubits = self.nqubits
        phys_dims = []
        for entry in self.adjacency_table:
            out_edges = entry.get('out_edge_list', [])
            if out_edges:
                phys_dims.append(out_edges[0]['edge_rank'])
            else:
                in_edges = entry.get('in_edge_list', [])
                if in_edges:
                    phys_dims.append(in_edges[0]['edge_rank'])
                else:
                    phys_dims.append(2)

        if len(set(phys_dims)) == 1:
            phys_dim = phys_dims[0]
        else:
            phys_dim = phys_dims[0]

        from ..utils.graph_generators import QCTNHelper
        bra_graph = QCTNHelper.state_bra(nqubits, phys_dim=phys_dim)
        bra_qctn = QCTN(bra_graph, backend=self.backend)

        for c_name in self.cores:
            t = self.cores_weights.get(c_name)
            if t is None:
                continue
            if c_name in bra_qctn.cores_weights:
                if isinstance(t, TNTensor):
                    bra_qctn.cores_weights[c_name] = TNTensor(
                        t.tensor.conj(), scale=t.scale)
                else:
                    bra_qctn.cores_weights[c_name] = t.conj()

        return bra_qctn

    # ================================================================
    # nn.Module-style interface
    # ================================================================

    def define(self):
        """Define the tensor network topology.

        Subclasses should override this method to declare the network
        structure (e.g. graph string, sub-module relationships).  The
        default implementation returns the parsed ``tn_graph``.

        Returns:
            TNGraph: The tensor network graph for this QCTN.
        """
        return self.tn_graph

    def forward(self, *args, **kwargs):
        """Execute the forward computation.

        Subclasses should override this method to implement custom
        computation logic.  The base implementation raises
        ``NotImplementedError`` to remind users to either use the
        Engine/Strategy pipeline or provide a subclass override.
        """
        raise NotImplementedError(
            "QCTN.forward() is not implemented in the base class. "
            "Use Engine + ContractionStrategy for contraction, or "
            "override forward() in a subclass."
        )

    # ================================================================
    # Module management (R2)
    # ================================================================

    @classmethod
    def from_graph(cls, graph_str: str, backend=None) -> "QCTN":
        """Create a QCTN from an ASCII graph string.

        Convenience classmethod that wraps the constructor.

        Args:
            graph_str: ASCII graph string defining the tensor network topology.
            backend: Optional compute backend; if None the default backend is used.

        Returns:
            QCTN: New QCTN instance.
        """
        return cls(graph_str, backend=backend)

    def register_module(self, name: str, module: "QCTN") -> None:
        """Register *module* as a named sub-QCTN.

        The sub-module's cores are accessible via ``named_cores()`` with the
        prefix ``"<name>."`` prepended to each core name.

        Args:
            name: Attribute name under which the sub-module will be stored.
            module: A QCTN instance to register.

        Raises:
            TypeError: If *module* is not a QCTN instance.
            ValueError: If *name* is empty or contains a dot.
        """
        if not isinstance(module, QCTN):
            raise TypeError(
                f"register_module: expected a QCTN instance, got {type(module).__name__}"
            )
        if not name or "." in name:
            raise ValueError(
                f"register_module: name must be a non-empty string without dots, got {name!r}"
            )
        self._submodules[name] = module

    def named_cores(self, prefix: str = ""):
        """Iterate over ``(name, tensor)`` pairs for all core tensors.

        Yields own cores first (in ``self.cores`` order), then recursively
        yields sub-module cores with ``"<submodule_name>."`` prefixed.

        Args:
            prefix: String prepended to every yielded name (used internally
                by recursive calls).

        Yields:
            tuple[str, Any]: ``(full_name, core_tensor)`` pairs.
        """
        for core_name in self.cores:
            full_name = f"{prefix}{core_name}" if prefix else core_name
            yield full_name, self.cores_weights[core_name]
        for mod_name, sub in self._submodules.items():
            sub_prefix = f"{prefix}{mod_name}." if prefix else f"{mod_name}."
            yield from sub.named_cores(prefix=sub_prefix)

    @property
    def all_cores(self) -> dict:
        """All core tensors including sub-modules, keyed by their full names.

        Own cores use their bare names; sub-module cores use
        ``"<submodule_name>.<core_name>"`` as keys.

        Returns:
            dict[str, Any]: Flat mapping of full core name → tensor.
        """
        return dict(self.named_cores())

    @property
    def named_weights(self) -> dict:
        """Core tensors keyed by readable name (from ``core_names``).

        Unlike ``cores_weights`` (keyed by einsum symbol), this dict uses
        the human-readable names assigned during ``concat``.

        Returns:
            dict[str, Any]: ``{readable_name: tensor}`` in core order.
        """
        names = getattr(self, 'core_names', {})
        return {
            names.get(sym, sym): self.cores_weights[sym]
            for sym in self.cores
            if sym in self.cores_weights
        }

    # ================================================================
    # Core name helpers
    # ================================================================

    def cores_by_prefix(self, prefix: str) -> dict:
        """Return cores whose readable name starts with ``prefix.``.

        Args:
            prefix: Name prefix to filter by (without trailing dot).

        Returns:
            dict: ``{symbol: tensor}`` for matching cores.
        """
        names = getattr(self, 'core_names', {})
        dot_prefix = f"{prefix}."
        return {
            sym: self.cores_weights[sym]
            for sym, name in names.items()
            if name.startswith(dot_prefix) or name == prefix
        }

    def get_core_by_name(self, name: str):
        """Look up a core tensor by its readable name.

        Args:
            name: Readable name (e.g. ``'mx.A'``).

        Returns:
            The core tensor (TNTensor or raw tensor).

        Raises:
            KeyError: If no core with that name exists.
        """
        names = getattr(self, 'core_names', {})
        for sym, n in names.items():
            if n == name:
                return self.cores_weights[sym]
        raise KeyError(f"No core with name {name!r}")

    # ================================================================
    # Trace operations
    # ================================================================

    def set_trace(self, qubit_indices='all'):
        """Mark qubits for trace (close boundary in/out edges).

        Only effective with ``RowPriorityStrategy``.

        Args:
            qubit_indices: ``'all'`` to trace every qubit, or a list of
                integer qubit indices.
        """
        if qubit_indices == 'all':
            self.trace_qubits = set(self.qubit_indices)
        else:
            self.trace_qubits = set(qubit_indices)
        # Invalidate compiled strategy caches.
        for attr in list(vars(self)):
            if attr.startswith('_compiled_strategy_'):
                delattr(self, attr)

    def clear_trace(self):
        """Remove all trace marks and invalidate strategy caches."""
        self.trace_qubits = set()
        for attr in list(vars(self)):
            if attr.startswith('_compiled_strategy_'):
                delattr(self, attr)

    # ================================================================
    # Chunk / Concat operations  (renamed from Split / Merge)
    # ================================================================

    def chunk(self, split_idx=None):
        """Split the QCTN into two QCTNs by core tensor index.

        Cores are divided into two groups:

        * **Group 1**: ``self.cores[:split_idx]``
        * **Group 2**: ``self.cores[split_idx:]``

        For each qubit line that contains cores from both groups, the bond
        dimension at the boundary becomes the output dimension for Group 1
        and the input dimension for Group 2.  Qubit lines that only
        contain cores from a single group are assigned entirely to that
        group's QCTN.

        Args:
            split_idx (int, optional): Index at which to split the core
                list.  Defaults to ``ncores // 2``.

        Returns:
            tuple[QCTN, QCTN]: Two new QCTN instances with the
            corresponding core weights copied.

        Raises:
            ValueError: If *split_idx* is out of range, or if cores from
                both groups are interleaved on any qubit line (i.e. a
                Group-1 core appears **after** a Group-2 core).
        """
        if split_idx is None:
            split_idx = self.ncores // 2

        if split_idx <= 0 or split_idx >= self.ncores:
            raise ValueError(
                f"split_idx must be between 1 and {self.ncores - 1}, "
                f"got {split_idx}"
            )

        cores_group1 = set(self.cores[:split_idx])
        cores_group2 = set(self.cores[split_idx:])

        lines_group1: list[str] = []
        lines_group2: list[str] = []

        for qubit_idx, line in enumerate(self.qubits):
            tokens = QCTN._parse_qubit_line(line)

            core_positions = [
                (i, tok[1])
                for i, tok in enumerate(tokens)
                if tok[0] == 'core'
            ]
            g1_pos = [(i, c) for i, c in core_positions if c in cores_group1]
            g2_pos = [(i, c) for i, c in core_positions if c in cores_group2]

            if g1_pos and g2_pos:
                last_g1 = max(i for i, _ in g1_pos)
                first_g2 = min(i for i, _ in g2_pos)

                if last_g1 >= first_g2:
                    raise ValueError(
                        f"Cannot chunk: cores from both groups are "
                        f"interleaved on qubit {qubit_idx}. Ensure that "
                        f"all Group-1 cores appear before Group-2 cores "
                        f"on every qubit line."
                    )

                g1_tokens = tokens[: last_g1 + 2]
                g2_tokens = tokens[first_g2 - 1 :]

                lines_group1.append(QCTN._rebuild_qubit_line(g1_tokens))
                lines_group2.append(QCTN._rebuild_qubit_line(g2_tokens))
            elif g1_pos:
                lines_group1.append(QCTN._rebuild_qubit_line(tokens))
            elif g2_pos:
                lines_group2.append(QCTN._rebuild_qubit_line(tokens))

        if not lines_group1:
            raise ValueError(
                "After chunk, Group 1 has no qubit lines. "
                "All qubits belong to Group 2."
            )
        if not lines_group2:
            raise ValueError(
                "After chunk, Group 2 has no qubit lines. "
                "All qubits belong to Group 1."
            )

        graph1 = "\n".join(lines_group1)
        graph2 = "\n".join(lines_group2)

        qctn1 = QCTN(graph1, backend=self.backend)
        qctn2 = QCTN(graph2, backend=self.backend)

        src_names = getattr(self, 'core_names', {})
        for core_name in self.cores[:split_idx]:
            if core_name in self.cores_weights:
                qctn1.cores_weights[core_name] = self.cores_weights[core_name]
            if core_name in src_names:
                qctn1.core_names[core_name] = src_names[core_name]
        for core_name in self.cores[split_idx:]:
            if core_name in self.cores_weights:
                qctn2.cores_weights[core_name] = self.cores_weights[core_name]
            if core_name in src_names:
                qctn2.core_names[core_name] = src_names[core_name]

        return qctn1, qctn2

    @staticmethod
    def concat(*args):
        """Left-right merge of multiple QCTNs into a single new QCTN (static method).

        The merged QCTN places QCTNs from left to right, concatenating each
        qubit line horizontally.

        Rules:

        1. The resulting number of qubits is ``max(qctn.nqubits for qctn in qctns)``.
        2. QCTNs with fewer qubits are padded at the bottom with
           dash-only lines so that all have the same number of rows.
        3. The right boundary (``-dim-`` at end) of one QCTN and the left
           boundary (``-dim-`` at start) of the next overlap – only one
           copy is kept.  The boundary from the QCTN that originally has
           **more qubits** is preserved (if equal, the left one is kept).
        4. Core tensors are renamed contiguously via
           ``opt_einsum.get_symbol(0, 1, 2, …)``.

        Each element can be a bare QCTN or a ``(name, qctn)`` tuple.  When
        names are provided, core names in the merged QCTN use
        ``"<name>.<original_name>"`` notation (similar to PyTorch modules).
        Duplicate prefixes are deduplicated by appending ``_1``, ``_2``, etc.

        Args:
            *args: Either a single list/tuple of QCTNs (or named tuples),
                or multiple QCTN arguments.

        Returns:
            QCTN: A new merged QCTN with renamed cores and copied weights.

        Examples:
            >>> # Bare QCTNs (backward-compatible)
            >>> result = QCTN.concat([q1, q2, q3])
            >>> # Named QCTNs
            >>> result = QCTN.concat([('mps', mps), ('mx', mx)])
        """
        # Parse arguments
        if len(args) == 0:
            raise ValueError("QCTN.concat() requires at least one argument")

        # If first arg is a list/tuple, use it as the list of QCTNs
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            items = args[0]
            # Distinguish list-of-QCTNs from a single (name, qctn) tuple
            if len(items) == 2 and isinstance(items[0], str) and isinstance(items[1], QCTN):
                # Single named tuple — wrap it
                items = [items]
        else:
            items = list(args)

        # Normalize: ensure each item is (prefix, qctn)
        named_qctns: list = []
        used_prefixes: dict = {}  # prefix → count for dedup
        for item in items:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                prefix, qctn_obj = item
            elif isinstance(item, QCTN):
                prefix, qctn_obj = None, item
            else:
                raise TypeError(
                    f"QCTN.concat: expected QCTN or (name, QCTN) tuple, got {type(item)}"
                )
            # Deduplicate prefix
            if prefix is not None:
                if prefix in used_prefixes:
                    used_prefixes[prefix] += 1
                    prefix = f"{prefix}_{used_prefixes[prefix]}"
                else:
                    used_prefixes[prefix] = 0
            named_qctns.append((prefix, qctn_obj))

        if len(named_qctns) == 0:
            raise ValueError("Cannot concat empty list of QCTNs")

        if len(named_qctns) == 1:
            return named_qctns[0][1]

        # Sequentially merge from left to right
        prefix_l, result = named_qctns[0]
        for prefix_r, qctn_r in named_qctns[1:]:
            result = QCTN._concat_impl(result, qctn_r, prefix_l, prefix_r)
            prefix_l = None  # already merged into result's core_names

        return result

    @staticmethod
    def _concat_impl(qctn1, qctn2, prefix1=None, prefix2=None):
        """Internal implementation shared by concat() and merge().

        Args:
            qctn1: Left QCTN.
            qctn2: Right QCTN.
            prefix1: Optional human-readable prefix for qctn1's cores.
                     ``None`` means reuse qctn1's existing ``core_names``.
            prefix2: Optional human-readable prefix for qctn2's cores.
                     ``None`` means reuse qctn2's existing ``core_names``.
        """
        import opt_einsum

        n1, n2 = qctn1.nqubits, qctn2.nqubits
        max_qubits = max(n1, n2)

        total_cores = qctn1.ncores + qctn2.ncores
        new_symbols = [opt_einsum.get_symbol(i) for i in range(total_cores)]

        core_map1 = {
            old: new_symbols[i] for i, old in enumerate(qctn1.cores)
        }
        core_map2 = {
            old: new_symbols[qctn1.ncores + i]
            for i, old in enumerate(qctn2.cores)
        }

        backend = qctn1.backend if qctn1.backend is not None else qctn2.backend
        new_qctn = QCTN(graph=None, backend=backend, _defer_init=True)
        new_qctn.nqubits = max_qubits
        new_qctn.qubit_indices = list(range(max_qubits))
        new_qctn.cores = list(new_symbols)
        new_qctn.ncores = total_cores
        new_qctn.trace_qubits = set(getattr(qctn1, 'trace_qubits', set())) | set(getattr(qctn2, 'trace_qubits', set()))
        new_qctn._submodules = {}
        new_qctn._loaded_metadata = None

        entry_map1 = {entry['core_name']: entry for entry in qctn1.adjacency_table}
        entry_map2 = {entry['core_name']: entry for entry in qctn2.adjacency_table}

        def _remap_edge(edge, core_map):
            new_edge = edge.copy()
            old_neighbor = edge.get('neighbor_name', '')
            if old_neighbor:
                new_edge['neighbor_name'] = core_map[old_neighbor]
            return new_edge

        adjacency_table = []
        for old_name in qctn1.cores:
            entry = entry_map1[old_name]
            adjacency_table.append({
                'core_idx': len(adjacency_table),
                'core_name': core_map1[old_name],
                'in_edge_list': [_remap_edge(edge, core_map1) for edge in entry['in_edge_list']],
                'out_edge_list': [_remap_edge(edge, core_map1) for edge in entry['out_edge_list']],
                'input_shape': list(entry['input_shape']),
                'output_shape': list(entry['output_shape']),
                'input_dim': entry['input_dim'],
                'output_dim': entry['output_dim'],
            })
        for old_name in qctn2.cores:
            entry = entry_map2[old_name]
            adjacency_table.append({
                'core_idx': len(adjacency_table),
                'core_name': core_map2[old_name],
                'in_edge_list': [_remap_edge(edge, core_map2) for edge in entry['in_edge_list']],
                'out_edge_list': [_remap_edge(edge, core_map2) for edge in entry['out_edge_list']],
                'input_shape': list(entry['input_shape']),
                'output_shape': list(entry['output_shape']),
                'input_dim': entry['input_dim'],
                'output_dim': entry['output_dim'],
            })

        left_entries = adjacency_table[:qctn1.ncores]
        right_entries = adjacency_table[qctn1.ncores:]

        def _find_boundary_edge(entries, qubit_idx, direction):
            for entry in entries:
                edge_list = entry[f'{direction}_edge_list']
                for edge in edge_list:
                    if edge['qubit_idx'] == qubit_idx and edge.get('neighbor_name', '') == '':
                        return entry, edge
            return None, None

        for qubit_idx in range(min(n1, n2)):
            left_entry, left_edge = _find_boundary_edge(left_entries, qubit_idx, 'out')
            right_entry, right_edge = _find_boundary_edge(right_entries, qubit_idx, 'in')

            if left_edge is None or right_edge is None:
                continue
            if left_edge['edge_rank'] != right_edge['edge_rank']:
                raise ValueError(
                    f"Cannot concat qubit {qubit_idx}: boundary rank mismatch "
                    f"{left_edge['edge_rank']} != {right_edge['edge_rank']}."
                )

            left_edge['neighbor_name'] = right_entry['core_name']
            right_edge['neighbor_name'] = left_entry['core_name']

        dict_core2idx = {core_name: idx for idx, core_name in enumerate(new_qctn.cores)}
        for entry in adjacency_table:
            entry['core_idx'] = dict_core2idx[entry['core_name']]
            for edge in entry['in_edge_list'] + entry['out_edge_list']:
                neighbor_name = edge.get('neighbor_name', '')
                edge['neighbor_idx'] = dict_core2idx[neighbor_name] if neighbor_name else -1
            entry['input_shape'] = [edge['edge_rank'] for edge in entry['in_edge_list']]
            entry['output_shape'] = [edge['edge_rank'] for edge in entry['out_edge_list']]
            in_dim = 1
            for rank in entry['input_shape']:
                in_dim *= rank
            out_dim = 1
            for rank in entry['output_shape']:
                out_dim *= rank
            entry['input_dim'] = in_dim
            entry['output_dim'] = out_dim

        new_qctn.adjacency_table = adjacency_table
        new_qctn.dict_core2idx = dict_core2idx

        new_qctn.cores_weights = {}
        for old_name, new_name in core_map1.items():
            if old_name in qctn1.cores_weights:
                new_qctn.cores_weights[new_name] = qctn1.cores_weights[old_name]
        for old_name, new_name in core_map2.items():
            if old_name in qctn2.cores_weights:
                new_qctn.cores_weights[new_name] = qctn2.cores_weights[old_name]

        new_qctn.core_names = {}
        q1_names = getattr(qctn1, 'core_names', {s: s for s in qctn1.cores})
        q2_names = getattr(qctn2, 'core_names', {s: s for s in qctn2.cores})
        for old, new in core_map1.items():
            orig_name = q1_names.get(old, old)
            new_qctn.core_names[new] = (
                f"{prefix1}.{orig_name}" if prefix1 is not None else orig_name
            )
        for old, new in core_map2.items():
            orig_name = q2_names.get(old, old)
            new_qctn.core_names[new] = (
                f"{prefix2}.{orig_name}" if prefix2 is not None else orig_name
            )

        remapped_fixed = {}
        for old, new in core_map1.items():
            if old in getattr(qctn1, '_fixed_identity_cores', {}):
                remapped_fixed[new] = qctn1._fixed_identity_cores[old]
        for old, new in core_map2.items():
            if old in getattr(qctn2, '_fixed_identity_cores', {}):
                remapped_fixed[new] = qctn2._fixed_identity_cores[old]
        new_qctn._fixed_identity_cores = remapped_fixed

        new_qctn.qubits = ['-'] * max_qubits
        new_graph = new_qctn.to_graph_string()
        new_qctn.graph = new_graph
        new_qctn._source_graph = new_graph
        new_qctn.qubits = new_graph.splitlines()
        new_qctn.tn_graph = TNGraph(new_graph, max_qubits)

        return new_qctn
    def concat_with(self, other):
        """Merge *self* with *other* and return a new QCTN.

        Equivalent to ``QCTN.concat([self, other])``.  The result has
        *self*'s cores first (preserving relative order), followed by
        *other*'s cores, with all core names reassigned contiguously.

        Args:
            other (QCTN): Another QCTN to merge with.

        Returns:
            QCTN: A new merged QCTN.
        """
        return QCTN.concat([self, other])

    # ================================================================
    # Phase 2: Reference semantics for siamese networks (R2-引用语义)
    # ================================================================

    def conjugate_transpose_cores(self) -> dict:
        """Return a dict of conj_transpose() references for all cores.

        For each core in ``self.cores_weights``, returns a
        :class:`~tneq_qc.core.tn_tensor.TNTensor` that is a zero-copy
        conjugate-transpose view of the original (``is_ref=True``,
        ``is_transposed=True``).

        This is used to implement the siamese right-side sharing pattern:
        the right-side core tensors in ``A · Mx · A†`` are derived from the
        same underlying data as the left-side cores, so gradients propagate
        correctly.

        Returns:
            dict[str, TNTensor]: ``{core_name: conj_transpose_view, ...}``
        """
        result: dict = {}
        for name in self.cores:
            tensor = self.cores_weights[name]
            if isinstance(tensor, TNTensor):
                result[name] = tensor.conj_transpose()
            else:
                result[name] = TNTensor(tensor).conj_transpose()
        return result

    # ================================================================
    # Phase 2.6.2: Clone and Hermitian conjugate operations
    # ================================================================

    def clone(self):
        """Return a cloned QCTN with independent core tensors.

        Creates a new QCTN with the same structure but with cloned core
        tensors. Each core tensor is independently trainable and maintains
        its own gradients during backpropagation.

        Returns:
            QCTN: New QCTN with cloned cores.

        Example:
            >>> q1 = QCTN(graph, backend)
            >>> q2 = q1.clone()
            >>> # q2 has independent cores, training updates them separately
        """
        cloned_qctn = QCTN(graph=None, backend=self.backend, _defer_init=True)

        # Copy structure
        cloned_qctn.qubits = self.qubits.copy()
        cloned_qctn.nqubits = self.nqubits
        cloned_qctn.qubit_indices = self.qubit_indices.copy()
        cloned_qctn.cores = self.cores.copy()
        cloned_qctn.ncores = self.ncores
        cloned_qctn.adjacency_table = [entry.copy() for entry in self.adjacency_table]
        cloned_qctn.graph = self.graph
        cloned_qctn.tn_graph = self.tn_graph

        # Clone each core tensor
        cloned_qctn.cores_weights = {}
        for core_name, tensor in self.cores_weights.items():
            if isinstance(tensor, TNTensor):
                cloned_qctn.cores_weights[core_name] = tensor.clone()
            else:
                # Wrap raw tensor first
                cloned_qctn.cores_weights[core_name] = TNTensor(tensor).clone()

        cloned_qctn.core_names = dict(getattr(self, 'core_names', {}))
        cloned_qctn.trace_qubits = set(getattr(self, 'trace_qubits', set()))
        cloned_qctn._fixed_identity_cores = dict(getattr(self, '_fixed_identity_cores', {}))
        cloned_qctn._source_graph = getattr(self, '_source_graph', self.graph)

        # Clone submodules
        cloned_qctn._submodules = {}
        for name, submodule in self._submodules.items():
            if hasattr(submodule, 'clone'):
                cloned_qctn._submodules[name] = submodule.clone()
            else:
                cloned_qctn._submodules[name] = submodule

        return cloned_qctn

    def hermit(self):
        """Return Hermitian conjugate of this QCTN.

        Creates a new QCTN where each core tensor is the Hermitian conjugate
        (conjugate transpose) of the original. During training, gradients
        flow back through the hermit transformation automatically via PyTorch's
        autograd mechanism.

        Returns:
            QCTN: New QCTN with Hermitian conjugate cores.

        Example:
            >>> q1 = QCTN(graph, backend)
            >>> q2 = q1.hermit()
            >>> # q2's cores are hermitian conjugates of q1's cores
            >>> # Gradients automatically transform during backprop
        """
        hermit_qctn = QCTN(graph=None, backend=self.backend, _defer_init=True)

        # Copy structure
        hermit_qctn.qubits = self.qubits.copy()
        hermit_qctn.nqubits = self.nqubits
        hermit_qctn.qubit_indices = self.qubit_indices.copy()
        hermit_qctn.cores = self.cores.copy()
        hermit_qctn.ncores = self.ncores
        hermit_qctn.graph = self.graph
        hermit_qctn._source_graph = getattr(self, '_source_graph', self.graph)
        hermit_qctn._fixed_identity_cores = dict(getattr(self, '_fixed_identity_cores', {}))
        hermit_qctn.tn_graph = self.tn_graph

        # Reverse adjacency_table: swap in_edge_list ↔ out_edge_list for every
        # core, then recompute the derived shape/dim fields.  This reflects that
        # Hermitian conjugation reverses all edge directions in the tensor network.
        hermit_table = []
        for entry in self.adjacency_table:
            new_in  = [e.copy() for e in entry['out_edge_list']]
            new_out = [e.copy() for e in entry['in_edge_list']]
            input_shape  = [e['edge_rank'] for e in new_in]
            output_shape = [e['edge_rank'] for e in new_out]
            in_dim  = 1
            for r in input_shape:
                in_dim *= r
            out_dim = 1
            for r in output_shape:
                out_dim *= r
            hermit_table.append({
                'core_idx':     entry['core_idx'],
                'core_name':    entry['core_name'],
                'in_edge_list':  new_in,
                'out_edge_list': new_out,
                'input_shape':   input_shape,
                'output_shape':  output_shape,
                'input_dim':  in_dim,
                'output_dim': out_dim,
            })
        hermit_qctn.adjacency_table = hermit_table

        # Apply hermit to each core tensor. Swap the full output block with
        # the full input block, not just the last two axes.
        hermit_qctn.cores_weights = {}
        entry_map = {entry['core_name']: entry for entry in self.adjacency_table}
        for core_name, tensor in self.cores_weights.items():
            entry = entry_map[core_name]
            n_in = len(entry['input_shape'])
            n_out = len(entry['output_shape'])
            offset = 1 if isinstance(tensor, TNTensor) and tensor.has_batch else 0
            axes = list(range(offset + n_in, offset + n_in + n_out)) + list(range(offset, offset + n_in))
            if offset:
                axes = [0] + axes

            if isinstance(tensor, TNTensor):
                hermit_qctn.cores_weights[core_name] = tensor.hermit(axes=axes)
            else:
                # Wrap raw tensor first
                hermit_qctn.cores_weights[core_name] = TNTensor(tensor).hermit(axes=axes)

        hermit_qctn.core_names = dict(getattr(self, 'core_names', {}))

        # Apply hermit to submodules
        hermit_qctn._submodules = {}
        for name, submodule in self._submodules.items():
            if hasattr(submodule, 'hermit'):
                hermit_qctn._submodules[name] = submodule.hermit()
            else:
                hermit_qctn._submodules[name] = submodule

        return hermit_qctn
