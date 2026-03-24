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
                topology.  Pass ``None`` to create a composite module with no
                own core tensors (submodules are registered via
                :meth:`register_module`).
            backend: Compute backend instance (e.g. BackendPyTorch).  Required
                for :meth:`auto_init`; may be ``None`` for structure-only use.
            _defer_init (bool): Internal keyword-only flag.  When ``True``,
                skip the automatic :meth:`_init_cores` call even when
                *backend* is provided.  Used by small-module subclasses that
                want to control initialization timing via :meth:`auto_init`.
        """
        # ---- Composite mode: no graph, act as a pure container ----
        if graph is None:
            self.qubits = []
            self.nqubits = 0
            self.qubit_indices = []
            self.graph = None
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
            return

        # ---- Normal graph-based mode ----
        self.qubits = graph.strip().splitlines()
        self.nqubits = len(self.qubits)
        self.qubit_indices = list(range(self.nqubits))

        self.graph = graph
        self.tn_graph = TNGraph(graph, self.nqubits)

        full_cores = set(opt_einsum.get_symbol(i) for i in range(10000))
        core2idx = {opt_einsum.get_symbol(i): i for i in range(10000)}
        self.cores = sorted(set(c for c in graph if c in full_cores), key=lambda x: core2idx[x])
        self.ncores = len(self.cores)

        self.adjacency_table = []
        self._circuit_to_adjacency()

        self.graph = graph
        self.backend = backend
        self._loaded_metadata = None
        self.cores_weights: dict = {}
        self.core_names: dict = {s: s for s in self.cores}
        self.trace_qubits: set = set()
        self._submodules: dict = {}

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
        Python object identity (so hermit views remain valid).  Otherwise
        replaces the entry in ``cores_weights`` directly.

        Args:
            key: Readable name (e.g. ``'mx.a'``) or einsum symbol.
            value: New tensor (TNTensor or raw tensor).
        """
        # Resolve to symbol.
        if key in self.cores_weights:
            sym = key
        else:
            sym = self._symbol_for_name(key)

        existing = self.cores_weights[sym]
        if isinstance(existing, TNTensor) and isinstance(value, TNTensor):
            existing.set(value.tensor, value.scale, has_batch=value.has_batch)
        else:
            self.cores_weights[sym] = value
        """Pretty-print tensor network structure based on adjacency_table.

        Uses linked-list logic to trace cores on each qubit from start to end,
        handling both boundary-connected and internal-only cores.

        Example output:
            -2-A-5-B-----3-
            -2-----B-6-C-2-
        """
        if not self.qubits:
            return f"QCTN(composite, submodules={list(self._submodules.keys())})"
        if not self.cores:
            return "QCTN(empty)"

        # Helper function to check if neighbor is boundary
        def is_boundary(neighbor_name):
            return neighbor_name is None or neighbor_name == ''

        # Build a map: core_name -> core_info for quick lookup
        core_map = {info['core_name']: info for info in self.adjacency_table}

        # Build output lines by tracing each qubit
        lines = []
        for qubit_idx in range(self.nqubits):
            # Find all cores that touch this qubit
            cores_on_qubit = []
            for core_info in self.adjacency_table:
                has_in_edge = any(e['qubit_idx'] == qubit_idx for e in core_info['in_edge_list'])
                has_out_edge = any(e['qubit_idx'] == qubit_idx for e in core_info['out_edge_list'])
                if has_in_edge or has_out_edge:
                    cores_on_qubit.append(core_info['core_name'])

            # If no cores on this qubit, output empty line
            if not cores_on_qubit:
                lines.append("-")
                continue

            # Find start cores (two cases):
            # 1. Has input from boundary on this qubit
            # 2. Has no input edge but has output edge on this qubit
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

            # Find end cores (two cases):
            # 1. Has output to boundary on this qubit
            # 2. Has input edge but no output edge on this qubit
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

            # Validate start and end cores
            if len(start_cores) == 0:
                print(f"WARNING: No start core found on qubit {qubit_idx}")
                lines.append("-")
                continue
            elif len(start_cores) > 1:
                print(f"WARNING: Multiple start cores found on qubit {qubit_idx}: {start_cores}")

            if len(end_cores) == 0:
                print(f"WARNING: No end core found on qubit {qubit_idx}")
                lines.append("-")
                continue
            elif len(end_cores) > 1:
                print(f"WARNING: Multiple end cores found on qubit {qubit_idx}: {end_cores}")

            # Use the first start and end cores
            start_core = start_cores[0]
            end_core = end_cores[0]

            # Trace the chain from start to end
            parts = []
            current_core = start_core

            # Get left boundary dimension
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

            # Trace the chain
            visited = set()
            while current_core is not None:
                if current_core in visited:
                    print(f"WARNING: Circular reference detected on qubit {qubit_idx} at core {current_core}")
                    break
                visited.add(current_core)

                parts.append(current_core)

                # If we reached the end core, finish
                if current_core == end_core:
                    # Get right boundary dimension
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

                # Find next core
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

                # Validate connection
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

                # Add connection dimension
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

        Only effective with ``RowPriorityStrategy`` (``strategy_mode='full'``
        or ``'balanced'``).

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

    def split(self, split_idx=None):
        """.. deprecated:: Use :meth:`chunk` instead."""
        warnings.warn(
            "QCTN.split() is deprecated, use QCTN.chunk() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.chunk(split_idx)

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

        remapped1 = QCTN._remap_graph(qctn1.qubits, core_map1)
        remapped2 = QCTN._remap_graph(qctn2.qubits, core_map2)

        pad_width1 = max(len(l) for l in remapped1) - 3
        pad_width2 = max(len(l) for l in remapped2) - 3

        new_lines = []
        for qi in range(max_qubits):
            has_l1 = qi < n1
            has_l2 = qi < n2

            l1 = remapped1[qi] if has_l1 else ("-" * pad_width1)
            l2 = remapped2[qi] if has_l2 else ("-" * pad_width2)

            m1 = re.search(r'-\d+-$', l1)
            dim_l1 = m1.group() if has_l1 else ""
            stripped_l1 = l1[:m1.start()] if has_l1 else l1

            m2 = re.match(r'^-\d+-', l2)
            dim_l2 = (m2.group() if m2 else "") if has_l2 else ""
            stripped_l2 = (l2[m2.end():] if m2 else l2) if has_l2 else l2

            if has_l1 and has_l2:
                merged = stripped_l1 + dim_l1 + stripped_l2
            elif has_l1:
                dim_l2 = '---'
                merged = stripped_l1 + stripped_l2 + dim_l1
            else:
                dim_l1 = '---'
                merged = dim_l2 + stripped_l1 + stripped_l2

            new_lines.append(merged)

        new_graph = "\n".join(new_lines)

        backend = qctn1.backend if qctn1.backend is not None else qctn2.backend
        new_qctn = QCTN(new_graph, backend=backend)

        # Copy weights (shallow — shares TNTensor references).
        for old_name, new_name in core_map1.items():
            if old_name in qctn1.cores_weights:
                new_qctn.cores_weights[new_name] = qctn1.cores_weights[old_name]
        for old_name, new_name in core_map2.items():
            if old_name in qctn2.cores_weights:
                new_qctn.cores_weights[new_name] = qctn2.cores_weights[old_name]

        # Build core_names: symbol → readable name.
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

        return new_qctn

    @staticmethod
    def merge(qctn1, qctn2):
        """.. deprecated:: Use :meth:`concat` instead."""
        warnings.warn(
            "QCTN.merge() is deprecated, use QCTN.concat([qctn1, qctn2]) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return QCTN.concat([qctn1, qctn2])

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

    def merge_with(self, other):
        """.. deprecated:: Use :meth:`concat_with` instead."""
        warnings.warn(
            "QCTN.merge_with() is deprecated, use QCTN.concat_with() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.concat_with(other)

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

        # Apply hermit to each core tensor
        hermit_qctn.cores_weights = {}
        for core_name, tensor in self.cores_weights.items():
            if isinstance(tensor, TNTensor):
                hermit_qctn.cores_weights[core_name] = tensor.hermit()
            else:
                # Wrap raw tensor first
                hermit_qctn.cores_weights[core_name] = TNTensor(tensor).hermit()

        hermit_qctn.core_names = dict(getattr(self, 'core_names', {}))

        # Apply hermit to submodules
        hermit_qctn._submodules = {}
        for name, submodule in self._submodules.items():
            if hasattr(submodule, 'hermit'):
                hermit_qctn._submodules[name] = submodule.hermit()
            else:
                hermit_qctn._submodules[name] = submodule

        return hermit_qctn


