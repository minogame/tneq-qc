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
        self._submodules: dict = {}

        if not _defer_init and backend is not None:
            self._init_cores()

    def __repr__(self):
        """String representation of the QCTN object."""
        circuit_input = [str(info['input_shape']) for info in self.adjacency_table]
        circuit_output = [str(info['output_shape']) for info in self.adjacency_table]

        return (
            f"circuit_input = \n{circuit_input}\n"
            f" adjacency_table = \n{self.adjacency_table}\n"
            f" circuit_output = \n{circuit_output}\n"
        )

    def __str__(self) -> str:
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

        for core_name in self.cores[:split_idx]:
            if core_name in self.cores_weights:
                qctn1.cores_weights[core_name] = self.cores_weights[core_name]
        for core_name in self.cores[split_idx:]:
            if core_name in self.cores_weights:
                qctn2.cores_weights[core_name] = self.cores_weights[core_name]

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
    def concat(qctn1, qctn2):
        """Left-right merge of two QCTNs into a single new QCTN (static method).

        The merged QCTN places *qctn1*'s graph on the left and *qctn2*'s
        graph on the right, concatenating each qubit line horizontally.

        Rules:

        1. The resulting number of qubits is ``max(qctn1.nqubits, qctn2.nqubits)``.
        2. The QCTN with fewer qubits is padded at the bottom with
           dash-only lines so that both sides have the same number of rows.
        3. The right boundary (``-dim-`` at end) of *qctn1* and the left
           boundary (``-dim-`` at start) of *qctn2* overlap – only one
           copy is kept.  The boundary from the QCTN that originally has
           **more qubits** is preserved (if equal, *qctn1*'s is kept).
        4. Core tensors are renamed contiguously via
           ``opt_einsum.get_symbol(0, 1, 2, …)``.

        Args:
            qctn1 (QCTN): Left QCTN.
            qctn2 (QCTN): Right QCTN.

        Returns:
            QCTN: A new merged QCTN with renamed cores and copied weights.
        """
        return QCTN._concat_impl(qctn1, qctn2)

    @staticmethod
    def _concat_impl(qctn1, qctn2):
        """Internal implementation shared by concat() and merge()."""
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

        for old_name, new_name in core_map1.items():
            if old_name in qctn1.cores_weights:
                new_qctn.cores_weights[new_name] = qctn1.cores_weights[old_name]
        for old_name, new_name in core_map2.items():
            if old_name in qctn2.cores_weights:
                new_qctn.cores_weights[new_name] = qctn2.cores_weights[old_name]

        return new_qctn

    @staticmethod
    def merge(qctn1, qctn2):
        """.. deprecated:: Use :meth:`concat` instead."""
        warnings.warn(
            "QCTN.merge() is deprecated, use QCTN.concat() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return QCTN.concat(qctn1, qctn2)

    def concat_with(self, other):
        """Merge *self* with *other* and return a new QCTN.

        Equivalent to ``QCTN.concat(self, other)``.  The result has
        *self*'s cores first (preserving relative order), followed by
        *other*'s cores, with all core names reassigned contiguously.

        Args:
            other (QCTN): Another QCTN to merge with.

        Returns:
            QCTN: A new merged QCTN.
        """
        return QCTN.concat(self, other)

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

