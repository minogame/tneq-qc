import warnings
import numpy as np
import re
from pathlib import Path
from typing import Union, Optional, Mapping
from copy import deepcopy
from enum import Enum
from .tn_tensor import TNTensor
from .tn_graph import TNGraph
from ..utils.graph_generators import QCTNHelper


class TensorSide(Enum):
    """Enum to track tensor side in symmetric expansion: left, middle, or right."""
    LEFT = "L"
    MIDDLE = "M"
    RIGHT = "R"


class QCTN:
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
                ``register_module``).
            backend (ComputeBackend): The backend to use for computation.
            _defer_init (bool): If ``True``, skip automatic core-tensor
                initialization.  Subclasses that want deferred initialization
                (e.g. ``MPS``, ``CircuitState``) should pass
                ``_defer_init=True`` and call :meth:`auto_init` explicitly.
        """
        # ---- composite mode (no graph) ----
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
            self.cores_weights = {}
            self._submodules: dict = {}
            return

        # ---- graph-based mode ----
        self.qubits = graph.strip().splitlines()
        self.nqubits = len(self.qubits)
        self.qubit_indices = list(range(self.nqubits))

        self.graph = graph
        self.tn_graph = TNGraph(graph, self.nqubits)

        import opt_einsum
        idx2core = [opt_einsum.get_symbol(i) for i in range(10000)]
        core2idx = {c: i for i, c in enumerate(idx2core)}

        full_cores = set([opt_einsum.get_symbol(i) for i in range(10000)])

        self.cores = list(set([c for c in graph if c in full_cores]))
        self.cores.sort(key=lambda x: core2idx[x])

        self.ncores = len(self.cores)

        # Build adjacency_table from graph string
        self._circuit_to_adjacency()

        self.backend = backend
        self._loaded_metadata: Optional[Mapping[str, str]] = None

        self.cores_weights = {}
        if not _defer_init and backend is not None:
            self._init_cores()

        # Phase 2: submodule registry (nn.Module-style nesting)
        self._submodules: dict = {}

    @classmethod
    def envolve_from_another_qctn(cls, qctn, strategies=None):
        """Create a new QCTN instance by evolving from another QCTN instance.

        .. deprecated::
            This method will be moved to the ``genetic`` module in a future
            release.  Use ``genetic.evolve(qctn, strategies)`` instead.

        Args:
            qctn (QCTN): The original QCTN instance to evolve from.
            strategies (list, optional): Strategies for evolution.

        Returns:
            QCTN: A new QCTN instance evolved from the original.
        """
        warnings.warn(
            "QCTN.envolve_from_another_qctn is deprecated and will be moved "
            "to the genetic module in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        if strategies is None \
           or (isinstance(strategies, list) and not strategies):
            if isinstance(qctn, cls):
                return cls.copy(qctn)
            else:
                raise TypeError("qctn must be an instance of QCTN.")

        if callable(strategies):
            new_graph = strategies(qctn.graph)
            return cls(new_graph)
        elif isinstance(strategies, list):
            new_graph = qctn.graph
            for strategy in strategies:
                if callable(strategy):
                    new_graph = strategy(new_graph)
                else:
                    raise TypeError("Each strategy must be a callable function.")
            return cls(new_graph)

    def __repr__(self):
        """String representation of the QCTN object."""
        circuit_input = [str(info['input_shape']) for info in self.adjacency_table]
        circuit_output = [str(info['output_shape']) for info in self.adjacency_table]

        return (
            f"circuit_input = \n{circuit_input}\n"
            f" adjacency_table = \n{self.adjacency_table}\n"
            f" circuit_output = \n{circuit_output}\n"
        )

    def _circuit_to_adjacency(self,):
        """
        Convert the quantum circuit graph to adjacency table.
        
        This method builds self.adjacency_table, a list where each element corresponds to a core
        and contains a dict with:
        - 'core_idx': int, index of the core
        - 'core_name': str, name of the core
        - 'in_edge_list': list of dicts with keys:
            {'neighbor_idx', 'neighbor_name', 'edge_rank', 'qubit_idx'}
            For input edges (from circuit input), neighbor_idx = -1, neighbor_name = ""
        - 'out_edge_list': list of dicts with keys:
            {'neighbor_idx', 'neighbor_name', 'edge_rank', 'qubit_idx'}
            For output edges (to circuit output), neighbor_idx = -1, neighbor_name = ""
        - 'input_shape': list of input ranks (from in_edge_list)
        - 'output_shape': list of output ranks (from out_edge_list)
        - 'input_dim': int, product of input_shape
        - 'output_dim': int, product of output_shape
        """
        
        cores = "".join(self.cores)
        dict_core2idx = {core: idx for idx, core in enumerate(self.cores)}
        self.dict_core2idx = dict_core2idx  # Store for later use
        
        # Initialize adjacency_table
        self.adjacency_table = []
        for idx, core_name in enumerate(self.cores):
            self.adjacency_table.append({
                'core_idx': idx,
                'core_name': core_name,
                'in_edge_list': [],
                'out_edge_list': [],
                'input_shape': [],
                'output_shape': [],
                'input_dim': 1,
                'output_dim': 1,
            })
        
        input_pattern = re.compile(rf"^(\d+)([{cores}])")
        output_pattern = re.compile(rf"([{cores}])(\d+)$")
        connect_pattern = re.compile(rf"([{cores}])(\d+)(?=[{cores}])")

        for qubit_idx, line in enumerate(self.qubits):
            # print(f'qubit_idx: {qubit_idx}, line: {len(line)}, {line[-10:]}')
            # if qubit_idx == 2000:
            #     print(line)
            line = line.strip().replace("-", "")
            m_input = input_pattern.match(line)
            m_output = output_pattern.search(line)
            
            if m_input is None or m_output is None:
                # raise ValueError(
                #     f"Qubit {qubit_idx}: failed to parse line '{self.qubits[qubit_idx].strip()}' "
                #     f"(cleaned: '{line}'). "
                #     f"{'input pattern not matched' if m_input is None else 'output pattern not matched'}."
                # )

                return
            
            input_rank, input_core = m_input.groups() if m_input else (0, None)
            output_core, output_rank = m_output.groups() if m_output else (None, 0)
            input_rank, output_rank = int(input_rank), int(output_rank)
            input_core_idx = dict_core2idx[input_core]
            output_core_idx = dict_core2idx[output_core]
            
            # Add input edge: from circuit input (-1, "") to input_core
            self.adjacency_table[input_core_idx]['in_edge_list'].append({
                'neighbor_idx': -1,
                'neighbor_name': "",
                'edge_rank': input_rank,
                'qubit_idx': qubit_idx
            })
            
            # Add output edge: from output_core to circuit output (-1, "")
            self.adjacency_table[output_core_idx]['out_edge_list'].append({
                'neighbor_idx': -1,
                'neighbor_name': "",
                'edge_rank': output_rank,
                'qubit_idx': qubit_idx
            })
            
            for match in connect_pattern.finditer(line):
                end_pos = match.end()
                if end_pos >= len(line):
                    print(f"Warning: end_pos {end_pos} out of range for line '{line}'")
                    break

                core1, rank1 = match.groups()
                core2 = line[end_pos]

                core1_idx = dict_core2idx[core1]
                core2_idx = dict_core2idx[core2]
                rank1 = int(rank1)
                
                # Add to adjacency table
                # core1 -> core2: out_edge for core1, in_edge for core2
                self.adjacency_table[core1_idx]['out_edge_list'].append({
                    'neighbor_idx': core2_idx,
                    'neighbor_name': core2,
                    'edge_rank': rank1,
                    'qubit_idx': qubit_idx
                })
                self.adjacency_table[core2_idx]['in_edge_list'].append({
                    'neighbor_idx': core1_idx,
                    'neighbor_name': core1,
                    'edge_rank': rank1,
                    'qubit_idx': qubit_idx
                })

        # Compute input_shape, output_shape, input_dim, output_dim for each core
        for core_info in self.adjacency_table:
            core_info['input_shape'] = [edge['edge_rank'] for edge in core_info['in_edge_list']]
            core_info['output_shape'] = [edge['edge_rank'] for edge in core_info['out_edge_list']]
            core_info['input_dim'] = int(np.prod(core_info['input_shape'])) if core_info['input_shape'] else 1
            core_info['output_dim'] = int(np.prod(core_info['output_shape'])) if core_info['output_shape'] else 1


    def _init_cores(self):
        """
        Initialize the cores of the quantum circuit with random values.
        
        For each core, use the pre-computed values from adjacency_table:
        - input_shape: ranks from in_edge_list (already ordered by qubit_idx)
        - output_shape: ranks from out_edge_list (already ordered by qubit_idx)
        - input_dim: product of input_shape
        - output_dim: product of output_shape
        
        The core tensor is initialized with shape [input_dim, output_dim], 
        then reshaped to input_shape + output_shape.
        
        Returns:
            None: The cores are stored in the `cores_weights` attribute.
        """

        for idx, core_info in enumerate(self.adjacency_table):
            core_name = core_info['core_name']
            input_shape = core_info['input_shape']
            output_shape = core_info['output_shape']
            input_dim = core_info['input_dim']
            output_dim = core_info['output_dim']
            
            # print(f"_init_cores: {idx} {input_shape} {output_shape} {input_dim} {output_dim}")

            full_shape = input_shape + output_shape

            if input_dim == output_dim:
                # Square case: use orthogonal (QR) initialization
                core = self.backend.init_random_core([input_dim, output_dim])
                core = self.backend.reshape(core, full_shape)
            else:
                # Non-square case: orthogonal init is not applicable;
                # fall back to random Gaussian (normalized by sqrt of max dim).
                max_dim = max(input_dim, output_dim)
                core = self.backend.init_random_core([max_dim, max_dim])
                # Slice to [input_dim, output_dim] then reshape
                raw = core.tensor if isinstance(core, TNTensor) else core
                raw_sliced = raw[:input_dim, :output_dim].contiguous()
                core = self.backend.reshape(
                    self.backend.wrap_tensor(raw_sliced) if isinstance(core, TNTensor) else raw_sliced,
                    full_shape,
                )

            self.cores_weights[core_name] = core

    def set_cores(self, cores, strict: bool = True):
        """
        Set core tensors from a list or dict.

        Each supplied tensor is validated to have the **same total number of
        elements** (numel) as the corresponding existing core weight.  If the
        shapes differ but the sizes match, the tensor is reshaped to the
        target core weight's shape.

        Args:
            cores (list | dict):
                * **list** – tensors are matched to ``self.cores`` by
                  positional order.

                  - *strict=True*: ``len(cores)`` must equal ``self.ncores``.
                  - *strict=False*: only the first ``min(len(cores), ncores)``
                    cores are set; a warning is emitted if the lengths differ.

                * **dict** – keys are core names (single-character symbols).

                  - *strict=True*: the key set must exactly equal
                    ``set(self.cores)`` (no missing, no extra keys).
                  - *strict=False*: only the intersection of keys is used;
                    warnings list any missing or extra keys.

            strict (bool): Whether to require an exact one-to-one match.
                Defaults to ``True``.

        Raises:
            TypeError: If *cores* is neither a list nor a dict.
            ValueError: If *strict=True* and the sizes / keys do not match,
                or if any tensor's total element count differs from its
                target core weight.
        """
        import warnings

        if isinstance(cores, list):
            self._set_cores_from_list(cores, strict)
        elif isinstance(cores, dict):
            self._set_cores_from_dict(cores, strict)
        else:
            raise TypeError(
                f"cores must be a list or dict, got {type(cores).__name__}"
            )

    # ------------------------------------------------------------------
    # Internal helpers for set_cores
    # ------------------------------------------------------------------

    def _set_single_core(self, core_name: str, tensor):
        """
        Validate *tensor* against the existing weight for *core_name*,
        reshape if necessary, and store it.

        Raises:
            ValueError: If the total number of elements does not match.
        """
        target = self.cores_weights[core_name]
        target_shape = tuple(target.shape)
        target_numel = int(np.prod(target_shape))

        src_shape = tuple(tensor.shape)
        src_numel = int(np.prod(src_shape))

        if src_numel != target_numel:
            raise ValueError(
                f"Core '{core_name}': size mismatch — input has "
                f"{src_numel} elements (shape {src_shape}) but target "
                f"has {target_numel} elements (shape {target_shape})."
            )

        if src_shape != target_shape:
            tensor = self.backend.reshape(tensor, list(target_shape))

        self.cores_weights[core_name] = tensor

    def _set_cores_from_list(self, cores: list, strict: bool):
        import warnings

        if strict:
            if len(cores) != self.ncores:
                raise ValueError(
                    f"strict=True: expected {self.ncores} core tensors, "
                    f"got {len(cores)}."
                )
            for idx, tensor in enumerate(cores):
                self._set_single_core(self.cores[idx], tensor)
        else:
            n = min(len(cores), self.ncores)
            if len(cores) != self.ncores:
                warnings.warn(
                    f"strict=False: input list has {len(cores)} tensors but "
                    f"QCTN has {self.ncores} cores. Only the first {n} will "
                    f"be set.",
                    stacklevel=3,
                )
            for idx in range(n):
                self._set_single_core(self.cores[idx], cores[idx])

    def _set_cores_from_dict(self, cores: dict, strict: bool):
        import warnings

        input_keys = set(cores.keys())
        self_keys = set(self.cores)

        if strict:
            if input_keys != self_keys:
                missing = self_keys - input_keys
                extra = input_keys - self_keys
                parts = []
                if missing:
                    parts.append(f"missing keys ({len(missing)}): {missing}")
                if extra:
                    parts.append(f"extra keys ({len(extra)}): {extra}")
                raise ValueError(
                    f"strict=True: key mismatch — {'; '.join(parts)}."
                )
            for core_name in self.cores:
                self._set_single_core(core_name, cores[core_name])
        else:
            common = input_keys & self_keys
            missing = self_keys - input_keys
            extra = input_keys - self_keys
            if missing:
                warnings.warn(
                    f"strict=False: {len(missing)} core(s) missing from "
                    f"input dict and will keep their current weights: "
                    f"{missing}",
                    stacklevel=3,
                )
            if extra:
                warnings.warn(
                    f"strict=False: {len(extra)} extra key(s) in input dict "
                    f"will be ignored: {extra}",
                    stacklevel=3,
                )
            for core_name in self.cores:
                if core_name in common:
                    self._set_single_core(core_name, cores[core_name])

    def save_cores(self, file_path: Union[str, Path], metadata: Optional[Mapping[str, str]] = None):
        """Save all core tensors into a safetensors file."""

        if self.backend is None:
            raise RuntimeError("Backend must be initialized before saving cores.")

        try:
            from safetensors.numpy import save_file
        except ImportError as exc:
            raise ImportError("safetensors is required to save cores; install it with `pip install safetensors`.") from exc

        tensor_dict = {}
        for core_name, tensor in self.cores_weights.items():
            if isinstance(tensor, TNTensor):
                arr = self.backend.tensor_to_numpy(tensor.tensor * tensor.scale)
            else:
                arr = self.backend.tensor_to_numpy(tensor)
            if np.iscomplexobj(arr):
                tensor_dict[f"core_{core_name}_real"] = np.ascontiguousarray(arr.real)
                tensor_dict[f"core_{core_name}_imag"] = np.ascontiguousarray(arr.imag)
            else:
                tensor_dict[f"core_{core_name}"] = np.ascontiguousarray(arr)

        metadata_dict = {} if metadata is None else {str(k): str(v) for k, v in metadata.items()}
        save_file(tensor_dict, str(file_path), metadata=metadata_dict)

    def load_cores(self, file_path: Union[str, Path], strict: bool = True) -> Mapping[str, str]:
        """Load saved core tensors from a safetensors file."""

        if self.backend is None:
            raise RuntimeError("Backend must be initialized before loading cores.")

        try:
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required to load cores; install it with `pip install safetensors`.") from exc

        result = load_file(str(file_path))
        if isinstance(result, tuple) and len(result) == 2:
            tensor_dict, metadata = result
        else:
            tensor_dict = result
            metadata = {}

        for core_name in self.cores:
            key = f"core_{core_name}"
            key_real, key_imag = f"core_{core_name}_real", f"core_{core_name}_imag"
            if key_real in tensor_dict:
                array = tensor_dict[key_real] + 1j * tensor_dict[key_imag]
            elif key in tensor_dict:
                array = tensor_dict[key]
            else:
                if strict:
                    raise KeyError(f"Missing tensor for core {core_name} in {file_path}")
                continue
            tensor = self.backend.convert_to_tensor(array)
            tn_tensor = TNTensor(tensor)
            tn_tensor.auto_scale()
            self.cores_weights[core_name] = tn_tensor

        metadata_dict = {str(k): str(v) for k, v in metadata.items()}
        self._loaded_metadata = metadata_dict
        return metadata_dict

    @classmethod
    def from_pretrained(
        cls,
        graph: str,
        file_path: Union[str, Path],
        backend=None,
        strict: bool = True,
    ) -> "QCTN":
        """Create a QCTN instance loading core tensors from safetensors."""

        if backend is None:
            from ..backends.backend_factory import BackendFactory

            backend = BackendFactory.get_default_backend()

        instance = cls(graph, backend=backend)
        instance.load_cores(file_path, strict=strict)
        return instance


    # ================================================================
    # Graph manipulation helpers
    # ================================================================

    @staticmethod
    def _parse_qubit_line(line):
        """
        Parse a qubit line into a sequence of tokens.

        Given a graph line like ``-2-A-5-B-3-``, returns::

            [('dim', 2), ('core', 'A'), ('dim', 5), ('core', 'B'), ('dim', 3)]

        After stripping all dashes, the remaining characters are either
        digit-sequences (dimensions) or single non-digit characters (core
        symbols).  They always alternate ``dim, core, dim, core, …, dim``.

        Args:
            line (str): A single qubit line from the graph.

        Returns:
            list[tuple]: ``[(type, value), ...]`` where *type* is ``'dim'``
            or ``'core'``.
        """
        cleaned = line.strip().replace("-", "")
        result = []
        i = 0
        while i < len(cleaned):
            if cleaned[i].isdigit():
                j = i
                while j < len(cleaned) and cleaned[j].isdigit():
                    j += 1
                result.append(('dim', int(cleaned[i:j])))
                i = j
            else:
                result.append(('core', cleaned[i]))
                i += 1
        return result

    @staticmethod
    def _rebuild_qubit_line(tokens):
        """
        Rebuild a qubit line string from parsed tokens.

        Args:
            tokens: list of ``(type, value)`` tuples, e.g.
                ``[('dim', 2), ('core', 'A'), ('dim', 5)]``

        Returns:
            str: Rebuilt qubit line, e.g. ``-2-A-5-``
        """
        parts = [str(val) for _, val in tokens]
        return "-" + "-".join(parts) + "-"

    @staticmethod
    def _remap_graph(graph_lines, core_map):
        """
        Remap core symbols in graph lines according to *core_map*.

        Each character in every line is independently looked up in
        *core_map*; if found it is replaced, otherwise kept as-is.  This
        is safe because core symbols are single, non-digit, non-dash
        characters.

        Args:
            graph_lines (list[str]): Qubit line strings.
            core_map (dict[str, str]): ``{old_symbol: new_symbol}``.

        Returns:
            list[str]: Remapped qubit line strings.
        """
        new_lines = []
        for line in graph_lines:
            new_line = []
            for ch in line:
                new_line.append(core_map.get(ch, ch))
            new_lines.append("".join(new_line))
        return new_lines

    # ================================================================
    # Core tensor initialization
    # ================================================================

    def auto_init(self, dtype=None, device=None) -> "QCTN":
        """Initialize (or re-initialize) all core tensors with random orthogonal values.

        For graph-based modules, calls :meth:`_init_cores` to populate
        ``cores_weights``.  For composite modules (``graph=None``), recursively
        calls ``auto_init`` on every registered submodule.

        Args:
            dtype: Optional dtype hint forwarded to submodule ``auto_init``
                calls.  Not yet used by :meth:`_init_cores` directly; reserved
                for future backend-level dtype control.
            device: Optional device hint forwarded to submodule ``auto_init``
                calls.

        Returns:
            self — supports chaining, e.g. ``MPS(3, 4).auto_init()``.
        """
        if self.graph is not None:
            self._init_cores()
        for sub in self._submodules.values():
            sub.auto_init(dtype=dtype, device=device)
        return self

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

            # Locate core tokens that belong to each group
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

                # Group 1: [start … last_g1_core, dim_after_last_g1_core]
                g1_tokens = tokens[: last_g1 + 2]
                # Group 2: [dim_before_first_g2_core, first_g2_core … end]
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

        # Copy core weights (shapes are unchanged by the chunk)
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

        # ---- core symbol renaming ----
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

        # ---- determine padding widths ----
        # Use the max width of each side's real lines as the padding width
        # for the extra qubit rows added to the shorter side.
        # stripped_l1 = l1 without right boundary, stripped_l2 = l2 without left boundary
        pad_width1 = max(len(l) for l in remapped1) - 3
        pad_width2 = max(len(l) for l in remapped2) - 3

        # ---- horizontal merge ----
        new_lines = []
        for qi in range(max_qubits):
            has_l1 = qi < n1
            has_l2 = qi < n2

            l1 = remapped1[qi] if has_l1 else ("-" * pad_width1)
            l2 = remapped2[qi] if has_l2 else ("-" * pad_width2)

            # Extract 4 segments:
            #   stripped_l1: l1 with right boundary removed  (e.g. "-3-A-5-B")
            #   dim_l1:      right boundary of l1            (e.g. "-3-")
            #   dim_l2:      left boundary of l2             (e.g. "-3-")
            #   stripped_l2: l2 with left boundary removed   (e.g. "C-5-D-3-")
            m1 = re.search(r'-\d+-$', l1)
            dim_l1 = m1.group() if has_l1 else ""
            stripped_l1 = l1[:m1.start()] if has_l1 else l1

            m2 = re.match(r'^-\d+-', l2)
            dim_l2 = m2.group() if has_l2 else ""
            stripped_l2 = l2[m2.end():] if has_l2 else l2

            if has_l1 and has_l2:
                # Both exist: keep qctn1's right boundary as the shared dim
                merged = stripped_l1 + dim_l1 + stripped_l2
            elif has_l1:
                # Only l1 exists: pad the right side
                dim_l2 = '---'
                merged = stripped_l1 + stripped_l2 + dim_l1
            else:
                # Only l2 exists: pad the left side
                dim_l1 = '---'
                merged = dim_l2 + stripped_l1 + stripped_l2

            new_lines.append(merged)

        new_graph = "\n".join(new_lines)

        backend = qctn1.backend if qctn1.backend is not None else qctn2.backend
        new_qctn = QCTN(new_graph, backend=backend)

        # Copy core weights under their new names
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
    # Phase 2: Graph-parsing separation (R5) – einsum info & core list
    # ================================================================

    def get_einsum_info(
        self,
        circuit_states_shapes=None,
        measure_shapes=None,
        measure_is_matrix: bool = True,
    ):
        """Build the einsum equation and shape list for self-contraction.

        Computes the ``A · Mx · A†`` einsum expression for this QCTN.
        Extracted from ``EinsumStrategy.build_with_self_expression`` so that
        the graph-parsing logic lives in the QCTN rather than the contractor.

        Args:
            circuit_states_shapes: Shape(s) of circuit input states.
                * ``None``: no circuit states.
                * ``tuple``: shape of a single state tensor.
                * ``tuple of tuples``: shapes for a list of per-qubit state vectors.
            measure_shapes: Shape(s) of measurement matrices.
                * ``None``: no measurement.
                * ``tuple``: shape of a single Mx matrix.
                * ``tuple of tuples``: shapes for a list of per-qubit Mx matrices.
            measure_is_matrix: Ignored (kept for API compatibility; always treated
                as ``True`` internally – pass the outer-product matrix Mx).

        Returns:
            tuple[str, list]: ``(einsum_equation, tensor_shapes)`` where
                *einsum_equation* is an opt_einsum-style string and
                *tensor_shapes* is the list of shape tuples in contraction order.
        """
        import opt_einsum

        is_states_list = (
            isinstance(circuit_states_shapes, tuple)
            and circuit_states_shapes
            and isinstance(circuit_states_shapes[0], tuple)
        )
        is_measure_list = (
            isinstance(measure_shapes, tuple)
            and measure_shapes
            and isinstance(measure_shapes[0], tuple)
        )

        cores_name = self.cores
        symbol_id = 0

        edge_symbol_map: dict = {}
        input_symbols_stack: list = []
        output_symbols_stack: list = []
        new_symbol_mapping: dict = {}
        equation_list: list = []

        # ---- LEFT side cores ------------------------------------------------
        for core_info in self.adjacency_table:
            core_idx = core_info["core_idx"]
            core_equation = ""

            for edge in core_info["in_edge_list"]:
                if edge["neighbor_idx"] == -1:
                    symbol = opt_einsum.get_symbol(symbol_id)
                    input_symbols_stack.append(symbol)
                    symbol_id += 1
                else:
                    key = tuple(sorted([edge["neighbor_idx"], core_idx])) + (
                        edge["qubit_idx"],
                    )
                    if key not in edge_symbol_map:
                        edge_symbol_map[key] = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                    symbol = edge_symbol_map[key]
                core_equation += symbol

            for edge in core_info["out_edge_list"]:
                if edge["neighbor_idx"] == -1:
                    symbol = opt_einsum.get_symbol(symbol_id)
                    output_symbols_stack.append(symbol)
                    symbol_id += 1
                else:
                    key = tuple(sorted([core_idx, edge["neighbor_idx"]])) + (
                        edge["qubit_idx"],
                    )
                    if key not in edge_symbol_map:
                        edge_symbol_map[key] = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                    symbol = edge_symbol_map[key]
                core_equation += symbol

            equation_list.append(core_equation)

        # ---- Middle (measurement) block ------------------------------------
        middle_block_list: list = []
        middle_symbols_mapping = {char: char for char in output_symbols_stack}
        batch_symbol = ""
        if measure_shapes is not None:
            batch_symbol = opt_einsum.get_symbol(symbol_id)
            symbol_id += 1
            for char in output_symbols_stack:
                symbol = opt_einsum.get_symbol(symbol_id)
                symbol_id += 1
                middle_symbols_mapping[char] = symbol
                middle_block_list.append(batch_symbol + char + symbol)
            if len(middle_block_list) >= 2:
                middle_block_list = middle_block_list[:-2] + middle_block_list[-2:][::-1]

        # ---- RIGHT side cores (conjugate / inverse) ------------------------
        real_output_symbols_stack: list = []
        inv_equation_list: list = []
        for core_equation in equation_list[::-1]:
            new_equation = ""
            for char in core_equation:
                if char in output_symbols_stack:
                    new_equation += middle_symbols_mapping[char]
                else:
                    if char in new_symbol_mapping:
                        symbol = new_symbol_mapping[char]
                    else:
                        symbol = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                        new_symbol_mapping[char] = symbol
                        if char in input_symbols_stack:
                            real_output_symbols_stack.append(symbol)
                    new_equation += symbol
            inv_equation_list.append(new_equation)

        equation_list = equation_list + middle_block_list + inv_equation_list
        einsum_equation_lefthand = ",".join(equation_list)

        # ---- Circuit state symbols -----------------------------------------
        if is_states_list:
            circuit_states_symbols = ",".join(input_symbols_stack)
            output_states_symbols = ""
            for char in circuit_states_symbols[::-1]:
                output_states_symbols += char if char == "," else new_symbol_mapping[char]
        else:
            circuit_states_symbols = "".join(input_symbols_stack)
            output_states_symbols = "".join(
                new_symbol_mapping[char] for char in circuit_states_symbols[::-1]
            )

        # ---- Assemble full LHS + RHS of equation ---------------------------
        left_parts = []
        if circuit_states_shapes is not None:
            left_parts.append(circuit_states_symbols)
        left_parts.append(einsum_equation_lefthand)
        if circuit_states_shapes is not None:
            left_parts.append(output_states_symbols)
        einsum_equation_lefthand = ",".join(left_parts)
        einsum_equation = f"{einsum_equation_lefthand}->{batch_symbol}"

        # ---- Assemble shapes list ------------------------------------------
        left_core_shapes = [
            self.cores_weights[n].shape for n in cores_name
        ]
        right_core_shapes = [
            self.cores_weights[n].shape for n in cores_name[::-1]
        ]

        shapes_list: list = []
        if circuit_states_shapes is not None:
            if is_states_list:
                shapes_list.extend(list(circuit_states_shapes))
            else:
                shapes_list.append(circuit_states_shapes)
        shapes_list.extend(left_core_shapes)
        if measure_shapes is not None:
            if is_measure_list:
                shapes_list.extend(list(measure_shapes))
            else:
                shapes_list.append(measure_shapes)
        shapes_list.extend(right_core_shapes)
        if circuit_states_shapes is not None:
            if is_states_list:
                shapes_list.extend(list(circuit_states_shapes))
            else:
                shapes_list.append(circuit_states_shapes)

        return einsum_equation, shapes_list

    def build_core_list(
        self,
        cores_dict=None,
        circuit_states=None,
        measure_matrices=None,
    ) -> list:
        """Build the ordered tensor list for einsum contraction.

        Returns tensors in the canonical order expected by the einsum
        equation produced by :meth:`get_einsum_info`:

            ``[circuit_states, left_cores, measure_matrices,
               right_cores (reversed), circuit_states]``

        Args:
            cores_dict: Mapping of core name → tensor.  Defaults to
                ``self.cores_weights`` when ``None``.
            circuit_states: Single tensor or list of per-qubit state tensors.
                Pass ``None`` to omit (no circuit input).
            measure_matrices: Single matrix or list of per-qubit Mx matrices.
                Pass ``None`` to omit (no measurement).

        Returns:
            list: Ordered tensor list for einsum execution.
        """
        if cores_dict is None:
            cores_dict = self.cores_weights

        tensors: list = []

        if circuit_states is not None:
            if isinstance(circuit_states, list):
                tensors.extend(circuit_states)
            else:
                tensors.append(circuit_states)

        for name in self.cores:
            tensors.append(cores_dict[name])

        if measure_matrices is not None:
            if isinstance(measure_matrices, list):
                tensors.extend(measure_matrices)
            else:
                tensors.append(measure_matrices)

        for name in reversed(self.cores):
            tensors.append(cores_dict[name])

        if circuit_states is not None:
            if isinstance(circuit_states, list):
                # Right-side states are reversed to match the equation symbol order
                # produced by get_einsum_info (output_states_symbols is the
                # string-reversal of input_symbols_stack).
                tensors.extend(reversed(circuit_states))
            else:
                tensors.append(circuit_states)

        return tensors

    # ================================================================
    # Phase 2.5: Symmetric expansion graph for row-priority contraction
    # ================================================================

    def build_symmetric_expansion_graph(
        self,
        circuit_states_shapes=None,
        measure_shapes=None,
        right_qctn="symmetric",
    ):
        """Build an expanded L-M-R graph for row-priority (greedy) contraction.

        Constructs a ``core_tensor_list`` containing LEFT cores, LEFT circuit
        states, MIDDLE measurement matrices (Mx), RIGHT cores, and RIGHT
        circuit states, with all neighbor connections wired and einsum symbols
        assigned.  This is the graph-structure counterpart of
        :meth:`get_einsum_info` (which serves EinsumStrategy).

        Actual tensors are embedded into each entry's ``'tensor'`` key
        (for core and transpose sources) so that downstream contraction code
        only needs ``entry['tensor']``.  For ``'circuit'`` and ``'mx'``
        sources, tensors are embedded only when ``self.circuit_states`` /
        ``self.measure_matrices`` are set as instance attributes (legacy
        pattern); pass shapes explicitly for pure graph-structure queries.

        Args:
            circuit_states_shapes: Per-qubit circuit-state shapes.
                * ``None`` — no circuit states (omit circuit entries).
                * ``list[tuple]`` — ``circuit_states_shapes[qubit_idx]`` is the
                  shape of that qubit's state vector, e.g. ``[(3,), (3,)]``.
            measure_shapes: Per-qubit measurement-matrix shapes.
                * ``None`` — no measurement matrices (omit Mx entries).
                * ``list[tuple|None]`` — ``measure_shapes[qubit_idx]`` is the
                  shape of Mx, e.g. ``[(10, 3, 3), ...]``.  ``None`` entries
                  mean "no measure on this qubit".
            right_qctn: How to build the right (conjugate) side.
                * ``"symmetric"`` (default) — mirror of left with reversed edges.
                * A :class:`QCTN` instance — use its ``adjacency_table``.
                * ``None`` — no right side.

        Returns:
            tuple[list[dict], dict]:
                ``(core_tensor_list, maps)`` where *core_tensor_list* is the
                fully-wired list of entry dicts and *maps* contains
                ``left_core_map``, ``right_core_map``, ``mx_map``,
                ``left_circuit_map``, ``right_circuit_map``.
        """
        import opt_einsum

        # Auto-derive shapes from dynamic instance attributes when not provided
        # (legacy pattern: qctn.circuit_states = [...]; qctn.measure_matrices = [...])
        _cs_attr = getattr(self, 'circuit_states', None)
        _mx_attr = getattr(self, 'measure_matrices', None)
        if circuit_states_shapes is None and _cs_attr is not None:
            cs = _cs_attr
            if isinstance(cs, dict):
                circuit_states_shapes = [
                    cs[i].shape if i in cs else None for i in range(self.nqubits)
                ]
            elif isinstance(cs, (list, tuple)):
                circuit_states_shapes = [
                    cs[i].shape if i < len(cs) and cs[i] is not None else None
                    for i in range(self.nqubits)
                ]
        if measure_shapes is None and _mx_attr is not None:
            mx = _mx_attr
            if isinstance(mx, dict):
                measure_shapes = [
                    mx[i].shape if i in mx else None for i in range(self.nqubits)
                ]
            elif isinstance(mx, (list, tuple)):
                measure_shapes = [
                    mx[i].shape if i < len(mx) and mx[i] is not None else None
                    for i in range(self.nqubits)
                ]

        core_tensor_list = []

        def _get_uid():
            return len(core_tensor_list)

        # ------------------------------------------------------------------
        # 1.1  LEFT cores
        # ------------------------------------------------------------------
        left_core_map = {}  # original_idx -> uid
        for core_info in self.adjacency_table:
            uid = _get_uid()
            left_core_map[core_info['core_idx']] = uid
            core_tensor_list.append({
                'core_idx': uid,
                'core_name': f"{core_info['core_name']}_L",
                'tensor_source': 'core',
                'tensor_key': core_info['core_name'],
                'in_edge_list': deepcopy(core_info['in_edge_list']),
                'out_edge_list': deepcopy(core_info['out_edge_list']),
                'side': TensorSide.LEFT,
                'original_core_idx': core_info['core_idx'],
                'original_in_count': len(core_info['in_edge_list']),
                'original_out_count': len(core_info['out_edge_list']),
                'batch_symbol': "",
            })

        # ------------------------------------------------------------------
        # 1.2  LEFT circuit states
        # ------------------------------------------------------------------
        left_circuit_map = {}  # qubit_idx -> uid
        if circuit_states_shapes is not None:
            for qubit_idx in self.qubit_indices:
                if qubit_idx >= len(circuit_states_shapes):
                    continue
                shape = circuit_states_shapes[qubit_idx]
                if shape is None:
                    continue
                uid = _get_uid()
                left_circuit_map[qubit_idx] = uid
                core_tensor_list.append({
                    'core_idx': uid,
                    'core_name': f"circuit_L_{qubit_idx}",
                    'tensor_source': 'circuit',
                    'tensor_key': qubit_idx,
                    'in_edge_list': [],
                    'out_edge_list': [{
                        'neighbor_idx': -1,
                        'neighbor_name': "",
                        'edge_rank': shape[0],
                        'qubit_idx': qubit_idx,
                    }],
                    'side': TensorSide.LEFT,
                    'batch_symbol': "",
                })

        # ------------------------------------------------------------------
        # 1.3  MIDDLE Mx
        # ------------------------------------------------------------------
        mx_map = {}  # qubit_idx -> uid
        if measure_shapes is not None:
            for qubit_idx in self.qubit_indices:
                if qubit_idx >= len(measure_shapes):
                    continue
                mx_shape = measure_shapes[qubit_idx]
                if mx_shape is None:
                    continue
                ndim = len(mx_shape)
                batch_sym = ""
                if ndim == 3:
                    batch_sym = "a"
                elif ndim == 4:
                    batch_sym = "ab"
                uid = _get_uid()
                mx_map[qubit_idx] = uid
                core_tensor_list.append({
                    'core_idx': uid,
                    'core_name': f"mx_{qubit_idx}",
                    'tensor_source': 'mx',
                    'tensor_key': qubit_idx,
                    'in_edge_list': [{
                        'neighbor_idx': -1,
                        'neighbor_name': "",
                        'edge_rank': mx_shape[-2],
                        'qubit_idx': qubit_idx,
                    }],
                    'out_edge_list': [{
                        'neighbor_idx': -1,
                        'neighbor_name': "",
                        'edge_rank': mx_shape[-1],
                        'qubit_idx': qubit_idx,
                    }],
                    'side': TensorSide.MIDDLE,
                    'batch_symbol': batch_sym,
                })

        # ------------------------------------------------------------------
        # 1.4  RIGHT cores
        # ------------------------------------------------------------------
        right_core_map = {}  # original_idx -> uid
        if isinstance(right_qctn, str) and right_qctn == "symmetric":
            for core_info in self.adjacency_table:
                uid = _get_uid()
                right_core_map[core_info['core_idx']] = uid
                new_in_edges = deepcopy(core_info['out_edge_list'])[::-1]
                new_out_edges = deepcopy(core_info['in_edge_list'])[::-1]
                core_tensor_list.append({
                    'core_idx': uid,
                    'core_name': f"{core_info['core_name']}_R",
                    'tensor_source': 'transpose',
                    'tensor_key': core_info['core_name'],
                    'in_edge_list': new_in_edges,
                    'out_edge_list': new_out_edges,
                    'side': TensorSide.RIGHT,
                    'original_core_idx': core_info['core_idx'],
                    'original_in_count': len(core_info['in_edge_list']),
                    'original_out_count': len(core_info['out_edge_list']),
                    'batch_symbol': "",
                })
        elif isinstance(right_qctn, QCTN):
            for core_info in right_qctn.adjacency_table:
                uid = _get_uid()
                core_idx = core_info['core_idx'] + len(left_core_map)
                right_core_map[core_idx] = uid
                core_tensor_list.append({
                    'core_idx': uid,
                    'core_name': f"{core_info['core_name']}_R",
                    'tensor_source': 'core',
                    'tensor_key': "right_" + core_info['core_name'],
                    'in_edge_list': deepcopy(core_info['in_edge_list']),
                    'out_edge_list': deepcopy(core_info['out_edge_list']),
                    'side': TensorSide.RIGHT,
                    'original_core_idx': core_idx,
                    'original_in_count': len(core_info['in_edge_list']),
                    'original_out_count': len(core_info['out_edge_list']),
                    'batch_symbol': "",
                })
        elif right_qctn is not None:
            raise ValueError("right_qctn must be 'symmetric', a QCTN instance, or None.")

        # ------------------------------------------------------------------
        # 1.5  RIGHT circuit states
        # ------------------------------------------------------------------
        right_circuit_map = {}  # qubit_idx -> uid
        if circuit_states_shapes is not None:
            for qubit_idx in self.qubit_indices:
                if qubit_idx >= len(circuit_states_shapes):
                    continue
                shape = circuit_states_shapes[qubit_idx]
                if shape is None:
                    continue
                uid = _get_uid()
                right_circuit_map[qubit_idx] = uid
                core_tensor_list.append({
                    'core_idx': uid,
                    'core_name': f"circuit_R_{qubit_idx}",
                    'tensor_source': 'circuit',
                    'tensor_key': qubit_idx,
                    'in_edge_list': [{
                        'neighbor_idx': -1,
                        'neighbor_name': "",
                        'edge_rank': shape[0],
                        'qubit_idx': qubit_idx,
                    }],
                    'out_edge_list': [],
                    'side': TensorSide.RIGHT,
                    'batch_symbol': "",
                })

        # ==================================================================
        # 2.  Wire neighbor connections
        # ==================================================================

        # 2.1  LEFT cores
        for original_idx, new_idx in left_core_map.items():
            entry = core_tensor_list[new_idx]
            for edge in entry['in_edge_list']:
                if edge.get('is_cross_partition'):
                    continue
                if edge['neighbor_idx'] == -1:
                    qubit_idx = edge['qubit_idx']
                    if qubit_idx in left_circuit_map:
                        neighbor_uid = left_circuit_map[qubit_idx]
                        edge['neighbor_idx'] = neighbor_uid
                        edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']
                        circ_entry = core_tensor_list[neighbor_uid]
                        circ_entry['out_edge_list'][0]['neighbor_idx'] = new_idx
                        circ_entry['out_edge_list'][0]['neighbor_name'] = entry['core_name']
                else:
                    if edge['neighbor_idx'] in left_core_map:
                        neighbor_uid = left_core_map[edge['neighbor_idx']]
                        edge['neighbor_idx'] = neighbor_uid
                        edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']

            for edge in entry['out_edge_list']:
                if edge.get('is_cross_partition'):
                    continue
                if edge['neighbor_idx'] == -1:
                    qubit_idx = edge['qubit_idx']
                    if qubit_idx in mx_map:
                        neighbor_uid = mx_map[qubit_idx]
                        edge['neighbor_idx'] = neighbor_uid
                        edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']
                        mx_entry = core_tensor_list[neighbor_uid]
                        mx_entry['in_edge_list'][0]['neighbor_idx'] = new_idx
                        mx_entry['in_edge_list'][0]['neighbor_name'] = entry['core_name']
                else:
                    if edge['neighbor_idx'] in left_core_map:
                        neighbor_uid = left_core_map[edge['neighbor_idx']]
                        edge['neighbor_idx'] = neighbor_uid
                        edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']

        # 2.2  RIGHT cores
        for original_idx, new_idx in right_core_map.items():
            entry = core_tensor_list[new_idx]
            for edge in entry['in_edge_list']:
                if edge.get('is_cross_partition'):
                    continue
                if edge['neighbor_idx'] == -1:
                    qubit_idx = edge['qubit_idx']
                    if qubit_idx in mx_map:
                        neighbor_uid = mx_map[qubit_idx]
                        edge['neighbor_idx'] = neighbor_uid
                        edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']
                        mx_entry = core_tensor_list[neighbor_uid]
                        mx_entry['out_edge_list'][0]['neighbor_idx'] = new_idx
                        mx_entry['out_edge_list'][0]['neighbor_name'] = entry['core_name']
                else:
                    if edge['neighbor_idx'] in right_core_map:
                        neighbor_uid = right_core_map[edge['neighbor_idx']]
                        edge['neighbor_idx'] = neighbor_uid
                        edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']

            for edge in entry['out_edge_list']:
                if edge.get('is_cross_partition'):
                    continue
                if edge['neighbor_idx'] == -1:
                    qubit_idx = edge['qubit_idx']
                    if qubit_idx in right_circuit_map:
                        neighbor_uid = right_circuit_map[qubit_idx]
                        edge['neighbor_idx'] = neighbor_uid
                        edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']
                        circ_entry = core_tensor_list[neighbor_uid]
                        circ_entry['in_edge_list'][0]['neighbor_idx'] = new_idx
                        circ_entry['in_edge_list'][0]['neighbor_name'] = entry['core_name']
                else:
                    if edge['neighbor_idx'] in right_core_map:
                        neighbor_uid = right_core_map[edge['neighbor_idx']]
                        edge['neighbor_idx'] = neighbor_uid
                        edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']

        # ==================================================================
        # 2.5  Assign symbols to edges
        # ==================================================================
        def _symbol_generator():
            i = 0
            while True:
                sym = opt_einsum.get_symbol(i)
                if sym not in ('a', 'b'):  # reserved for batch dims
                    yield sym
                i += 1

        symbol_gen = _symbol_generator()

        for entry in core_tensor_list:
            for edge in entry['out_edge_list']:
                if 'symbol' in edge:
                    continue
                sym = next(symbol_gen)
                edge['symbol'] = sym
                neighbor_idx = edge['neighbor_idx']
                if neighbor_idx >= 0:
                    neighbor_entry = core_tensor_list[neighbor_idx]
                    for in_edge in neighbor_entry['in_edge_list']:
                        if (in_edge['neighbor_idx'] == entry['core_idx']
                                and in_edge['qubit_idx'] == edge['qubit_idx']):
                            in_edge['symbol'] = sym
                            break

        for entry in core_tensor_list:
            for edge in entry['in_edge_list']:
                if 'symbol' not in edge:
                    edge['symbol'] = next(symbol_gen)

        # ==================================================================
        # 3.  Embed actual tensors into entries
        # ==================================================================
        for entry in core_tensor_list:
            source = entry['tensor_source']
            key = entry['tensor_key']
            if source == 'core':
                if isinstance(right_qctn, QCTN) and isinstance(key, str) and key.startswith('right_'):
                    actual_key = key[len('right_'):]
                    entry['tensor'] = right_qctn.cores_weights[actual_key]
                else:
                    entry['tensor'] = self.cores_weights[key]
            elif source == 'transpose':
                t = self.cores_weights[key]
                if self.backend is not None and self.backend.is_complex(t):
                    entry['tensor'] = t.conj()
                else:
                    entry['tensor'] = t
            elif source == 'circuit':
                _cs = getattr(self, 'circuit_states', None)
                if _cs is not None:
                    entry['tensor'] = _cs[key]
            elif source == 'mx':
                _mx = getattr(self, 'measure_matrices', None)
                if _mx is not None:
                    entry['tensor'] = _mx[key]

        maps = {
            'left_core_map': left_core_map,
            'right_core_map': right_core_map,
            'mx_map': mx_map,
            'left_circuit_map': left_circuit_map,
            'right_circuit_map': right_circuit_map,
        }
        return core_tensor_list, maps

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

