"""Contraction-interface Mixin for QCTN.

Contains the graph-structure methods used by EinsumStrategy and
RowPriorityStrategy:

- ``get_einsum_info``           — einsum equation + shapes for EinsumStrategy
- ``build_core_list``           — ordered tensor list for einsum execution
- ``build_graph``                    — LEFT-RIGHT wired graph for RowPriorityStrategy

Also defines ``TensorSide``, re-exported by ``qctn.py`` for public use.
"""
from copy import deepcopy
from enum import Enum


class TensorSide(Enum):
    """Enum to track tensor side in symmetric expansion: left, middle, or right."""
    LEFT = "L"
    MIDDLE = "M"
    RIGHT = "R"


class QCTNContractorMixin:
    """Mixin: einsum info, core list, and symmetric expansion graph."""

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

    def build_graph(self):
        """Build a wired graph from adjacency_table for contraction.

        Purely maps ``self.adjacency_table`` + ``self.cores_weights`` into a
        flat ``core_tensor_list``.  Makes no assumptions about topology or
        symmetry — all structure (conjugates, transposes, extra nodes such as
        circuit-state or Mx tensors) must already be encoded in the QCTN.

        Steps
        -----
        1. **Embed**    — for each core in adjacency_table, attach the tensor
                          from ``cores_weights`` and classify ``tensor_source``.
        2. **Wire**     — remap every ``neighbor_idx`` from the original
                          ``core_idx`` space to the new uid space.
        3. **Symbolize**— assign a unique einsum symbol to every edge;
                          paired edges (connected by a bond) share one symbol.

        ``tensor_source`` values
        ------------------------
        * ``'core'``      — plain ``torch.Tensor``, or ``TNTensor`` with
                            ``is_ref=False`` and ``is_transposed=False``
        * ``'ref'``       — ``TNTensor`` with ``is_ref=True`` only
                            (e.g. a conjugate view)
        * ``'transpose'`` — ``TNTensor`` with ``is_transposed=True``
                            (conjugate-transpose / dagger view)

        Returns
        -------
        tuple[list[dict], dict]
            ``(core_tensor_list, maps)`` where
            ``maps = {'core_map': {original_core_idx → uid}}``.
        """
        import opt_einsum

        try:
            from .tn_tensor import TNTensor as _TNTensor
        except ImportError:
            _TNTensor = None

        core_tensor_list = []
        core_map = {}  # original core_idx -> uid

        # ==================================================================
        # Step 1: Embed
        # ==================================================================
        for core_info in self.adjacency_table:
            uid = len(core_tensor_list)
            core_map[core_info['core_idx']] = uid

            t = self.cores_weights[core_info['core_name']]

            if _TNTensor is not None and isinstance(t, _TNTensor):
                if t.is_transposed:
                    tensor_source = 'transpose'
                elif t.is_ref:
                    tensor_source = 'ref'
                else:
                    tensor_source = 'core'
            else:
                tensor_source = 'core'

            # Detect batch dimension: if the TNTensor was created with
            # has_batch=True, reserve 'a' as the batch symbol so that
            # RowPriorityStrategy automatically prepends it to the einsum
            # subscript for this core.
            batch_symbol = (
                'a' if (_TNTensor is not None
                        and isinstance(t, _TNTensor)
                        and t.has_batch)
                else ''
            )

            core_tensor_list.append({
                'core_idx':        uid,
                'core_name':       core_info['core_name'],
                'tensor_source':   tensor_source,
                'tensor_key':      core_info['core_name'],
                'tensor':          t,
                'in_edge_list':    deepcopy(core_info['in_edge_list']),
                'out_edge_list':   deepcopy(core_info['out_edge_list']),
                'side':            TensorSide.LEFT,
                'original_core_idx': core_info['core_idx'],
                'batch_symbol':    batch_symbol,
            })

        # ==================================================================
        # Step 2: Wire
        # ==================================================================
        for entry in core_tensor_list:
            for edge in entry['in_edge_list'] + entry['out_edge_list']:
                if edge.get('is_cross_partition'):
                    continue
                old_nb = edge['neighbor_idx']
                if old_nb != -1 and old_nb in core_map:
                    nb_new = core_map[old_nb]
                    edge['neighbor_idx'] = nb_new
                    edge['neighbor_name'] = core_tensor_list[nb_new]['core_name']

        # ==================================================================
        # Step 3: Symbolize
        # ==================================================================
        def _symbol_gen():
            i = 0
            while True:
                sym = opt_einsum.get_symbol(i)
                if sym not in ('a', 'b'):  # reserved for batch dims
                    yield sym
                i += 1

        sym_gen = _symbol_gen()

        # Assign symbols from out_edge side; propagate to the paired in_edge
        for entry in core_tensor_list:
            for edge in entry['out_edge_list']:
                if 'symbol' in edge:
                    continue
                sym = next(sym_gen)
                edge['symbol'] = sym
                nb_idx = edge['neighbor_idx']
                if nb_idx >= 0:
                    nb_entry = core_tensor_list[nb_idx]
                    for in_edge in nb_entry['in_edge_list']:
                        if (in_edge['neighbor_idx'] == entry['core_idx']
                                and in_edge['qubit_idx'] == edge['qubit_idx']):
                            in_edge['symbol'] = sym
                            break

        # Open in_edges (boundary, no out_edge partner) get their own symbol
        for entry in core_tensor_list:
            for edge in entry['in_edge_list']:
                if 'symbol' not in edge:
                    edge['symbol'] = next(sym_gen)

        # ==================================================================
        # Step 4: Close trace — assign matching symbols to boundary pairs
        # ==================================================================
        trace_qubits = getattr(self, 'trace_qubits', set())
        if trace_qubits:
            for q_idx in trace_qubits:
                # Collect boundary out edges on this qubit
                boundary_out = []
                for entry in core_tensor_list:
                    for edge in entry['out_edge_list']:
                        if edge['qubit_idx'] == q_idx and edge['neighbor_idx'] == -1:
                            boundary_out.append(edge)

                # Collect boundary in edges on this qubit
                boundary_in = []
                for entry in core_tensor_list:
                    for edge in entry['in_edge_list']:
                        if edge['qubit_idx'] == q_idx and edge['neighbor_idx'] == -1:
                            boundary_in.append(edge)

                if len(boundary_out) != len(boundary_in):
                    raise ValueError(
                        f"Trace qubit {q_idx}: mismatched boundary count "
                        f"(out={len(boundary_out)}, in={len(boundary_in)})"
                    )
                for out_e, in_e in zip(boundary_out, boundary_in):
                    if out_e['edge_rank'] != in_e['edge_rank']:
                        raise ValueError(
                            f"Trace qubit {q_idx}: boundary rank mismatch "
                            f"(out={out_e['edge_rank']}, in={in_e['edge_rank']})"
                        )
                    # Close: make in_edge share the out_edge symbol → einsum contracts.
                    # Mark both edges so _contract_group skips collecting them
                    # as open boundaries.
                    in_e['symbol'] = out_e['symbol']
                    in_e['is_trace_closed'] = True
                    out_e['is_trace_closed'] = True

        return core_tensor_list, {'core_map': core_map}
