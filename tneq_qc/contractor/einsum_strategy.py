"""
Einsum-based contraction strategy.

This module provides the EinsumStrategy class that uses opt_einsum for tensor contractions.
"""

from __future__ import annotations
import itertools
import opt_einsum
from typing import Tuple, List, Dict, Any, Callable

from .base import ContractionStrategy

from ..core.tn_tensor import TNTensor


class EinsumStrategy(ContractionStrategy):
    """Strategy using opt_einsum directly (Fast mode)"""
    
    def check_compatibility(self, qctn, shapes_info: Dict[str, Any]) -> bool:
        """einsum can handle any structure"""
        return True
    
    def get_compute_function(self, qctn, shapes_info: Dict[str, Any], backend, **_kwargs) -> Callable:
        """Return an opt_einsum compute function over the wired graph.

        Wires from ``qctn.build_graph()`` (via ``assemble_global_einsum``) — the
        same correct wiring ``RowPriorityStrategy`` and the cotengra strategy
        use — rather than ``qctn.get_einsum_info()``, which mis-wires the generic
        concat/trace case (it returned ``K^N`` independent of the weights).  Like
        those strategies it ignores the ``cores_dict`` ordering args and reads
        live tensors from the graph, but routes a supplied ``cores_dict``
        override through so functional-autodiff backends (JAX) get gradients.
        """
        import opt_einsum
        from .cotengra_strategy import assemble_global_einsum  # lazy: avoid cycle
        from ..config import Configuration

        def compute_fn(_cores_dict=None, _circuit_states=None, _measure_matrices=None, **_):
            eq, raw_tensors, total_scale, total_log_scale, _out = assemble_global_einsum(
                qctn, cores_override=_cores_dict
            )

            cache = getattr(qctn, "_einsum_default_expr", None)
            if cache is None or cache[0] != eq:
                expr = opt_einsum.contract_expression(
                    eq,
                    *[tuple(int(d) for d in t.shape) for t in raw_tensors],
                    optimize=Configuration.opt_einsum_optimize,
                )
                qctn._einsum_default_expr = (eq, expr)
            else:
                expr = cache[1]

            result = backend.execute_expression(expr, *raw_tensors)

            if total_scale is not None:
                result = TNTensor(result, scale=total_scale, log_scale=total_log_scale)
                result = backend.maybe_auto_scale(result)
            return result

        return compute_fn
    
    def estimate_cost(self, qctn, shapes_info: Dict[str, Any]) -> float:
        """Estimate cost using opt_einsum"""
        circuit_states_shapes = shapes_info.get('circuit_states_shapes')
        measure_shapes = shapes_info.get('measure_shapes')
        measure_is_matrix = shapes_info.get('measure_is_matrix', True)
        
        einsum_eq, tensor_shapes = self.build_with_self_expression(
            qctn, circuit_states_shapes, measure_shapes, measure_is_matrix
        )
        
        print(f'einsum_eq for cost estimation: {einsum_eq}')
        print(f'tensor_shapes for cost estimation: {tensor_shapes}')

        # TODO: fix contract_path params
        # Estimate cost using opt_einsum
        # path_info = opt_einsum.contract_path(einsum_eq, *tensor_shapes, optimize='auto')
        # return float(path_info[1].opt_cost)

        return 1.0
    
    @property
    def name(self) -> str:
        return "einsum_default"

    # ================================================================
    # Legacy static methods (deprecated — use QCTN.get_einsum_info instead)
    # These directly parse qctn.adjacency_table to build einsum expressions.
    # Kept for backward compatibility with engine.py and existing tests.
    # ================================================================

    @staticmethod
    def build_core_only_expression(qctn) -> Tuple[str, List]:
        """Build einsum expression for contracting cores only (no inputs).

        .. deprecated:: Use ``qctn.get_einsum_info()`` instead.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """
        cores_name = qctn.cores
        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''
        
        # Map edge connections to symbols
        edge_symbol_map = {}  # (core1_idx, core2_idx, qubit_idx) -> symbol
        
        for core_info in qctn.adjacency_table:
            core_idx = core_info['core_idx']
            core_eq = ''
            
            # Add input edges
            for edge in core_info['in_edge_list']:
                if edge['neighbor_idx'] == -1:  # Circuit input
                    symbol = opt_einsum.get_symbol(symbol_id)
                    einsum_equation_righthand += symbol
                    symbol_id += 1
                else:  # Connection from another core
                    key = tuple(sorted([edge['neighbor_idx'], core_idx])) + (edge['qubit_idx'],)
                    if key not in edge_symbol_map:
                        edge_symbol_map[key] = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                    symbol = edge_symbol_map[key]
                core_eq += symbol
            
            # Add output edges
            for edge in core_info['out_edge_list']:
                if edge['neighbor_idx'] == -1:  # Circuit output
                    symbol = opt_einsum.get_symbol(symbol_id)
                    einsum_equation_righthand += symbol
                    symbol_id += 1
                else:  # Connection to another core
                    key = tuple(sorted([core_idx, edge['neighbor_idx']])) + (edge['qubit_idx'],)
                    if key not in edge_symbol_map:
                        edge_symbol_map[key] = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                    symbol = edge_symbol_map[key]
                core_eq += symbol
            
            einsum_equation_lefthand += core_eq + ','
        
        einsum_equation_lefthand = einsum_equation_lefthand[:-1]
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'
        
        tensor_shapes = [qctn.cores_weights[core_name].shape for core_name in cores_name]
        
        return einsum_equation, tensor_shapes

    @staticmethod
    def build_with_inputs_expression(qctn, inputs_shape) -> Tuple[str, List]:
        """Build einsum expression for contracting with single input tensor.

        .. deprecated:: Use ``qctn.get_einsum_info()`` instead.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs_shape (tuple): Shape of the input tensor.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """
        cores_name = qctn.cores
        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''
        inputs_equation_lefthand = ''
        
        # Map edge connections to symbols
        edge_symbol_map = {}  # (core1_idx, core2_idx, qubit_idx) -> symbol
        
        for core_info in qctn.adjacency_table:
            core_idx = core_info['core_idx']
            core_eq = ''
            
            # Add input edges
            for edge in core_info['in_edge_list']:
                if edge['neighbor_idx'] == -1:  # Circuit input
                    symbol = opt_einsum.get_symbol(symbol_id)
                    inputs_equation_lefthand += symbol
                    symbol_id += 1
                else:  # Connection from another core
                    key = tuple(sorted([edge['neighbor_idx'], core_idx])) + (edge['qubit_idx'],)
                    if key not in edge_symbol_map:
                        edge_symbol_map[key] = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                    symbol = edge_symbol_map[key]
                core_eq += symbol
            
            # Add output edges
            for edge in core_info['out_edge_list']:
                if edge['neighbor_idx'] == -1:  # Circuit output
                    symbol = opt_einsum.get_symbol(symbol_id)
                    einsum_equation_righthand += symbol
                    symbol_id += 1
                else:  # Connection to another core
                    key = tuple(sorted([core_idx, edge['neighbor_idx']])) + (edge['qubit_idx'],)
                    if key not in edge_symbol_map:
                        edge_symbol_map[key] = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                    symbol = edge_symbol_map[key]
                core_eq += symbol
            
            einsum_equation_lefthand += core_eq + ','
        
        einsum_equation_lefthand = f'{inputs_equation_lefthand},{einsum_equation_lefthand[:-1]}'
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'
        
        tensor_shapes = [inputs_shape] + [qctn.cores_weights[core_name].shape for core_name in cores_name]
        
        return einsum_equation, tensor_shapes

    @staticmethod
    def build_with_vector_inputs_expression(qctn, inputs_shapes: List) -> Tuple[str, List]:
        """Build einsum expression for contracting with vector inputs.

        .. deprecated:: Use ``qctn.get_einsum_info()`` instead.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs_shapes (list): List of input tensor shapes.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """
        cores_name = qctn.cores
        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''
        inputs_equation_lefthand = ''
        
        # Map edge connections to symbols
        edge_symbol_map = {}  # (core1_idx, core2_idx, qubit_idx) -> symbol
        
        for core_info in qctn.adjacency_table:
            core_idx = core_info['core_idx']
            core_eq = ''
            
            # Add input edges
            for edge in core_info['in_edge_list']:
                if edge['neighbor_idx'] == -1:  # Circuit input
                    symbol = opt_einsum.get_symbol(symbol_id)
                    inputs_equation_lefthand += symbol + ','
                    symbol_id += 1
                else:  # Connection from another core
                    key = tuple(sorted([edge['neighbor_idx'], core_idx])) + (edge['qubit_idx'],)
                    if key not in edge_symbol_map:
                        edge_symbol_map[key] = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                    symbol = edge_symbol_map[key]
                core_eq += symbol
            
            # Add output edges
            for edge in core_info['out_edge_list']:
                if edge['neighbor_idx'] == -1:  # Circuit output
                    symbol = opt_einsum.get_symbol(symbol_id)
                    einsum_equation_righthand += symbol
                    symbol_id += 1
                else:  # Connection to another core
                    key = tuple(sorted([core_idx, edge['neighbor_idx']])) + (edge['qubit_idx'],)
                    if key not in edge_symbol_map:
                        edge_symbol_map[key] = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                    symbol = edge_symbol_map[key]
                core_eq += symbol
            
            einsum_equation_lefthand += core_eq + ','
        
        einsum_equation_lefthand = f'{inputs_equation_lefthand}{einsum_equation_lefthand[:-1]}'
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'
        
        tensor_shapes = inputs_shapes + [qctn.cores_weights[core_name].shape for core_name in cores_name]
        
        return einsum_equation, tensor_shapes

    @staticmethod
    def build_with_qctn_expression(qctn, target_qctn) -> Tuple[str, List]:
        """Build einsum expression for contracting two QCTNs together.

        .. deprecated:: Use ``qctn.get_einsum_info()`` instead.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            target_qctn (QCTN): The target quantum circuit tensor network.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """
        cores_name = qctn.cores
        target_cores_name = target_qctn.cores
        
        symbol_id = 0
        einsum_equation_lefthand = ''
        target_einsum_equation_lefthand = ''
        
        # Map edge connections to symbols for qctn
        edge_symbol_map = {}  # (core1_idx, core2_idx, qubit_idx) -> symbol
        input_symbols_stack = []
        output_symbols_stack = []
        
        for core_info in qctn.adjacency_table:
            core_idx = core_info['core_idx']
            core_eq = ''
            
            # Add input edges
            for edge in core_info['in_edge_list']:
                if edge['neighbor_idx'] == -1:  # Circuit input
                    symbol = opt_einsum.get_symbol(symbol_id)
                    input_symbols_stack.append(symbol)
                    symbol_id += 1
                else:  # Connection from another core
                    key = tuple(sorted([edge['neighbor_idx'], core_idx])) + (edge['qubit_idx'],)
                    if key not in edge_symbol_map:
                        edge_symbol_map[key] = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                    symbol = edge_symbol_map[key]
                core_eq += symbol
            
            # Add output edges
            for edge in core_info['out_edge_list']:
                if edge['neighbor_idx'] == -1:  # Circuit output
                    symbol = opt_einsum.get_symbol(symbol_id)
                    output_symbols_stack.append(symbol)
                    symbol_id += 1
                else:  # Connection to another core
                    key = tuple(sorted([core_idx, edge['neighbor_idx']])) + (edge['qubit_idx'],)
                    if key not in edge_symbol_map:
                        edge_symbol_map[key] = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                    symbol = edge_symbol_map[key]
                core_eq += symbol
            
            einsum_equation_lefthand += core_eq + ','
        
        # Map edge connections to symbols for target_qctn
        target_edge_symbol_map = {}
        
        for core_info in target_qctn.adjacency_table:
            core_idx = core_info['core_idx']
            core_eq = ''
            
            # Add input edges - reuse symbols from output_symbols_stack
            for edge in core_info['in_edge_list']:
                if edge['neighbor_idx'] == -1:  # Circuit input (connects to qctn output)
                    symbol = input_symbols_stack.pop(0)
                else:  # Connection from another core
                    key = tuple(sorted([edge['neighbor_idx'], core_idx])) + (edge['qubit_idx'],)
                    if key not in target_edge_symbol_map:
                        target_edge_symbol_map[key] = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                    symbol = target_edge_symbol_map[key]
                core_eq += symbol
            
            # Add output edges - reuse symbols from output_symbols_stack
            for edge in core_info['out_edge_list']:
                if edge['neighbor_idx'] == -1:  # Circuit output
                    symbol = output_symbols_stack.pop(0)
                else:  # Connection to another core
                    key = tuple(sorted([core_idx, edge['neighbor_idx']])) + (edge['qubit_idx'],)
                    if key not in target_edge_symbol_map:
                        target_edge_symbol_map[key] = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                    symbol = target_edge_symbol_map[key]
                core_eq += symbol
            
            target_einsum_equation_lefthand += core_eq + ','
        
        einsum_equation = f'{einsum_equation_lefthand}{target_einsum_equation_lefthand[:-1]}->'
        
        tensor_shapes = [qctn.cores_weights[core_name].shape for core_name in cores_name] + \
                        [target_qctn.cores_weights[core_name].shape for core_name in target_cores_name]
        
        return einsum_equation, tensor_shapes

    @staticmethod
    def build_with_self_expression(qctn, circuit_states_shape=None, measure_shape=None, measure_is_matrix=True) -> Tuple[str, List]:
        """Backward-compatible wrapper – delegates to ``qctn.get_einsum_info()``."""
        return qctn.get_einsum_info(circuit_states_shape, measure_shape, measure_is_matrix)

    @staticmethod
    def _build_with_self_expression_legacy(qctn, circuit_states_shape=None, measure_shape=None, measure_is_matrix=False) -> Tuple[str, List]:
        """Build einsum expression for contracting QCTN with itself (hermitian conjugate).

        .. deprecated:: Kept for reference/validation. Use ``qctn.get_einsum_info()`` instead.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states_shape (tuple or tuple of tuples, optional): Shape(s) of circuit states input.
                Can be a single shape tuple or tuple of shape tuples for list inputs.
            measure_shape (tuple or tuple of tuples, optional): Shape(s) of measurement input.
                Can be a single shape tuple or tuple of shape tuples for list inputs.
            measure_is_matrix (bool): If True, measure_input is the outer product matrix Mx;
                If False, measure_input is the vector phi_x.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """

        # Determine if we have list inputs
        is_states_list = isinstance(circuit_states_shape, tuple) and circuit_states_shape and isinstance(circuit_states_shape[0], tuple)
        is_measure_list = isinstance(measure_shape, tuple) and measure_shape and isinstance(measure_shape[0], tuple)
        
        cores_name = qctn.cores
        symbol_id = 0
        
        # Map edge connections to symbols
        edge_symbol_map = {}  # (core1_idx, core2_idx, qubit_idx) -> symbol
        input_symbols_stack = []
        output_symbols_stack = []
        
        new_symbol_mapping = {}
        
        equation_list = []

        # Build equations for LEFT side cores
        for core_info in qctn.adjacency_table:
            core_idx = core_info['core_idx']
            core_equation = ""
            
            # Add input edges
            for edge in core_info['in_edge_list']:
                if edge['neighbor_idx'] == -1:  # Circuit input
                    symbol = opt_einsum.get_symbol(symbol_id)
                    input_symbols_stack.append(symbol)
                    symbol_id += 1
                else:  # Connection from another core
                    key = tuple(sorted([edge['neighbor_idx'], core_idx])) + (edge['qubit_idx'],)
                    if key not in edge_symbol_map:
                        edge_symbol_map[key] = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                    symbol = edge_symbol_map[key]
                core_equation += symbol
            
            # Add output edges
            for edge in core_info['out_edge_list']:
                if edge['neighbor_idx'] == -1:  # Circuit output
                    symbol = opt_einsum.get_symbol(symbol_id)
                    output_symbols_stack.append(symbol)
                    symbol_id += 1
                else:  # Connection to another core
                    key = tuple(sorted([core_idx, edge['neighbor_idx']])) + (edge['qubit_idx'],)
                    if key not in edge_symbol_map:
                        edge_symbol_map[key] = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                    symbol = edge_symbol_map[key]
                core_equation += symbol
            
            print(f"{cores_name[core_idx]} core_equation before sorting: {core_equation}")
            # TODO: use better strategy
            # ll = core_equation[:len(core_info['in_edge_list'])]
            # rr = core_equation[len(core_info['in_edge_list']):]
            # # sort string characters
            # ll = list(ll)
            # ll.sort()
            # rr = list(rr)
            # rr.sort()
            # core_equation = "".join(ll + rr[::-1])
            # print(f"{cores_name[core_idx]} core_equation after sorting: {core_equation}")
            
            equation_list.append(core_equation)

        middle_block_list = []
        middle_symbols_mapping = {
            char: char for char in output_symbols_stack
        }
        batch_symbol = ''
        if measure_shape is not None:
            # Add batch size dimension
            batch_symbol = opt_einsum.get_symbol(symbol_id)
            symbol_id += 1
            
            middle_block_list = []
            for char in output_symbols_stack:
                symbol = opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

                middle_symbols_mapping[char] = symbol

                middle_block_list += [batch_symbol + char + symbol]
            # swap last two 
            if len(middle_block_list) >=2:
                middle_block_list = middle_block_list[:-2] + middle_block_list[-2:][::-1]

        real_output_symbols_stack = []
        inv_equation_list = []
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
        
        # Handle circuit_states and measure_input
        if is_states_list:
            circuit_states_symbols = ','.join(input_symbols_stack)
            output_states_symbols = ''
            for char in circuit_states_symbols[::-1]:
                output_states_symbols += char if char==',' else new_symbol_mapping[char]
        else:
            circuit_states_symbols = ''.join(input_symbols_stack)
            output_states_symbols = ''
            for char in circuit_states_symbols[::-1]:
                output_states_symbols += new_symbol_mapping[char]

        # Build equation parts
        left_parts = []
        
        # Add circuit_states
        if circuit_states_shape is not None:
            left_parts.append(circuit_states_symbols)
        
        # Add cores equations
        left_parts.append(einsum_equation_lefthand)
        
        # Add conjugate side inputs
        if circuit_states_shape is not None:
            left_parts.append(output_states_symbols)
        
        einsum_equation_lefthand = ",".join(left_parts)

        einsum_equation = f'{einsum_equation_lefthand}->{batch_symbol}'

        tensor_shapes = [qctn.cores_weights[core_name].shape for core_name in cores_name]
        inv_tensor_shapes = [qctn.cores_weights[core_name].shape for core_name in cores_name[::-1]]

        # Prepare tensor_shapes list
        shapes_list = []
        
        # Add circuit_states shapes
        if circuit_states_shape is not None:
            if is_states_list:
                shapes_list.extend(list(circuit_states_shape))
            else:
                shapes_list.append(circuit_states_shape)
        
        print(f"shapes_list after adding circuit_states: {len(shapes_list)} items")

        # Add core shapes
        shapes_list.extend(tensor_shapes)

        print(f"shapes_list after adding cores: {len(shapes_list)} items")

        # Add measure_input shapes
        if measure_shape is not None:
            if is_measure_list:
                shapes_list.extend(list(measure_shape))
            else:
                shapes_list.append(measure_shape)
        
        print(f"shapes_list after adding measure_input: {len(shapes_list)} items")
        # Add inverse core shapes
        shapes_list.extend(inv_tensor_shapes)

        print(f"shapes_list after adding inverse cores: {len(shapes_list)} items")
        
        # Add conjugate side shapes
        if circuit_states_shape is not None:
            if is_states_list:
                shapes_list.extend(list(circuit_states_shape))
            else:
                shapes_list.append(circuit_states_shape)
        
        print(f"shapes_list after adding conjugate circuit_states: {len(shapes_list)} items")
        tensor_shapes = shapes_list

        return einsum_equation, tensor_shapes

    @staticmethod
    def create_contract_expression(einsum_equation: str, tensor_shapes: List, optimize='auto'):
        """
        Create optimized contraction expression using opt_einsum.
        
        Args:
            einsum_equation (str): The einsum equation string.
            tensor_shapes (list): List of tensor shapes.
            optimize (str or bool): Optimization strategy for opt_einsum.
        
        Returns:
            opt_einsum.ContractExpression: Optimized contraction expression.
        """
        from ..config import Configuration
        print('einsum_equation', einsum_equation)
        print('tensor_shapes', tensor_shapes)

        return opt_einsum.contract_expression(
            einsum_equation, 
            *tensor_shapes, 
            optimize=optimize if optimize != 'auto' else Configuration.opt_einsum_optimize
        )
