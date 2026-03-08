from __future__ import annotations
from ..config import Configuration
import itertools
import opt_einsum
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass
import math

local_debug = True

def local_print(*args, **kwargs):
    """Print function that only prints when local_debug is True."""
    if local_debug:
        print(*args, **kwargs)

def _build_circuit_from_adjacency_table(qctn):
    """Build (input_ranks, adjacency_matrix, output_ranks) from qctn.adjacency_table.

    This is a compatibility shim so that legacy ContractorOptEinsum methods can
    work without the removed ``qctn.adjacency_matrix`` / ``qctn.circuit`` attrs.
    """
    import numpy as np
    ncores = qctn.ncores
    adjacency_table = qctn.adjacency_table

    adjacency_matrix = np.empty((ncores, ncores), dtype=object)
    for i in range(ncores):
        for j in range(ncores):
            adjacency_matrix[i, j] = []

    for core_info in adjacency_table:
        core_idx = core_info['core_idx']
        for edge in core_info['out_edge_list']:
            if edge['neighbor_idx'] >= 0:
                adjacency_matrix[core_idx, edge['neighbor_idx']].append(edge['edge_rank'])
                adjacency_matrix[edge['neighbor_idx'], core_idx].append(edge['edge_rank'])

    input_ranks = np.empty(ncores, dtype=object)
    output_ranks = np.empty(ncores, dtype=object)
    for i in range(ncores):
        input_ranks[i] = adjacency_table[i]['input_shape'].copy()
        output_ranks[i] = adjacency_table[i]['output_shape'].copy()

    return input_ranks, adjacency_matrix, output_ranks


class ContractorOptEinsum:
    """
    ContractorOptEinsum class for optimized tensor contraction using opt_einsum.

    This class provides methods to contract tensors using the opt_einsum library,
    which is optimized for performance and memory efficiency.
    """

    # tenmul_qc.py
    @staticmethod
    def contract_core_only(qctn):
        """
        Contract the given QCTN.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
        
        Returns:
            jnp.ndarray: The result of the tensor contraction.
        """

        input_ranks, adjacency_matrix, output_ranks = _build_circuit_from_adjacency_table(qctn)
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from ..utils.graph_generators import QCTNHelper
        for element in QCTNHelper.triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols  # Ensure symmetry
                
        for idx, _ in enumerate(cores_name):
            for _ in input_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                einsum_equation_righthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

            einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                einsum_equation_righthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1
            
            einsum_equation_lefthand += ','

        einsum_equation_lefthand = einsum_equation_lefthand[:-1]  # Remove the last comma
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'

        tensor_shapes = [cores_weights[core_name].shape for core_name in cores_name]

        for core_name in cores_name:
            local_print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')
        
        local_print(f'QCTN cores: {qctn.cores}')
        local_print(f'Einsum Equation: {einsum_equation}')
        local_print(f'Tensor Shapes: {tensor_shapes}')

        qctn.einsum_expr = opt_einsum.contract_expression(einsum_equation, *tensor_shapes, optimize=Configuration.opt_einsum_optimize)
        jit_retraction = jax.jit(qctn.einsum_expr)
        retracted_QCTN = jit_retraction(*[cores_weights[core_name] for core_name in cores_name])

        return retracted_QCTN
    
    @staticmethod
    def contract_with_inputs(qctn, inputs):
        """
        Contract the given QCTN with specified inputs.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs (jnp.ndarray): The inputs for the contraction operation.
        
        Returns:
            jnp.ndarray: The result of the tensor contraction.
        """

        local_print(f'ContractorOptEinsum.contract_with_inputs called with inputs shape: {inputs.shape}')

        input_ranks, adjacency_matrix, output_ranks = _build_circuit_from_adjacency_table(qctn)
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

        local_print(f'ContractorOptEinsum.contract_with_inputs called with cores_weights shapes: {[w.shape for w in cores_weights.values()]}')
        local_print(f'ContractorOptEinsum.contract_with_inputs called with input_ranks: {input_ranks}')
        local_print(f'ContractorOptEinsum.contract_with_inputs called with output_ranks: {output_ranks}')
        local_print(f'ContractorOptEinsum.contract_with_inputs called with adjacency_matrix: {adjacency_matrix}')
        local_print(f'ContractorOptEinsum.contract_with_inputs called with cores_name: {cores_name}')

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from ..utils.graph_generators import QCTNHelper
        for element in QCTNHelper.triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols  # Ensure symmetry
                
        
        inputs_equation_lefthand = ''
        for idx, _ in enumerate(cores_name):
            local_print(f"Processing core: {cores_name[idx]} with input ranks {input_ranks[idx]}, adjacency matrix {adjacency_matrix[idx, :]}, output ranks {output_ranks[idx]}")
            for _ in input_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                inputs_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

            einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                einsum_equation_righthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1
            
            einsum_equation_lefthand += ','

        einsum_equation_lefthand = f'{inputs_equation_lefthand},{einsum_equation_lefthand[:-1]}'
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'

        tensor_shapes = [inputs.shape] + [cores_weights[core_name].shape for core_name in cores_name]

        for core_name in cores_name:
            local_print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')
        
        local_print(f'QCTN cores: {qctn.cores}')
        local_print(f'Einsum Equation: {einsum_equation}')
        local_print(f'Tensor Shapes: {tensor_shapes}')

        qctn.einsum_expr = opt_einsum.contract_expression(einsum_equation, *tensor_shapes, optimize=Configuration.opt_einsum_optimize)
        jit_retraction = jax.jit(qctn.einsum_expr)

        inputs_cores = [inputs] + [cores_weights[core_name] for core_name in cores_name]
        retracted_QCTN = jit_retraction(*inputs_cores)

        return retracted_QCTN
    
    @staticmethod
    def contract_with_vector_inputs(qctn, inputs):
        """
        Contract the given QCTN with vector inputs.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs (list[jnp.ndarray]): The vector inputs for the contraction operation.

        Returns:
            jnp.ndarray: The result of the tensor contraction.
        """

        input_ranks, adjacency_matrix, output_ranks = _build_circuit_from_adjacency_table(qctn)
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from ..utils.graph_generators import QCTNHelper
        for element in QCTNHelper.triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols  # Ensure symmetry

        inputs_equation_lefthand = ''
        for idx, _ in enumerate(cores_name):
            local_print(f"Processing core: {cores_name[idx]} with input ranks {input_ranks[idx]}, adjacency matrix {adjacency_matrix[idx, :]}, output ranks {output_ranks[idx]}")
            for _ in input_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                inputs_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                inputs_equation_lefthand += ','
                symbol_id += 1

            einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                einsum_equation_righthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1
            
            einsum_equation_lefthand += ','

        einsum_equation_lefthand = f'{inputs_equation_lefthand}{einsum_equation_lefthand[:-1]}'
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'

        tensor_shapes = [v.shape for v in inputs] + [cores_weights[core_name].shape for core_name in cores_name]

        for core_name in cores_name:
            local_print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')
        
        local_print(f'QCTN cores: {qctn.cores}')
        local_print(f'Einsum Equation: {einsum_equation}')
        local_print(f'Tensor Shapes: {tensor_shapes}')

        qctn.einsum_expr = opt_einsum.contract_expression(einsum_equation, *tensor_shapes, optimize=Configuration.opt_einsum_optimize)
        jit_retraction = jax.jit(qctn.einsum_expr)

        inputs_cores = inputs + [cores_weights[core_name] for core_name in cores_name]
        retracted_QCTN = jit_retraction(*inputs_cores)

        return retracted_QCTN
    
    @staticmethod
    def contract_with_QCTN(qctn, target_qctn, initialization_mode=False):
        """
        Contract the given QCTN with a target QCTN.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            target_qctn (QCTN): The target quantum circuit tensor network for contraction.
        
        Returns:
            jnp.ndarray: The result of the tensor contraction.
        """

        # print("ContractorOptEinsum.contract_with_QCTN called")

        input_ranks, adjacency_matrix, output_ranks = _build_circuit_from_adjacency_table(qctn)
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

        target_input_ranks, target_adjacency_matrix, target_output_ranks = _build_circuit_from_adjacency_table(target_qctn)
        target_cores_name = target_qctn.cores
        target_cores_weights = target_qctn.cores_weights

        symbol_id = 0
        einsum_equation_lefthand = ''
        target_einsum_equation_lefthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()
        local_print(f'adjacency_matrix for interaction: \n{adjacency_matrix_for_interaction}')
        
        from ..utils.graph_generators import QCTNHelper
        for element in QCTNHelper.triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols  # Ensure symmetry

        local_print(f"adjacency_matrix after symbol assignment: \n{adjacency_matrix}")

        target_adjacency_matrix_for_interaction = target_adjacency_matrix.copy()
        for element in QCTNHelper.triu_ndindex(len(target_cores_name)):
            i, j = element
            if target_adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(target_adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                target_adjacency_matrix[i, j] = connection_symbols
                target_adjacency_matrix[j, i] = connection_symbols

        local_print(f"target_adjacency_matrix after symbol assignment: \n{target_adjacency_matrix}")

        input_symbols_stack = []
        output_symbols_stack = []

        # self.einsum_equation_lefthand
        for idx, _ in enumerate(cores_name):
            for _ in input_ranks[idx]:
                symbol = opt_einsum.get_symbol(symbol_id)
                einsum_equation_lefthand += symbol
                input_symbols_stack.append(symbol)
                symbol_id += 1

            einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                symbol = opt_einsum.get_symbol(symbol_id)
                einsum_equation_lefthand += symbol
                output_symbols_stack.append(symbol)
                symbol_id += 1
            
            einsum_equation_lefthand += ','
        
        local_print(f'input_symbols_stack after source QCTN processing: {input_symbols_stack}')
        local_print(f'output_symbols_stack after source QCTN processing: {output_symbols_stack}')

        # target.einsum_equation_lefthand
        for idx, _ in enumerate(target_cores_name):
            for _ in target_input_ranks[idx]:
                target_einsum_equation_lefthand += input_symbols_stack.pop(0)

            target_einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(target_adjacency_matrix[idx])))

            for _ in target_output_ranks[idx]:
                target_einsum_equation_lefthand += output_symbols_stack.pop(0)
            
            target_einsum_equation_lefthand += ','

        einsum_equation = f'{einsum_equation_lefthand}{target_einsum_equation_lefthand[:-1]}->'

        tensor_shapes = [cores_weights[core_name].shape for core_name in cores_name] + \
                        [target_cores_weights[core_name].shape for core_name in target_cores_name]


        for core_name in cores_name:
            local_print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')

        for core_name in target_cores_name:
            local_print(f'Target Core: {core_name}, Shape: {target_cores_weights[core_name].shape}')
        
        local_print(f'QCTN cores: {qctn.cores}')
        local_print(f'Target QCTN cores: {target_qctn.cores}')
        local_print(f'Einsum Equation: {einsum_equation}')
        local_print(f'Tensor Shapes: {tensor_shapes}')

        qctn.einsum_expr = opt_einsum.contract_expression(einsum_equation, *tensor_shapes, optimize=Configuration.opt_einsum_optimize)
        jit_retraction = jax.jit(qctn.einsum_expr)

        local_print(f'qctn.einsum_expr: {qctn.einsum_expr is None}')

        if initialization_mode:
            # In initialization mode, we do not actually contract the target QCTN
            return qctn.einsum_expr

        inputs_cores = [cores_weights[core_name] for core_name in cores_name] + \
                       [target_cores_weights[core_name] for core_name in target_cores_name]
        
        retracted_QCTN = jit_retraction(*inputs_cores)

        local_print(f"retracted_QCTN: {retracted_QCTN}")

        return retracted_QCTN
    
    @staticmethod
    def contract_with_self(qctn, circuit_states=None, mx=None,  initialization_mode=False):
        """
        Contract the given QCTN with itself.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
        
        Returns:
            jnp.ndarray: The result of the tensor contraction.
        """
        

        input_ranks, adjacency_matrix, output_ranks = _build_circuit_from_adjacency_table(qctn)
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

        symbol_id = 0
        einsum_equation_lefthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()
        local_print(f'adjacency_matrix for interaction: \n{adjacency_matrix_for_interaction}')
        
        from ..utils.graph_generators import QCTNHelper
        for element in QCTNHelper.triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols  # Ensure symmetry

        local_print(f"adjacency_matrix after symbol assignment: \n{adjacency_matrix}")

        input_symbols_stack = []
        output_symbols_stack = []

        equation_list = []
        new_symbol_mapping = {}

        # self.einsum_equation_lefthand
        for idx, _ in enumerate(cores_name):
            core_equation = ""

            for _ in input_ranks[idx]:
                symbol = opt_einsum.get_symbol(symbol_id)
                core_equation += symbol
                input_symbols_stack.append(symbol)
                symbol_id += 1

            core_equation += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                symbol = opt_einsum.get_symbol(symbol_id)
                core_equation += symbol
                output_symbols_stack.append(symbol)
                symbol_id += 1
            
            einsum_equation_lefthand += core_equation
            equation_list.append(core_equation)

        real_output_symbols_stack = []

        inv_equation_list = []
        for core_equation in equation_list[::-1]:
            new_equation = ""
            for char in core_equation:
                if char in output_symbols_stack:
                    new_equation += char
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

        equation_list += inv_equation_list

        einsum_equation_lefthand = ",".join(equation_list)

        local_print(f'einsum_equation_lefthand: {einsum_equation_lefthand}')
        local_print(f'input_symbols_stack after source QCTN processing: {input_symbols_stack}')
        local_print(f'output_symbols_stack after source QCTN processing: {output_symbols_stack}')

        if circuit_array_input is not None:
            local_print(f"check circuit_array_input shape: {circuit_array_input.shape}, expected input_ranks: {input_ranks}")
            # TODO: correct this code, check the circuit_array_input shape
            # for idx, core_name in enumerate(cores_name):
            #     expected_input_rank = len(input_ranks[idx])
            #     if circuit_array_input[idx].ndim != expected_input_rank:
            #         raise ValueError(f'circuit_array_input[{idx}] has shape {circuit_array_input[idx].shape}, expected input rank {expected_input_rank}')
            einsum_equation_output = ''.join(real_output_symbols_stack)

            einsum_equation_lefthand = f"{''.join(input_symbols_stack)},{einsum_equation_lefthand},{einsum_equation_output}"

        einsum_equation = f'{einsum_equation_lefthand}->'

        tensor_shapes = [cores_weights[core_name].shape for core_name in cores_name] + \
                        [cores_weights[core_name].shape for core_name in cores_name[::-1]]

        if circuit_array_input is not None:
            tensor_shapes = [circuit_array_input.shape] + tensor_shapes + [circuit_array_input.shape]

        for core_name in cores_name:
            local_print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')

        local_print(f'QCTN cores: {qctn.cores}')
        local_print(f'Einsum Equation: {einsum_equation}')
        local_print(f'Tensor Shapes: {tensor_shapes}')

        qctn.einsum_expr = opt_einsum.contract_expression(einsum_equation, *tensor_shapes, optimize=Configuration.opt_einsum_optimize)
        jit_retraction = jax.jit(qctn.einsum_expr)

        local_print(f'qctn.einsum_expr: {qctn.einsum_expr is None}')

        if initialization_mode:
            # In initialization mode, we do not actually contract the target QCTN
            return qctn.einsum_expr

        inputs_cores =  [cores_weights[core_name] for core_name in cores_name] + \
                        [cores_weights[core_name] for core_name in cores_name[::-1]]
        
        if circuit_array_input is not None:
            inputs_cores = [circuit_array_input] + inputs_cores + [circuit_array_input]


        retracted_QCTN = jit_retraction(*inputs_cores)

        local_print(f"retracted_QCTN: {retracted_QCTN}")

        return retracted_QCTN

    @staticmethod
    def contract_with_self_for_gradient(qctn, circuit_array_input=None, circuit_list_input=None):
        """
        We use JAX's autograd to compute the core gradient.
        """
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

        inputs_cores =  [cores_weights[core_name] for core_name in cores_name] + \
                        [cores_weights[core_name] for core_name in cores_name[::-1]]

        if circuit_array_input is not None:
            inputs_cores = [circuit_array_input] + inputs_cores + [circuit_array_input]

        def mse_loss_fn(*inputs_cores):
            retracted_QCTN = qctn.einsum_expr(*inputs_cores)
            return jnp.mean((retracted_QCTN - 1.0) ** 2) 

        # print(f'ContractorOptEinsum.contract_with_QCTN_for_gradient called {qctn.einsum_expr is None}')

        if qctn.einsum_expr is None:
            ContractorOptEinsum.contract_with_self(qctn, circuit_array_input=circuit_array_input, initialization_mode=True)
            argnums = list(range(len(cores_name)))
            if circuit_array_input is not None:
                argnums = list(range(len(cores_name) + 2))  # +2 for the circuit_array_input at start and end

            qctn.jit_retraction_with_self_value_gradient = jax.jit(jax.value_and_grad(mse_loss_fn, 
                                                                                      argnums=argnums))

        loss, grad_cores = qctn.jit_retraction_with_self_value_gradient(*inputs_cores)
        
        if circuit_array_input is not None:
            grad_cores = grad_cores[1:-1]  # Remove gradients for circuit_array_input

        # local_print('inputs_cores', [x.shape for x in inputs_cores])
        # local_print('mean', [x.mean() for x in inputs_cores])
        # local_print('var', [x.var() for x in inputs_cores])
        # local_print('grad_cores', [x.shape for x in grad_cores])

        local_print(f'Loss: {loss}')
        
        # local_print(f"-"*70)

        return loss, grad_cores

    @staticmethod
    def contract_with_QCTN_for_gradient(qctn, target_qctn):
        """
        We use JAX's autograd to compute the core gradient.
        """

        cores_name = qctn.cores
        cores_weights = qctn.cores_weights
        target_cores_name = target_qctn.cores
        target_cores_weights = target_qctn.cores_weights
        
        inputs_cores = [cores_weights[core_name] for core_name in cores_name] + \
            [target_cores_weights[core_name] for core_name in target_cores_name]

        def mse_loss_fn(*inputs_cores):
            retracted_QCTN = qctn.einsum_expr(*inputs_cores)
            return jnp.mean((retracted_QCTN - 1.0) ** 2) 

        # print(f'ContractorOptEinsum.contract_with_QCTN_for_gradient called {qctn.einsum_expr is None}')

        if qctn.einsum_expr is None:
            # print("contract with QCTN")
            
            ContractorOptEinsum.contract_with_QCTN(qctn, target_qctn, initialization_mode=True)
            argnums = list(range(len(cores_name)))

            # print(f'argnums: {argnums}')

            qctn.jit_retraction_with_QCTN_value_gradient = jax.jit(jax.value_and_grad(mse_loss_fn, 
                                                                                      argnums=argnums))

        # print(f"qctn.jit_retraction_with_QCTN_value_gradient: {qctn.jit_retraction_with_QCTN_value_gradient is None}")

        # print('inputs_cores', inputs_cores, len(inputs_cores))
        # print('inputs_cores', [x.shape for x in inputs_cores])

        local_print(f"-"*70)

        local_print('inputs_cores', [x.shape for x in inputs_cores])
        local_print('mean', [x.mean() for x in inputs_cores])
        local_print('var', [x.var() for x in inputs_cores])

        # local_print(f"-"*70)

        loss, grad_cores = qctn.jit_retraction_with_QCTN_value_gradient(*inputs_cores)
        
        # local_print('inputs_cores', [x.shape for x in inputs_cores])
        # local_print('mean', [x.mean() for x in inputs_cores])
        # local_print('var', [x.var() for x in inputs_cores])

        # local_print(f"-"*70)

        local_print(f'Loss: {loss}')
        

        return loss, grad_cores