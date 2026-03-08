"""
Unified engine that combines contractor (expression generation) with backend (execution).

This module provides high-level functions that use EinsumStrategy to generate
expressions and then execute them using the specified backend.

Now supports strategy-based compilation for optimized contraction paths.
"""

from __future__ import annotations
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np
import torch

from ..contractor import EinsumStrategy, StrategyCompiler
from ..backends.backend_factory import BackendFactory, ComputeBackend


class Engine:
    """
    Engine that combines tensor contraction expression generation with backend execution.
    
    This class separates concerns:
    - EinsumStrategy: Generates einsum expressions using opt_einsum (legacy)
    - StrategyCompiler: Compiles optimal strategies based on network structure
    - ComputeBackend: Executes expressions using JAX, PyTorch, etc.
    """

    def __init__(self, backend: Optional[Union[str, ComputeBackend]] = None, strategy_mode: str = 'balanced'):
        """
        Initialize the engine with a specific backend and strategy mode.
        
        Args:
            backend (str or ComputeBackend, optional): Backend to use. 
                Can be 'jax', 'pytorch', or a ComputeBackend instance.
                If None, uses the default backend.
            strategy_mode (str): Contraction strategy mode:
                - 'fast': Use einsum only (fastest compilation)
                - 'balanced': Use einsum + MPS chain (balanced)
                - 'full': Use all available strategies (slowest compilation, best runtime)
        """
        if backend is None:
            self.backend = BackendFactory.get_default_backend()
        elif isinstance(backend, str):
            self.backend = BackendFactory.create_backend(backend, device="cuda")
        else:
            self.backend = backend

        self.contractor = EinsumStrategy()  # Keep for legacy methods
        self.strategy_compiler = StrategyCompiler(mode=strategy_mode)
        self.strategy_mode = strategy_mode

    # ============================================================================
    # Strategy-based Compilation Methods (NEW API)
    # ============================================================================

    def contract_with_compiled_strategy(self, qctn, circuit_states=None, measure_input=None, 
                                       measure_is_matrix=True, force_recompile: bool = False):
        """
        Contract using compiled strategy (auto-selected based on mode).
        
        This is the NEW recommended API that automatically selects the best strategy.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states (array or list, optional): Circuit input states.
            measure_input (array or list, optional): Measurement input.
            measure_is_matrix (bool): If True, measure_input is the outer product matrix.
            force_recompile (bool): Force recompilation even if cached.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        # Prepare shapes_info
        states_shape = None
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                circuit_states = [self.backend.convert_to_tensor(s) for s in circuit_states]
                states_shape = tuple([s.shape for s in circuit_states])
            else:
                circuit_states = self.backend.convert_to_tensor(circuit_states)
                states_shape = circuit_states.shape
        
        measure_shape = None
        if measure_input is not None:
            if isinstance(measure_input, list):
                measure_input = [self.backend.convert_to_tensor(m) for m in measure_input]
                measure_shape = tuple([m.shape for m in measure_input])
            else:
                measure_input = self.backend.convert_to_tensor(measure_input)
                measure_shape = measure_input.shape
        
        shapes_info = {
            'circuit_states_shapes': states_shape,
            'measure_shapes': measure_shape,
            'measure_is_matrix': measure_is_matrix
        }
        
        # Check cache
        cache_key = f'_compiled_strategy_{self.strategy_mode}_{states_shape}_{measure_shape}_{measure_is_matrix}'
        
        if force_recompile or not hasattr(qctn, cache_key):
            # Compile strategy
            compute_fn, strategy_name, cost = self.strategy_compiler.compile(qctn, shapes_info, self.backend)
            
            # Cache the result
            setattr(qctn, cache_key, {
                'compute_fn': compute_fn,
                'strategy_name': strategy_name,
                'cost': cost
            })
            # print(f"[Engine] Compiled and cached strategy: {strategy_name}")
        else:
            cached = getattr(qctn, cache_key)
            compute_fn = cached['compute_fn']
            strategy_name = cached['strategy_name']
            # print(f"[Engine] Using cached strategy: {strategy_name}")
        
        # Prepare data
        cores_dict = {name: self.backend.convert_to_tensor(qctn.cores_weights[name]) for name in qctn.cores}
        
        # Execute
        result = compute_fn(cores_dict, circuit_states, measure_input)
        
        return result

    def contract_with_compiled_strategy_for_gradient(self, qctn, circuit_states_list=None, measure_input_list=None, 
                                                    measure_is_matrix=True, force_recompile: bool = False) -> Tuple:
        """
        Contract using compiled strategy and compute gradients.
        
        This is the NEW recommended API for gradient computation.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states_list (array or list, optional): Circuit input states.
            measure_input_list (array or list, optional): Measurement input.
            measure_is_matrix (bool): If True, measure_input is the outer product matrix.
            force_recompile (bool): Force recompilation even if cached.
        
        Returns:
            tuple: (loss, gradients)
        """
        circuit_states = circuit_states_list
        measure_input = measure_input_list

        # Prepare shapes_info
        states_shape = None
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                circuit_states = [self.backend.convert_to_tensor(s) for s in circuit_states]
                states_shape = tuple([s.shape for s in circuit_states])
            else:
                circuit_states = self.backend.convert_to_tensor(circuit_states)
                states_shape = circuit_states.shape
        
        measure_shape = None
        if measure_input is not None:
            if isinstance(measure_input, list):
                measure_input = [self.backend.convert_to_tensor(m) for m in measure_input]
                measure_shape = tuple([m.shape for m in measure_input])
            else:
                measure_input = self.backend.convert_to_tensor(measure_input)
                measure_shape = measure_input.shape
        
        shapes_info = {
            'circuit_states_shapes': states_shape,
            'measure_shapes': measure_shape,
            'measure_is_matrix': measure_is_matrix
        }
        
        # Check cache
        cache_key = f'_compiled_strategy_{self.strategy_mode}_{states_shape}_{measure_shape}_{measure_is_matrix}'
        
        if force_recompile or not hasattr(qctn, cache_key):
            # Compile strategy
            compute_fn, strategy_name, cost = self.strategy_compiler.compile(qctn, shapes_info, self.backend)
            
            # Cache the result
            setattr(qctn, cache_key, {
                'compute_fn': compute_fn,
                'strategy_name': strategy_name,
                'cost': cost
            })
            print(f"[Engine] Compiled and cached strategy: {strategy_name}")
        else:
            cached = getattr(qctn, cache_key)
            compute_fn = cached['compute_fn']
            strategy_name = cached['strategy_name']
            # print(f"[Engine] Using cached strategy: {strategy_name}")
            
        # Define loss function
        def loss_fn(*core_tensors):
            # Reconstruct cores_dict
            cores_dict = {name: tensor for name, tensor in zip(qctn.cores, core_tensors)}
            
            # Execute contraction
            result = compute_fn(cores_dict, circuit_states, measure_input)
            
            # Compute Cross Entropy loss
            # Target is all ones (maximizing probability)
            target = torch.ones_like(result)
            
            # Avoid log(0)
            result = torch.clamp(result, min=1e-10)
            log_result = torch.log(result)
            return -torch.mean(target * log_result)
        
        # Prepare core tensors
        core_tensors = [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]
        
        # Compute gradients
        # We want gradients with respect to all cores
        argnums = list(range(len(core_tensors)))
        
        # Create value_and_grad function
        value_and_grad_fn = self.backend.compute_value_and_grad(loss_fn, argnums=argnums)
        
        # Execute
        loss, grads = value_and_grad_fn(*core_tensors)
        
        return loss, grads

    # ============================================================================
    # Legacy Methods (kept for backward compatibility)
    # ============================================================================

    def contract_core_only(self, qctn):
        """
        Contract QCTN cores only (no inputs).
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_core_only_expression(qctn)
        
        # Create optimized expression
        if not hasattr(qctn, '_contract_expr_core_only'):
            qctn._contract_expr_core_only = self.contractor.create_contract_expression(
                einsum_eq, tensor_shapes
            )
        
        # Prepare tensors
        tensors = [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]
        
        # Execute
        jit_fn = self.backend.jit_compile(qctn._contract_expr_core_only)
        return self.backend.execute_expression(jit_fn, *tensors)

    def contract_with_inputs(self, qctn, inputs):
        """
        Contract QCTN with input tensor.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs (array): Input tensor.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        inputs = self.backend.convert_to_tensor(inputs)
        
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_with_inputs_expression(qctn, inputs.shape)
        
        # Create optimized expression if not cached
        cache_key = f'_contract_expr_inputs_{inputs.shape}'
        if not hasattr(qctn, cache_key):
            expr = self.contractor.create_contract_expression(einsum_eq, tensor_shapes)
            setattr(qctn, cache_key, expr)
        else:
            expr = getattr(qctn, cache_key)
        
        # Prepare tensors
        tensors = [inputs] + [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]
        
        # Execute
        jit_fn = self.backend.jit_compile(expr)
        return self.backend.execute_expression(jit_fn, *tensors)

    def contract_with_vector_inputs(self, qctn, inputs: List):
        """
        Contract QCTN with vector inputs.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs (list): List of input tensors.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        inputs = [self.backend.convert_to_tensor(inp) for inp in inputs]
        inputs_shapes = [inp.shape for inp in inputs]
        
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_with_vector_inputs_expression(qctn, inputs_shapes)
        
        # Create optimized expression if not cached
        cache_key = f'_contract_expr_vector_inputs_{tuple(inputs_shapes)}'
        if not hasattr(qctn, cache_key):
            expr = self.contractor.create_contract_expression(einsum_eq, tensor_shapes)
            setattr(qctn, cache_key, expr)
        else:
            expr = getattr(qctn, cache_key)
        
        # Prepare tensors
        tensors = inputs + [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]
        
        # Execute
        jit_fn = self.backend.jit_compile(expr)
        return self.backend.execute_expression(jit_fn, *tensors)

    def contract_with_qctn(self, qctn, target_qctn):
        """
        Contract QCTN with another QCTN.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            target_qctn (QCTN): The target QCTN.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_with_qctn_expression(qctn, target_qctn)
        
        # Create optimized expression if not cached
        if not hasattr(qctn, '_contract_expr_with_qctn'):
            qctn._contract_expr_with_qctn = self.contractor.create_contract_expression(
                einsum_eq, tensor_shapes
            )
        
        # Prepare tensors
        tensors = [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores] + \
                  [self.backend.convert_to_tensor(target_qctn.cores_weights[c]) for c in target_qctn.cores]
        
        # Execute
        jit_fn = self.backend.jit_compile(qctn._contract_expr_with_qctn)
        return self.backend.execute_expression(jit_fn, *tensors)

    def contract_with_self(self, qctn, circuit_states=None, measure_input=None, measure_is_matrix=False):
        """
        Contract QCTN with itself (hermitian conjugate).
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states (array or list, optional): Circuit input states. 
                Can be numpy array, torch tensor, jax ndarray, or a list of such tensors.
            measure_input (array or list, optional): Measurement input.
                Can be numpy array, torch tensor, jax ndarray, or a list of such tensors.
            measure_is_matrix (bool): If True, measure_input is the outer product matrix Mx;
                If False, measure_input is the vector phi_x.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        # Convert circuit_states
        states_shape = None
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                circuit_states = [self.backend.convert_to_tensor(s) for s in circuit_states]
                states_shape = tuple([s.shape for s in circuit_states])
            else:
                circuit_states = self.backend.convert_to_tensor(circuit_states)
                states_shape = circuit_states.shape
        
        # Convert measure_input
        measure_shape = None
        if measure_input is not None:
            if isinstance(measure_input, list):
                measure_input = [self.backend.convert_to_tensor(m) for m in measure_input]
                measure_shape = tuple([m.shape for m in measure_input])
            else:
                measure_input = self.backend.convert_to_tensor(measure_input)
                measure_shape = measure_input.shape
        
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_with_self_expression(
            qctn, states_shape, measure_shape, measure_is_matrix
        )

        # print(f'ori einsum_eq : {einsum_eq}')
        # einsum_eq = "c,d,f,h,cdea,afgb,bhji,kel,kgm,kjo,kin,qpon,srmq,utls,p,r,t,u->k"
        
        # Create optimized expression if not cached
        cache_key = f'_contract_expr_self_{states_shape}_{measure_shape}_{measure_is_matrix}'
        if not hasattr(qctn, cache_key):
            expr = self.contractor.create_contract_expression(einsum_eq, tensor_shapes)
            setattr(qctn, cache_key, expr)
        else:
            expr = getattr(qctn, cache_key)
        
        # Prepare tensors
        tensors = []
        
        # Add circuit_states
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                tensors.extend(circuit_states)
            else:
                tensors.append(circuit_states)
        
        # Add cores
        tensors += [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]

        # Add measure_input
        if measure_input is not None:
            if isinstance(measure_input, list):
                tensors.extend(measure_input)
            else:
                tensors.append(measure_input)
        
        # Add inverse cores
        tensors += [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores[::-1]]
        
        # Add circuit_states again (conjugate side)
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                tensors.extend(circuit_states)
            else:
                tensors.append(circuit_states)
        
        # TODO: use general interface, this will cause error with jax
        for i in range(len(tensors)):
            tensors[i] = tensors[i].cuda()

        # Execute
        jit_fn = self.backend.jit_compile(expr)
        return self.backend.execute_expression(jit_fn, *tensors)

    def contract_with_self_for_gradient(self, qctn, circuit_states=None, measure_input=None, measure_is_matrix=False) -> Tuple:
        """
        Contract QCTN with itself and compute gradients.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states (array or list, optional): Circuit input states. 
                Can be numpy array, torch tensor, jax ndarray, or a list of such tensors.
            measure_input (array or list, optional): Measurement input.
                Can be numpy array, torch tensor, jax ndarray, or a list of such tensors.
            measure_is_matrix (bool): If True, measure_input is the outer product matrix Mx;
                If False, measure_input is the vector phi_x.
        
        Returns:
            tuple: (loss, gradients)
        """
        # Convert circuit_states
        states_shape = None
        num_states_tensors = 0
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                circuit_states = [self.backend.convert_to_tensor(s) for s in circuit_states]
                states_shape = tuple([s.shape for s in circuit_states])
                num_states_tensors = len(circuit_states)
            else:
                circuit_states = self.backend.convert_to_tensor(circuit_states)
                states_shape = circuit_states.shape
                num_states_tensors = 1
        
        # Convert measure_input
        measure_shape = None
        num_measure_tensors = 0
        if measure_input is not None:
            if isinstance(measure_input, list):
                measure_input = [self.backend.convert_to_tensor(m) for m in measure_input]
                measure_shape = tuple([m.shape for m in measure_input])
                num_measure_tensors = len(measure_input)
            else:
                measure_input = self.backend.convert_to_tensor(measure_input)
                measure_shape = measure_input.shape
                num_measure_tensors = 1
        
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_with_self_expression(
            qctn, states_shape, measure_shape, measure_is_matrix
        )
        
        # Create optimized expression if not cached
        cache_key = f'_contract_expr_self_{states_shape}_{measure_shape}_{measure_is_matrix}'
        if not hasattr(qctn, cache_key):
            expr = self.contractor.create_contract_expression(einsum_eq, tensor_shapes)
            setattr(qctn, cache_key, expr)
        else:
            expr = getattr(qctn, cache_key)

        # Define loss function
        def loss_fn(*tensors):
            jit_fn = self.backend.jit_compile(expr)
            result = self.backend.execute_expression(jit_fn, *tensors)
            # Compute MSE loss
            diff = result - 1.0
            return (diff * diff).mean()

            # compute cross entropy loss
            # target = torch.ones_like(result)
            # log_result = torch.log(result + 1e-10)
            # return -torch.mean(target * log_result)

        # Determine which arguments to compute gradients for
        # Skip circuit_states and measure_input, only compute gradients for cores
        num_cores = len(qctn.cores)
        total_args = num_states_tensors * 2 + num_cores * 2 + num_measure_tensors
        # argnums = list(range(0, total_args))

        cores_index = list(range(num_states_tensors, num_states_tensors + num_cores))
        inv_cores_index = list(range(num_states_tensors + num_cores + num_measure_tensors,
                                   num_states_tensors + num_cores + num_measure_tensors + num_cores))
        argnums = cores_index + inv_cores_index
        
        # Create value_and_grad function
        cache_key_grad = f'_grad_fn_self_{states_shape}_{measure_shape}_{measure_is_matrix}'
        if not hasattr(qctn, cache_key_grad):
            value_and_grad_fn = self.backend.compute_value_and_grad(loss_fn, argnums=argnums)
            value_and_grad_fn = self.backend.jit_compile(value_and_grad_fn)
            setattr(qctn, cache_key_grad, value_and_grad_fn)
        else:
            value_and_grad_fn = getattr(qctn, cache_key_grad)
        
        # Prepare tensors
        tensors = []
        
        # Add circuit_states
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                tensors.extend(circuit_states)
            else:
                tensors.append(circuit_states)
        
        # Add cores
        tensors += [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]

        # Add measure_input
        if measure_input is not None:
            if isinstance(measure_input, list):
                tensors.extend(measure_input)
            else:
                tensors.append(measure_input)

        # Add inverse cores
        tensors += [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores[::-1]]
        
        # Add circuit_states again (conjugate side)
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                tensors.extend(circuit_states)
            else:
                tensors.append(circuit_states)
        
        # TODO: use general interface, this will cause error with jax
        for i in range(len(tensors)):
            tensors[i] = tensors[i].cuda()

        # print('input tensors', len(tensors))

        # Compute value and gradients
        loss, grads = value_and_grad_fn(*tensors)


        grads = [grads[i] + grads[-j] for i, j in zip(range(0, len(cores_index)), range(1, len(cores_index)+1))]
        
        # print(f'grads for cores: {len(grads)}')

        return loss, grads

    def contract_with_qctn_for_gradient(self, qctn, target_qctn) -> Tuple:
        """
        Contract QCTN with target QCTN and compute gradients.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            target_qctn (QCTN): The target QCTN.
        
        Returns:
            tuple: (loss, gradients)
        """
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_with_qctn_expression(qctn, target_qctn)
        
        # Create optimized expression if not cached
        if not hasattr(qctn, '_contract_expr_with_qctn'):
            qctn._contract_expr_with_qctn = self.contractor.create_contract_expression(
                einsum_eq, tensor_shapes
            )
        
        expr = qctn._contract_expr_with_qctn
        
        # Define loss function
        def loss_fn(*tensors):
            jit_fn = self.backend.jit_compile(expr)
            result = self.backend.execute_expression(jit_fn, *tensors)
            # # Compute MSE loss
            # diff = result - 1.0
            # return (diff * diff).mean()

            # compute cross entropy loss
            target = torch.ones_like(result)
            log_result = torch.log(result + 1e-10)
            return -torch.mean(target * log_result)
        
        # Only compute gradients for qctn cores, not target_qctn
        argnums = list(range(len(qctn.cores)))
        
        # Create value_and_grad function
        if not hasattr(qctn, '_grad_fn_with_qctn'):
            value_and_grad_fn = self.backend.compute_value_and_grad(loss_fn, argnums=argnums)
            value_and_grad_fn = self.backend.jit_compile(value_and_grad_fn)
            qctn._grad_fn_with_qctn = value_and_grad_fn
        else:
            value_and_grad_fn = qctn._grad_fn_with_qctn
        
        # Prepare tensors
        tensors = [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores] + \
                  [self.backend.convert_to_tensor(target_qctn.cores_weights[c]) for c in target_qctn.cores]
        
        # Compute value and gradients
        loss, grads = value_and_grad_fn(*tensors)
        
        return loss, grads

    def contract_with_std_graph_mini(self, qctn, circuit_states_list, measure_input_list):
        """
        Contract QCTN using standard graph method with pre-contracted circuit states.
        
        This method first contracts circuit_states with cores to create 
        cores_weight_with_circuit_states, then uses those for efficient computation.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states_list (list): List of circuit input states (one per qubit).
                Each element is a tensor representing the state of one qubit.
            measure_input_list (list): List of measurement input matrices.
                Each element is a matrix (not vector) for measurement.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        
        # Ensure inputs are lists
        assert isinstance(circuit_states_list, list), "circuit_states_list must be a list"
        assert isinstance(measure_input_list, list), "measure_input_list must be a list"
        
        # Step 1: Initialize cores_weight_with_circuit_states if first time
        if not hasattr(qctn, 'cores_weight_with_circuit_states'):
            # Convert circuit_states to backend tensors
            circuit_states_tensors = [self.backend.convert_to_tensor(s) for s in circuit_states_list]
            
            # Initialize the dictionary to store contracted cores
            qctn.cores_weight_with_circuit_states = {}
            
            # Verify dimensions: cores should be one less than circuit_states
            num_cores = len(qctn.cores)
            num_states = len(circuit_states_tensors)
            assert num_cores == num_states - 1, \
                f"Number of cores ({num_cores}) should be one less than circuit_states ({num_states})"
            
            # Contract each core with its corresponding circuit_state(s)
            for idx, core_name in enumerate(qctn.cores):
                core_tensor = self.backend.convert_to_tensor(qctn.cores_weights[core_name])
                
                if idx == 0:
                    state1 = circuit_states_tensors[0]
                    state2 = circuit_states_tensors[1]
                    
                    # Contract with both states
                    # core_tensor shape: (input_dim1, input_dim2, bond_dims..., output_dim)
                    # state1 shape: (input_dim1,)
                    # state2 shape: (input_dim2,)
                    # result shape: (bond_dims..., output_dim)
                    contracted = torch.einsum('i,j,ij...->...', state1, state2, core_tensor)
                    qctn.cores_weight_with_circuit_states[core_name] = contracted
                else:
                    state = circuit_states_tensors[idx + 1]
                    
                    # Contract along the first dimension
                    # core_tensor shape: (input_dim, bond_dims..., output_dim)
                    # state shape: (input_dim,)
                    # result shape: (bond_dims..., output_dim)
                    contracted = torch.einsum('i,ji...->j...', state, core_tensor)
                    qctn.cores_weight_with_circuit_states[core_name] = contracted
        
        print(f"cores_weight_with_circuit_states shape {[(k, v.shape) for k, v in qctn.cores_weight_with_circuit_states.items()]}")

        # Step 2: Define computation function
        def compute_with_measure(qctn_inner, measure_matrices):
            """
            Compute contraction with measure_input matrices.
            
            Args:
                qctn_inner: QCTN object with cores_weight_with_circuit_states
                measure_matrices: List of measurement matrices (one per output)
            
            Returns:
                Result of contraction
            """
            
            n = len(qctn_inner.cores_weight_with_circuit_states)

            for idx in range(n):
                if idx == 0:
                    core_tensor = qctn_inner.cores_weight_with_circuit_states[qctn_inner.cores[0]]
                    measure_matrix = measure_matrices[0]

                    # A * M * A_T
                    contracted = torch.einsum('ka,zkl,lb->zab', core_tensor, measure_matrix, core_tensor)
                elif idx < n - 1:
                    core_tensor = qctn_inner.cores_weight_with_circuit_states[qctn_inner.cores[idx]]
                    measure_matrix = measure_matrices[idx]

                    # contracted * B * M * B_T
                    contracted = torch.einsum('zab,akc,zkl,bld->zcd', contracted, core_tensor, measure_matrix, core_tensor)
                else:
                    core_tensor = qctn_inner.cores_weight_with_circuit_states[qctn_inner.cores[idx]]
                    measure_matrix_1 = measure_matrices[idx]
                    measure_matrix_2 = measure_matrices[idx + 1]

                    # contracted * Z * M * Z_T
                    contracted = torch.einsum('zab,akc,zkl,zcd,bld->z', contracted, core_tensor, measure_matrix_1, measure_matrix_2, core_tensor)
            return contracted
        
        # Step 3: Call the computation function and return result
        # Convert measure_input_list to backend tensors
        measure_matrices = [self.backend.convert_to_tensor(m) for m in measure_input_list]
        
        # Call computation function
        result = compute_with_measure(qctn, measure_matrices)
        
        return result

    def contract_with_std_graph(self, qctn, circuit_states_list, measure_input_list):
        """
        Contract QCTN using standard graph method with pre-contracted circuit states.
        
        This method first contracts circuit_states with cores to create 
        cores_weight_with_circuit_states, then uses those for efficient computation.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states_list (list): List of circuit input states (one per qubit).
                Each element is a tensor representing the state of one qubit.
            measure_input_list (list): List of measurement input matrices.
                Each element is a matrix (not vector) for measurement.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        def compute_fn(qctn_inner, circuit_states, measure_matrices):
            """
            Compute contraction with measure_input matrices.
            
            Args:
                qctn_inner: QCTN object with cores_weight_with_circuit_states
                measure_matrices: List of measurement matrices (one per output)
            
            Returns:
                Result of contraction
            """

            new_core_dict = {}

            total_mem = 0

            # Contract each core with its corresponding circuit_state(s)
            for idx, core_name in enumerate(qctn.cores):
                core_tensor = self.backend.convert_to_tensor(qctn.cores_weights[core_name])
                
                if idx == 0:
                    state1 = circuit_states[0]
                    state2 = circuit_states[1]
                    
                    # Contract with both states
                    # core_tensor shape: (input_dim1, input_dim2, bond_dims..., output_dim)
                    # state1 shape: (input_dim1,)
                    # state2 shape: (input_dim2,)
                    # result shape: (bond_dims..., output_dim)
                    contracted = torch.einsum('i,j,ij...->...', state1, state2, core_tensor)

                    new_core_dict[core_name] = contracted
                else:
                    state = circuit_states[idx + 1]
                    
                    # Contract along the first dimension
                    # core_tensor shape: (input_dim, bond_dims..., output_dim)
                    # state shape: (input_dim,)
                    # result shape: (bond_dims..., output_dim)
                    contracted = torch.einsum('i,ji...->j...', state, core_tensor)
                    new_core_dict[core_name] = contracted
                
                # print(f'Contract core {core_name}, result shape: {contracted.shape}')
                total_mem += contracted.numel() * contracted.element_size()

            print('new_core_dict', [(core_name, new_core_dict[core_name].shape) for core_name in new_core_dict])
            
            n = len(new_core_dict)

            # for idx in range(n):
            #     if idx == 0:
            #         core_tensor = new_core_dict[qctn_inner.cores[0]]
            #         measure_matrix = measure_matrices[0]

            #         # A * M * A_T
            #         contracted = torch.einsum('ka,zkl,lb->zab', core_tensor, measure_matrix, core_tensor)
            #     elif idx < n - 1:
            #         core_tensor = new_core_dict[qctn_inner.cores[idx]]
            #         measure_matrix = measure_matrices[idx]

            #         # contracted * B * M * B_T
            #         contracted = torch.einsum('zab,akc,zkl,bld->zcd', contracted, core_tensor, measure_matrix, core_tensor)
            #     else:
            #         core_tensor = new_core_dict[qctn_inner.cores[idx]]
            #         measure_matrix_1 = measure_matrices[idx]
            #         measure_matrix_2 = measure_matrices[idx + 1]

            #         # contracted * Z * M * Z_T
            #         contracted = torch.einsum('zab,akc,zkl,zcd,bld->z', contracted, core_tensor, measure_matrix_1, measure_matrix_2, core_tensor)

            #     print(f"Contract step {idx}, intermediate shape: {contracted.shape}")

            #     total_mem += contracted.numel() * contracted.element_size()
            
            if n == 1:
                core_tensor = new_core_dict[qctn_inner.cores[0]]
                measure_matrix_1 = measure_matrices[0]
                measure_matrix_2 = measure_matrices[1]

                # A * M * M * A_T
                contracted = torch.einsum('ka,zkl,zab,lb->z', core_tensor, measure_matrix_1, measure_matrix_2, core_tensor)

            # for idx in range(n):
            #     if idx == 0:
            #         core_tensor = new_core_dict[qctn_inner.cores[0]]
            #         measure_matrix_1 = measure_matrices[idx]
            #         measure_matrix_2 = measure_matrices[idx+1]

            #         # A * M * M * A_T
            #         contracted = torch.einsum('ka,zkl,z,lb->zab', core_tensor, measure_matrix, core_tensor)
            #     elif idx < n - 1:
            #         core_tensor = new_core_dict[qctn_inner.cores[idx]]
            #         measure_matrix = measure_matrices[idx]

            #         # contracted * B * M * B_T
            #         contracted = torch.einsum('zab,akc,zkl,bld->zcd', contracted, core_tensor, measure_matrix, core_tensor)
            #     else:
            #         core_tensor = new_core_dict[qctn_inner.cores[idx]]
            #         measure_matrix_1 = measure_matrices[idx]
            #         measure_matrix_2 = measure_matrices[idx + 1]

            #         # contracted * Z * M * Z_T
            #         contracted = torch.einsum('zab,akc,zkl,zcd,bld->z', contracted, core_tensor, measure_matrix_1, measure_matrix_2, core_tensor)

            #     print(f"Contract step {idx}, intermediate shape: {contracted.shape}")
                
            #     total_mem += contracted.numel() * contracted.element_size()

            print(f"Total memory used in contraction: {total_mem / (1024 ** 2):.2f} MB")

            return contracted
        
        # Ensure inputs are lists
        assert isinstance(circuit_states_list, list), "circuit_states_list must be a list"
        assert isinstance(measure_input_list, list), "measure_input_list must be a list"
        
        circuit_states_tensors = [self.backend.convert_to_tensor(s) for s in circuit_states_list]
        
        measure_matrices = [self.backend.convert_to_tensor(m) for m in measure_input_list]
        
        result = compute_fn(qctn, circuit_states_tensors, measure_matrices)
        
        return result

    def contract_with_std_graph_for_gradient(self, qctn, circuit_states_list, measure_input_list) -> Tuple:
        """
        Contract QCTN using standard graph method and compute gradients.
        
        This method computes the contraction and gradients with respect to core weights.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states_list (list): List of circuit input states (one per qubit).
                Each element is a tensor representing the state of one qubit.
            measure_input_list (list): List of measurement input matrices.
                Each element is a matrix (not vector) for measurement.
        
        Returns:
            tuple: (loss, gradients)
        """
        
        # Ensure inputs are lists
        assert isinstance(circuit_states_list, list), "circuit_states_list must be a list"
        assert isinstance(measure_input_list, list), "measure_input_list must be a list"
        
        # Convert inputs to backend tensors
        circuit_states_tensors = [self.backend.convert_to_tensor(s) for s in circuit_states_list]
        measure_matrices = [self.backend.convert_to_tensor(m) for m in measure_input_list]
        
        # Define the computation function
        def compute_fn(qctn_inner, circuit_states, measure_matrices):
            """
            Compute contraction with measure_input matrices.
            
            Args:
                qctn_inner: QCTN object
                circuit_states: List of circuit state tensors
                measure_matrices: List of measurement matrices
            
            Returns:
                Result of contraction
            """
            new_core_dict = {}

            # Contract each core with its corresponding circuit_state(s)
            for idx, core_name in enumerate(qctn_inner.cores):
                core_tensor = qctn_inner.cores_weights[core_name]
                
                if idx == 0:
                    state1 = circuit_states[0]
                    state2 = circuit_states[1]
                    
                    # Contract with both states
                    contracted = torch.einsum('i,j,ij...->...', state1, state2, core_tensor)
                    new_core_dict[core_name] = contracted
                else:
                    state = circuit_states[idx + 1]
                    
                    # Contract along the first dimension
                    contracted = torch.einsum('i,ji...->j...', state, core_tensor)
                    new_core_dict[core_name] = contracted

            n = len(new_core_dict)

            if n == 1:
                core_tensor = new_core_dict[qctn_inner.cores[0]]
                measure_matrix_1 = measure_matrices[0]
                measure_matrix_2 = measure_matrices[1]

                # A * M * M * A_T
                contracted = torch.einsum('ka,zkl,zab,lb->z', core_tensor, measure_matrix_1, measure_matrix_2, core_tensor)
                
                return contracted
            
            for idx in range(n):
                # print(f'Contract shape {qctn_inner.cores[idx]} {qctn_inner.cores_weights[qctn_inner.cores[idx]].shape} {new_core_dict[qctn_inner.cores[idx]].shape} {measure_matrices[idx].shape} at step {idx}')
                if idx == 0:
                    core_tensor = new_core_dict[qctn_inner.cores[idx]]
                    measure_matrix = measure_matrices[idx]

                    # A * M * A_T
                    contracted = torch.einsum('ka,zkl,lb->zab', core_tensor, measure_matrix, core_tensor)
                elif idx < n - 1:
                    core_tensor = new_core_dict[qctn_inner.cores[idx]]
                    measure_matrix = measure_matrices[idx]

                    # contracted * B * M * B_T
                    contracted = torch.einsum('zab,akc,zkl,bld->zcd', contracted, core_tensor, measure_matrix, core_tensor)
                else:
                    core_tensor = new_core_dict[qctn_inner.cores[idx]]
                    measure_matrix_1 = measure_matrices[idx]
                    measure_matrix_2 = measure_matrices[idx + 1]

                    # contracted * Z * M * Z_T
                    contracted = torch.einsum('zab,akc,zkl,zcd,bld->z', contracted, core_tensor, measure_matrix_1, measure_matrix_2, core_tensor)
            return contracted
        
        # Define loss function
        def loss_fn(*core_tensors):
            """
            Loss function that takes core tensors as input.
            
            Args:
                *core_tensors: Core weight tensors to optimize
            
            Returns:
                Loss value
            """
            # Create a temporary qctn-like object with updated cores
            class TempQCTN:
                def __init__(self, cores_list, core_tensors):
                    self.cores = cores_list
                    self.cores_weights = {
                        core_name: core_tensors[idx] 
                        for idx, core_name in enumerate(cores_list)
                    }
            
            temp_qctn = TempQCTN(qctn.cores, core_tensors)
            result = compute_fn(temp_qctn, circuit_states_tensors, measure_matrices)
            
            # Compute Cross entropy loss
            target = torch.ones_like(result)

            # TODO: avoid result <= 0
            result = torch.clamp(result, min=1e-10)
            log_result = torch.log(result)
            return -torch.mean(target * log_result)
        
        # Prepare core tensors for gradient computation
        core_tensors = [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]
        
        # Compute gradients for all cores
        num_cores = len(qctn.cores)
        argnums = list(range(num_cores))
        
        # Create value_and_grad function
        cache_key = '_grad_fn_std_graph'
        if not hasattr(qctn, cache_key):
            value_and_grad_fn = self.backend.compute_value_and_grad(loss_fn, argnums=argnums)
            # Note: JIT compilation might not work well with the TempQCTN class
            # value_and_grad_fn = self.backend.jit_compile(value_and_grad_fn)
            setattr(qctn, cache_key, value_and_grad_fn)
        else:
            value_and_grad_fn = getattr(qctn, cache_key)
        
        # Compute value and gradients
        loss, grads = value_and_grad_fn(*core_tensors)
        
        return loss, grads


