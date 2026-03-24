"""
Unified engine that combines contractor (expression generation) with backend (execution).

This module provides high-level functions that use EinsumStrategy to generate
expressions and then execute them using the specified backend.

Now supports strategy-based compilation for optimized contraction paths.
"""

from __future__ import annotations
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np
import math

from ..contractor import EinsumStrategy, StrategyCompiler, GreedyStrategy
from ..backends.backend_factory import BackendFactory, ComputeBackend
from .tn_tensor import TNTensor
from tqdm import tqdm
from .qctn import QCTN

class EngineSiamese:
    """
    EngineSiamese that combines tensor contraction expression generation with backend execution.
    
    This class separates concerns:
    - EinsumStrategy: Generates einsum expressions using opt_einsum (legacy)
    - StrategyCompiler: Compiles optimal strategies based on network structure
    - ComputeBackend: Executes expressions using JAX, PyTorch, etc.
    """

    def __init__(self, backend: Optional[Union[str, ComputeBackend]] = None, strategy_mode: str = 'balanced', mx_K: int = 100):
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
            mx_K (int): Maximum order for Hermite polynomials (for data generation).
        """
        if backend is None:
            self.backend = BackendFactory.get_default_backend()
        elif isinstance(backend, str):
            self.backend = BackendFactory.create_backend(backend, device="cpu")
        else:
            self.backend = backend

        self.contractor = EinsumStrategy()  # Keep for legacy methods
        self.strategy_compiler = StrategyCompiler(mode=strategy_mode)
        self.strategy_mode = strategy_mode
        self.strategy_mode = strategy_mode
        self.mx_K = mx_K
        self.mx_weights = self._init_mx_weights(mx_K)

    def _init_mx_weights(self, k_max):
        """Initialize normalization weights for Hermite polynomials.
        
        为了兼容不同 backend（以及复数 dtype），这里在 CPU/NumPy 上用实数先算出权重，
        再通过 backend.convert_to_tensor 转成对应后端/设备上的张量。
        """
        # k = 0, 1, ..., k_max
        k = np.arange(k_max + 1, dtype=np.float64)

        # log(k!) = lgamma(k+1)
        log_factorial = np.array([math.lgamma(int(ki) + 1) for ki in k], dtype=np.float64)
        log_2pi = math.log(2 * math.pi)
        log_factor = -0.5 * (0.5 * log_2pi + log_factorial)

        weights_np = np.exp(log_factor).astype(np.float64)  # 实数权重

        # 缓存一份 NumPy 权重，方便 complex 情况下在 CPU/实数域复用
        self._mx_weights_np = weights_np

        # 转成 backend 张量（会自动放到 backend 默认 device / dtype，上层若是 complex，会自动提升为复数）
        weights = self.backend.convert_to_tensor(weights_np)
        return weights

    def _eval_hermitenorm_batch(self, n_max, x):
        """Evaluate Hermite polynomials up to n_max."""
        
        # Ensure x is a tensor
        if not hasattr(x, 'shape'):
             x = self.backend.convert_to_tensor(x)

        # Assuming x is already on correct device or backend handles it
        # We need generic way to create zeros and ones with same shape/device/dtype
        
        # If backend supports zeros/ones with explicit shape/device/dtype
        # backend.zeros(shape, dtype) uses default device in backend_info
        
        # Better to access shape from x
        x_shape = x.shape
        full_shape = (n_max + 1,) + x_shape
        dtype = x.dtype
        
        # H = torch.zeros((n_max + 1,) + x.shape, dtype=x.dtype, device=device)
        H = self.backend.zeros(full_shape, dtype=dtype)
        
        # H[0] = torch.ones_like(x)
        H[0] = self.backend.ones_like(x)

        if n_max >= 1:
            H[1] = x
            for i in range(2, n_max + 1):
                H[i] = x * H[i-1] - (i-1) * H[i-2]

        return H

    def _eval_hermitenorm_batch_np(self, n_max, x_np: np.ndarray):
        """
        NumPy 版本的 Hermite 多项式计算，行为对齐 reference_code 中的实现。

        Args:
            n_max (int): 最高阶数（包含），即计算 k = 0..n_max。
            x_np (ndarray): 实数输入，形状 [B, D]。

        Returns:
            ndarray: 形状为 [n_max+1, B, D] 的实数数组。
        """
        x_np = np.asarray(x_np, dtype=np.float64)
        H = np.zeros((n_max + 1,) + x_np.shape, dtype=np.float64)
        H[0] = 1.0
        if n_max >= 1:
            H[1] = x_np
            for i in range(2, n_max + 1):
                H[i] = x_np * H[i - 1] - (i - 1) * H[i - 2]
        return H

    def generate_data(self, x, K: int = None, ret_type='tensor'):
        """
        Generate data (Mx and phi_x) for a given batch of x.

        Args:
            x (Tensor): Input batch [Batch, D].
            K (int): Number of Hermite polynomials to use.
        
        Returns:
             tuple: (Mx_list, phi_x)
        """
        if K is None:
            K = self.mx_K

        # 确保 x 使用 backend 的设备和 dtype（包括复数 dtype）
        x = self.backend.convert_to_tensor(x)

        num_qubits = x.shape[1]
        
        # 若 K 超过已预计算的范围，则扩展权重
        if K > self.mx_K or K > self.mx_weights.shape[0]:
            self.mx_weights = self._init_mx_weights(K)
            self.mx_K = K

        # 检查 backend 是否为复数 dtype
        backend_info = getattr(self.backend, "backend_info", None)
        backend_dtype = getattr(backend_info, "dtype", None) if backend_info is not None else None
        is_complex_backend = backend_dtype is not None and "complex" in str(backend_dtype)

        # ================================
        # complex 后端：完全在 CPU/实数域中按参考代码计算，再转换回 complex
        # ================================
        if is_complex_backend:
            # 统一使用实数 x（与 reference_code 一致），忽略虚部
            x_np = self.backend.tensor_to_numpy(x)
            x_np = np.asarray(x_np.real, dtype=np.float64)  # [B, D]

            # 确保有 NumPy 形式的权重
            if not hasattr(self, "_mx_weights_np") or self._mx_weights_np.shape[0] < self.mx_K + 1:
                # 触发一次初始化来刷新 _mx_weights_np
                self.mx_weights = self._init_mx_weights(self.mx_K)

            # 取前 K 项权重，并 reshape 成 [1, 1, K]
            weights_np = np.asarray(self._mx_weights_np[:K], dtype=np.float64)
            weights_np = weights_np[None, None, :]  # [1, 1, K]

            # 计算 Hermite 多项式：shape [K, B, D]
            H_np = self._eval_hermitenorm_batch_np(K - 1, x_np)

            # 高斯因子 sqrt(exp(-x^2/2))，shape [B, D, 1]
            gaussian_np = np.sqrt(np.exp(-np.square(x_np) / 2.0))[..., None]

            # 调整维度为 [B, D, K] 并应用权重与高斯因子
            phi_x_np = np.transpose(H_np, (1, 2, 0))  # [B, D, K]
            phi_x_np = weights_np * gaussian_np * phi_x_np

            # Mx: [B, D, K, K]，完全实数计算
            Mx_np = np.einsum("bdk,bdl->bdkl", phi_x_np, phi_x_np)

            # 转回 backend tensor（在 complex 后端会被提升到 complex dtype，但数值仍为实数）
            phi_x_tensor = self.backend.convert_to_tensor(phi_x_np)

            Mx_list = []
            for i in range(num_qubits):
                tmp_np = Mx_np[:, i, :, :]
                tmp_tensor = self.backend.convert_to_tensor(tmp_np)

                if ret_type == "TNTensor":
                    tmp_tt = TNTensor(tmp_tensor)
                    tmp_tt.auto_scale()
                    Mx_list.append(tmp_tt)
                else:
                    Mx_list.append(tmp_tensor)

            return Mx_list, phi_x_tensor

        # ================================
        # 非 complex 后端：保持原有 backend 上的实现
        # ================================
        weights = self.mx_weights[:K]
        # weights = weights[None, None, :] # [1, 1, K]
        # Use unsqueeze
        weights = self.backend.unsqueeze(weights, 0)  # [1, K]
        weights = self.backend.unsqueeze(weights, 0)  # [1, 1, K]

        # Calculate Hermite polynomials
        out = self._eval_hermitenorm_batch(K - 1, x)  # [K, B, D]
        
        # out = out.permute(1, 2, 0) # [B, D, K]
        out = self.backend.permute(out, (1, 2, 0))
        
        # Apply weights and Gaussian factor
        # x is [B, D]
        # gaussian_factor = torch.sqrt(torch.exp(- x**2 / 2))[:, :, None] # [B, D, 1]
        
        # - x**2 / 2
        neg_half_x_sq = - self.backend.square(x) / 2
        exp_val = self.backend.exp(neg_half_x_sq)
        sqrt_val = self.backend.sqrt(exp_val)
        
        gaussian_factor = self.backend.unsqueeze(sqrt_val, -1)
        
        out = weights * gaussian_factor * out  # [B, D, K]
        
        # Calculate Mx
        # 对于复数域，应该使用共轭内积：out_conj * out^T
        # 使用 out.conj() 在实数 dtype 下也是 no-op。
        Mx = self.backend.einsum("bdk,bdl->bdkl", out.conj(), out)
        
        # Split into list of Mx for each qubit
        # Mx_list = [Mx[:, i, :, :] for i in range(num_qubits)] # List of [B, K, K]
        Mx_list = []
        for i in range(num_qubits):
            tmp = Mx[:, i, :, :]

            if ret_type == "TNTensor":
                tmp = TNTensor(tmp)
                tmp.auto_scale()

            Mx_list.append(tmp)
        
        return Mx_list, out


    # ============================================================================
    # Strategy-based Compilation Methods (NEW API)
    # ============================================================================

    def contract_with_compiled_strategy(self, qctn, circuit_states_list, measure_input_list, measure_is_matrix=True, right_qctn="symmetric", ret_type='tensor') -> Any:
        """
        Contract using compiled strategy (auto-selected based on mode).
        
        This is the NEW recommended API that automatically selects the best strategy.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states (array or list, optional): Circuit input states.
            measure_input (array or list, optional): Measurement input.
            measure_is_matrix (bool): If True, measure_input is the outer product matrix.
        
        Returns:
            Backend tensor: Result of the contraction.
        """

        circuit_states = circuit_states_list
        if circuit_states_list is None:
            states_shape = None
        elif isinstance(circuit_states_list, list):
            states_shape = tuple([s.shape if s is not None else () for s in circuit_states_list])
        elif isinstance(circuit_states_list, dict):
            states_shape = tuple([circuit_states_list[i].shape if circuit_states_list[i] is not None else () 
                                  for i in sorted(circuit_states_list.keys())])
    
        if isinstance(measure_input_list, list):
            measure_shape = tuple([m.shape if m is not None else () for m in measure_input_list])
        elif isinstance(measure_input_list, dict):
            measure_shape = tuple([measure_input_list[i].shape if measure_input_list[i] is not None else () 
                                  for i in sorted(measure_input_list.keys())])
        measure_input = measure_input_list

        shapes_info = {
            'circuit_states_shapes': states_shape,
            'measure_shapes': measure_shape,
            'measure_is_matrix': measure_is_matrix
        }
        
        # Check cache
        cache_key = f'_compiled_strategy_{self.strategy_mode}_{states_shape}_{measure_shape}_{measure_is_matrix}'
        
        if not hasattr(qctn, cache_key):
            # Compile strategy
            compute_fn, strategy_name, cost = self.strategy_compiler.compile(qctn, shapes_info, self.backend, right_qctn=right_qctn)
            
            # Cache the result
            setattr(qctn, cache_key, {
                'compute_fn': compute_fn,
                'strategy_name': strategy_name,
                'cost': cost
            })
            # print(f"[EngineSiamese] Compiled and cached strategy: {strategy_name}")
        else:
            cached = getattr(qctn, cache_key)
            compute_fn = cached['compute_fn']
            strategy_name = cached['strategy_name']
            # print(f"[EngineSiamese] Using cached strategy: {strategy_name}")
        
        # Prepare data
        # Pass cores weights directly to support TNTensor
        cores_dict = {name: qctn.cores_weights[name] for name in qctn.cores}

        right_cores_dict = None
        if right_qctn is not None and isinstance(right_qctn, QCTN):
            right_cores_dict = {}
            for name in right_qctn.cores:
                right_cores_dict[name] = right_qctn.cores_weights[name]

        # Execute
        result = compute_fn(cores_dict, circuit_states, measure_input, right_cores_dict=right_cores_dict)
        
        if isinstance(result, TNTensor):
            
            # result.scale_to(1.0)
            if ret_type == 'TNTensor':
                if self.backend.is_complex(result.tensor):
                    result = TNTensor(self.backend.abs_square(result.tensor), result.scale, result.log_scale)
                return result
            else:
                result.scale_to(1.0)

                if self.backend.is_complex(result.tensor):
                    return self.backend.abs_square(result.tensor)

                return result.tensor
        else:
            if self.backend.is_complex(result):
                result = self.backend.abs_square(result)
            return result

    def contract_with_compiled_strategy_for_gradient(self, qctn, circuit_states_list, measure_input_list, measure_is_matrix=True, right_qctn="symmetric") -> Tuple:
        """
        Contract using compiled strategy and compute gradients.
        
        This is the NEW recommended API for gradient computation.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states_list (array or list, optional): Circuit input states.
            measure_input_list (array or list, optional): Measurement input.
            measure_is_matrix (bool): If True, measure_input is the outer product matrix.
        
        Returns:
            tuple: (loss, gradients)
        """

        circuit_states = circuit_states_list
        if circuit_states_list is not None:
            states_shape = tuple([s.shape if s is not None else () for s in circuit_states_list])
        else:
            states_shape = None

        if isinstance(measure_input_list[0], TNTensor):
            measure_shape = tuple([m.tensor.shape if m is not None else () for m in measure_input_list])
        else:
            measure_shape = tuple([m.shape if m is not None else () for m in measure_input_list])
        measure_input = measure_input_list
        
        shapes_info = {
            'circuit_states_shapes': states_shape,
            'measure_shapes': measure_shape,
            'measure_is_matrix': measure_is_matrix
        }
        
        # Check cache
        cache_key = f'_compiled_strategy_{self.strategy_mode}_{states_shape}_{measure_shape}_{measure_is_matrix}'
        
        if not hasattr(qctn, cache_key):
            # Compile strategy
            compute_fn, strategy_name, cost = self.strategy_compiler.compile(qctn, shapes_info, self.backend, right_qctn=right_qctn)
            
            # Cache the result
            setattr(qctn, cache_key, {
                'compute_fn': compute_fn,
                'strategy_name': strategy_name,
                'cost': cost
            })
            print(f"[EngineSiamese] Compiled and cached strategy: {strategy_name}")
        else:
            cached = getattr(qctn, cache_key)
            compute_fn = cached['compute_fn']
            strategy_name = cached['strategy_name']
            # print(f"[EngineSiamese] Using cached strategy: {strategy_name}")

        trainable_cores = []
        for c_name in qctn.cores:
            c = qctn.cores_weights[c_name]
            if isinstance(c, TNTensor) and c.tensor.requires_grad:
                trainable_cores.append(c)
            elif not isinstance(c, TNTensor) and c.requires_grad:
                trainable_cores.append(c)
        
        if right_qctn is not None and isinstance(right_qctn, QCTN):
            for c_name in right_qctn.cores:
                c = right_qctn.cores_weights[c_name]
                if isinstance(c, TNTensor) and c.tensor.requires_grad:
                    trainable_cores.append(c)
                elif not isinstance(c, TNTensor) and c.requires_grad:
                    trainable_cores.append(c)

        # Prepare tensors for gradient calculation
        # We need to separate tensors (which require grad) from scales (constants)
        # raw_core_tensors = []
        # core_scales = []
        # for c_name in qctn.cores:
        #     c = qctn.cores_weights[c_name]
        #     # print(f"qctn cores grad {c_name} {c.tensor.requires_grad}")
        #     if isinstance(c, TNTensor) and not c.tensor.requires_grad:
        #         continue
        #     if not isinstance(c, TNTensor) and not c.requires_grad:
        #         continue
        #     if isinstance(c, TNTensor):
        #         raw_core_tensors.append(c.tensor)
        #         core_scales.append(c.scale)
        #     else:
        #         raw_core_tensors.append(c)
        #         core_scales.append(1.0)
        
        # # print(f"raw_core_tensors {len(raw_core_tensors)}")

        # if right_qctn is not None and isinstance(right_qctn, QCTN):
        #     for c_name in right_qctn.cores:
        #         c = right_qctn.cores_weights[c_name]
        #         if isinstance(c, TNTensor) and not c.tensor.requires_grad:
        #             continue
        #         if not isinstance(c, TNTensor) and not c.requires_grad:
        #             continue
        #         c = right_qctn.cores_weights[c_name]
        #         if isinstance(c, TNTensor):
        #             raw_core_tensors.append(c.tensor)
        #             core_scales.append(c.scale)
        #         else:
        #             raw_core_tensors.append(c)
        #             core_scales.append(1.0)
        
        # Define loss function
        def loss_fn(*core_tensors_args):
            
            offset = 0

            cores_dict = qctn.cores_weights

            right_cores_dict = {}
            if right_qctn is not None and isinstance(right_qctn, QCTN):
                right_cores_dict = right_qctn.cores_weights

            # cores_dict = {}
            # for c_name in qctn.cores:
            #     c = qctn.cores_weights[c_name]
            #     is_tntensor = isinstance(c, TNTensor)

            #     if isinstance(c, TNTensor):
            #         c = c.tensor
            #     if c.requires_grad:
            #         if is_tntensor:
            #             tensor = TNTensor(core_tensors_args[offset], core_scales[offset])
            #         else:
            #             tensor = core_tensors_args[offset]
            #         offset += 1
            #     else:
            #         tensor = c
                
            #     cores_dict[c_name] = tensor
            
            # right_cores_dict = {}
            # if right_qctn is not None and isinstance(right_qctn, QCTN):
            #     for c_name in right_qctn.cores:
            #         # print(f"right_qctn iter core: {c_name}")
            #         c = right_qctn.cores_weights[c_name]
            #         is_tntensor = isinstance(c, TNTensor)
                    
            #         if isinstance(c, TNTensor):
            #             c = c.tensor
            #         if c.requires_grad:
            #             if is_tntensor:
            #                 tensor = TNTensor(core_tensors_args[offset], core_scales[offset])
            #             else:
            #                 tensor = core_tensors_args[offset]
            #             offset += 1
            #         else:
            #             tensor = c
            #         right_cores_dict[c_name] = tensor
            
            # print(f'cores_dict keys: {list(cores_dict.keys())}')
            # print(f'right_cores_dict keys: {list(right_cores_dict.keys())}')
            # print(f"core a tensor {cores_dict['a']}")
            # print(f"right core a tensor {right_cores_dict['a']}")
            # print(f'right_cores_dict keys: {list(right_cores_dict.keys())}')


            result = compute_fn(cores_dict, circuit_states, measure_input, right_cores_dict=right_cores_dict)
            
            # print(f'result {result.shape}')

            # Result might be TNTensor or raw tensor
            if isinstance(result, TNTensor):
                res_tensor = result.tensor
                res_scale = result.scale
                res_log_scale = result.log_scale
            else:
                res_tensor = result
                res_scale = 1.0
                res_log_scale = 0.0

            # Born rule: 若为复数则转为实数概率 P = |ψ|^2
            if self.backend.is_complex(res_tensor):
                res_tensor = self.backend.abs_square(res_tensor)

            # Compute Cross Entropy loss; target 全 1（最大化概率）
            target = self.backend.ones(res_tensor.shape, dtype=res_tensor.dtype)

            # Avoid log(0)
            res_tensor = self.backend.clamp(res_tensor, min=1e-10)
            log_result = self.backend.log(res_tensor)

            # MSE loss

            loss = self.backend.mean((res_tensor - target) ** 2)
            return loss

            # print(f"res_tensor : {res_tensor}, res_scale: {res_scale}")
            # print(f"log_result: {log_result.mean().item()}")
            # print(f"res_scale: {res_scale}")
            # print(f"res_log_scale: {res_log_scale}")

            # Add log(scale) for correct loss value (log(P*S) = log(P) + log(S))
            # log(S) is constant w.r.t parameters, so gradients are correct
            # detached_scale = self.backend.detach(res_scale)
            # # detached_scale = res_scale
            
            # # # Handle float/scalar scale for log
            # # import torch
            # if isinstance(detached_scale, (int, float)):
            #      log_scale = np.log(detached_scale)
            # else:
            #      # Check if 0-dim tensor
            #      if detached_scale.ndim == 0:
            #           log_scale = self.backend.log(detached_scale)
            #      else:
            #           log_scale = self.backend.log(detached_scale)

            # log_scale = np.log(res_scale)
            # log_scale = self.backend.log(self.backend.detach(res_scale))
            log_scale = self.backend.detach(res_log_scale)

            # print('log_scale', log_scale)

            log_total = log_result + log_scale

            return -self.backend.mean(target * log_total)
        
        # Compute gradients
        # We want gradients with respect to all cores
        argnums = list(range(len(trainable_cores)))
        
        # Create value_and_grad function
        value_and_grad_fn = self.backend.compute_value_and_grad(loss_fn, argnums=argnums)
        
        # Execute
        loss, grads = value_and_grad_fn(*trainable_cores)

        # print(f'input num {len(raw_core_tensors)} grad output num {len(grads)}')
        
        # grads = [grads[i] / core_scales[i] for i in range(len(core_scales))]

        # tmp = {i: (grads[i], core_scales[i]) for i in range(len(grads))}
        # print(f"grads : {tmp}")
        # print(f"scale : {{i: core_scales[i] for i in range(len(core_scales))}}")

        # print(f"core_weights names: {[(name, qctn.cores_weights[name].tensor.mean() if isinstance(qctn.cores_weights[name], TNTensor) else qctn.cores_weights[name].mean()) for name in qctn.cores]}")
        # print(f"Loss: {loss.item()}, Collected {[grad.mean().item() for grad in grads]} gradients.")
        # print(f"measure_input_list mean: {[m.mean().item() for m in measure_input_list]}")

        return loss, trainable_cores, grads
        

    # ============================================================================
    # Probability Calculation Methods
    # .. deprecated:: Use EngineCommon equivalents instead.
    # ============================================================================

    def calculate_full_probability(self, qctn, circuit_states_list, measure_input_list):
        """Calculate the full probability of observing a specific bitstring.

        .. deprecated:: Use :meth:`EngineCommon.calculate_full_probability`.

        Args:
            qctn (QCTN): The quantum circuit tensor network.
            circuit_states_list (list): List of circuit input states.
            measure_input_list (list): List of measurement input matrices (complete).
            
        Returns:
            Backend tensor: The calculated probability.
        """
        res =  self.contract_with_compiled_strategy(
            qctn, 
            circuit_states_list=circuit_states_list, 
            measure_input_list=measure_input_list, 
            measure_is_matrix=True
        )

        res.scale_to(1.0)

        return res.tensor

    def calculate_marginal_probability(self, qctn, circuit_states_list, measure_input_list, qubit_indices: List[int]):
        """Calculate the marginal probability of a subset of qubits.

        .. deprecated:: Use :meth:`EngineCommon.calculate_marginal_probability`.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network.
            circuit_states_list (list): List of circuit input states.
            measure_input_list (list): List of measurement input matrices (partial).
            qubit_indices (list[int]): Indices of qubits corresponding to measure_input_list.
            
        Returns:
            Backend tensor: The calculated probability (or batch of probabilities).
        """

        if len(qubit_indices) != len(measure_input_list):
            raise ValueError("Length of qubit_indices must match length of measure_input_list")
        
        dim = 1
        for m in measure_input_list:
            if m is not None:
                dim = m.shape[-1]
                break

        full_measure_input_list = []
        
        # Create Identity matrix
        ident = self.backend.eye(dim)
        # If measure_input_list has batch dim, ident should broadcast or match?
        # Usually measure_input_list elements are (B, K, K) or (K, K).
        # We assume (B, K, K) or compatible.
        # If we need batch dim for identity, we can add it later or rely on broadcasting.
        # But contract_with_compiled_strategy expects consistent batch dims if present.
        # Let's check the first element of measure_input_list to see if it has batch dim.
        has_batch = False
        batch_size = 1
        if len(measure_input_list) > 0:
            if measure_input_list[0].ndim == 3:
                has_batch = True
                batch_size = measure_input_list[0].shape[0]
                ident = self.backend.unsqueeze(ident, 0)
                ident = self.backend.expand(ident, batch_size, -1, -1)

        for i in range(qctn.nqubits):
            if i in qubit_indices:
                idx = qubit_indices.index(i)
                full_measure_input_list.append(measure_input_list[idx])
            else:
                full_measure_input_list.append(ident)
        
        res =  self.contract_with_compiled_strategy(
            qctn, 
            circuit_states_list=circuit_states_list, 
            measure_input_list=full_measure_input_list, 
            measure_is_matrix=True
        )

        if isinstance(res, TNTensor):
            res.scale_to(1.0)

            return res.tensor
        else:
            return res

    def calculate_conditional_probability(self, qctn, circuit_states_list, measure_input_list,
                                          qubit_indices: List[int], target_indices: List[int]):
        """Calculate conditional probability P(target | condition).

        .. deprecated:: Use :meth:`EngineCommon.calculate_conditional_probability`.

        Calculate the conditional probability P(target | condition).
        
        Args:
            qctn (QCTN): The quantum circuit tensor network.
            circuit_states_list (list): List of circuit input states.
            measure_input_list (list): List of measurement input matrices (covering target + condition).
            qubit_indices (list[int]): Indices of qubits corresponding to measure_input_list.
            target_indices (list[int]): Indices of target qubits (subset of qubit_indices).
            
        Returns:
            Backend tensor: The calculated conditional probability.
        """
        # Check inputs
        if len(qubit_indices) != len(measure_input_list):
            raise ValueError("Length of qubit_indices must match length of measure_input_list")
        
        dim = 1
        for m in measure_input_list:
            if m is not None:
                dim = m.shape[-1]
                break
        # Create Identity matrix (B, K, K)
        ident = self.backend.eye(dim)

        has_batch = False
        batch_size = 1
        if len(measure_input_list) > 0:
            if measure_input_list[0].ndim == 3:
                has_batch = True
                batch_size = measure_input_list[0].shape[0]
                ident = self.backend.unsqueeze(ident, 0)
                ident = self.backend.expand(ident, batch_size, -1, -1)

        # Prepare stacked measurements
        # We want output shape (B, 2) -> effectively batch size 2*B? Or B*2?
        # The user requested: "change shape to B*2*K*K".
        # Index 0: Original (Joint P(A,B))
        # Index 1: Identity on Target (Marginal P(B))
        
        full_measure_input_list = []
        
        for i in range(qctn.nqubits):
            # Prepare tensor of shape (B, 2, K, K)
            
            if i in qubit_indices:
                idx = qubit_indices.index(i)
                measure_tensor = measure_input_list[idx] # (B, K, K)
                
                if i in target_indices:
                    # Target qubit: [Measure, Identity]
                    # Stack along dim 1
                    stacked = self.backend.stack([measure_tensor, ident], dim=1) # (B, 2, K, K)
                else:
                    # Condition qubit: [Measure, Measure]
                    stacked = self.backend.stack([measure_tensor, measure_tensor], dim=1) # (B, 2, K, K)
            else:
                # Unused qubit: [Identity, Identity]
                stacked = self.backend.stack([ident, ident], dim=1) # (B, 2, K, K)
            
            full_measure_input_list.append(stacked)
        
        # Contract
        # The engine's einsum strategy should handle the extra dimension '2' via broadcasting '...'
        # Result shape should be (B, 2)
        result = self.contract_with_compiled_strategy(
            qctn, 
            circuit_states_list=circuit_states_list, 
            measure_input_list=full_measure_input_list, 
            measure_is_matrix=True
        )
        
        # Calculate conditional probability
        # result[:, 0] is Joint P(A, B)
        # result[:, 1] is Marginal P(B)
        # P(A|B) = P(A, B) / P(B)

        result.scale_to(1.0)

        result = result.tensor
        
        prob_joint = result[:, 0]
        prob_condition = result[:, 1]
        
        epsilon = 1e-10
        return prob_joint / (prob_condition + epsilon)

    # ============================================================================
    # Sampling Methods
    # .. deprecated:: Use EngineCommon.sample instead.
    # ============================================================================

    def sample(self, qctn, circuit_states_list, num_samples, K, bounds=[-5, 5], grid_size=1000):
        """Sample values from the quantum circuit using Numerical Inverse CDF method.

        .. deprecated:: Use :meth:`EngineCommon.sample`.
        
        Args:
            qctn: QCTN object
            circuit_states_list: List of input states
            num_samples: Number of samples (batch size)
            K: Dimension of each qubit (used for Hermite polynomial calculation)
            bounds: Sampling range [min, max]
            grid_size: Number of grid points for potential calculation
            
        Returns:
            samples: Tensor of shape (num_samples, nqubits) containing sampled values (continuous).
        """
        print(f"qctn cores_weights: {qctn.cores_weights['a']}")
        # print(f"qctn cores_weights: {qctn.cores_weights['a'].tensor}")
        # print(f"qctn scale: {qctn.cores_weights['a'].scale}")
        
        # 1. Prepare Grid and Basis
        x_min, x_max = bounds
        grid_x = self.backend.linspace(x_min, x_max, steps=grid_size) # (Grid,)

        # 2. Check Input States Batch Size
        expanded_circuit_states = []
        for s in circuit_states_list:
            if s.ndim == 1:
                s_expanded = self.backend.unsqueeze(s, 0)
                s_expanded = self.backend.expand(s_expanded, num_samples, -1)
                expanded_circuit_states.append(s_expanded)
            else:
                if s.shape[0] == 1 and num_samples > 1:
                    s_expanded = self.backend.expand(s, num_samples, -1)
                    expanded_circuit_states.append(s_expanded)
                else:
                    expanded_circuit_states.append(s)

        # 3. Initialize Persistent Measurements
        ident = self.backend.eye(K)
        ident_batch = self.backend.unsqueeze(ident, 0)
        ident_batch = self.backend.expand(ident_batch, num_samples, -1, -1)
        
        persistent_measures = [ident_batch for _ in range(qctn.nqubits)]
        
        samples = self.backend.zeros((num_samples, qctn.nqubits))

        # 4. Sampling Loop
        for q_idx in tqdm(range(qctn.nqubits)):
            # Step A: Generate Mx for Grid
            grid_x_input = self.backend.unsqueeze(grid_x, 1) # (G, 1)
            mx_list_grid, _ = self.generate_data(grid_x_input, K=K)
            Mx_grid = mx_list_grid[0] # (G, K, K)

            print(f"q_idx: {q_idx}")
            print(f"Mx_grid: {Mx_grid.shape}")
            print(f"Mx_grid: {Mx_grid[0, :, :]}")
            print(f"Mx_grid: {Mx_grid[1, :, :]}")
            

            # Step B: Prepare Temporary Measurements
            temp_measure_list = []
            
            for i in range(qctn.nqubits):
                if i == q_idx:
                    # Current Qubit: Use Grid
                    m = self.backend.unsqueeze(Mx_grid, 0)
                    m = self.backend.expand(m, num_samples, -1, -1, -1)
                elif i < q_idx:
                     # Previous Qubits: Use Persistent (Sampled values)
                     # Persistent measures are (S, K, K). Expand to (S, G, K, K)
                    p = persistent_measures[i]
                    m = self.backend.unsqueeze(p, 1)
                    m = self.backend.expand(m, -1, grid_size, -1, -1)
                else:
                    # Future Qubits: Use Identity (Trace out)
                    # Use identity batch (S, K, K)
                    p = ident_batch
                    m = self.backend.unsqueeze(p, 1)
                    m = self.backend.expand(m, -1, grid_size, -1, -1)

                # Reshape to (S*G, K, K)
                m = self.backend.reshape(m, (num_samples * grid_size, K, K))
                temp_measure_list.append(m)
            
            # Step C: Prepare Temporary Inputs
            temp_input_list = []
            for s in circuit_states_list:
                # s_exp = self.backend.unsqueeze(s, 1)
                # s_exp = self.backend.expand(s_exp, -1, grid_size, -1)
                # s_reshaped = self.backend.reshape(s_exp, (num_samples * grid_size, -1))
                temp_input_list.append(s)

            # Step D: Contract
            # print(f"[EngineSiamese.sample] Step {q_idx}: Contraction (Batch={num_samples*grid_size})")
    
            print(f"[EngineSiamese.sample] Sampling qubit {q_idx+1}/{qctn.nqubits}...")
            print(f"  Contracting with batch size: {num_samples * grid_size}...")
            # print(f". Temp input list shape: {[x.shape for x in temp_input_list]}")
            # print(f". Temp measure shape: {[x.shape for x in temp_measure_list]}")

            # print(f". Temp measure: {temp_measure_list}")
            
            results = self.contract_with_compiled_strategy(
                 qctn,
                 circuit_states_list=temp_input_list,
                 measure_input_list=temp_measure_list,
                 measure_is_matrix=True
            )
            
            if isinstance(results, TNTensor):
                results = results.tensor
                
            # print(f"results: {results.shape} {results}")

            # Step E: CDF & Sample
            density = self.backend.reshape(results, (num_samples, grid_size))
            # density = self.backend.real(density)
            density = self.backend.abs_square(density)
            
            # print(f"density: {density.shape} {density}")

            density = self.backend.clamp(density, min=0.0)
            
            cdf = self.backend.cumsum(density, dim=1)
            
            # print(f"cdf: {cdf.shape} {cdf}")

            total_sum = self.backend.unsqueeze(cdf[:, -1], 1)
            
            # print(f"total_sum: {total_sum.shape} {total_sum}")

            cdf = cdf / (total_sum + 1e-10) # (S, G)
            
            # print(f"rescale cdf: {cdf.shape} {cdf}")

            u = self.backend.rand((num_samples, 1), dtype=self.backend.torch.float32)

            # print(f"u: {u.shape} {u}")

            # 逆 CDF 采样需要实数比较，复数 backend 时 u 取实部
            # if self.backend.is_complex(u):
            #     u = self.backend.abs_square(u)
            # self.backend.real(u)
            
            # print(f"u: {u.shape} {u}")

            mask = (cdf < u).float()
            indices = self.backend.sum(mask, dim=1).long() # (S,)
            indices = self.backend.clamp(indices, max=grid_size - 2)
            
            indices = self.backend.unsqueeze(indices, 1) # (S, 1)
            indices_next = indices + 1
            
            cdf_L = self.backend.gather(cdf, 1, indices)
            cdf_R = self.backend.gather(cdf, 1, indices_next)
            
            grid_expanded = self.backend.unsqueeze(grid_x, 0)
            grid_expanded = self.backend.expand(grid_expanded, num_samples, -1)
            
            x_L_val = self.backend.gather(grid_expanded, 1, indices)
            x_R_val = self.backend.gather(grid_expanded, 1, indices_next)
            
            fraction = (u - cdf_L) / (cdf_R - cdf_L + 1e-10)
            sampled_y = x_L_val + fraction * (x_R_val - x_L_val) # (S, 1)
            
            samples[:, q_idx] = self.backend.squeeze(sampled_y, 1)
            
            print(f"sampled_y: {sampled_y.shape} {sampled_y}")

            # Step F: Update Persistent Measure
            mx_list_y, _ = self.generate_data(sampled_y, K=K)
            Mx_y = mx_list_y[0] # (S, K, K)
            
            persistent_measures[q_idx] = Mx_y
            
        if isinstance(samples, TNTensor):
            return samples.tensor
        return samples


