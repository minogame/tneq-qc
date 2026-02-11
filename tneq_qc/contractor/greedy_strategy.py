"""
Greedy contraction strategy.

This module provides the GreedyStrategy class for greedy tensor contraction.
The strategy processes qubits one by one, contracting connected cores and measurements.

The strategy uses symmetric expansion:
1. Left version (L): original cores and circuit_states
2. Right version (R): conjugate transposed version, edges reversed
3. Middle (M): measurement matrices Mx

The contraction computes: A * Mx * A^T for each qubit.
"""

from __future__ import annotations
from typing import Dict, Any, Callable, List, Set, Optional
from copy import deepcopy
from enum import Enum
from ..core.tn_tensor import TNTensor
from ..core.qctn import QCTN

from .base import ContractionStrategy


class TensorSide(Enum):
    """Enum to track tensor side: left, middle, or right"""
    LEFT = "L"
    MIDDLE = "M"
    RIGHT = "R"


class GreedyStrategy(ContractionStrategy):
    """Greedy contraction strategy that processes qubits sequentially"""
    
    def check_compatibility(self, qctn, shapes_info: Dict[str, Any]) -> bool:
        """
        Check compatibility - always returns True as this strategy is general purpose.
        """
        return True
    
    def get_compute_function(self, qctn, shapes_info: Dict[str, Any], backend, right_qctn="symmetric") -> Callable:
        """
        Return computation function for greedy contraction.
        """
        self.backend = backend

        def compute_fn(_cores_dict, circuit_states, measure_matrices, right_cores_dict=None):
            """
            Greedy contraction strategy with symmetric expansion.
            
            The contraction computes: sum_i (A_i * Mx_i * A_i^T) for all qubits.
            
            Process:
            1. Build symmetric expanded tensor list with Left, Middle, Right versions
            2. For each qubit, find connected cores and contract them
            
            Args:
                cores_dict: {core_name: core_tensor} dictionary
                circuit_states: List of circuit input states (one per qubit)
                measure_matrices: List of measurement matrices (one per qubit)
            
            Returns:
                Contraction result
            """
            import torch
            import opt_einsum
            
            nqubits = qctn.nqubits
            ncores = qctn.ncores
            cores_dict = {k: v for k, v in _cores_dict.items()}
            
            # ========================================
            # Step 1: Build core_tensor_list with basic info
            # ========================================
            core_tensor_list = []
            
            # Helper to get unique ID
            def get_uid():
                return len(core_tensor_list)

            # 1.1 Add LEFT version cores
            # Map original core_idx to new core_idx in list
            left_core_map = {}  # original_idx -> new_idx
            
            # print(f"qctn.adjacency_table: {len(qctn.adjacency_table)} entries")
            for core_info in qctn.adjacency_table:
                uid = get_uid()
                left_core_map[core_info['core_idx']] = uid
                
                # print(f'add core_entry {core_info["core_name"]}_L with \n in_edge_list {core_info["in_edge_list"]} \n out_edge_list {core_info["out_edge_list"]}')

                core_entry = {
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
                }
                core_tensor_list.append(core_entry)
            
            # 1.2 Add LEFT version circuit_states
            left_circuit_map = {} # qubit_idx -> new_idx
            
            for qubit_idx in qctn.qubit_indices:
                if circuit_states is None:
                    continue
                
                if isinstance(circuit_states, dict):
                    if qubit_idx not in circuit_states:
                        continue
                
                if isinstance(circuit_states, list) or isinstance(circuit_states, tuple):
                    if qubit_idx >= len(circuit_states):
                        continue

                uid = get_uid()
                left_circuit_map[qubit_idx] = uid
                
                circuit_entry = {
                    'core_idx': uid,
                    'core_name': f"circuit_L_{qubit_idx}",
                    'tensor_source': 'circuit',
                    'tensor_key': qubit_idx,
                    'in_edge_list': [],
                    'out_edge_list': [{
                        'neighbor_idx': -1,
                        'neighbor_name': "",
                        'edge_rank': circuit_states[qubit_idx].shape[0],
                        'qubit_idx': qubit_idx,
                    }],
                    'side': TensorSide.LEFT,
                    'batch_symbol': "",
                }
                core_tensor_list.append(circuit_entry)
            
            # 1.3 Add MIDDLE version Mx
            mx_map = {} # qubit_idx -> new_idx
            
            for qubit_idx in qctn.qubit_indices:
                if measure_matrices is None:
                    continue
                
                if isinstance(measure_matrices, dict):
                    if qubit_idx not in measure_matrices:
                        continue
                
                if isinstance(measure_matrices, list) or isinstance(measure_matrices, tuple):
                    if qubit_idx >= len(measure_matrices):
                        continue
                
                mx = measure_matrices[qubit_idx]

                if mx is None:
                    continue
                
                uid = get_uid()
                mx_map[qubit_idx] = uid

                # Mx shape: (B, d, d) or (B, 2, d, d)
                batch_sym = ""
                if mx.ndim == 3:
                    batch_sym = "a"
                elif mx.ndim == 4:
                    batch_sym = "ab"
                
                mx_entry = {
                    'core_idx': uid,
                    'core_name': f"mx_{qubit_idx}",
                    'tensor_source': 'mx',
                    'tensor_key': qubit_idx,
                    'in_edge_list': [{
                        'neighbor_idx': -1,
                        'neighbor_name': "",
                        'edge_rank': mx.shape[-2],
                        'qubit_idx': qubit_idx,
                    }],
                    'out_edge_list': [{
                        'neighbor_idx': -1,
                        'neighbor_name': "",
                        'edge_rank': mx.shape[-1],
                        'qubit_idx': qubit_idx,
                    }],
                    'side': TensorSide.MIDDLE,
                    'batch_symbol': batch_sym,
                }
                core_tensor_list.append(mx_entry)
            
            # 1.4 Add RIGHT version cores
            # Right version: in becomes out, out becomes in. Lists reversed. Tensor NOT transposed.
            right_core_map = {} # original_idx -> new_idx
            if isinstance(right_qctn, str) and right_qctn == "symmetric":
                
                for core_info in qctn.adjacency_table:
                    uid = get_uid()
                    right_core_map[core_info['core_idx']] = uid
                    
                    # print(f'add core_entry {core_info["core_name"]}_R with \n in_edge_list {core_info["in_edge_list"]} \n out_edge_list {core_info["out_edge_list"]}')

                    # Reverse edges: in becomes out, out becomes in
                    # Also reverse the order within each list

                    new_in_edges = deepcopy(core_info['out_edge_list'])[::-1]
                    new_out_edges = deepcopy(core_info['in_edge_list'])[::-1]
                    
                    core_entry = {
                        'core_idx': uid,
                        'core_name': f"{core_info['core_name']}_R",
                        'tensor_source': 'transpose' ,
                        # 'tensor': cores_dict[core_info['core_name']].conj(),
                        'tensor_key': core_info['core_name'],
                        'in_edge_list': new_in_edges,
                        'out_edge_list': new_out_edges,
                        'side': TensorSide.RIGHT,
                        'original_core_idx': core_info['core_idx'],
                        'original_in_count': len(core_info['in_edge_list']),
                        'original_out_count': len(core_info['out_edge_list']),
                        'batch_symbol': "",
                    }
                    core_tensor_list.append(core_entry)
            elif isinstance(right_qctn, QCTN):
                for core_info in right_qctn.adjacency_table:
                    uid = get_uid()

                    # print(f"core_info {core_info}")
                    
                    cores_dict["right_" + core_info['core_name']] = right_cores_dict[core_info['core_name']]

                    core_idx = core_info['core_idx'] + len(left_core_map)
                    right_core_map[core_idx] = uid
                    
                    # print(f'add core_entry {core_info["core_name"]}_R with \n in_edge_list {core_info["in_edge_list"]} \n out_edge_list {core_info["out_edge_list"]}')

                    # Reverse edges: in becomes out, out becomes in
                    # Also reverse the order within each list

                    new_in_edges = deepcopy(core_info['in_edge_list'])
                    new_out_edges = deepcopy(core_info['out_edge_list'])
                    
                    core_entry = {
                        'core_idx': uid,
                        'core_name': f"{core_info['core_name']}_R",
                        'tensor_source': 'core',
                        'tensor_key': "right_" + core_info['core_name'],
                        'in_edge_list': new_in_edges,
                        'out_edge_list': new_out_edges,
                        'side': TensorSide.RIGHT,
                        'original_core_idx': core_idx,
                        'original_in_count': len(core_info['in_edge_list']),
                        'original_out_count': len(core_info['out_edge_list']),
                        'batch_symbol': "",
                    }
                    core_tensor_list.append(core_entry)
            elif right_qctn is None:
                pass
            else:
                raise ValueError("Invalid right_qctn parameter.")
            
            # 1.5 Add RIGHT version circuit_states
            right_circuit_map = {} # qubit_idx -> new_idx
            
            for qubit_idx in qctn.qubit_indices:
                if circuit_states is None:
                    continue
                
                if isinstance(circuit_states, dict):
                    if qubit_idx not in circuit_states:
                        continue
                
                if isinstance(circuit_states, list) or isinstance(circuit_states, tuple):
                    if qubit_idx >= len(circuit_states):
                        continue
                
                uid = get_uid()
                right_circuit_map[qubit_idx] = uid
                
                circuit_entry = {
                    'core_idx': uid,
                    'core_name': f"circuit_R_{qubit_idx}",
                    'tensor_source': 'circuit',
                    'tensor_key': qubit_idx,
                    'in_edge_list': [{
                        'neighbor_idx': -1,
                        'neighbor_name': "",
                        'edge_rank': circuit_states[qubit_idx].shape[0],
                        'qubit_idx': qubit_idx,
                    }],
                    'out_edge_list': [],
                    'side': TensorSide.RIGHT,
                    'batch_symbol': "",
                }
                core_tensor_list.append(circuit_entry)
            
            # ========================================
            # Step 2: Update neighbor connections
            # ========================================
            
            # 2.1 Update LEFT cores
            for original_idx, new_idx in left_core_map.items():
                entry = core_tensor_list[new_idx]
                
                # Update in_edges
                for edge in entry['in_edge_list']:
                    if 'is_cross_partition' in edge and edge['is_cross_partition']:
                        continue
                    if edge['neighbor_idx'] == -1:
                        # Connect to Left Circuit
                        qubit_idx = edge['qubit_idx']
                        if qubit_idx in left_circuit_map:
                            neighbor_uid = left_circuit_map[qubit_idx]
                            edge['neighbor_idx'] = neighbor_uid
                            edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']
                            
                            # Also update circuit's out_edge
                            circuit_entry = core_tensor_list[neighbor_uid]
                            circuit_entry['out_edge_list'][0]['neighbor_idx'] = new_idx
                            circuit_entry['out_edge_list'][0]['neighbor_name'] = entry['core_name']
                    else:
                        # Connect to another Left Core
                        original_neighbor = edge['neighbor_idx']
                        if original_neighbor in left_core_map:
                            neighbor_uid = left_core_map[original_neighbor]
                            edge['neighbor_idx'] = neighbor_uid
                            edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']
                
                # Update out_edges
                for edge in entry['out_edge_list']:
                    if 'is_cross_partition' in edge and edge['is_cross_partition']:
                        continue
                    if edge['neighbor_idx'] == -1:
                        # Connect to Mx
                        qubit_idx = edge['qubit_idx']
                        if qubit_idx in mx_map:
                            neighbor_uid = mx_map[qubit_idx]
                            edge['neighbor_idx'] = neighbor_uid
                            edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']
                            
                            # Also update Mx's in_edge
                            mx_entry = core_tensor_list[neighbor_uid]
                            mx_entry['in_edge_list'][0]['neighbor_idx'] = new_idx
                            mx_entry['in_edge_list'][0]['neighbor_name'] = entry['core_name']
                    else:
                        # Connect to another Left Core
                        original_neighbor = edge['neighbor_idx']
                        if original_neighbor in left_core_map:
                            neighbor_uid = left_core_map[original_neighbor]
                            edge['neighbor_idx'] = neighbor_uid
                            edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']

            # 2.2 Update RIGHT cores
            for original_idx, new_idx in right_core_map.items():
                entry = core_tensor_list[new_idx]
                
                # Update in_edges (originally out_edges)
                for edge in entry['in_edge_list']:
                    if 'is_cross_partition' in edge and edge['is_cross_partition']:
                        continue
                    if edge['neighbor_idx'] == -1:
                        # Originally out_edge to Mx (in reverse view)
                        # Connect to Mx's out_edge
                        qubit_idx = edge['qubit_idx']
                        if qubit_idx in mx_map:
                            neighbor_uid = mx_map[qubit_idx]
                            edge['neighbor_idx'] = neighbor_uid
                            edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']
                            
                            # Also update Mx's out_edge
                            mx_entry = core_tensor_list[neighbor_uid]
                            mx_entry['out_edge_list'][0]['neighbor_idx'] = new_idx
                            mx_entry['out_edge_list'][0]['neighbor_name'] = entry['core_name']
                    else:
                        # Connect to another Right Core
                        original_neighbor = edge['neighbor_idx']
                        if original_neighbor in right_core_map:
                            neighbor_uid = right_core_map[original_neighbor]
                            edge['neighbor_idx'] = neighbor_uid
                            edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']
                
                # Update out_edges (originally in_edges)
                for edge in entry['out_edge_list']:
                    if 'is_cross_partition' in edge and edge['is_cross_partition']:
                        continue

                    if edge['neighbor_idx'] == -1:
                        # Originally in_edge from Circuit (in reverse view)
                        # Connect to Right Circuit
                        qubit_idx = edge['qubit_idx']
                        if qubit_idx in right_circuit_map:
                            neighbor_uid = right_circuit_map[qubit_idx]
                            edge['neighbor_idx'] = neighbor_uid
                            edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']
                            
                            # Also update circuit's in_edge
                            circuit_entry = core_tensor_list[neighbor_uid]
                            circuit_entry['in_edge_list'][0]['neighbor_idx'] = new_idx
                            circuit_entry['in_edge_list'][0]['neighbor_name'] = entry['core_name']
                    else:
                        # Connect to another Right Core
                        original_neighbor = edge['neighbor_idx']
                        if original_neighbor in right_core_map:
                            neighbor_uid = right_core_map[original_neighbor]
                            edge['neighbor_idx'] = neighbor_uid
                            edge['neighbor_name'] = core_tensor_list[neighbor_uid]['core_name']

            # ========================================
            # Step 2.5: Assign symbols to edges
            # ========================================
            def symbol_generator():
                i = 0
                while True:
                    sym = opt_einsum.get_symbol(i)
                    if sym not in ['a', 'b']:
                        yield sym
                    i += 1
            
            symbol_gen = symbol_generator()
            
            for entry in core_tensor_list:
                # Assign symbols to out_edges and propagate to neighbor's in_edges
                for edge in entry['out_edge_list']:
                    if 'symbol' in edge:
                        continue
                    
                    sym = next(symbol_gen)
                    edge['symbol'] = sym
                    
                    neighbor_idx = edge['neighbor_idx']
                    if neighbor_idx >= 0:
                        neighbor_entry = core_tensor_list[neighbor_idx]
                        # Find corresponding in_edge in neighbor
                        # Match by neighbor_idx (which is current entry) and qubit_idx
                        for in_edge in neighbor_entry['in_edge_list']:
                            if in_edge['neighbor_idx'] == entry['core_idx'] and \
                               in_edge['qubit_idx'] == edge['qubit_idx']:
                                in_edge['symbol'] = sym
                                break

            # process remaining in_edges that might not have been assigned
            for entry in core_tensor_list:
                # Assign symbols to out_edges and propagate to neighbor's in_edges
                for edge in entry['in_edge_list']:
                    if 'symbol' in edge:
                        continue
                    
                    sym = next(symbol_gen)
                    edge['symbol'] = sym

            # print(f'core_tensor_list :\n{core_tensor_list}')
            # print(f'left_core_map :\n{left_core_map}')
            # print(f'right_core_map :\n{right_core_map}')
    

            # ========================================
            # Step 3: Process each qubit
            # ========================================
            next_uid = len(core_tensor_list)
            
            for qubit_idx in qctn.qubit_indices:
                # Find all entries with edges on current qubit
                entries_on_qubit = []
                for entry in core_tensor_list:
                    has_edge = False
                    for edge in entry['in_edge_list']:
                        if edge['qubit_idx'] == qubit_idx:
                            has_edge = True
                            break
                    if not has_edge:
                        for edge in entry['out_edge_list']:
                            if edge['qubit_idx'] == qubit_idx:
                                has_edge = True
                                break
                    if has_edge:
                        entries_on_qubit.append(entry)
                
                if not entries_on_qubit:
                    continue

                # print(f'entries_on_qubit \n{entries_on_qubit}')

                # 遍历所有entries_on_qubit，如果circuit_state是某个core入边或出边，则把这个circuit_state也加进来
                # 在这里，neighbor_idx已经没办法直接和core_tensor_list对上了，所以需要再遍历一遍core_tensor_list来找
                additional_entries = []
                for entry in entries_on_qubit:
                    # print(f'preprocess entry {entry["core_name"]}')
                    # Check in_edges
                    for edge in entry['in_edge_list']:
                        # print(f'preprocess in edge {edge}')
                        neighbor_idx = edge['neighbor_idx']
                        if neighbor_idx >= 0:
                            # Find neighbor entry by core_idx
                            neighbor_entry = None
                            for cand in core_tensor_list:
                                if cand['core_idx'] == neighbor_idx:
                                    neighbor_entry = cand
                                    break
                            
                            if neighbor_entry and neighbor_entry['tensor_source'] == 'circuit' and \
                                 neighbor_entry not in entries_on_qubit and \
                                    neighbor_entry not in additional_entries:
                                additional_entries.append(neighbor_entry)
                    # Check out_edges
                    for edge in entry['out_edge_list']:
                        # print(f'preprocess out edge {edge}')
                        neighbor_idx = edge['neighbor_idx']
                        if neighbor_idx >= 0:
                            # Find neighbor entry by core_idx
                            neighbor_entry = None
                            for cand in core_tensor_list:
                                if cand['core_idx'] == neighbor_idx:
                                    neighbor_entry = cand
                                    break
                            
                            if neighbor_entry and neighbor_entry['tensor_source'] == 'circuit' and \
                                 neighbor_entry not in entries_on_qubit and \
                                    neighbor_entry not in additional_entries:
                                additional_entries.append(neighbor_entry)
                
                entries_on_qubit.extend(additional_entries)

                # Find connected groups
                groups = self._find_connected_groups_symmetric(entries_on_qubit, qubit_idx)
                
                # if len(groups) > 1:
                #     raise NotImplementedError(
                #         f"Multiple disconnected groups ({len(groups)}) found at qubit {qubit_idx}. "
                #         "Currently only single connected group is supported."
                #     )
                
                # Contract groups and collect updates
                new_entries = []
                ids_to_remove = set()
                old_to_new_map = {} # old_core_idx -> new_entry
                
                for group_idx, group in enumerate(groups):

                    #TODO: 只有一个元素时，存在bug，结果不对，先跳过。
                    # if len(group) == 1:
                    #     continue

                    new_entry = self._contract_symmetric_group(
                        group, qubit_idx, backend, nqubits,
                        cores_dict, circuit_states, measure_matrices
                    )
                    
                    if new_entry is None:
                        continue

                    # Check if new_entry is just one of the old entries (no contraction happened)
                    is_existing = any(new_entry is member for member in group)
                    if is_existing:
                        continue
                        
                    # Assign unique ID and Name
                    new_entry['core_idx'] = next_uid
                    new_entry['core_name'] = f"merged_q{qubit_idx}_g{group_idx}_{next_uid}"
                    next_uid += 1
                    
                    new_entries.append(new_entry)
                    
                    for member in group:
                        ids_to_remove.add(member['core_idx'])
                        old_to_new_map[member['core_idx']] = new_entry
                
                if not new_entries:
                    continue

                # Update core_tensor_list
                core_tensor_list = [
                    entry for entry in core_tensor_list 
                    if entry['core_idx'] not in ids_to_remove
                ]
                core_tensor_list.extend(new_entries)
                
                # Update neighbor connections for all edges
                for entry in core_tensor_list:
                    for edge in entry['in_edge_list'] + entry['out_edge_list']:
                        neighbor_idx = edge['neighbor_idx']
                        if neighbor_idx in old_to_new_map:
                            new_neighbor = old_to_new_map[neighbor_idx]
                            edge['neighbor_idx'] = new_neighbor['core_idx']
                            edge['neighbor_name'] = new_neighbor['core_name']
            
                # print(f"core_tensor_list after all qubits processed:\n{core_tensor_list}")
            # ========================================
            # Step 4: Return final result
            # ========================================
            # print(f"Final core_tensor_list length: {len(core_tensor_list)}")
            if len(core_tensor_list) == 1:
                return self._get_tensor(core_tensor_list[0], cores_dict, circuit_states, measure_matrices)
            elif len(core_tensor_list) == 0:
                raise RuntimeError("No tensor left after contraction")
            else:
                # Contract remaining tensors
                result = self._contract_remaining(core_tensor_list, backend, cores_dict, circuit_states, measure_matrices)
                return result
        
        return compute_fn
    
    def estimate_cost(self, qctn, shapes_info: Dict[str, Any]) -> float:
        """
        Estimate cost of greedy contraction.
        
        Returns a moderate fixed value - greedy is usually between einsum and MPS.
        """
        return 5e5  # Moderate cost estimate
    
    @property
    def name(self) -> str:
        return "greedy"


    def _find_connected_groups_symmetric(self, entries_on_qubit: List[Dict], qubit_idx: int) -> List[List[Dict]]:
        """
        Find connected components among entries on a given qubit.
        
        Two entries are connected if they share an edge (neighbor relationship).
        """
        if not entries_on_qubit:
            return []
        
        n = len(entries_on_qubit)
        if n == 1:
            return [entries_on_qubit]
        
        # Build index mapping
        core_indices = [entry['core_idx'] for entry in entries_on_qubit]
        idx_to_pos = {idx: pos for pos, idx in enumerate(core_indices)}
        
        # Union-Find
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Check connectivity via edges
        for i, entry in enumerate(entries_on_qubit):
            for edge in entry['out_edge_list']:
                neighbor_idx = edge['neighbor_idx']
                if neighbor_idx >= 0 and neighbor_idx in idx_to_pos:
                    union(i, idx_to_pos[neighbor_idx])
            for edge in entry['in_edge_list']:
                neighbor_idx = edge['neighbor_idx']
                if neighbor_idx >= 0 and neighbor_idx in idx_to_pos:
                    union(i, idx_to_pos[neighbor_idx])
        
        # Group by root
        groups_dict = {}
        for i in range(n):
            root = find(i)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(entries_on_qubit[i])
        
        return list(groups_dict.values())


    def _get_tensor(self, entry: Dict, cores_dict, circuit_states, measure_matrices):
        """Helper to retrieve tensor from source info."""
        if 'tensor' in entry:
            return entry['tensor']
        
        source = entry['tensor_source']
        key = entry['tensor_key']
        
        if source == 'core':
            return cores_dict[key]
        elif source == 'transpose':
            if self.backend.is_complex(cores_dict[key]):
                return cores_dict[key].conj()
            else:
                return cores_dict[key]
        elif source == 'circuit':
            return circuit_states[key]
        elif source == 'mx':
            return measure_matrices[key]
        else:
            raise ValueError(f"Unknown tensor source: {source}")


    def _contract_symmetric_group(
        self,
        group: List[Dict], 
        qubit_idx: int, 
        backend,
        nqubits: int,
        cores_dict,
        circuit_states,
        measure_matrices
    ) -> Optional[Dict]:
        """
        Contract all entries in a group, eliminating edges on the current qubit.
        
        This handles the symmetric structure: Left cores, Mx, Right cores, circuit states.
        
        Key: For RIGHT version tensors, the edge index order is reversed relative to tensor dims.
        When building einsum, we need to handle this reversal.
        
        Args:
            group: List of entries to contract
            qubit_idx: Current qubit index being processed
            backend: Computation backend
            nqubits: Total number of qubits
        
        Returns:
            New entry with contracted tensor and updated edges
        """
        import torch
        import opt_einsum
        
        if len(group) == 1:
            entry = group[0]
            # Check if this single entry has any edges on current qubit that need removal
            has_qubit_edge = False
            for edge in entry['in_edge_list'] + entry['out_edge_list']:
                if edge['qubit_idx'] == qubit_idx:
                    has_qubit_edge = True
                    break
            if not has_qubit_edge:
                return entry
        
        # print(f"\n  _contract_symmetric_group at qubit {qubit_idx}")
        # print(f"  Group members: {[e['core_name'] for e in group]}")
        
        # ========================================
        # Build einsum expression
        # ========================================
        tensor_list = []
        einsum_parts = []
        
        # Group core indices
        group_indices = set(entry['core_idx'] for entry in group)
        
        # Collect output edges and symbols
        collected_in_edges = []   # List of (symbol, edge)
        collected_out_edges = []  # List of (symbol, edge)
        batch_symbols = set()
        
        for entry in group:
            
            # print(f"Processing entry: {entry['core_name']}")

            core_idx = entry['core_idx']
            tensor = self._get_tensor(entry, cores_dict, circuit_states, measure_matrices)
            side = entry['side']
            
            tensor_list.append(tensor)
            
            # Handle batch symbols
            batch_sym = entry.get('batch_symbol', '')
            for char in batch_sym:
                batch_symbols.add(char)
            
            part = batch_sym
            
            if side == TensorSide.RIGHT:
                # RIGHT version: tensor dims are [original_in..., original_out...]
                # But edge lists are reversed: in_edge_list is reversed(original_out_edge_list)
                #                              out_edge_list is reversed(original_in_edge_list)
                
                original_in_count = entry.get('original_in_edge_count', len(entry['out_edge_list']))
                original_out_count = entry.get('original_out_edge_count', len(entry['in_edge_list']))
                
                # Build symbols in tensor dimension order (original_in first, then original_out)
                # Note: batch_sym is already added to part, so we only need to handle non-batch dims
                dim_symbols = [None] * (tensor.ndim - len(batch_sym))
                
                # out_edge_list (was original in_edges, reversed)
                # out_edge_list[i] <-> tensor dim (original_in_count - 1 - i)
                for edge_list_idx, edge in enumerate(entry['out_edge_list']):
                    
                    # print(f"  RIGHT out_edge_list idx {edge_list_idx}, edge: {edge}")
                    
                    symbol = edge['symbol']
                    tensor_dim = original_in_count - 1 - edge_list_idx
                    dim_symbols[tensor_dim] = symbol
                    
                    # Check if edge should be kept
                    neighbor_idx = edge['neighbor_idx']
                    is_internal = neighbor_idx >= 0 and neighbor_idx in group_indices
                    is_on_current_qubit = edge['qubit_idx'] == qubit_idx
                    
                    # empty_edge = edge['neighbor_idx'] == -1 and edge['neighbor_name'] == ""
                    empty_edge = edge['neighbor_idx'] == -1 # and edge['neighbor_name'] == ""
                    
                    # print(f"out edge {edge} empty_edge: {empty_edge}, is_internal: {is_internal}, is_on_current_qubit: {is_on_current_qubit}")

                    if empty_edge or not is_internal and not is_on_current_qubit:
                        collected_out_edges.append((symbol, edge.copy()))
                
                # in_edge_list (was original out_edges, reversed)
                # in_edge_list[i] <-> tensor dim (original_in_count + original_out_count - 1 - i)
                for edge_list_idx, edge in enumerate(entry['in_edge_list']):

                    # print(f"  RIGHT in_edge_list idx {edge_list_idx}, edge: {edge}")

                    symbol = edge['symbol']
                    tensor_dim = original_in_count + original_out_count - 1 - edge_list_idx
                    dim_symbols[tensor_dim] = symbol
                    
                    neighbor_idx = edge['neighbor_idx']
                    is_internal = neighbor_idx >= 0 and neighbor_idx in group_indices
                    is_on_current_qubit = edge['qubit_idx'] == qubit_idx
                    
                    # empty_edge = edge['neighbor_idx'] == -1 and edge['neighbor_name'] == ""
                    empty_edge = edge['neighbor_idx'] == -1 # and edge['neighbor_name'] == ""

                    # print(f"in edge {edge} empty_edge: {empty_edge}, is_internal: {is_internal}, is_on_current_qubit: {is_on_current_qubit}")
                    
                    if empty_edge or not is_internal and not is_on_current_qubit:
                        collected_in_edges.append((symbol, edge.copy()))
                
                part += "".join(s for s in dim_symbols if s is not None)
                einsum_parts.append(part)
                
            else:
                # LEFT version, MIDDLE (Mx), Merged, or circuit states
                # Standard order: batch + in_edges + out_edges
                
                # In edges
                for edge in entry['in_edge_list']:
                    symbol = edge['symbol']
                    part += symbol
                    
                    neighbor_idx = edge['neighbor_idx']
                    is_internal = neighbor_idx >= 0 and neighbor_idx in group_indices
                    is_on_current_qubit = edge['qubit_idx'] == qubit_idx

                    # empty_edge = edge['neighbor_idx'] == -1 and edge['neighbor_name'] == ""
                    empty_edge = edge['neighbor_idx'] == -1 # and edge['neighbor_name'] == ""
                    
                    # print(f"in edge {edge} empty_edge: {empty_edge}, is_internal: {is_internal}, is_on_current_qubit: {is_on_current_qubit}")

                    if empty_edge or not is_internal and not is_on_current_qubit:
                        collected_in_edges.append((symbol, edge.copy()))
                
                # Out edges
                for edge in entry['out_edge_list']:
                    symbol = edge['symbol']
                    part += symbol
                    
                    neighbor_idx = edge['neighbor_idx']
                    is_internal = neighbor_idx >= 0 and neighbor_idx in group_indices
                    is_on_current_qubit = edge['qubit_idx'] == qubit_idx

                    # empty_edge = edge['neighbor_idx'] == -1 and edge['neighbor_name'] == ""
                    empty_edge = edge['neighbor_idx'] == -1 # and edge['neighbor_name'] == ""
                    
                    # print(f"out edge {edge} empty_edge: {empty_edge}, is_internal: {is_internal}, is_on_current_qubit: {is_on_current_qubit}")

                    if empty_edge or not is_internal and not is_on_current_qubit:
                        collected_out_edges.append((symbol, edge.copy()))
                
                einsum_parts.append(part)
        
        # Construct output_symbols in specific order: a, b, in_edges, out_edges
        output_symbols = []
        new_batch_symbol = ""
        if 'a' in batch_symbols:
            output_symbols.append('a')
            new_batch_symbol += 'a'
        if 'b' in batch_symbols:
            output_symbols.append('b')
            new_batch_symbol += 'b'

        # print(f'collected_in_edges : {collected_in_edges}')
        # print(f'collected_out_edges : {collected_out_edges}')

        for sym, _ in collected_in_edges:
            output_symbols.append(sym)
        for sym, _ in collected_out_edges:
            output_symbols.append(sym)
        
        # Build einsum equation
        einsum_eq = ",".join(einsum_parts) + "->" + "".join(output_symbols)
        
        # mapping symbol to standard einsum symbols
        def remap_symbols(einsum_eq: str) -> str:
            import opt_einsum
            symbol_map = {
                'a': 'a',
                'b': 'b',
                ',': ',',
                '-': '-',
                '>': '>',
            }

            idx = 2
            for i, c in enumerate(einsum_eq):
                if c not in symbol_map:
                    symbol_map[c] = opt_einsum.get_symbol(idx)
                    idx += 1
            remapped = "".join(symbol_map.get(c, c) for c in einsum_eq)
            return remapped

        einsum_eq = remap_symbols(einsum_eq)

        # print(f"  Einsum equation: {einsum_eq}")
        # print(f"  Tensor shapes: {[t.shape for t in tensor_list]}")
        
        # Execute contraction
        # result_tensor = torch.einsum(einsum_eq, *tensor_list)
        
        # Check for TNTensors and prepare inputs for einsum
        raw_tensor_list = []
        total_scale = None
        total_log_scale = None
        has_tntensor = False

        for t in tensor_list:
            if isinstance(t, TNTensor):
                has_tntensor = True
                raw_tensor_list.append(t.tensor)
                if total_scale is None:
                    # total_scale = 1.0 * t.scale
                    total_scale = t.scale #.detach().clone()
                else:
                    total_scale *= t.scale #.detach().clone()
                
                if total_log_scale is None:
                    total_log_scale = t.log_scale
                else:
                    total_log_scale += t.log_scale
            else:
                raw_tensor_list.append(t)
        
        # Execute contraction
        if has_tntensor:
            # print("  Using TNTensor contraction with scale handling.")
            # print(f"  einsum eq: {einsum_eq} raw_tensor_list shapes: {[t.shape for t in raw_tensor_list]}")
            # Use raw tensors for contraction
            raw_result = torch.einsum(einsum_eq, *raw_tensor_list)
        
            # import numpy as np
            # raw_result = raw_result.contiguous()
            # tmp = raw_result.abs().max().detach().item()
            # raw_result /= tmp
            # total_scale *= tmp
            # total_log_scale += np.log(tmp)

            # result_tensor = TNTensor(raw_result / tmp, scale=total_scale * tmp, log_scale=total_log_scale + np.log(tmp))

            # Create result TNTensor
            result_tensor = TNTensor(raw_result, scale=total_scale, log_scale=total_log_scale)
            # result_tensor = TNTensor(raw_result, scale=total_scale)

            # tmp = result_tensor.tensor.abs().max()
            
            # result_tensor.auto_scale()
        else:
            result_tensor = torch.einsum(einsum_eq, *tensor_list)

            # print(f"torch einsum einsum_eq: {einsum_eq}")
            # for tt in tensor_list:
            #     print(f"tt: {tt.shape} {tt}")
            # print(f"result_tensor: {result_tensor.shape} {result_tensor}")

        # print(f"  Result shape: {result_tensor.shape}")
        
        # Build remaining edges
        remaining_in_edges = [edge for _, edge in collected_in_edges]
        remaining_out_edges = [edge for _, edge in collected_out_edges]
        
        # Create new entry
        new_core_idx = -1 - qubit_idx
        
        new_entry = {
            'core_idx': new_core_idx,
            'core_name': f"merged_{qubit_idx}",
            'tensor': result_tensor,
            'tensor_source': 'merged',
            'tensor_key': None,
            'in_edge_list': remaining_in_edges,
            'out_edge_list': remaining_out_edges,
            'side': TensorSide.MIDDLE,  # Merged is considered middle
            'batch_symbol': new_batch_symbol,
        }

        # print(f"  Created new merged entry: {new_entry['core_name']} with shape {result_tensor.shape}")
        # print(f"new_entry :\n{new_entry}")
        
        return new_entry


    def _contract_remaining(self, core_tensor_list: List[Dict], backend, cores_dict, circuit_states, measure_matrices) -> Any:
        """
        Contract any remaining tensors into final result.
        """
        import torch
        import opt_einsum
        
        if len(core_tensor_list) == 0:
            raise RuntimeError("No tensors to contract")
        
        if len(core_tensor_list) == 1:
            return self._get_tensor(core_tensor_list[0], cores_dict, circuit_states, measure_matrices)
        
        # Simple contraction of remaining tensors
        tensors = [self._get_tensor(entry, cores_dict, circuit_states, measure_matrices) for entry in core_tensor_list]
        
        # Build simple einsum to contract all
        # We can use the symbols already assigned to edges!
        parts = []
        output_symbols = []
        
        for entry in core_tensor_list:
            part = ""
            # We need to reconstruct the part string based on side and edges, similar to _contract_symmetric_group
            # But here we assume all edges are contracted except maybe batch dims?
            # Actually, if we are at the end, all edges should be contracted or open.
            # Let's reuse the logic from _contract_symmetric_group but for all tensors at once.
            
            # ... Actually, simpler to just use the symbols we assigned.
            # But we need to handle the RIGHT version index reversal again.
            
            side = entry['side']
            tensor = tensors[len(parts)] # corresponding tensor
            
            if side == TensorSide.MIDDLE:
                mx_shape = tensor.shape
                batch_ndim = len(mx_shape) - 2
                if batch_ndim >= 1: part += 'a'
                if batch_ndim >= 2: part += 'b'
                if entry['in_edge_list']: part += entry['in_edge_list'][0]['symbol']
                if entry['out_edge_list']: part += entry['out_edge_list'][0]['symbol']
                
            elif side == TensorSide.RIGHT:
                original_in_count = entry.get('original_in_edge_count', len(entry['out_edge_list']))
                original_out_count = entry.get('original_out_edge_count', len(entry['in_edge_list']))
                dim_symbols = [None] * tensor.ndim
                
                for i, edge in enumerate(entry['out_edge_list']):
                    dim_symbols[original_in_count - 1 - i] = edge['symbol']
                for i, edge in enumerate(entry['in_edge_list']):
                    dim_symbols[original_in_count + original_out_count - 1 - i] = edge['symbol']
                part = "".join(s for s in dim_symbols if s is not None)
                
            else: # LEFT or merged
                for edge in entry['in_edge_list']: part += edge['symbol']
                for edge in entry['out_edge_list']: part += edge['symbol']
                
                # For merged tensors, we might have batch dims 'a', 'b' if they were preserved
                # But merged tensors store 'tensor' directly, so we don't need special handling for 'a','b' 
                # UNLESS they are part of the tensor shape but not in edge list.
                # In _contract_symmetric_group, we added 'a','b' to output_symbols.
                # So the merged tensor will have 'a','b' as first dimensions.
                # We need to account for that.
                if entry.get('tensor_source') == 'merged':
                    # Check if 'a' or 'b' are in the tensor shape
                    # We can check if they were in the output symbols of the contraction that created it
                    # But we don't have that info here easily.
                    # However, 'a' and 'b' are global symbols for batch.
                    # If the tensor has extra dims at start, they are likely batch dims.
                    num_edges = len(entry['in_edge_list']) + len(entry['out_edge_list'])
                    extra_dims = tensor.ndim - num_edges
                    prefix = ""
                    if extra_dims >= 1: prefix += 'a'
                    if extra_dims >= 2: prefix += 'b'
                    part = prefix + part
            
            parts.append(part)
            
            # Collect output symbols (batch dims)
            if 'a' in part and 'a' not in output_symbols: output_symbols.append('a')
            if 'b' in part and 'b' not in output_symbols: output_symbols.append('b')

        einsum_eq = ",".join(parts) + "->" + "".join(output_symbols)
        
        # print(f"[Greedy] Final contraction")
        # print(f"  Einsum: {einsum_eq}")
        
        return torch.einsum(einsum_eq, *tensors)
