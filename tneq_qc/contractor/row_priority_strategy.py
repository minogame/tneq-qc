"""
Row-priority contraction strategy.

This module provides the RowPriorityStrategy class, which processes qubits
one by one (row-by-row) using symmetric expansion: L cores, Mx, R cores.

Unlike GreedyStrategy, the graph construction and wiring logic lives in
QCTN.build_graph().  This strategy only handles the
per-qubit contraction (Stage 3) and result assembly (Stage 4).

The contractor is agnostic to tensor semantics (cores, circuit states,
measure matrices, etc.) — it only sees entries with 'tensor' and edge info.
"""

from __future__ import annotations
from typing import Dict, Any, Callable, List, Optional
from ..core.tn_tensor import TNTensor
from ..core.qctn import TensorSide

from .base import ContractionStrategy


class RowPriorityStrategy(ContractionStrategy):
    """Row-priority contraction strategy that processes qubits sequentially.

    Delegates graph construction to ``QCTN.build_graph``
    and focuses on per-qubit contraction logic.
    """

    def check_compatibility(self, qctn, shapes_info: Dict[str, Any]) -> bool:
        return True

    def get_compute_function(self, qctn, shapes_info: Dict[str, Any], backend, **_kwargs) -> Callable:
        """Return a compute function for symmetric (A · A†) contraction.

        ``qctn`` and ``backend`` are captured via closure.  Core tensors are
        read directly from ``qctn.cores_weights`` inside
        ``qctn.build_graph()``.

        An engine-adapter wrapper is returned so that the engine's standard
        calling convention ``(cores_dict, circuit_states, measure_matrices, …)``
        is accepted without modification (unused arguments are discarded).
        """
        self.backend = backend

        def compute_fn():
            """Symmetric contraction: Tr(A† · A).

            Delegates graph construction to
            ``qctn.build_graph()``, which only processes
            ``qctn.adjacency_table``.  No circuit states or measurement
            matrices are referenced here.
            """
            # Build graph — every entry already has 'tensor' embedded
            core_tensor_list, _maps = qctn.build_graph()

            # ==============================================================
            # Per-qubit contraction
            # ==============================================================
            next_uid = len(core_tensor_list)

            for qubit_idx in qctn.qubit_indices:
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

                # Pull in circuit entries that are neighbors of entries_on_qubit
                additional_entries = []
                for entry in entries_on_qubit:
                    for edge in entry['in_edge_list'] + entry['out_edge_list']:
                        neighbor_idx = edge['neighbor_idx']
                        if neighbor_idx >= 0:
                            for cand in core_tensor_list:
                                if cand['core_idx'] == neighbor_idx:
                                    if (cand['tensor_source'] == 'circuit'
                                            and cand not in entries_on_qubit
                                            and cand not in additional_entries):
                                        additional_entries.append(cand)
                                    break
                entries_on_qubit.extend(additional_entries)

                groups = _find_connected_groups(entries_on_qubit)

                new_entries = []
                ids_to_remove = set()
                old_to_new_map = {}

                for group_idx, group in enumerate(groups):
                    new_entry = _contract_group(group, qubit_idx, backend)
                    if new_entry is None:
                        continue
                    is_existing = any(new_entry is member for member in group)
                    if is_existing:
                        continue

                    new_entry['core_idx'] = next_uid
                    new_entry['core_name'] = f"merged_q{qubit_idx}_g{group_idx}_{next_uid}"
                    next_uid += 1

                    new_entries.append(new_entry)
                    for member in group:
                        ids_to_remove.add(member['core_idx'])
                        old_to_new_map[member['core_idx']] = new_entry

                if not new_entries:
                    continue

                core_tensor_list = [
                    entry for entry in core_tensor_list
                    if entry['core_idx'] not in ids_to_remove
                ]
                core_tensor_list.extend(new_entries)

                for entry in core_tensor_list:
                    for edge in entry['in_edge_list'] + entry['out_edge_list']:
                        if edge['neighbor_idx'] in old_to_new_map:
                            new_neighbor = old_to_new_map[edge['neighbor_idx']]
                            edge['neighbor_idx'] = new_neighbor['core_idx']
                            edge['neighbor_name'] = new_neighbor['core_name']

            # ==============================================================
            # Return final result
            # ==============================================================
            if len(core_tensor_list) == 1:
                return core_tensor_list[0]['tensor']
            elif len(core_tensor_list) == 0:
                raise RuntimeError("No tensor left after contraction")
            else:
                return _contract_remaining(core_tensor_list, backend)

        # ------------------------------------------------------------------
        # Engine adapter
        # The engine calls compute_fn with the standard signature:
        #   (cores_dict, circuit_states, measure_matrices, right_cores_dict=None)
        # All arguments are intentionally ignored here — tensors and graph
        # structure are read directly from qctn inside build_graph.
        # ------------------------------------------------------------------
        def _engine_fn(_cores_dict, _circuit_states, _measure_matrices, **_):
            return compute_fn()

        return _engine_fn

    def estimate_cost(self, qctn, shapes_info: Dict[str, Any]) -> float:
        return 5e5

    @property
    def name(self) -> str:
        return "row_priority"


# ======================================================================
# Helper functions (module-level to keep strategy class clean)
# ======================================================================

def _find_connected_groups(entries_on_qubit: List[Dict]) -> List[List[Dict]]:
    """Find connected components via Union-Find."""
    if not entries_on_qubit:
        return []
    n = len(entries_on_qubit)
    if n == 1:
        return [entries_on_qubit]

    core_indices = [entry['core_idx'] for entry in entries_on_qubit]
    idx_to_pos = {idx: pos for pos, idx in enumerate(core_indices)}

    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, entry in enumerate(entries_on_qubit):
        for edge in entry['out_edge_list'] + entry['in_edge_list']:
            neighbor_idx = edge['neighbor_idx']
            if neighbor_idx >= 0 and neighbor_idx in idx_to_pos:
                union(i, idx_to_pos[neighbor_idx])

    groups_dict: Dict[int, list] = {}
    for i in range(n):
        root = find(i)
        groups_dict.setdefault(root, []).append(entries_on_qubit[i])
    return list(groups_dict.values())


def _contract_group(
    group: List[Dict],
    qubit_idx: int,
    backend,
) -> Optional[Dict]:
    """Contract all entries in a group, eliminating edges on the current qubit."""
    import opt_einsum

    if len(group) == 1:
        entry = group[0]
        has_qubit_edge = False
        for edge in entry['in_edge_list'] + entry['out_edge_list']:
            if edge['qubit_idx'] == qubit_idx:
                has_qubit_edge = True
                break
        if not has_qubit_edge:
            return entry

    tensor_list = []
    einsum_parts = []
    group_indices = set(entry['core_idx'] for entry in group)
    collected_in_edges = []
    collected_out_edges = []
    batch_symbols = set()

    for entry in group:
        tensor = entry['tensor']
        side = entry['side']
        tensor_list.append(tensor)

        batch_sym = entry.get('batch_symbol', '')
        for char in batch_sym:
            batch_symbols.add(char)

        part = batch_sym

        if side == TensorSide.RIGHT:
            original_in_count = entry.get('original_in_edge_count', len(entry['out_edge_list']))
            original_out_count = entry.get('original_out_edge_count', len(entry['in_edge_list']))

            dim_symbols = [None] * (tensor.ndim - len(batch_sym))

            for edge_list_idx, edge in enumerate(entry['out_edge_list']):
                symbol = edge['symbol']
                tensor_dim = original_in_count - 1 - edge_list_idx
                dim_symbols[tensor_dim] = symbol

                neighbor_idx = edge['neighbor_idx']
                is_internal = neighbor_idx >= 0 and neighbor_idx in group_indices
                is_on_current_qubit = edge['qubit_idx'] == qubit_idx
                empty_edge = edge['neighbor_idx'] == -1

                if empty_edge or (not is_internal and not is_on_current_qubit):
                    collected_out_edges.append((symbol, edge.copy()))

            for edge_list_idx, edge in enumerate(entry['in_edge_list']):
                symbol = edge['symbol']
                tensor_dim = original_in_count + original_out_count - 1 - edge_list_idx
                dim_symbols[tensor_dim] = symbol

                neighbor_idx = edge['neighbor_idx']
                is_internal = neighbor_idx >= 0 and neighbor_idx in group_indices
                is_on_current_qubit = edge['qubit_idx'] == qubit_idx
                empty_edge = edge['neighbor_idx'] == -1

                if empty_edge or (not is_internal and not is_on_current_qubit):
                    collected_in_edges.append((symbol, edge.copy()))

            part += "".join(s for s in dim_symbols if s is not None)
            einsum_parts.append(part)

        else:
            # LEFT, MIDDLE, merged, circuit
            for edge in entry['in_edge_list']:
                symbol = edge['symbol']
                part += symbol
                neighbor_idx = edge['neighbor_idx']
                is_internal = neighbor_idx >= 0 and neighbor_idx in group_indices
                is_on_current_qubit = edge['qubit_idx'] == qubit_idx
                empty_edge = edge['neighbor_idx'] == -1
                if empty_edge or (not is_internal and not is_on_current_qubit):
                    collected_in_edges.append((symbol, edge.copy()))

            for edge in entry['out_edge_list']:
                symbol = edge['symbol']
                part += symbol
                neighbor_idx = edge['neighbor_idx']
                is_internal = neighbor_idx >= 0 and neighbor_idx in group_indices
                is_on_current_qubit = edge['qubit_idx'] == qubit_idx
                empty_edge = edge['neighbor_idx'] == -1
                if empty_edge or (not is_internal and not is_on_current_qubit):
                    collected_out_edges.append((symbol, edge.copy()))

            einsum_parts.append(part)

    # Build output symbols
    output_symbols = []
    new_batch_symbol = ""
    if 'a' in batch_symbols:
        output_symbols.append('a')
        new_batch_symbol += 'a'
    if 'b' in batch_symbols:
        output_symbols.append('b')
        new_batch_symbol += 'b'

    for sym, _ in collected_in_edges:
        output_symbols.append(sym)
    for sym, _ in collected_out_edges:
        output_symbols.append(sym)

    einsum_eq = ",".join(einsum_parts) + "->" + "".join(output_symbols)
    einsum_eq = _remap_symbols(einsum_eq)

    # Execute contraction — handle TNTensor scale propagation
    raw_tensor_list = []
    total_scale = None
    total_log_scale = None
    has_tntensor = False

    for t in tensor_list:
        if isinstance(t, TNTensor):
            has_tntensor = True
            raw_tensor_list.append(t.tensor)
            if total_scale is None:
                total_scale = t.scale
            else:
                total_scale *= t.scale
            if total_log_scale is None:
                total_log_scale = t.log_scale
            else:
                total_log_scale += t.log_scale
        else:
            raw_tensor_list.append(t)

    if has_tntensor:
        raw_result = backend.einsum(einsum_eq, *raw_tensor_list)
        result_tensor = TNTensor(raw_result, scale=total_scale, log_scale=total_log_scale)
    else:
        result_tensor = backend.einsum(einsum_eq, *tensor_list)

    remaining_in_edges = [edge for _, edge in collected_in_edges]
    remaining_out_edges = [edge for _, edge in collected_out_edges]

    return {
        'core_idx': -1 - qubit_idx,
        'core_name': f"merged_{qubit_idx}",
        'tensor': result_tensor,
        'tensor_source': 'merged',
        'tensor_key': None,
        'in_edge_list': remaining_in_edges,
        'out_edge_list': remaining_out_edges,
        'side': TensorSide.MIDDLE,
        'batch_symbol': new_batch_symbol,
    }


def _remap_symbols(einsum_eq: str) -> str:
    """Remap arbitrary symbols to standard opt_einsum symbols."""
    import opt_einsum
    symbol_map = {'a': 'a', 'b': 'b', ',': ',', '-': '-', '>': '>'}
    idx = 2
    for c in einsum_eq:
        if c not in symbol_map:
            symbol_map[c] = opt_einsum.get_symbol(idx)
            idx += 1
    return "".join(symbol_map.get(c, c) for c in einsum_eq)


def _contract_remaining(core_tensor_list: List[Dict], backend):
    """Contract any remaining tensors into final result."""
    import opt_einsum

    if len(core_tensor_list) == 0:
        raise RuntimeError("No tensors to contract")
    if len(core_tensor_list) == 1:
        return core_tensor_list[0]['tensor']

    tensors = [entry['tensor'] for entry in core_tensor_list]

    parts = []

    for entry in core_tensor_list:
        part = ""
        side = entry['side']
        tensor = tensors[len(parts)]

        if side == TensorSide.MIDDLE:
            mx_shape = tensor.shape if not isinstance(tensor, TNTensor) else tensor.tensor.shape
            batch_ndim = len(mx_shape) - 2
            if batch_ndim >= 1:
                part += 'a'
            if batch_ndim >= 2:
                part += 'b'
            if entry['in_edge_list']:
                part += entry['in_edge_list'][0]['symbol']
            if entry['out_edge_list']:
                part += entry['out_edge_list'][0]['symbol']

        elif side == TensorSide.RIGHT:
            original_in_count = entry.get('original_in_edge_count', len(entry['out_edge_list']))
            original_out_count = entry.get('original_out_edge_count', len(entry['in_edge_list']))
            t_shape = tensor.shape if not isinstance(tensor, TNTensor) else tensor.tensor.shape
            dim_symbols = [None] * len(t_shape)

            for edge_list_idx, edge in enumerate(entry['out_edge_list']):
                tensor_dim = original_in_count - 1 - edge_list_idx
                dim_symbols[tensor_dim] = edge['symbol']
            for edge_list_idx, edge in enumerate(entry['in_edge_list']):
                tensor_dim = original_in_count + original_out_count - 1 - edge_list_idx
                dim_symbols[tensor_dim] = edge['symbol']

            part = "".join(s for s in dim_symbols if s is not None)

        else:
            batch_sym = entry.get('batch_symbol', '')
            part = batch_sym
            for edge in entry['in_edge_list']:
                part += edge['symbol']
            for edge in entry['out_edge_list']:
                part += edge['symbol']

        parts.append(part)

    batch_sym_out = ""
    if any('a' in p for p in parts):
        batch_sym_out += 'a'
    if any('b' in p for p in parts):
        batch_sym_out += 'b'

    einsum_eq = ",".join(parts) + "->" + batch_sym_out
    einsum_eq = _remap_symbols(einsum_eq)

    # Handle TNTensor scale
    raw_tensors = []
    total_scale = None
    total_log_scale = None
    has_tntensor = False

    for t in tensors:
        if isinstance(t, TNTensor):
            has_tntensor = True
            raw_tensors.append(t.tensor)
            if total_scale is None:
                total_scale = t.scale
            else:
                total_scale *= t.scale
            if total_log_scale is None:
                total_log_scale = t.log_scale
            else:
                total_log_scale += t.log_scale
        else:
            raw_tensors.append(t)

    if has_tntensor:
        raw_result = backend.einsum(einsum_eq, *raw_tensors)
        return TNTensor(raw_result, scale=total_scale, log_scale=total_log_scale)
    else:
        return backend.einsum(einsum_eq, *tensors)
