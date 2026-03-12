"""
Tests for QCTN.build_graph and RowPriorityStrategy.

build_graph design (pure, 3-step):
  1. Embed    — iterate adjacency_table, embed tensor, classify tensor_source
  2. Wire     — remap neighbor_idx to new uid space
  3. Symbolize— assign unique einsum symbols; paired edges share one symbol

Returns (core_tensor_list, {'core_map': {orig_idx → uid}}).
No circuit states, no measure matrices, no auto-created mirror copies.
All structure must already be encoded in the QCTN before calling build_graph.
"""

import torch
import pytest
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN, TensorSide
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.contractor.row_priority_strategy import RowPriorityStrategy


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def backend():
    return BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')


@pytest.fixture
def qctn_2bit(backend):
    """2-qubit, 2-core QCTN: -3-A-3-B-3- on each qubit."""
    return QCTN("-3-A-3-B-3-\n-3-A-3-B-3-", backend=backend)


@pytest.fixture
def qctn_4bit(backend):
    """4-qubit, 2-core QCTN: a(q0,q1), b(q2,q3) with bond on q1-q2 line."""
    graph = (
        "-3-a--------3-\n"
        "-3-a--3--b--3-\n"
        "-3-------b--3-\n"
        "-3-------b--3-"
    )
    return QCTN(graph, backend=backend)


# ======================================================================
# Tests: build_graph
# ======================================================================

class TestBuildGraph:

    def test_entry_count_equals_ncores(self, qctn_2bit):
        """build_graph produces exactly ncores entries (one per adjacency_table row)."""
        entries, maps = qctn_2bit.build_graph()
        assert len(entries) == qctn_2bit.ncores

    def test_maps_has_core_map(self, qctn_2bit):
        """maps must contain 'core_map' with ncores entries."""
        _, maps = qctn_2bit.build_graph()
        assert 'core_map' in maps
        assert len(maps['core_map']) == qctn_2bit.ncores

    def test_maps_no_legacy_keys(self, qctn_2bit):
        """maps must not contain old left_core_map / right_core_map keys."""
        _, maps = qctn_2bit.build_graph()
        assert 'left_core_map'  not in maps
        assert 'right_core_map' not in maps

    def test_tensors_embedded(self, qctn_2bit):
        """Every entry must have a 'tensor' key."""
        entries, _ = qctn_2bit.build_graph()
        for entry in entries:
            assert 'tensor' in entry, (
                f"Entry {entry['core_name']} missing 'tensor' key"
            )

    def test_all_sides_are_left(self, qctn_2bit):
        """All entries use TensorSide.LEFT (no RIGHT/MIDDLE in pure graph)."""
        entries, _ = qctn_2bit.build_graph()
        for entry in entries:
            assert entry['side'] == TensorSide.LEFT

    def test_core_names_match_adjacency_table(self, qctn_2bit):
        """Entry core_name must equal the original adjacency_table core_name."""
        entries, _ = qctn_2bit.build_graph()
        orig_names = {info['core_name'] for info in qctn_2bit.adjacency_table}
        for entry in entries:
            assert entry['core_name'] in orig_names

    def test_all_edges_have_symbols(self, qctn_4bit):
        """Every edge in every entry must have a 'symbol' key after build."""
        entries, _ = qctn_4bit.build_graph()
        for entry in entries:
            for edge in entry['in_edge_list'] + entry['out_edge_list']:
                assert 'symbol' in edge, (
                    f"Edge missing symbol in entry {entry['core_name']}: {edge}"
                )

    def test_connected_edges_share_symbol(self, qctn_2bit):
        """For every bond: out_edge.symbol == paired in_edge.symbol."""
        entries, _ = qctn_2bit.build_graph()
        for entry in entries:
            for out_edge in entry['out_edge_list']:
                nb_idx = out_edge['neighbor_idx']
                if nb_idx < 0:
                    continue  # open boundary — no pair to check
                nb_entry = entries[nb_idx]
                matched = False
                for in_edge in nb_entry['in_edge_list']:
                    if (in_edge['neighbor_idx'] == entry['core_idx']
                            and in_edge['qubit_idx'] == out_edge['qubit_idx']):
                        assert in_edge['symbol'] == out_edge['symbol'], (
                            f"Symbol mismatch on bond "
                            f"{entry['core_name']}→{nb_entry['core_name']}"
                        )
                        matched = True
                        break
                assert matched, (
                    f"Could not find paired in_edge for out_edge "
                    f"{out_edge} in {entry['core_name']}"
                )

    def test_tensor_source_plain_tensor_is_core(self, qctn_2bit):
        """Plain torch.Tensor cores must get tensor_source='core'."""
        entries, _ = qctn_2bit.build_graph()
        for entry in entries:
            t = entry['tensor']
            if isinstance(t, torch.Tensor):
                assert entry['tensor_source'] == 'core'

    def test_tensor_source_tntensor_ref(self, qctn_2bit, backend):
        """TNTensor with is_ref=True (and not is_transposed) → tensor_source='ref'."""
        # Inject a ref TNTensor into one core weight
        core_name = qctn_2bit.cores[0]
        orig = qctn_2bit.cores_weights[core_name]
        raw = orig if isinstance(orig, torch.Tensor) else orig.tensor
        ref_tnt = TNTensor(raw.conj(), is_ref=True)
        qctn_2bit.cores_weights[core_name] = ref_tnt

        entries, maps = qctn_2bit.build_graph()
        uid = maps['core_map'][qctn_2bit.adjacency_table[0]['core_idx']]
        assert entries[uid]['tensor_source'] == 'ref'

        # Restore original weight
        qctn_2bit.cores_weights[core_name] = orig

    def test_tensor_source_tntensor_transpose(self, qctn_2bit, backend):
        """TNTensor with is_transposed=True → tensor_source='transpose'."""
        core_name = qctn_2bit.cores[0]
        orig = qctn_2bit.cores_weights[core_name]
        raw = orig if isinstance(orig, torch.Tensor) else orig.tensor
        trans_tnt = TNTensor(raw, is_ref=True, is_transposed=True)
        qctn_2bit.cores_weights[core_name] = trans_tnt

        entries, maps = qctn_2bit.build_graph()
        uid = maps['core_map'][qctn_2bit.adjacency_table[0]['core_idx']]
        assert entries[uid]['tensor_source'] == 'transpose'

        # Restore
        qctn_2bit.cores_weights[core_name] = orig

    def test_internal_bonds_wired(self, qctn_2bit):
        """Neighbor indices must point into core_tensor_list (not adjacency_table)."""
        entries, maps = qctn_2bit.build_graph()
        valid_uids = set(range(len(entries)))
        for entry in entries:
            for edge in entry['in_edge_list'] + entry['out_edge_list']:
                nb = edge['neighbor_idx']
                if nb != -1:
                    assert nb in valid_uids, (
                        f"neighbor_idx={nb} not in valid uid set {valid_uids}"
                    )

    def test_multi_qubit_entry_count(self, qctn_4bit):
        """4-qubit QCTN should produce exactly ncores entries."""
        entries, _ = qctn_4bit.build_graph()
        assert len(entries) == qctn_4bit.ncores


# ======================================================================
# Tests: RowPriorityStrategy
# ======================================================================

class TestRowPriorityStrategy:

    def test_strategy_name(self):
        assert RowPriorityStrategy().name == "row_priority"

    def test_check_compatibility(self, qctn_2bit):
        assert RowPriorityStrategy().check_compatibility(qctn_2bit, {}) is True

    def test_estimate_cost(self, qctn_2bit):
        assert RowPriorityStrategy().estimate_cost(qctn_2bit, {}) == 5e5

    def test_strategy_registered(self):
        """RowPriorityStrategy must be registered in the compiler."""
        from tneq_qc.contractor.compiler import StrategyCompiler
        strategies = StrategyCompiler.get_registered_strategies()
        assert "row_priority" in strategies

    def test_engine_fn_ignores_all_args(self, qctn_2bit):
        """Engine adapter must accept and ignore cores_dict / circuit / measure args."""
        backend = qctn_2bit.backend
        row_fn = RowPriorityStrategy().get_compute_function(qctn_2bit, {}, backend)
        cs = [torch.randn(3), torch.randn(3)]
        mx = [torch.randn(10, 3, 3), torch.randn(10, 3, 3)]
        # Both calls must return the same value regardless of cs/mx
        result_none = row_fn(qctn_2bit.cores_weights, None, None)
        result_with = row_fn(qctn_2bit.cores_weights, cs, mx)
        assert torch.allclose(result_none, result_with)
