"""
Tests for QCTN.build_symmetric_expansion_graph and RowPriorityStrategy.
"""

import torch
import numpy as np
import pytest
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN, TensorSide
from tneq_qc.contractor.row_priority_strategy import RowPriorityStrategy
from tneq_qc.contractor.greedy_strategy import GreedyStrategy


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


@pytest.fixture
def circuit_state_qctn_2bit():
    """Circuit states for 2-qubit QCTN (bond dim 3)."""
    return [torch.randn(3), torch.randn(3)]


@pytest.fixture
def circuit_state_qctn_4bit():
    """Circuit states for 4-qubit QCTN (bond dim 3)."""
    return [torch.randn(3) for _ in range(4)]


@pytest.fixture
def measure_matrices_qctn_2bit():
    """Measure matrices for 2-qubit QCTN (batch=10, K=3)."""
    return [torch.randn(10, 3, 3), torch.randn(10, 3, 3)]


@pytest.fixture
def measure_matrices_qctn_4bit():
    """Measure matrices for 4-qubit QCTN (batch=10, K=3)."""
    return [torch.randn(10, 3, 3) for _ in range(4)]


# ======================================================================
# Tests: build_symmetric_expansion_graph
# ======================================================================

class TestBuildSymmetricExpansionGraph:

    def test_no_circuit_no_measure(self, qctn_2bit):
        """With no circuits/measures, should have 2*ncores entries (L + R)."""
        entries, maps = qctn_2bit.build_symmetric_expansion_graph()
        assert len(entries) == 2 * qctn_2bit.ncores
        assert len(maps['left_core_map']) == qctn_2bit.ncores
        assert len(maps['right_core_map']) == qctn_2bit.ncores
        assert len(maps['mx_map']) == 0
        assert len(maps['left_circuit_map']) == 0
        assert len(maps['right_circuit_map']) == 0

    def test_with_circuits_and_measures(
        self, qctn_2bit, circuit_state_qctn_2bit, measure_matrices_qctn_2bit
    ):
        """Full expansion: L_cores + L_circuits + Mx + R_cores + R_circuits."""
        qctn_2bit.circuit_states = circuit_state_qctn_2bit
        qctn_2bit.measure_matrices = measure_matrices_qctn_2bit
        nq = qctn_2bit.nqubits
        nc = qctn_2bit.ncores
        entries, maps = qctn_2bit.build_symmetric_expansion_graph()
        expected = nc * 2 + nq * 2 + nq
        assert len(entries) == expected

    def test_auto_derive_shapes(
        self, qctn_2bit, circuit_state_qctn_2bit, measure_matrices_qctn_2bit
    ):
        """Shapes should be auto-derived from qctn.circuit_states/measure_matrices."""
        qctn_2bit.circuit_states = circuit_state_qctn_2bit
        qctn_2bit.measure_matrices = measure_matrices_qctn_2bit
        entries, maps = qctn_2bit.build_symmetric_expansion_graph()
        nq = qctn_2bit.nqubits
        nc = qctn_2bit.ncores
        expected = nc * 2 + nq * 2 + nq
        assert len(entries) == expected

    def test_tensors_embedded(
        self, qctn_2bit, circuit_state_qctn_2bit, measure_matrices_qctn_2bit
    ):
        """All entries should have 'tensor' key after build."""
        qctn_2bit.circuit_states = circuit_state_qctn_2bit
        qctn_2bit.measure_matrices = measure_matrices_qctn_2bit
        entries, _ = qctn_2bit.build_symmetric_expansion_graph()
        for entry in entries:
            assert 'tensor' in entry, (
                f"Entry {entry['core_name']} missing 'tensor' key"
            )

    def test_sides_correct(
        self, qctn_2bit, circuit_state_qctn_2bit, measure_matrices_qctn_2bit
    ):
        """LEFT, MIDDLE, RIGHT sides should be assigned correctly."""
        qctn_2bit.circuit_states = circuit_state_qctn_2bit
        qctn_2bit.measure_matrices = measure_matrices_qctn_2bit
        entries, _ = qctn_2bit.build_symmetric_expansion_graph()
        left_entries = [e for e in entries if e['side'] == TensorSide.LEFT]
        mid_entries = [e for e in entries if e['side'] == TensorSide.MIDDLE]
        right_entries = [e for e in entries if e['side'] == TensorSide.RIGHT]
        assert len(left_entries) == qctn_2bit.ncores + qctn_2bit.nqubits
        assert len(mid_entries) == qctn_2bit.nqubits
        assert len(right_entries) == qctn_2bit.ncores + qctn_2bit.nqubits

    def test_all_edges_have_symbols(
        self, qctn_4bit, circuit_state_qctn_4bit, measure_matrices_qctn_4bit
    ):
        """Every edge in every entry should have a 'symbol' key."""
        qctn_4bit.circuit_states = circuit_state_qctn_4bit
        qctn_4bit.measure_matrices = measure_matrices_qctn_4bit
        entries, _ = qctn_4bit.build_symmetric_expansion_graph()
        for entry in entries:
            for edge in entry['in_edge_list'] + entry['out_edge_list']:
                assert 'symbol' in edge, (
                    f"Edge missing symbol in entry {entry['core_name']}: {edge}"
                )

    def test_symmetric_right_reversed_edges(self, qctn_2bit):
        """RIGHT entries in symmetric mode should have reversed edge lists."""
        entries, maps = qctn_2bit.build_symmetric_expansion_graph()
        for orig_idx, right_uid in maps['right_core_map'].items():
            right_entry = entries[right_uid]
            left_uid = maps['left_core_map'][orig_idx]
            left_entry = entries[left_uid]
            assert len(right_entry['in_edge_list']) == len(left_entry['out_edge_list'])
            assert len(right_entry['out_edge_list']) == len(left_entry['in_edge_list'])

    def test_right_qctn_none(self, qctn_2bit):
        """right_qctn=None should produce no right entries."""
        entries, maps = qctn_2bit.build_symmetric_expansion_graph(right_qctn=None)
        assert len(entries) == qctn_2bit.ncores
        assert len(maps['right_core_map']) == 0

    def test_batch_symbol_3d_mx(self, qctn_2bit):
        """3D Mx shape should produce batch_symbol='a'."""
        mx_shapes = [(10, 3, 3)] * qctn_2bit.nqubits
        entries, maps = qctn_2bit.build_symmetric_expansion_graph(measure_shapes=mx_shapes)
        for qidx, uid in maps['mx_map'].items():
            assert entries[uid]['batch_symbol'] == 'a'

    def test_batch_symbol_4d_mx(self, qctn_2bit):
        """4D Mx shape should produce batch_symbol='ab'."""
        mx_shapes = [(10, 2, 3, 3)] * qctn_2bit.nqubits
        entries, maps = qctn_2bit.build_symmetric_expansion_graph(measure_shapes=mx_shapes)
        for qidx, uid in maps['mx_map'].items():
            assert entries[uid]['batch_symbol'] == 'ab'

    def test_multi_qubit_entry_count(
        self, qctn_4bit, circuit_state_qctn_4bit, measure_matrices_qctn_4bit
    ):
        """Multi-qubit QCTN should produce correct entry count."""
        qctn_4bit.circuit_states = circuit_state_qctn_4bit
        qctn_4bit.measure_matrices = measure_matrices_qctn_4bit
        nq = qctn_4bit.nqubits
        nc = qctn_4bit.ncores
        entries, _ = qctn_4bit.build_symmetric_expansion_graph()
        expected = nc * 2 + nq * 2 + nq
        assert len(entries) == expected


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

    def test_compute_2bit_matches_greedy(
        self, qctn_2bit, circuit_state_qctn_2bit, measure_matrices_qctn_2bit
    ):
        """RowPriority and Greedy should match on 2-qubit QCTN."""
        backend = qctn_2bit.backend
        cs = circuit_state_qctn_2bit
        mx = measure_matrices_qctn_2bit

        shapes_info = {
            'circuit_states_shapes': [c.shape for c in cs],
            'measure_shapes': [m.shape for m in mx],
        }

        greedy_fn = GreedyStrategy().get_compute_function(qctn_2bit, shapes_info, backend)
        row_fn = RowPriorityStrategy().get_compute_function(qctn_2bit, shapes_info, backend)

        result_greedy = greedy_fn(qctn_2bit.cores_weights, cs, mx)

        qctn_2bit.circuit_states = cs
        qctn_2bit.measure_matrices = mx
        result_row = row_fn(qctn_2bit)

        assert torch.allclose(result_greedy, result_row, atol=1e-5), (
            f"Max diff: {(result_greedy - result_row).abs().max()}"
        )

    def test_compute_4bit_matches_greedy(
        self, qctn_4bit, circuit_state_qctn_4bit, measure_matrices_qctn_4bit
    ):
        """RowPriority and Greedy should match on 4-qubit QCTN."""
        backend = qctn_4bit.backend
        cs = circuit_state_qctn_4bit
        mx = measure_matrices_qctn_4bit

        shapes_info = {
            'circuit_states_shapes': [c.shape for c in cs],
            'measure_shapes': [m.shape for m in mx],
        }

        greedy_fn = GreedyStrategy().get_compute_function(qctn_4bit, shapes_info, backend)
        row_fn = RowPriorityStrategy().get_compute_function(qctn_4bit, shapes_info, backend)

        result_greedy = greedy_fn(qctn_4bit.cores_weights, cs, mx)

        qctn_4bit.circuit_states = cs
        qctn_4bit.measure_matrices = mx
        result_row = row_fn(qctn_4bit)

        assert torch.allclose(result_greedy, result_row, atol=1e-5), (
            f"Max diff: {(result_greedy - result_row).abs().max()}"
        )

    def test_compute_conditional_measure_matches_greedy(
        self, qctn_4bit, circuit_state_qctn_4bit
    ):
        """RowPriority and Greedy should match when some qubits have no measure."""
        backend = qctn_4bit.backend
        nq = qctn_4bit.nqubits
        cs = circuit_state_qctn_4bit
        mx = [torch.randn(10, 3, 3)] + [None] * (nq - 1)

        shapes_info = {
            'circuit_states_shapes': [c.shape for c in cs],
            'measure_shapes': [m.shape if m is not None else None for m in mx],
        }

        greedy_fn = GreedyStrategy().get_compute_function(qctn_4bit, shapes_info, backend)
        row_fn = RowPriorityStrategy().get_compute_function(qctn_4bit, shapes_info, backend)

        result_greedy = greedy_fn(qctn_4bit.cores_weights, cs, mx)

        qctn_4bit.circuit_states = cs
        qctn_4bit.measure_matrices = mx
        result_row = row_fn(qctn_4bit)

        assert torch.allclose(result_greedy, result_row, atol=1e-5), (
            f"Max diff: {(result_greedy - result_row).abs().max()}"
        )

    def test_strategy_registered(self):
        """RowPriorityStrategy should be registered in the compiler."""
        from tneq_qc.contractor.compiler import StrategyCompiler
        strategies = StrategyCompiler.get_registered_strategies()
        assert "row_priority" in strategies

    def test_qctn_has_circuit_and_measure_attrs(self, qctn_2bit):
        """QCTN should have circuit_states and measure_matrices attributes."""
        assert qctn_2bit.circuit_states is None or isinstance(qctn_2bit.circuit_states, list)
        assert qctn_2bit.measure_matrices is None or isinstance(qctn_2bit.measure_matrices, list)
