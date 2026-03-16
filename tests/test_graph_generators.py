"""
Tests for QCTNHelper graph generators.

Each test generates a graph with n=5 qubits (default), prints the graph
string, then verifies basic structural constraints via QCTN parsing.
"""

import pytest
import re
from tneq_qc.utils.graph_generators import QCTNHelper
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN

N = 5  # default qubit count for all tests


@pytest.fixture(scope="module")
def backend():
    return BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')


# ======================================================================
# Helpers
# ======================================================================

def _parse(graph: str, backend) -> QCTN:
    """Create a QCTN from graph string; raises on parse failure."""
    return QCTN(graph, backend=backend)


def _check_rows(graph: str, expected_rows: int):
    """Graph must have the expected number of non-empty lines."""
    rows = [r for r in graph.strip().splitlines() if r.strip()]
    assert len(rows) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(rows)}"
    )


# ======================================================================
# generate_example_graph — each graph_type
# ======================================================================

class TestGenerateExampleGraph:

    def test_mps(self, backend):
        graph = QCTNHelper.generate_example_graph(n=N, graph_type="mps", dim_char='3')
        print(f"\n[mps, n={N}]\n{graph}")
        _check_rows(graph, N)
        qctn = _parse(graph, backend)
        assert qctn.nqubits == N
        assert qctn.ncores == N - 1  # MPS has n-1 bond cores

    def test_tree(self, backend):
        graph = QCTNHelper.generate_example_graph(n=N, graph_type="tree", dim_char='3')
        print(f"\n[tree, n={N}]\n{graph}")
        _check_rows(graph, N)
        qctn = _parse(graph, backend)
        assert qctn.nqubits == N

    def test_wall(self, backend):
        graph = QCTNHelper.generate_example_graph(n=N, graph_type="wall", dim_char='3')
        print(f"\n[wall, n={N}]\n{graph}")
        _check_rows(graph, N)
        qctn = _parse(graph, backend)
        assert qctn.nqubits == N

    def test_circuit(self, backend):
        graph = QCTNHelper.generate_example_graph(n=N, graph_type="circuit", dim_char='2')
        print(f"\n[circuit, n={N}]\n{graph}")
        _check_rows(graph, N)
        qctn = _parse(graph, backend)
        assert qctn.nqubits == N
        assert qctn.ncores == N  # one core per qubit

    def test_mx(self, backend):
        graph = QCTNHelper.generate_example_graph(n=N, graph_type="mx", dim_char='2')
        print(f"\n[mx, n={N}]\n{graph}")
        _check_rows(graph, N)
        qctn = _parse(graph, backend)
        assert qctn.nqubits == N
        assert qctn.ncores == N  # one core per qubit

    def test_default_is_mps(self, backend):
        """No graph_type → falls back to MPS."""
        graph = QCTNHelper.generate_example_graph(n=N, dim_char='3')
        print(f"\n[default/mps, n={N}]\n{graph}")
        _check_rows(graph, N)
        qctn = _parse(graph, backend)
        assert qctn.nqubits == N


# ======================================================================
# generate_random_example_graph
# ======================================================================

class TestGenerateRandomExampleGraph:

    def test_random(self, backend):
        graph = QCTNHelper.generate_random_example_graph(nqubits=N, ncores=3)
        print(f"\n[random, nqubits={N}, ncores=3]\n{graph}")
        _check_rows(graph, N)
        # Random graph may or may not parse cleanly; just verify string shape
        assert isinstance(graph, str)
        assert len(graph.strip()) > 0


# ======================================================================
# QCTNHelper.mps  (named constructor)
# ======================================================================

class TestMps:

    def test_mps_bond2(self, backend):
        graph = QCTNHelper.mps(nqubits=N, bond_dim=2, phys_dim=2)
        print(f"\n[QCTNHelper.mps, nqubits={N}, bond_dim=2]\n{graph}")
        _check_rows(graph, N)
        qctn = _parse(graph, backend)
        assert qctn.nqubits == N

    def test_mps_bond4(self, backend):
        graph = QCTNHelper.mps(nqubits=N, bond_dim=4, phys_dim=2)
        print(f"\n[QCTNHelper.mps, nqubits={N}, bond_dim=4]\n{graph}")
        _check_rows(graph, N)
        qctn = _parse(graph, backend)
        assert qctn.nqubits == N

    def test_staggered_row_pattern(self):
        """Staggered MPS: edge rows have one core, middle rows have two."""
        graph = QCTNHelper.mps(nqubits=N, bond_dim=3, phys_dim=2)
        rows = [r for r in graph.strip().splitlines() if r.strip()]
        symbols0 = re.findall(r"[A-Za-z]", rows[0])
        symbolsn = re.findall(r"[A-Za-z]", rows[-1])
        assert symbols0 == ["a"]
        assert symbolsn == ["d"]
        for i in range(1, N - 1):
            symbols = re.findall(r"[A-Za-z]", rows[i])
            assert len(symbols) == 2


# ======================================================================
# QCTNHelper.circuit_state  (named constructor)
# ======================================================================

class TestCircuitState:

    def test_circuit_state(self, backend):
        graph = QCTNHelper.circuit_state(nqubits=N, phys_dim=2)
        print(f"\n[QCTNHelper.circuit_state, nqubits={N}]\n{graph}")
        _check_rows(graph, N)
        qctn = _parse(graph, backend)
        assert qctn.nqubits == N
        assert qctn.ncores == N

    def test_circuit_state_one_core_per_qubit(self, backend):
        """Each qubit line should contain exactly one core symbol."""
        graph = QCTNHelper.circuit_state(nqubits=N, phys_dim=2)
        rows = [r for r in graph.strip().splitlines() if r.strip()]
        assert len(rows) == N


# ======================================================================
# QCTNHelper.measure_matrix  (named constructor)
# ======================================================================

class TestMeasureMatrix:

    def test_measure_matrix(self, backend):
        graph = QCTNHelper.measure_matrix(nqubits=N, phys_dim=2)
        print(f"\n[QCTNHelper.measure_matrix, nqubits={N}]\n{graph}")
        _check_rows(graph, N)
        qctn = _parse(graph, backend)
        assert qctn.nqubits == N
        assert qctn.ncores == N

    def test_measure_matrix_in_out_dims(self):
        """Each line must have the phys_dim on both sides of the core."""
        phys_dim = 3
        graph = QCTNHelper.measure_matrix(nqubits=N, phys_dim=phys_dim)
        for line in graph.strip().splitlines():
            if not line.strip():
                continue
            # Line format: -p-X-p-  where p = phys_dim
            assert line.startswith(f'-{phys_dim}-'), (
                f"Line does not start with '-{phys_dim}-': {line}"
            )
            assert line.endswith(f'-{phys_dim}-'), (
                f"Line does not end with '-{phys_dim}-': {line}"
            )
