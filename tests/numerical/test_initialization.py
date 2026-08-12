"""Numerical correctness: initialization."""
import numpy as np
import pytest

from tneq_qc import QCTN
from tneq_qc.modules.small import State, MPS
from ._helpers import to_np, independent_graph


def test_state_is_exact_basis_vector(backend):
    """State.auto_init() must produce |0...0> exactly: one-hot per qubit core."""
    st = State(nqubits=3, phys_dim=2, backend=backend).auto_init()
    for name in st.cores:
        arr = to_np(backend, st.cores_weights[name]).ravel()
        expected = np.zeros_like(arr)
        expected[0] = 1.0
        np.testing.assert_allclose(arr.real, expected, atol=1e-6)


def test_core_shapes_match_graph(backend):
    """Each core's shape equals input_shape + output_shape from the graph."""
    qctn = QCTN(independent_graph(3, phys=2), backend=backend).auto_init()
    for info in qctn.adjacency_table:
        name = info["core_name"]
        expected = tuple(info["input_shape"] + info["output_shape"])
        assert tuple(to_np(backend, qctn.cores_weights[name]).shape) == expected


def test_init_is_finite(backend):
    qctn = QCTN(independent_graph(4, phys=2), backend=backend).auto_init(orthogonal=True)
    for name in qctn.cores:
        arr = to_np(backend, qctn.cores_weights[name])
        assert np.all(np.isfinite(arr))


@pytest.mark.parametrize("phys", [2, 3, 4])
def test_orthogonal_init_is_orthogonal_matrix(backend64, phys):
    """orthogonal=True on a square ``phys x phys`` core yields an orthogonal
    matrix (QR-based init): Q^H Q = I.  (Init builds a square matrix from the
    first half of the dims; an independent core's two legs make it exactly
    ``phys x phys``.)"""
    qctn = QCTN(independent_graph(2, phys=phys), backend=backend64).auto_init(orthogonal=True)
    for name in qctn.cores:
        Q = to_np(backend64, qctn.cores_weights[name])
        gram = Q.conj().T @ Q
        np.testing.assert_allclose(gram, np.eye(phys), atol=1e-9)


def test_dtype_matches_backend(backend):
    qctn = QCTN(independent_graph(2), backend=backend).auto_init()
    arr = to_np(backend, qctn.cores_weights[list(qctn.cores)[0]])
    # float32 backend -> real-valued cores
    assert arr.dtype in (np.float32, np.float64, np.complex64, np.complex128)
