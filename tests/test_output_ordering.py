"""
Tests for RowPriorityStrategy output dimension ordering.

Verifies that contraction results have dimensions ordered as:
  (in_q0, in_q1, ..., out_q0, out_q1, ...)

Uses asymmetric bond dimensions to make ordering unambiguous.
"""

import torch
import pytest
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.core.engine_common import EngineCommon


@pytest.fixture(scope="module")
def backend():
    return BackendFactory.create_backend("pytorch", device="cpu", dtype="float32")


@pytest.fixture(scope="module")
def engine(backend):
    return EngineCommon(backend=backend, strategy_mode="full")


def _val(result):
    if isinstance(result, TNTensor):
        result.scale_to(1.0)
        return result.tensor
    return result


# ======================================================================
# Graph: -2-A-3- / -4-A-5- / -6-B-7- / -8-B-9-
#
# Core A: shape (2, 4, 3, 5)  — (in_q0, in_q1, out_q0, out_q1)
# Core B: shape (6, 8, 7, 9)  — (in_q2, in_q3, out_q2, out_q3)
# A and B have NO internal connections → result = A ⊗ B
#
# Expected output shape (with correct ordering):
#   (in_q0, in_q1, in_q2, in_q3, out_q0, out_q1, out_q2, out_q3)
#   = (2, 4, 6, 8, 3, 5, 7, 9)
#
# result[i0,i1,i2,i3, j0,j1,j2,j3] = A[i0,i1,j0,j1] * B[i2,i3,j2,j3]
# ======================================================================

GRAPH_ASYM = "-2-A-3-\n-4-A-5-\n-6-B-7-\n-8-B-9-"


class TestOutputDimensionOrdering:
    """Verify output tensor dimensions follow (in_q0,...,out_q0,...) ordering."""

    def test_shape(self, backend, engine):
        """Result shape must be (2,4,6,8, 3,5,7,9)."""
        torch.manual_seed(0)
        q = QCTN(GRAPH_ASYM, backend=backend)
        # Set cores to known random values
        A = torch.randn(2, 4, 3, 5)
        B = torch.randn(6, 8, 7, 9)
        q.cores_weights['A'] = backend.convert_to_tensor(A)
        q.cores_weights['B'] = backend.convert_to_tensor(B)

        result = _val(engine.contract_with_compiled_strategy(q))
        assert tuple(result.shape) == (2, 4, 6, 8, 3, 5, 7, 9), (
            f"Expected shape (2,4,6,8, 3,5,7,9), got {tuple(result.shape)}"
        )

    def test_values_tensor_product(self, backend, engine):
        """result[i0,i1,i2,i3, j0,j1,j2,j3] = A[i0,i1,j0,j1] * B[i2,i3,j2,j3]."""
        torch.manual_seed(42)
        q = QCTN(GRAPH_ASYM, backend=backend)
        A = torch.randn(2, 4, 3, 5)
        B = torch.randn(6, 8, 7, 9)
        q.cores_weights['A'] = backend.convert_to_tensor(A)
        q.cores_weights['B'] = backend.convert_to_tensor(B)

        result = _val(engine.contract_with_compiled_strategy(q))

        # Manual reference: outer product with correct index ordering
        # A has dims (in_q0=2, in_q1=4, out_q0=3, out_q1=5)
        # B has dims (in_q2=6, in_q3=8, out_q2=7, out_q3=9)
        # result[i0,i1,i2,i3, j0,j1,j2,j3] = A[i0,i1,j0,j1] * B[i2,i3,j2,j3]
        expected = torch.einsum('abcd,efgh->abefcdgh', A, B)
        assert expected.shape == (2, 4, 6, 8, 3, 5, 7, 9)
        assert result.shape == (2, 4, 6, 8, 3, 5, 7, 9)

        assert torch.allclose(result, expected, atol=1e-5), (
            f"Max diff = {(result - expected).abs().max().item()}"
        )

    def test_single_core_shape(self, backend, engine):
        """Single core, 2 qubits, asymmetric dims → shape (2,4, 3,5)."""
        q = QCTN("-2-A-3-\n-4-A-5-", backend=backend)
        A = torch.randn(2, 4, 3, 5)
        q.cores_weights['A'] = backend.convert_to_tensor(A)

        result = _val(engine.contract_with_compiled_strategy(q))
        assert tuple(result.shape) == (2, 4, 3, 5), (
            f"Expected (2,4,3,5), got {tuple(result.shape)}"
        )
        # For a single core with no internal edges, result = core tensor itself
        assert torch.allclose(result, A, atol=1e-5)

    def test_connected_cores_shape(self, backend, engine):
        """Two connected cores → boundary only, correct ordering."""
        # A(in=2,3) -4-> B(out=5,6)  on 2 qubits
        q = QCTN("-2-A-4-B-5-\n-3-A-4-B-6-", backend=backend)
        A = torch.randn(2, 3, 4, 4)  # (in_q0, in_q1, out_q0, out_q1)
        B = torch.randn(4, 4, 5, 6)  # (in_q0, in_q1, out_q0, out_q1)
        q.cores_weights['A'] = backend.convert_to_tensor(A)
        q.cores_weights['B'] = backend.convert_to_tensor(B)

        result = _val(engine.contract_with_compiled_strategy(q))
        # Boundary: A.in (q0=2, q1=3), B.out (q0=5, q1=6)
        assert tuple(result.shape) == (2, 3, 5, 6), (
            f"Expected (2,3,5,6), got {tuple(result.shape)}"
        )

        # Manual: result[i0,i1, j0,j1] = sum_{m0,m1} A[i0,i1,m0,m1] * B[m0,m1,j0,j1]
        expected = torch.einsum('abcd,cdef->abef', A, B)
        assert torch.allclose(result, expected, atol=1e-5), (
            f"Max diff = {(result - expected).abs().max().item()}"
        )

    def test_partial_trace_shape(self, backend, engine):
        """Trace qubit 0 of the asymmetric graph → removes in_q0 and out_q0 dims."""
        q = QCTN(GRAPH_ASYM, backend=backend)
        # For trace, in_rank must equal out_rank on the traced qubit.
        # qubit 0: in=2, out=3 → rank mismatch → cannot trace qubit 0
        # Use a symmetric graph for partial trace test instead.
        q2 = QCTN("-2-A-3-\n-4-A-5-\n-6-B-7-\n-6-B-7-", backend=backend)
        A = torch.randn(2, 4, 3, 5)
        B = torch.randn(6, 6, 7, 7)  # q2,q3 both have in=6,out=7... no, in=6,out=7 mismatch
        # Need in_rank == out_rank for traced qubits. Trace qubit 2 and 3:
        # qubit 2: in=6, out=7 → mismatch. Let's use equal dims.
        q3 = QCTN("-2-A-3-\n-4-A-5-\n-6-B-6-\n-8-B-8-", backend=backend)
        A = torch.randn(2, 4, 3, 5)
        B = torch.randn(6, 8, 6, 8)
        q3.cores_weights['A'] = backend.convert_to_tensor(A)
        q3.cores_weights['B'] = backend.convert_to_tensor(B)
        q3.set_trace([2, 3])  # trace qubits 2 and 3

        result = _val(engine.contract_with_compiled_strategy(q3))
        # Traced: q2 and q3 → their in/out dims disappear
        # Remaining: in_q0(2), in_q1(4), out_q0(3), out_q1(5)
        assert tuple(result.shape) == (2, 4, 3, 5), (
            f"Expected (2,4,3,5), got {tuple(result.shape)}"
        )

        # Manual: A is unchanged (no trace on q0,q1).
        # B traced = sum over diagonal: Tr(B) = sum_{i,j} B[i,j,i,j]
        B_trace = 0.0
        for i in range(6):
            for j in range(8):
                B_trace += B[i, j, i, j]
        expected = A * B_trace
        assert torch.allclose(result, expected, atol=1e-4), (
            f"Max diff = {(result - expected).abs().max().item()}"
        )

    def test_full_trace_scalar(self, backend, engine):
        """Full trace of symmetric-dim graph → scalar."""
        # All qubit dims must match in/out for trace
        q = QCTN("-2-A-2-\n-3-A-3-\n-4-B-4-\n-5-B-5-", backend=backend)
        A = torch.randn(2, 3, 2, 3)
        B = torch.randn(4, 5, 4, 5)
        q.cores_weights['A'] = backend.convert_to_tensor(A)
        q.cores_weights['B'] = backend.convert_to_tensor(B)
        q.set_trace('all')

        result = _val(engine.contract_with_compiled_strategy(q))
        # Scalar
        assert result.numel() == 1, f"Expected scalar, got shape {tuple(result.shape)}"

        # Tr(A) * Tr(B), where each trace sums diagonal
        trA = 0.0
        for i in range(2):
            for j in range(3):
                trA += A[i, j, i, j].item()
        trB = 0.0
        for i in range(4):
            for j in range(5):
                trB += B[i, j, i, j].item()
        expected_scalar = trA * trB
        assert abs(result.item() - expected_scalar) < 1e-4, (
            f"Expected {expected_scalar}, got {result.item()}"
        )
