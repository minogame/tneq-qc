"""
Tests for QCTN.concat() with list-based API.
"""

import pytest
import torch
from tneq_qc.core.qctn import QCTN
from tneq_qc.backends.backend_pytorch import BackendPyTorch


@pytest.fixture
def backend():
    """Create a PyTorch backend for testing."""
    return BackendPyTorch()


class TestConcatListAPI:
    """Test QCTN.concat() with list-based API."""

    def test_concat_two_qctns(self, backend):
        """Test concatenating two QCTNs."""
        graph1 = "-2-A-2-"
        graph2 = "-2-B-2-"

        q1 = QCTN(graph1, backend=backend)
        q2 = QCTN(graph2, backend=backend)

        # Use new list-based API
        result = QCTN.concat([q1, q2])

        # Check that result has cores from both
        assert result.ncores == q1.ncores + q2.ncores
        assert result.nqubits == max(q1.nqubits, q2.nqubits)

    def test_concat_three_qctns(self, backend):
        """Test concatenating three QCTNs."""
        graph1 = "-2-A-2-"
        graph2 = "-2-B-2-"
        graph3 = "-2-C-2-"

        q1 = QCTN(graph1, backend=backend)
        q2 = QCTN(graph2, backend=backend)
        q3 = QCTN(graph3, backend=backend)

        # Concatenate three QCTNs
        result = QCTN.concat([q1, q2, q3])

        # Check that result has cores from all three
        assert result.ncores == q1.ncores + q2.ncores + q3.ncores
        assert result.nqubits == max(q1.nqubits, q2.nqubits, q3.nqubits)

    def test_concat_single_qctn(self, backend):
        """Test concatenating a single QCTN returns itself."""
        graph = "-2-A-2-"
        q1 = QCTN(graph, backend=backend)

        result = QCTN.concat([q1])

        # Should return the same QCTN
        assert result is q1

    def test_concat_empty_list_raises(self, backend):
        """Test that concatenating empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot concat empty list"):
            QCTN.concat([])

    def test_concat_with_different_nqubits(self, backend):
        """Test concatenating QCTNs with different numbers of qubits."""
        graph1 = "-2-A-2-"
        graph2 = "-2-B-2-\n-2-C-2-"

        q1 = QCTN(graph1, backend=backend)
        q2 = QCTN(graph2, backend=backend)

        result = QCTN.concat([q1, q2])

        # Result should have max qubits
        assert result.nqubits == 2
        assert result.ncores == 3

    def test_concat_preserves_weights(self, backend):
        """Test that concatenation preserves core weights."""
        graph1 = "-2-A-2-"
        graph2 = "-2-B-2-"

        q1 = QCTN(graph1, backend=backend)
        q2 = QCTN(graph2, backend=backend)

        # Get original weights
        a_weight = q1.cores_weights['A']
        b_weight = q2.cores_weights['B']

        result = QCTN.concat([q1, q2])

        # Check that weights are preserved (cores are renamed)
        # First core should be from q1, second from q2
        result_cores = list(result.cores_weights.values())
        assert len(result_cores) == 2

        # Extract raw tensors for comparison
        def get_raw(t):
            from tneq_qc.core.tn_tensor import TNTensor
            return t.tensor if isinstance(t, TNTensor) else t

        assert torch.allclose(get_raw(result_cores[0]), get_raw(a_weight))
        assert torch.allclose(get_raw(result_cores[1]), get_raw(b_weight))

    def test_concat_with_method(self, backend):
        """Test concat_with() instance method."""
        graph1 = "-2-A-2-"
        graph2 = "-2-B-2-"

        q1 = QCTN(graph1, backend=backend)
        q2 = QCTN(graph2, backend=backend)

        result = q1.concat_with(q2)

        assert result.ncores == q1.ncores + q2.ncores

    def test_concat_multiple_with_clone_and_hermit(self, backend):
        """Test concatenating cloned and hermit QCTNs."""
        graph = "-2-A-2-"
        q1 = QCTN(graph, backend=backend)

        # Create variations
        q2 = q1.clone()
        q3 = q1.hermit()

        # Concatenate all three
        result = QCTN.concat([q1, q2, q3])

        # Should have 3 cores (one from each)
        assert result.ncores == 3
        assert result.nqubits == 1


class TestBackwardCompatibility:
    """Test backward compatibility with old API."""

    def test_old_api_still_works(self, backend):
        """Test that old two-argument API still works."""
        graph1 = "-2-A-2-"
        graph2 = "-2-B-2-"

        q1 = QCTN(graph1, backend=backend)
        q2 = QCTN(graph2, backend=backend)

        # Old API should still work (now treated as varargs)
        result = QCTN.concat(q1, q2)

        assert result.ncores == q1.ncores + q2.ncores

    def test_varargs_api(self, backend):
        """Test that varargs API works."""
        graph1 = "-2-A-2-"
        graph2 = "-2-B-2-"
        graph3 = "-2-C-2-"

        q1 = QCTN(graph1, backend=backend)
        q2 = QCTN(graph2, backend=backend)
        q3 = QCTN(graph3, backend=backend)

        # Varargs API
        result = QCTN.concat(q1, q2, q3)

        assert result.ncores == q1.ncores + q2.ncores + q3.ncores

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
