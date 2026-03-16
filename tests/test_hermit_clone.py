"""
Tests for hermit and clone operations on TNTensor and QCTN.

Based on design document: docs/PHASE_2_6_2_QCTN_HERMIT.md
"""

import pytest
import torch
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.core.qctn import QCTN
from tneq_qc.backends.backend_pytorch import BackendPyTorch


@pytest.fixture
def backend():
    """Create a PyTorch backend for testing."""
    return BackendPyTorch()


class TestTNTensorClone:
    """Test TNTensor.clone() method."""

    def test_clone_creates_independent_copy(self, backend):
        """Test that clone creates an independent copy."""
        # Create a TNTensor
        data = torch.randn(2, 3, 4, requires_grad=True)
        t1 = TNTensor(data, scale=2.0, name="test")

        # Clone it
        t2 = t1.clone()

        # Check that data is equal but not the same object
        assert torch.allclose(t1.tensor, t2.tensor)
        assert t1.tensor is not t2.tensor

        # Check scale is preserved
        assert t1.scale == t2.scale

        # Check name suffix
        assert t2.name == "test_clone"

        # Check source info
        assert t2._source_info['type'] == 'clone'
        assert t2._source_info['base'] == 'test'

    def test_clone_maintains_computation_graph(self, backend):
        """Test that clone maintains computation graph for gradient flow."""
        # Create a TNTensor with requires_grad
        data = torch.randn(2, 3, requires_grad=True)
        t1 = TNTensor(data, name="original")

        # Clone it
        t2 = t1.clone(deep=True)

        # Perform operation on clone
        loss = (t2.tensor ** 2).sum()
        loss.backward()

        # Original tensor should have gradients
        assert data.grad is not None
        assert torch.allclose(data.grad, 2 * data)


class TestTNTensorHermit:
    """Test TNTensor.hermit() method."""

    def test_hermit_conjugate_transpose(self, backend):
        """Test that hermit applies conjugate and transpose."""
        # Create a complex tensor
        data = torch.randn(2, 3, 4, dtype=torch.complex64)
        t1 = TNTensor(data, name="test")

        # Apply hermit (default: transpose last two dims)
        t2 = t1.hermit()

        # Check shape (last two dims should be swapped)
        assert t2.shape == (2, 4, 3)

        # Check that it's conjugate transpose
        expected = torch.conj(data).permute(0, 2, 1)
        assert torch.allclose(t2.tensor, expected)

        # Check name suffix
        assert t2.name == "test_H"

        # Check source info
        assert t2._source_info['type'] == 'hermit'
        assert t2._source_info['base'] == 'test'

    def test_hermit_custom_axes(self, backend):
        """Test hermit with custom axes."""
        data = torch.randn(2, 3, 4, 5, dtype=torch.complex64)
        t1 = TNTensor(data, name="test")

        # Apply hermit with custom axes
        axes = [0, 2, 1, 3]  # Swap dims 1 and 2
        t2 = t1.hermit(axes=axes)

        # Check shape
        assert t2.shape == (2, 4, 3, 5)

        # Check that it's conjugate transpose with custom axes
        expected = torch.conj(data).permute(*axes)
        assert torch.allclose(t2.tensor, expected)

    def test_hermit_gradient_flow(self, backend):
        """Test that hermit maintains gradient flow."""
        # Create a complex tensor with requires_grad
        data = torch.randn(2, 3, 4, dtype=torch.complex64, requires_grad=True)
        t1 = TNTensor(data, name="original")

        # Apply hermit
        t2 = t1.hermit()

        # Perform operation
        loss = (t2.tensor.real ** 2).sum()
        loss.backward()

        # Original tensor should have gradients
        assert data.grad is not None


class TestQCTNClone:
    """Test QCTN.clone() method."""

    def test_clone_creates_independent_qctn(self, backend):
        """Test that QCTN.clone() creates independent cores."""
        # Create a simple QCTN
        graph = "-2-A-2-\n-2-B-2-"
        q1 = QCTN(graph, backend=backend)

        # Clone it
        q2 = q1.clone()

        # Check structure is preserved
        assert q2.nqubits == q1.nqubits
        assert q2.ncores == q1.ncores
        assert q2.cores == q1.cores

        # Check that cores are different objects
        for core_name in q1.cores:
            assert q1.cores_weights[core_name] is not q2.cores_weights[core_name]

            # But data should be equal
            t1 = q1.cores_weights[core_name]
            t2 = q2.cores_weights[core_name]

            # Extract raw tensors
            raw1 = t1.tensor if isinstance(t1, TNTensor) else t1
            raw2 = t2.tensor if isinstance(t2, TNTensor) else t2

            assert torch.allclose(raw1, raw2)


class TestQCTNHermit:
    """Test QCTN.hermit() method."""

    def test_hermit_creates_conjugate_qctn(self, backend):
        """Test that QCTN.hermit() creates hermitian conjugate cores."""
        # Create a simple QCTN
        graph = "-2-A-2-\n-2-B-2-"
        q1 = QCTN(graph, backend=backend)

        # Apply hermit
        q2 = q1.hermit()

        # Check structure is preserved
        assert q2.nqubits == q1.nqubits
        assert q2.ncores == q1.ncores
        assert q2.cores == q1.cores

        # Check that cores are hermitian conjugates
        for core_name in q1.cores:
            t1_obj = q1.cores_weights[core_name]
            t2_obj = q2.cores_weights[core_name]

            # Extract raw tensors
            t1 = t1_obj.tensor if isinstance(t1_obj, TNTensor) else t1_obj
            t2 = t2_obj.tensor if isinstance(t2_obj, TNTensor) else t2_obj

            # Default hermit transposes last two dims
            expected = torch.conj(t1).permute(
                *list(range(t1.ndim - 2)) + [t1.ndim - 1, t1.ndim - 2]
            )
            assert torch.allclose(t2, expected)


class TestReferenceVsCloneVsHermit:
    """Test the three operation modes: reference, clone, hermit."""

    def test_reference_shares_tensors(self, backend):
        """Test that direct reference shares the same tensor."""
        graph = "-2-A-2-"
        q1 = QCTN(graph, backend=backend)

        # Create a dict that references the same cores
        cores_ref = {}
        for name, tensor in q1.cores_weights.items():
            cores_ref[name] = tensor

        # Check that they are the same object
        for name in q1.cores:
            assert cores_ref[name] is q1.cores_weights[name]

    def test_clone_creates_independent_tensors(self, backend):
        """Test that clone creates independent tensors."""
        graph = "-2-A-2-"
        q1 = QCTN(graph, backend=backend)
        q2 = q1.clone()

        # Check that they are different objects
        for name in q1.cores:
            assert q2.cores_weights[name] is not q1.cores_weights[name]

    def test_hermit_maintains_gradient_connection(self, backend):
        """Test that hermit maintains gradient connection."""
        graph = "-2-A-2-"
        q1 = QCTN(graph, backend=backend)

        # Enable gradients on the core tensor
        core_name = q1.cores[0]
        t1_obj = q1.cores_weights[core_name]
        t1 = t1_obj.tensor if isinstance(t1_obj, TNTensor) else t1_obj
        t1.requires_grad_(True)

        # Now apply hermit
        q2 = q1.hermit()

        # Get the hermit core tensor
        t2_obj = q2.cores_weights[core_name]
        t2 = t2_obj.tensor if isinstance(t2_obj, TNTensor) else t2_obj

        # Check that t2 is in the computation graph of t1
        assert t1.requires_grad
        assert t2.requires_grad

        # Perform operation and check gradient flow
        loss = (t2 ** 2).sum()
        loss.backward()

        # t1 should have gradients
        assert t1.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
