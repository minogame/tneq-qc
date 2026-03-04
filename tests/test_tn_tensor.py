"""
Unit tests for TNTensor (Phase-1 refactor).

Covers:
- Metadata properties: shape, ndim, dtype, device
- Reference markers: is_ref, is_transposed, source
- Scale helpers: auto_scale, scale_to, scale_with
- Layout ops: reshape, transpose, conj, conj_transpose, clone, to
- Arithmetic with scale propagation: @, *, /, +, -, sum, mean, einsum
- Backend wrappers: tn_einsum, tn_reshape, backend.einsum (auto-unwrap)
"""

import math
import pytest
import torch
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.backends.backend_factory import BackendFactory


@pytest.fixture
def backend():
    return BackendFactory.create_backend('pytorch', device='cpu')


@pytest.fixture
def t():
    return TNTensor(torch.ones(3, 4), scale=2.0)


@pytest.fixture
def a():
    return TNTensor(torch.ones(3, 3), scale=3.0)


@pytest.fixture
def b():
    return TNTensor(torch.ones(3, 3) * 2, scale=5.0)


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------

class TestMetadata:
    def test_shape(self, t):
        assert t.shape == (3, 4)

    def test_ndim(self, t):
        assert t.ndim == 2

    def test_dtype(self, t):
        assert t.dtype == torch.float32

    def test_device(self, t):
        assert str(t.device) == "cpu"

    def test_scale(self, t):
        assert t.scale == 2.0
        assert math.isclose(t.log_scale, math.log(2.0))

    def test_is_ref_default(self, t):
        assert not t.is_ref
        assert not t.is_transposed
        assert t.source is None

    def test_repr(self, t):
        r = repr(t)
        assert "TNTensor" in r
        assert "3" in r


# --------------------------------------------------------------------------
# Scale helpers
# --------------------------------------------------------------------------

class TestScaleHelpers:
    def test_scale_to(self, t):
        original_value = t.tensor * t.scale
        t.scale_to(4.0)
        assert t.scale == 4.0
        assert torch.allclose(t.tensor * t.scale, original_value)

    def test_scale_to_zero_raises(self, t):
        with pytest.raises(ValueError):
            t.scale_to(0.0)

    def test_scale_with(self, t):
        original_value = t.tensor * t.scale
        t.scale_with(0.5)
        assert math.isclose(t.scale, 1.0)
        assert torch.allclose(t.tensor * t.scale, original_value)

    def test_scale_with_zero_raises(self, t):
        with pytest.raises(ValueError):
            t.scale_with(0.0)

    def test_auto_scale(self):
        raw = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
        t = TNTensor(raw, scale=1.0)
        t.auto_scale()
        assert math.isclose(t.tensor.abs().max().item(), 1.0, rel_tol=1e-6)
        assert math.isclose(t.scale, 4.0, rel_tol=1e-6)


# --------------------------------------------------------------------------
# Layout operations
# --------------------------------------------------------------------------

class TestLayoutOps:
    def test_reshape(self, t):
        r = t.reshape((12,))
        assert r.shape == (12,)
        assert r.scale == t.scale
        assert r.is_ref
        assert r.source is t

    def test_transpose_no_dims(self, t):
        tr = t.transpose()
        assert tr.shape == (4, 3)
        assert tr.scale == t.scale
        assert tr.is_transposed
        assert tr.is_ref

    def test_transpose_with_dims(self, t):
        tr = t.transpose(1, 0)
        assert tr.shape == (4, 3)
        assert tr.is_transposed

    def test_conj_real(self, t):
        c = t.conj()
        assert torch.allclose(c.tensor, t.tensor)
        assert c.scale == t.scale

    def test_conj_complex(self):
        raw = torch.tensor([[1 + 2j, 3 - 4j]], dtype=torch.complex64)
        t = TNTensor(raw, scale=1.0)
        c = t.conj()
        assert torch.allclose(c.tensor, raw.conj())

    def test_conj_transpose(self, t):
        ct = t.conj_transpose()
        assert ct.is_transposed
        assert ct.shape == (4, 3)

    def test_clone_independence(self, t):
        cl = t.clone()
        cl._tensor[0, 0] = 999.0
        assert t.tensor[0, 0].item() != 999.0
        assert cl.scale == t.scale
        assert not cl.is_ref

    def test_to_dtype(self, t):
        t2 = t.to(dtype=torch.float64)
        assert t2.dtype == torch.float64
        assert t2.scale == t.scale


# --------------------------------------------------------------------------
# Arithmetic / scale propagation
# --------------------------------------------------------------------------

class TestArithmetic:
    def test_matmul_scale(self, a, b):
        mm = a @ b
        assert mm.scale == 15.0
        # ones(3,3) @ 2*ones(3,3) = 6*ones(3,3)
        assert torch.allclose(mm.tensor, torch.full((3, 3), 6.0))

    def test_matmul_type_error(self, a):
        with pytest.raises(TypeError):
            _ = a @ torch.ones(3, 3)

    def test_scalar_mul(self, a):
        sm = a * 4
        assert sm.scale == 12.0
        assert torch.allclose(sm.tensor, torch.ones(3, 3))

    def test_rmul(self, a):
        sm = 4 * a
        assert sm.scale == 12.0

    def test_elementwise_mul(self, a, b):
        em = a * b
        assert em.scale == 15.0
        assert torch.allclose(em.tensor, torch.ones(3, 3) * 2)

    def test_div_scalar(self, a):
        dv = a / 2.0
        assert math.isclose(dv.scale, 1.5)

    def test_div_zero_raises(self, a):
        with pytest.raises(ZeroDivisionError):
            _ = a / 0.0

    def test_add_same_scale(self, a):
        result = a + a
        assert result.scale == 3.0
        assert torch.allclose(result.tensor, torch.ones(3, 3) * 2)

    def test_add_diff_scale(self, a):
        c = TNTensor(torch.ones(3, 3), scale=1.5)
        result = a + c  # normalise c to a.scale: factor = 1.5/3.0 = 0.5
        assert result.scale == 3.0
        assert torch.allclose(result.tensor, torch.full((3, 3), 1.5))

    def test_neg(self, a):
        neg = -a
        assert neg.scale == -3.0
        assert torch.allclose(neg.tensor, torch.ones(3, 3))

    def test_sum(self, a):
        s = a.sum()
        assert s.scale == 3.0
        assert torch.allclose(s.tensor, torch.tensor(9.0))

    def test_sum_dim(self, a):
        s = a.sum(dim=1)
        assert s.shape == (3,)
        assert s.scale == 3.0

    def test_mean(self, a):
        m = a.mean()
        assert m.scale == 3.0
        assert torch.allclose(m.tensor, torch.tensor(1.0))

    def test_einsum_method(self, a, b):
        e = a.einsum('ij,jk->ik', b)
        assert e.scale == 15.0
        assert torch.allclose(e.tensor, torch.full((3, 3), 6.0))

    def test_value_preserved_through_scale_ops(self, a):
        """The actual value (tensor * scale) must be preserved under scale ops."""
        original = a.tensor * a.scale
        a.scale_to(10.0)
        assert torch.allclose(a.tensor * a.scale, original)


# --------------------------------------------------------------------------
# Backend wrappers
# --------------------------------------------------------------------------

class TestBackendWrappers:
    def test_tn_einsum_returns_tntensor(self, backend, a, b):
        result = backend.tn_einsum('ij,jk->ik', a, b)
        assert isinstance(result, TNTensor)
        assert result.scale == 15.0
        assert torch.allclose(result.tensor, torch.full((3, 3), 6.0))

    def test_tn_einsum_raw_tensors(self, backend):
        x = torch.ones(3, 3)
        y = torch.ones(3, 3) * 2
        result = backend.tn_einsum('ij,jk->ik', x, y)
        assert not isinstance(result, TNTensor)

    def test_tn_reshape_tntensor(self, backend, a):
        result = backend.tn_reshape(a, (9,))
        assert isinstance(result, TNTensor)
        assert result.shape == (9,)
        assert result.scale == 3.0

    def test_tn_reshape_raw_tensor(self, backend):
        x = torch.ones(3, 3)
        result = backend.tn_reshape(x, (9,))
        assert result.shape == (9,)
        assert not isinstance(result, TNTensor)

    def test_backend_einsum_unwraps(self, backend, a, b):
        """backend.einsum should auto-unwrap TNTensor and return raw tensor."""
        result = backend.einsum('ij,jk->ik', a, b)
        assert not isinstance(result, TNTensor)
        assert torch.allclose(result, torch.full((3, 3), 6.0))

    def test_backend_reshape_unwraps(self, backend, a):
        """backend.reshape should auto-unwrap TNTensor and return raw tensor."""
        result = backend.reshape(a, (9,))
        assert not isinstance(result, TNTensor)
        assert result.shape == (9,)
