"""
Unit tests for TNTensor Phase-2.9 additions.

Covers:
- Indexing / slicing (__getitem__, __setitem__)
- Missing arithmetic (__sub__, __rsub__, __radd__, __pow__, __abs__)
- Enhanced __add__ (raw tensor, scalar, sum())
- Gradient proxy (requires_grad, grad, is_leaf, detach, backward)
- Utility methods (item, numpy, contiguous, abs, max, min, unsqueeze, squeeze, view, permute)
- Type conversions (__float__, __len__, __iter__, float, double)
- Scale correctness across all operations
"""

import math
import pytest
import torch
import numpy as np
from tneq_qc.core.tn_tensor import TNTensor


@pytest.fixture
def t():
    return TNTensor(torch.arange(12, dtype=torch.float32).reshape(3, 4), scale=2.0)


@pytest.fixture
def s():
    """Scalar TNTensor."""
    return TNTensor(torch.tensor(3.0), scale=5.0)


@pytest.fixture
def a():
    return TNTensor(torch.ones(3, 3), scale=3.0)


@pytest.fixture
def b():
    return TNTensor(torch.ones(3, 3) * 2, scale=5.0)


# --------------------------------------------------------------------------
# Indexing / slicing
# --------------------------------------------------------------------------

class TestIndexing:
    def test_getitem_single(self, t):
        row = t[0]
        assert row.shape == (4,)
        assert row.scale == t.scale

    def test_getitem_slice(self, t):
        sliced = t[:2]
        assert sliced.shape == (2, 4)
        assert sliced.scale == t.scale

    def test_getitem_fancy(self, t):
        elem = t[1, 2]
        assert elem.shape == ()
        assert elem.scale == t.scale
        # value check: raw tensor[1,2] = 1*4+2 = 6, logical = 6 * 2.0 = 12.0
        assert abs(elem.item() - 12.0) < 1e-6

    def test_getitem_ellipsis(self, t):
        col = t[..., 0]
        assert col.shape == (3,)

    def test_setitem_raw(self, t):
        t[0, 0] = torch.tensor(99.0)
        assert t.tensor[0, 0].item() == 99.0

    def test_setitem_tntensor(self, t):
        val = TNTensor(torch.tensor(42.0), scale=1.0)
        t[0, 0] = val
        assert t.tensor[0, 0].item() == 42.0


# --------------------------------------------------------------------------
# Missing arithmetic
# --------------------------------------------------------------------------

class TestArithmetic:
    def test_sub_tntensor(self, a, b):
        # a - b: a.tensor=1, a.scale=3; b.tensor=2, b.scale=5
        # logical: 1*3 - 2*5 = 3 - 10 = -7
        c = a - b
        logical = c.tensor * c.scale
        expected = torch.ones(3, 3) * (-7.0)
        assert torch.allclose(logical, expected, atol=1e-5)

    def test_sub_scalar(self):
        t = TNTensor(torch.tensor([4.0, 5.0]), scale=2.0)
        # logical: [8, 10] - 3 = [5, 7]
        c = t - 3.0
        logical = c.tensor * c.scale
        assert torch.allclose(logical, torch.tensor([5.0, 7.0]), atol=1e-5)

    def test_rsub_scalar(self):
        t = TNTensor(torch.tensor([4.0, 5.0]), scale=2.0)
        # 20 - [8, 10] = [12, 10]
        c = 20.0 - t
        logical = c.tensor * c.scale
        assert torch.allclose(logical, torch.tensor([12.0, 10.0]), atol=1e-5)

    def test_radd_zero(self, a):
        # 0 + a should return a (identity, for sum())
        c = 0 + a
        assert c is a

    def test_radd_sum_builtin(self):
        ts = [TNTensor(torch.ones(2), scale=float(i)) for i in range(1, 4)]
        # sum([1*1, 1*2, 1*3]) = 6 per element
        result = sum(ts)
        logical = result.tensor * result.scale
        assert torch.allclose(logical, torch.tensor([6.0, 6.0]), atol=1e-5)

    def test_pow(self):
        t = TNTensor(torch.tensor([2.0, 3.0]), scale=2.0)
        # (tensor * scale) ** 2 = [4, 6]**2 = [16, 36]
        c = t ** 2
        logical = c.tensor * c.scale
        expected = torch.tensor([2.0, 3.0]) ** 2 * (2.0 ** 2)
        assert torch.allclose(logical, expected, atol=1e-5)

    def test_abs_builtin(self):
        t = TNTensor(torch.tensor([-2.0, 3.0]), scale=-3.0)
        c = abs(t)
        # abs(tensor) = [2, 3], abs(scale) = 3
        assert c.scale == 3.0
        assert torch.allclose(c.tensor, torch.tensor([2.0, 3.0]))

    def test_add_raw_tensor(self):
        t = TNTensor(torch.tensor([1.0, 2.0]), scale=3.0)
        raw = torch.tensor([10.0, 20.0])
        c = t + raw
        # logical: [3, 6] + [10, 20] = [13, 26]
        logical = c.tensor * c.scale
        assert torch.allclose(logical, torch.tensor([13.0, 26.0]), atol=1e-5)

    def test_add_scalar(self):
        t = TNTensor(torch.tensor([1.0, 2.0]), scale=3.0)
        c = t + 10.0
        # logical: [3, 6] + 10 = [13, 16]
        logical = c.tensor * c.scale
        assert torch.allclose(logical, torch.tensor([13.0, 16.0]), atol=1e-5)


# --------------------------------------------------------------------------
# Gradient proxy
# --------------------------------------------------------------------------

class TestGradientProxy:
    def test_requires_grad_default(self):
        t = TNTensor(torch.ones(2, 3))
        assert t.requires_grad is False

    def test_requires_grad_set(self):
        t = TNTensor(torch.ones(2, 3))
        t.requires_grad_(True)
        assert t.requires_grad is True
        assert t.tensor.requires_grad is True

    def test_requires_grad_ignored_for_fixed_identity(self):
        t = TNTensor(torch.eye(2), is_fixed=True, fixed_kind="identity")
        with pytest.warns(UserWarning, match="fixed TNTensor"):
            t.requires_grad_(True)
        assert t.requires_grad is False

    def test_requires_grad_chain(self):
        t = TNTensor(torch.ones(2))
        result = t.requires_grad_(True)
        assert result is t  # returns self

    def test_is_leaf(self):
        t = TNTensor(torch.ones(2, requires_grad=True))
        assert t.is_leaf is True

    def test_grad_fn_none(self):
        t = TNTensor(torch.ones(2))
        assert t.grad_fn is None

    def test_grad_after_backward(self):
        raw = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        t = TNTensor(raw, scale=2.0)
        loss = (t.tensor * t.scale).sum()
        loss.backward()
        assert t.grad is not None
        assert t.grad.shape == (3,)

    def test_detach(self):
        raw = torch.tensor([1.0, 2.0], requires_grad=True)
        t = TNTensor(raw, scale=3.0)
        d = t.detach()
        assert d.requires_grad is False
        assert d.scale == 3.0

    def test_backward(self):
        raw = torch.tensor([2.0, 3.0], requires_grad=True)
        t = TNTensor(raw, scale=1.0)
        out = (t.tensor ** 2).sum()
        t.backward(torch.tensor([4.0, 6.0]))  # backward through raw tensor
        assert t.grad is not None


# --------------------------------------------------------------------------
# Utility methods
# --------------------------------------------------------------------------

class TestUtilityMethods:
    def test_item(self, s):
        # s: tensor=3.0, scale=5.0 → 15.0
        assert abs(s.item() - 15.0) < 1e-6

    def test_numpy(self):
        t = TNTensor(torch.tensor([1.0, 2.0, 3.0]), scale=2.0)
        arr = t.numpy()
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [2.0, 4.0, 6.0])

    def test_contiguous(self):
        raw = torch.ones(3, 4).t()  # non-contiguous
        t = TNTensor(raw, scale=2.0)
        c = t.contiguous()
        assert c.tensor.is_contiguous()
        assert c.scale == 2.0

    def test_abs_method(self):
        t = TNTensor(torch.tensor([-1.0, 2.0, -3.0]), scale=-2.0)
        c = t.abs()
        assert c.scale == 2.0
        assert torch.allclose(c.tensor, torch.tensor([1.0, 2.0, 3.0]))

    def test_max_global(self):
        t = TNTensor(torch.tensor([1.0, 5.0, 3.0]), scale=2.0)
        m = t.max()
        assert isinstance(m, TNTensor)
        assert abs(m.item() - 10.0) < 1e-6

    def test_max_dim(self):
        t = TNTensor(torch.tensor([[1.0, 3.0], [4.0, 2.0]]), scale=2.0)
        vals, indices = t.max(dim=1)
        assert vals.scale == 2.0
        assert torch.equal(indices, torch.tensor([1, 0]))

    def test_min_global(self):
        t = TNTensor(torch.tensor([5.0, 1.0, 3.0]), scale=3.0)
        m = t.min()
        assert abs(m.item() - 3.0) < 1e-6

    def test_unsqueeze(self):
        t = TNTensor(torch.ones(3), scale=2.0)
        u = t.unsqueeze(0)
        assert u.shape == (1, 3)
        assert u.scale == 2.0

    def test_squeeze(self):
        t = TNTensor(torch.ones(1, 3, 1), scale=2.0)
        s = t.squeeze()
        assert s.shape == (3,)
        s2 = t.squeeze(0)
        assert s2.shape == (3, 1)

    def test_view(self):
        t = TNTensor(torch.ones(3, 4), scale=2.0)
        v = t.view(12)
        assert v.shape == (12,)
        assert v.scale == 2.0

    def test_permute(self):
        t = TNTensor(torch.ones(2, 3, 4), scale=2.0)
        p = t.permute(2, 0, 1)
        assert p.shape == (4, 2, 3)
        assert p.scale == 2.0

    def test_expand(self):
        t = TNTensor(torch.ones(1, 3), scale=2.0)
        e = t.expand(4, 3)
        assert e.shape == (4, 3)
        assert e.scale == 2.0

    def test_cpu(self):
        t = TNTensor(torch.ones(3), scale=2.0)
        c = t.cpu()
        assert str(c.device) == "cpu"


# --------------------------------------------------------------------------
# Type conversions
# --------------------------------------------------------------------------

class TestTypeConversions:
    def test_float_builtin(self, s):
        assert abs(float(s) - 15.0) < 1e-6

    def test_len(self, t):
        assert len(t) == 3  # first dim

    def test_iter(self):
        t = TNTensor(torch.arange(6, dtype=torch.float32).reshape(3, 2), scale=2.0)
        rows = list(t)
        assert len(rows) == 3
        assert all(isinstance(r, TNTensor) for r in rows)
        assert rows[0].shape == (2,)
        assert rows[0].scale == 2.0

    def test_float_method(self):
        t = TNTensor(torch.ones(2, dtype=torch.float64), scale=1.0)
        f = t.float()
        assert f.dtype == torch.float32

    def test_double_method(self):
        t = TNTensor(torch.ones(2, dtype=torch.float32), scale=1.0)
        d = t.double()
        assert d.dtype == torch.float64


# --------------------------------------------------------------------------
# Scale correctness across operations
# --------------------------------------------------------------------------

class TestScaleCorrectness:
    """Verify that result.tensor * result.scale == expected for all new ops."""

    def test_getitem_preserves_value(self):
        t = TNTensor(torch.tensor([10.0, 20.0, 30.0]), scale=0.5)
        s = t[1]
        assert abs(s.tensor.item() * s.scale - 10.0) < 1e-6

    def test_sub_preserves_value(self):
        a = TNTensor(torch.tensor([6.0]), scale=2.0)  # val=12
        b = TNTensor(torch.tensor([3.0]), scale=3.0)  # val=9
        c = a - b
        assert abs(c.tensor.item() * c.scale - 3.0) < 1e-5

    def test_pow_preserves_value(self):
        t = TNTensor(torch.tensor([3.0]), scale=2.0)  # val=6
        c = t ** 3  # val=216
        assert abs(c.tensor.item() * c.scale - 216.0) < 1e-3

    def test_abs_preserves_value(self):
        t = TNTensor(torch.tensor([-5.0]), scale=-2.0)  # val = (-5)*(-2) = 10
        c = abs(t)
        # abs(tensor)=[5], abs(scale)=2 → logical = 5*2 = 10
        assert abs(c.tensor.item() * c.scale - 10.0) < 1e-6

    def test_item_includes_scale(self):
        t = TNTensor(torch.tensor(7.0), scale=3.0)
        assert abs(t.item() - 21.0) < 1e-6

    def test_numpy_includes_scale(self):
        t = TNTensor(torch.tensor([1.0, 2.0]), scale=4.0)
        arr = t.numpy()
        np.testing.assert_allclose(arr, [4.0, 8.0])
