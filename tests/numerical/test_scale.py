"""Numerical correctness: TNTensor scale and the backend auto-scale switch.

The backend-level switch is ``ComputeBackend.enable_auto_scale`` (default False),
set via ``BackendFactory.create_backend(..., enable_auto_scale=True)`` and applied
through ``backend.maybe_auto_scale``.  ``scale`` is pure numerical bookkeeping:
the *value* is always ``tensor * scale`` and must be invariant to scaling.
"""
import numpy as np
import pytest

from tneq_qc import EngineCommon, BackendFactory
from tneq_qc.core.tn_tensor import TNTensor
from ._helpers import STRATEGIES, to_scalar, to_np, set_core, make_tneq


# ----------------------------------------------------------------------
# TNTensor scale arithmetic — value (= tensor * scale) is preserved
# ----------------------------------------------------------------------
def test_value_is_tensor_times_scale(backend):
    raw = backend.convert_to_tensor(np.array([1.0, 2.0, 3.0]))
    raw = raw.tensor if isinstance(raw, TNTensor) else raw
    t = TNTensor(raw, scale=10.0)
    np.testing.assert_allclose(to_np(backend, t), np.array([10.0, 20.0, 30.0]), atol=1e-6)


def test_auto_scale_preserves_value_and_normalizes(backend):
    raw = backend.convert_to_tensor(np.array([2.0, -8.0, 4.0]))
    raw = raw.tensor if isinstance(raw, TNTensor) else raw
    t = TNTensor(raw, scale=1.0)
    before = to_np(backend, t).copy()
    t.auto_scale()
    after = to_np(backend, t)
    np.testing.assert_allclose(before, after, atol=1e-6)          # value preserved
    assert np.max(np.abs(np.asarray(backend.tensor_to_numpy(t.tensor)))) == pytest.approx(1.0, abs=1e-6)


def test_scale_to_preserves_value(backend):
    raw = backend.convert_to_tensor(np.array([1.0, 2.0, 4.0]))
    raw = raw.tensor if isinstance(raw, TNTensor) else raw
    t = TNTensor(raw, scale=3.0)
    before = to_np(backend, t).copy()
    t.scale_to(1.0)
    np.testing.assert_allclose(before, to_np(backend, t), atol=1e-6)


def test_mul_propagates_scale(backend):
    raw = backend.convert_to_tensor(np.array([1.0, 2.0]))
    raw = raw.tensor if isinstance(raw, TNTensor) else raw
    a = TNTensor(raw, scale=2.0)
    b = TNTensor(raw, scale=5.0)
    prod = a * b
    np.testing.assert_allclose(to_np(backend, prod),
                               to_np(backend, a) * to_np(backend, b), atol=1e-6)


# ----------------------------------------------------------------------
# Contraction is invariant to where scale lives
# ----------------------------------------------------------------------
def test_contraction_invariant_to_core_scale(backend64):
    """A student core with scale s gives the same result as folding s into values."""
    S = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    T = [np.array([[2.0, 0.0], [1.0, 3.0]])]
    eng = EngineCommon(backend=backend64, strategy="cotengra")

    plain, _, _ = make_tneq(backend64, n=1, student_vals=S, teacher_vals=T, train_student=False)
    v_plain = to_scalar(backend64, eng.contract(plain))

    # Same network, but the student core carries value S/5 with scale 5 (== S).
    scaled, _, _ = make_tneq(backend64, n=1, student_vals=S, teacher_vals=T, train_student=False)
    sname = [s for s in scaled.cores if scaled.core_names.get(s, "").startswith("u.")][0]
    rawt = backend64.convert_to_tensor(S[0] / 5.0)
    rawt = rawt.tensor if isinstance(rawt, TNTensor) else rawt
    scaled.cores_weights[sname] = TNTensor(rawt, scale=5.0)
    v_scaled = to_scalar(backend64, eng.contract(scaled))

    assert v_scaled == pytest.approx(v_plain, rel=1e-9)


# ----------------------------------------------------------------------
# Backend global switch: enable_auto_scale on/off -> identical value
# ----------------------------------------------------------------------
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_enable_auto_scale_does_not_change_value(strategy):
    rng = np.random.RandomState(5)
    S = [rng.randn(2, 2) for _ in range(3)]
    T = [rng.randn(2, 2) for _ in range(3)]

    b_off = BackendFactory.create_backend("pytorch", device="cpu", dtype="float64",
                                          enable_auto_scale=False)
    b_on = BackendFactory.create_backend("pytorch", device="cpu", dtype="float64",
                                         enable_auto_scale=True)
    assert b_off.enable_auto_scale is False
    assert b_on.enable_auto_scale is True

    c_off, _, _ = make_tneq(b_off, n=3, student_vals=S, teacher_vals=T, train_student=False)
    c_on, _, _ = make_tneq(b_on, n=3, student_vals=S, teacher_vals=T, train_student=False)
    v_off = to_scalar(b_off, EngineCommon(backend=b_off, strategy=strategy).contract(c_off))
    v_on = to_scalar(b_on, EngineCommon(backend=b_on, strategy=strategy).contract(c_on))
    assert v_on == pytest.approx(v_off, rel=1e-9)
