"""Numerical correctness: gradient back-propagation."""
import numpy as np
import pytest

from tneq_qc import EngineCommon, create_optimizer
from tneq_qc.losses import LossRegistry
from tneq_qc.losses.target import TargetResolver
from ._helpers import STRATEGIES, to_np, to_scalar, set_core, make_tneq


def _trainable_syms(combined):
    return [s for s in combined.cores
            if combined.cores_weights[s].requires_grad
            and combined.cores_weights[s].is_leaf]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_gradient_matches_finite_difference(backend64, strategy):
    """Analytic grad (autodiff) == central finite-difference of the same loss."""
    rng = np.random.RandomState(2)
    S = [rng.randn(2, 2) * 0.5 for _ in range(2)]
    T = [rng.randn(2, 2) * 0.5 for _ in range(2)]
    eng = EngineCommon(backend=backend64, strategy=strategy)
    loss_obj = LossRegistry.resolve("mse")

    # Analytic gradient.
    combined, _, _ = make_tneq(backend64, n=2, student_vals=S, teacher_vals=T)
    train_syms = _trainable_syms(combined)
    _, grads = eng.contract_for_gradient(combined, target=1.0, loss="mse")
    grads = [to_np(backend64, g) for g in grads]

    # Finite-difference gradient on an identical, grad-free model.
    fd = make_tneq(backend64, n=2, student_vals=S, teacher_vals=T, train_student=False)[0]

    def loss_of():
        r = eng.contract(fd)
        tgt = TargetResolver.resolve(1.0, r.shape, backend64, engine=eng)
        return to_scalar(backend64, loss_obj(r, tgt, backend64))

    eps = 1e-6
    for k, sym in enumerate(train_syms):
        base = to_np(backend64, fd.cores_weights[sym]).copy()
        num = np.zeros_like(base)
        for idx in np.ndindex(base.shape):
            p = base.copy(); p[idx] += eps
            set_core(backend64, fd, sym, p); lp = loss_of()
            p = base.copy(); p[idx] -= eps
            set_core(backend64, fd, sym, p); lm = loss_of()
            num[idx] = (lp - lm) / (2 * eps)
        set_core(backend64, fd, sym, base)
        np.testing.assert_allclose(grads[k].real, num, atol=1e-5, rtol=1e-4)


def test_strategies_agree_on_gradient(backend64):
    """All strategies produce the same gradient for the same network."""
    rng = np.random.RandomState(3)
    S = [rng.randn(2, 2) * 0.5 for _ in range(2)]
    T = [rng.randn(2, 2) * 0.5 for _ in range(2)]
    ref = None
    for strat in STRATEGIES:
        combined, _, _ = make_tneq(backend64, n=2, student_vals=S, teacher_vals=T)
        _, grads = EngineCommon(backend=backend64, strategy=strat).contract_for_gradient(
            combined, target=1.0, loss="mse")
        g = [to_np(backend64, x).real for x in grads]
        if ref is None:
            ref = g
        else:
            for a, b in zip(ref, g):
                np.testing.assert_allclose(a, b, atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_training_reduces_loss_to_target(backend, strategy):
    """1-qubit TNEQ trained so Tr(S @ T) -> 1.0: loss drops below 1e-3."""
    import torch
    torch.manual_seed(0)
    T = [np.array([[1.0, 0.2], [0.1, 0.9]])]
    S = [np.array([[0.3, 0.0], [0.0, 0.3]])]
    combined, _, _ = make_tneq(backend, n=1, student_vals=S, teacher_vals=T)
    eng = EngineCommon(backend=backend, strategy=strategy)
    opt = create_optimizer("adam", combined.parameters(), backend=backend, lr=0.05)

    losses = []
    for _ in range(300):
        loss, grads = eng.contract_for_gradient(combined, target=1.0, loss="mse")
        opt.step(list(grads))
        losses.append(float(to_np(backend, loss)))
    assert losses[-1] < 1e-3, f"did not converge: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert losses[-1] < losses[0]
