"""Numerical correctness: contraction / computation."""
import numpy as np
import pytest

from tneq_qc import QCTN, EngineCommon
from ._helpers import (
    STRATEGIES, to_scalar, make_tneq, tneq_trace_reference, contract_value,
)


def test_single_qubit_tneq_equals_trace_ST(backend64):
    """TNEQ trace of one qubit == Tr(S @ T) (analytic ground truth)."""
    S = np.array([[1.0, 2.0], [3.0, 4.0]])
    T = np.array([[5.0, 6.0], [7.0, 8.0]])
    combined, _, _ = make_tneq(backend64, n=1, student_vals=[S], teacher_vals=[T],
                               train_student=False)
    assert to_scalar(backend64, EngineCommon(backend=backend64, strategy="row_priority")
                     .contract(combined)) == pytest.approx(float(np.trace(S @ T)), abs=1e-9)


def test_multiqubit_tneq_equals_product_of_traces(backend64):
    """Independent qubits: trace == product of per-qubit Tr(S_i @ T_i)."""
    rng = np.random.RandomState(0)
    S = [rng.randn(2, 2) for _ in range(3)]
    T = [rng.randn(2, 2) for _ in range(3)]
    combined, _, _ = make_tneq(backend64, n=3, student_vals=S, teacher_vals=T,
                               train_student=False)
    got = to_scalar(backend64, EngineCommon(backend=backend64, strategy="row_priority")
                    .contract(combined))
    assert got == pytest.approx(tneq_trace_reference(S, T), rel=1e-9)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_strategies_match_reference(backend64, strategy):
    """Every strategy reproduces the analytic TNEQ trace."""
    rng = np.random.RandomState(1)
    S = [rng.randn(2, 2) for _ in range(3)]
    T = [rng.randn(2, 2) for _ in range(3)]
    combined, _, _ = make_tneq(backend64, n=3, student_vals=S, teacher_vals=T,
                               train_student=False)
    got = contract_value(backend64, combined, strategy=strategy)
    assert got == pytest.approx(tneq_trace_reference(S, T), rel=1e-8)


def test_all_strategies_agree_on_random_network(backend):
    """Cross-strategy agreement on a randomly-initialised network."""
    import torch
    torch.manual_seed(7)
    vals = {}
    for strat in STRATEGIES:
        combined, _, _ = make_tneq(backend, n=4, orthogonal=True, train_student=False,
                                   student_vals=None, teacher_vals=None)
        # rebuild deterministically so every strategy sees identical weights
        torch.manual_seed(7)
        combined, _, _ = make_tneq(backend, n=4, orthogonal=True, train_student=False)
        vals[strat] = contract_value(backend, combined, strategy=strat)
    ref = vals["row_priority"]
    for strat, v in vals.items():
        assert v == pytest.approx(ref, rel=1e-4, abs=1e-5), f"{strat} disagrees"


def test_contraction_is_deterministic(backend64):
    """Same network + same strategy twice -> identical scalar."""
    S = [np.eye(2) * 2.0]
    T = [np.eye(2) * 3.0]
    combined, _, _ = make_tneq(backend64, n=1, student_vals=S, teacher_vals=T,
                               train_student=False)
    eng = EngineCommon(backend=backend64, strategy="cotengra")
    a = to_scalar(backend64, eng.contract(combined))
    b = to_scalar(backend64, eng.contract(combined))
    assert a == pytest.approx(b, rel=1e-12)
    # Tr(2I @ 3I) = Tr(6I) = 12
    assert a == pytest.approx(12.0, abs=1e-9)
