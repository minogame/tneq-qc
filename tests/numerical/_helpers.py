"""Shared builders/asserts for the numerical correctness suite.

Kept backend-agnostic: everything goes through the ``ComputeBackend`` so the same
helpers can drive PyTorch or JAX.
"""
from __future__ import annotations

import numpy as np

from tneq_qc import QCTN, EngineCommon
from tneq_qc.core.tn_tensor import TNTensor

# The three contraction strategies that must agree numerically.
STRATEGIES = ["row_priority", "cotengra", "einsum_default"]


def to_np(backend, x):
    """Return the *true value* (tensor * scale) as a NumPy array."""
    if isinstance(x, TNTensor):
        x = x.tensor * x.scale
    return np.asarray(backend.tensor_to_numpy(x))


def to_scalar(backend, x):
    """Return a real Python float for a scalar contraction result."""
    a = to_np(backend, x)
    a = a.real if np.iscomplexobj(a) else a
    return float(a.reshape(-1)[0]) if a.size == 1 else float(a)


def set_core(backend, qctn, name, array):
    """Overwrite a core with a concrete NumPy array (scale 1)."""
    raw = backend.convert_to_tensor(np.asarray(array))
    qctn.cores_weights[name] = raw if isinstance(raw, TNTensor) else TNTensor(raw)


def independent_graph(n, phys=2):
    """``n`` qubits, one independent ``phys x phys`` core each."""
    return "\n".join(f"-{phys}-{chr(ord('A') + i)}-{phys}-" for i in range(n))


def make_tneq(backend, n=2, phys=2, student_vals=None, teacher_vals=None,
              orthogonal=False, train_student=True):
    """Build a traced student*teacher TNEQ QCTN.

    Returns ``(combined, student, teacher)``.  Optional ``*_vals`` are lists of
    per-core NumPy arrays (in core order) to set concrete weights.
    """
    g = independent_graph(n, phys)
    student = QCTN(g, backend=backend).auto_init(orthogonal=orthogonal)
    teacher = QCTN(g, backend=backend).auto_init(orthogonal=orthogonal)
    if student_vals is not None:
        for name, v in zip(student.cores, student_vals):
            set_core(backend, student, name, v)
    if teacher_vals is not None:
        for name, v in zip(teacher.cores, teacher_vals):
            set_core(backend, teacher, name, v)
    if train_student:
        student.requires_grad_(True)
    combined = QCTN.concat([("u", student), ("t", teacher)])
    combined.set_trace("all")
    return combined, student, teacher


def contract_value(backend, qctn, strategy="row_priority"):
    return to_scalar(backend, EngineCommon(backend=backend, strategy=strategy).contract(qctn))


def tneq_trace_reference(student_vals, teacher_vals):
    """Analytic TNEQ trace: product over qubits of Tr(S_i @ T_i)."""
    out = 1.0
    for s, t in zip(student_vals, teacher_vals):
        out *= float(np.trace(np.asarray(s) @ np.asarray(t)))
    return out
