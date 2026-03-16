"""
Train a small TN (student) to approximate a larger TN (teacher) via trace maximization.

Structure:
    combined = student · teacher†
    loss = MSE(Tr(combined), 1.0)

- teacher : fixed, larger bond dimension, orthogonal init
- student : trainable, smaller bond dimension, orthogonal init + SGDG optimizer
- trace   : all qubits traced → scalar fidelity measure

Usage::

    python examples/train_tneq.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.core.engine_common import EngineCommon
from tneq_qc.utils.graph_generators import QCTNHelper

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_QUBITS      = 4
PHYS_DIM      = 2
N_STEPS       = 500
LR            = 0.01
LOG_EVERY     = 10

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw(t):
    return t.tensor if isinstance(t, TNTensor) else t


def get_val(result):
    if isinstance(result, TNTensor):
        result.scale_to(1.0)
        v = result.tensor
    else:
        v = result
    if hasattr(v, 'is_complex') and v.is_complex():
        v = v.real
    return v.item() if hasattr(v, 'item') else float(v)


def print_qctn_info(qctn, label="QCTN"):
    print(f"\n[{label}]  nqubits={qctn.nqubits}  ncores={qctn.ncores}")
    for c in qctn.cores:
        t = qctn.cores_weights[c]
        raw = _raw(t)
        shape = tuple(raw.shape)
        rg = raw.requires_grad
        print(f"    '{c}': {shape}  requires_grad={rg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')
    engine = EngineCommon(backend=backend, strategy_mode="full")

    # ------------------------------------------------------------------
    # 1. Teacher TN (fixed)
    #    Each qubit has its own core (separate, no bonds).
    #    Use random orthogonal matrices as the target.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Teacher TN (fixed)")
    print("=" * 60)

    graph_lines = [f"-{PHYS_DIM}-{chr(ord('A') + i)}-{PHYS_DIM}-" for i in range(N_QUBITS)]
    graph_teacher = "\n".join(graph_lines)
    teacher = QCTN(graph_teacher, backend=backend).auto_init()
    print_qctn_info(teacher, "Teacher")

    # ------------------------------------------------------------------
    # 2. Student TN (trainable, same structure)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Student TN (trainable)")
    print("=" * 60)

    graph_student = "\n".join(
        [f"-{PHYS_DIM}-{chr(ord('A') + i)}-{PHYS_DIM}-" for i in range(N_QUBITS)]
    )
    student = QCTN(graph_student, backend=backend).auto_init()

    for c in student.cores:
        _raw(student.cores_weights[c]).requires_grad_(True)
    print_qctn_info(student, "Student")

    # ------------------------------------------------------------------
    # 3. Build combined: student · teacher†, set trace
    # ------------------------------------------------------------------
    # teacher_h = teacher.hermit()
    combined = QCTN.concat([
        ('u', student),
        # ('t', teacher_h),
        ('t', teacher),
    ])
    combined.set_trace('all')
    print_qctn_info(combined, "Combined (student · teacher†, all traced)")
    print(f"  core_names: {combined.core_names}")
    print(f"  trace_qubits: {combined.trace_qubits}")

    # ------------------------------------------------------------------
    # 4. Collect trainable params
    # ------------------------------------------------------------------
    params = [
        combined.cores_weights[c]
        for c in combined.cores
        if _raw(combined.cores_weights[c]).requires_grad
           and _raw(combined.cores_weights[c]).is_leaf
    ]
    print(f"\nTrainable cores: {len(params)}")

    # ------------------------------------------------------------------
    # 5. Verify forward pass
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Verifying one forward pass ...")
    print("=" * 60)
    result = engine.contract_with_compiled_strategy(combined)
    print(f"  Tr(student · teacher†) = {get_val(result)}")

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Training  steps={N_STEPS}  lr={LR}  optimizer=SGDG  loss=MSE(trace, 1.0)")
    print("=" * 60)

    optimizer_state: dict = {}
    loss_history = []

    for step in tqdm(range(1, N_STEPS + 1)):
        loss_tensor, grads = engine.contract_with_compiled_strategy_for_gradient(
            combined,
            target=1.0,
            loss='mse',
        )

        params, optimizer_state = backend.optimizer_update(
            params, list(grads), optimizer_state,
            method='sgdg',
            hyperparams={'learning_rate': LR},
        )

        loss_val = loss_tensor.item() if hasattr(loss_tensor, 'item') else float(loss_tensor)
        loss_history.append(loss_val)

        if step % LOG_EVERY == 0 or step == 1:
            trace_val = get_val(engine.contract_with_compiled_strategy(combined))
            print(f"  Step {step:4d}/{N_STEPS}  loss={loss_val:.6f}  trace={trace_val:.6f}")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Training complete")
    print(f"  Initial loss : {loss_history[0]:.6f}")
    print(f"  Final   loss : {loss_history[-1]:.6f}")
    print(f"  Loss reduced : {loss_history[0] - loss_history[-1]:.6f}")

    final_trace = get_val(engine.contract_with_compiled_strategy(combined))
    print(f"  Final trace  : {final_trace:.6f}  (target: 1.0)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    save_path = f"checkpoints/tneq_{N_QUBITS}q_P{PHYS_DIM}.safetensors"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined.save_cores(save_path, metadata={
        'n_qubits': str(N_QUBITS),
        'phys_dim': str(PHYS_DIM),
        'n_steps': str(N_STEPS),
        'final_loss': f"{loss_history[-1]:.6f}",
        'final_trace': f"{final_trace:.6f}",
    })
    print(f"  Saved to: {save_path}")


if __name__ == "__main__":
    main()
