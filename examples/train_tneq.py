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

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.engine_common import EngineCommon
from tneq_qc.optim import SGDG
from tneq_qc.trainer import Trainer, TrainConfig

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

def get_val(result):
    from tneq_qc.core.tn_tensor import TNTensor
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
        print(f"    '{c}': {tuple(t.shape)}  requires_grad={t.requires_grad}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')
    engine = EngineCommon(backend=backend, strategy_mode="full")

    # ------------------------------------------------------------------
    # 1. Teacher TN (fixed)
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
        student.cores_weights[c].requires_grad_(True)
    print_qctn_info(student, "Student")

    # ------------------------------------------------------------------
    # 3. Build combined: student · teacher, set trace
    # ------------------------------------------------------------------
    combined = QCTN.concat([
        ('u', student),
        ('t', teacher),
    ])
    combined.set_trace('all')
    print_qctn_info(combined, "Combined (student · teacher, all traced)")
    print(f"  core_names: {combined.core_names}")
    print(f"  trace_qubits: {combined.trace_qubits}")
    print(f"\nTrainable cores: {len(combined.parameters())}")

    # ------------------------------------------------------------------
    # 4. Verify forward pass
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Verifying one forward pass ...")
    print("=" * 60)
    result = engine.contract_with_compiled_strategy(combined)
    print(f"  Tr(student · teacher†) = {get_val(result)}")

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Training  steps={N_STEPS}  lr={LR}  optimizer=SGDG  loss=MSE(trace, 1.0)")
    print("=" * 60)

    optimizer = SGDG(combined.parameters(), backend, lr=LR)
    save_path = f"checkpoints/tneq_{N_QUBITS}q_P{PHYS_DIM}.safetensors"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    trainer = Trainer(engine, combined, optimizer,
        TrainConfig(max_steps=N_STEPS, log_every=LOG_EVERY, save_path=save_path))
    loss_history = trainer.fit(target=1.0, loss='mse')

    # ------------------------------------------------------------------
    # 6. Summary
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


if __name__ == "__main__":
    main()
