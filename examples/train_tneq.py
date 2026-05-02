"""Train a small TN (student) to approximate a larger TN (teacher) via trace.

Structure: combined = student * teacher_h, all qubits traced.
Loss: MSE(Tr(combined), 1.0)

Usage:
    python examples/train_tneq.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from tneq_qc import QCTN, EngineCommon, BackendFactory, create_optimizer
from tneq_qc.core.tn_tensor import TNTensor

N_QUBITS = 4
PHYS_DIM = 2
N_STEPS  = 500
LR       = 0.01
LOG_EVERY = 10
SAVE_PATH = "checkpoints/tneq_student.safetensors"


def get_val(result):
    if isinstance(result, TNTensor):
        result.scale_to(1.0)
        v = result.tensor
    else:
        v = result
    if hasattr(v, 'is_complex') and v.is_complex():
        v = v.real
    return v.item() if hasattr(v, 'item') else float(v)


def main():
    device = os.environ.get('TNEQ_DEVICE', 'cpu')
    backend = BackendFactory.create_backend('pytorch', device=device, dtype='float32')
    engine = EngineCommon(backend=backend, strategy="row_priority")

    # Teacher (fixed)
    graph = "\n".join(f"-{PHYS_DIM}-{chr(ord('A') + i)}-{PHYS_DIM}-" for i in range(N_QUBITS))
    teacher = QCTN(graph, backend=backend).auto_init(orthogonal=True)

    # Student (trainable, same structure)
    student = QCTN(graph, backend=backend).auto_init(orthogonal=True)
    student.requires_grad_(True)

    # Combined with trace
    combined = QCTN.concat([('u', student), ('t', teacher)])
    combined.set_trace('all')

    print(f"Device: {device}")
    print(f"Teacher: {teacher.ncores} cores")
    print(f"Student: {student.ncores} cores, {len(combined.parameters())} trainable")
    print(f"Tr(student * teacher) = {get_val(engine.contract(combined))}")

    # Train
    optimizer = create_optimizer("sgdg", combined.parameters(), backend=backend, lr=LR)
    loss_history = []

    for step in range(1, N_STEPS + 1):
        loss_val, grads = engine.contract_for_gradient(combined, target=1.0, loss='mse')
        optimizer.step(list(grads))
        lv = float(loss_val)
        loss_history.append(lv)
        if step % LOG_EVERY == 0 or step == 1:
            print(f"  Step {step:4d}/{N_STEPS}  loss={lv:.6f}")

    final_trace = get_val(engine.contract(combined))
    print(f"\nDone. Initial={loss_history[0]:.6f}  Final={loss_history[-1]:.6f}  "
          f"Trace={final_trace:.6f}")

    # Save model
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    student.save_cores(SAVE_PATH, metadata={
        'n_qubits': str(N_QUBITS),
        'n_steps': str(N_STEPS),
        'final_loss': f"{loss_history[-1]:.6f}",
        'final_trace': f"{final_trace:.6f}",
    })
    print(f"Model saved: {SAVE_PATH}")

    # Validation: reload and compare
    print("\n=== Validation ===")
    student_loaded = QCTN(graph, backend=backend)
    student_loaded.load_cores(SAVE_PATH)

    combined_val = QCTN.concat([('u', student_loaded), ('t', teacher)])
    combined_val.set_trace('all')
    loaded_trace = get_val(engine.contract(combined_val))
    print(f"  Loaded trace : {loaded_trace:.6f}")
    print(f"  Expected     : {final_trace:.6f}")
    print(f"  Error        : {abs(loaded_trace - final_trace):.2e}")

    # Per-core error
    print("\n  Per-core comparison (student vs loaded):")
    for core_name in student.cores:
        orig = student.cores_weights[core_name]
        load = student_loaded.cores_weights[core_name]
        orig_np = backend.tensor_to_numpy(orig.tensor * orig.scale if isinstance(orig, TNTensor) else orig)
        load_np = backend.tensor_to_numpy(load.tensor * load.scale if isinstance(load, TNTensor) else load)
        err = np.max(np.abs(orig_np - load_np))
        print(f"    Core '{core_name}': max_abs_error={err:.2e}")


if __name__ == "__main__":
    main()
