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

from tneq_qc import QCTN, EngineCommon, BackendFactory, SGDG, Trainer, TrainConfig
from tneq_qc.core.tn_tensor import TNTensor

N_QUBITS = 4
PHYS_DIM = 2
N_STEPS  = 500
LR       = 0.01
LOG_EVERY = 10

torch.manual_seed(42)
np.random.seed(42)


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
    backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')
    engine = EngineCommon(backend=backend, strategy_mode="full")

    # Teacher (fixed)
    graph = "\n".join(f"-{PHYS_DIM}-{chr(ord('A') + i)}-{PHYS_DIM}-" for i in range(N_QUBITS))
    teacher = QCTN(graph, backend=backend).auto_init()

    # Student (trainable, same structure)
    student = QCTN(graph, backend=backend).auto_init()
    student.requires_grad_(True)

    # Combined with trace
    combined = QCTN.concat([('u', student), ('t', teacher)])
    combined.set_trace('all')

    print(f"Teacher: {teacher.ncores} cores")
    print(f"Student: {student.ncores} cores, {len(combined.parameters())} trainable")
    print(f"Tr(student * teacher) = {get_val(engine.contract(combined))}")

    # Train
    optimizer = SGDG(combined.parameters(), backend, lr=LR)
    save_path = f"checkpoints/tneq_{N_QUBITS}q_P{PHYS_DIM}.safetensors"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    trainer = Trainer(engine, combined, optimizer,
        TrainConfig(max_steps=N_STEPS, log_every=LOG_EVERY, save_path=save_path))
    loss_history = trainer.fit(target=1.0, loss='mse')

    final_trace = get_val(engine.contract(combined))
    print(f"\nDone. Initial={loss_history[0]:.6f}  Final={loss_history[-1]:.6f}  "
          f"Trace={final_trace:.6f}")


if __name__ == "__main__":
    main()
