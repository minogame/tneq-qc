"""Quadratic form training example.

Structure: circuit + mps + mx + mps_h + circuit_bra
Loss: NLL -mean(log(P) + log_scale) on batch expectation value.

Usage:
    python examples/train_quadratic.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from tneq_qc import (
    QCTN, TNTensor, EngineCommon, BackendFactory, Quadratic,
    DataGenerator, make_data_fn, SGDG,
)

N_QUBITS   = 4
BOND_DIM   = 2
PHYS_DIM   = 2
BATCH_SIZE = 128
N_STEPS    = 1000
LR         = 0.01
LOG_EVERY  = 1

torch.manual_seed(42)
np.random.seed(42)


def init_circuit_01(qctn: QCTN, backend) -> QCTN:
    """Fill each circuit core with an alternating 0/1 pattern."""
    for c in qctn.cores:
        core = qctn.cores_weights[c]
        shape = tuple(core.shape)
        n = 1
        for d in shape:
            n *= d
        flat = torch.zeros(n, dtype=core.dtype)
        for i in range(n):
            flat[i] = float(i % 2)
        qctn.cores_weights[c] = backend.convert_to_tensor(flat.reshape(shape))
    return qctn


def main():
    backend  = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
    engine   = EngineCommon(backend=backend, strategy_mode="full")
    data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

    # Build model
    model = Quadratic(nqubits=N_QUBITS, bond_dim=BOND_DIM, phys_dim=PHYS_DIM,
                      backend=backend).auto_init()
    init_circuit_01(model._submodules['circuit'], backend)
    model._submodules['mps'].requires_grad_(True)
    combined = model.build()

    print(f"Combined: {combined.ncores} cores, "
          f"{len(combined.parameters())} trainable")

    # Verify forward pass
    result = engine.contract(combined)
    if isinstance(result, TNTensor):
        print(f"Forward check: shape={tuple(result.shape)} eff={result.tensor * result.scale}")

    # Train
    optimizer = SGDG(combined.parameters(), backend, lr=LR)
    data_fn = make_data_fn(data_gen, combined, batch_size=BATCH_SIZE, K=PHYS_DIM)
    loss_history = []

    for step in range(1, N_STEPS + 1):
        data_fn(step)
        loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
        optimizer.step(list(grads))
        lv = float(loss_val)
        loss_history.append(lv)
        if step % LOG_EVERY == 0 or step == 1:
            print(f"  Step {step:4d}/{N_STEPS}  loss={lv:.6f}")

    print(f"\nDone. Initial={loss_history[0]:.6f}  Final={loss_history[-1]:.6f}  "
          f"Reduced={loss_history[0] - loss_history[-1]:.6f}")


if __name__ == "__main__":
    main()
