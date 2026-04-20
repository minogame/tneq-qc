"""Quadratic training: 1024-qubit MPS with standard Gaussian data.

Structure: circuit + mps + mx + mps_h + circuit_bra
Data: standard Gaussian N(0, I)
dtype: complex64, CPU

Usage:
    python examples/train_mps_1024_gaussian.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import numpy as np

from tneq_qc import (
    QCTN, BackendFactory, EngineCommon, Quadratic,
    DataGenerator, create_optimizer,
)
from tneq_qc.utils.graph_generators import QCTNHelper

N_QUBITS   = 1024
BOND_DIM   = 2
PHYS_DIM   = 2
BATCH_SIZE = 128
N_STEPS    = 200
LR         = 0.01
LOG_EVERY  = 10
SAVE_DIR   = "checkpoints"


def init_circuit_01(qctn: QCTN, backend) -> QCTN:
    """Fill each circuit core with an alternating 0/1 pattern."""
    for core_info in qctn.adjacency_table:
        core_name = core_info['core_name']
        shape = tuple(core_info['input_shape'] + core_info['output_shape'])
        n = 1
        for d in shape:
            n *= d
        flat = torch.zeros(n, dtype=backend.default_dtype)
        for i in range(n):
            flat[i] = float(i % 2)
        qctn.cores_weights[core_name] = backend.convert_to_tensor(flat.reshape(shape))
    return qctn


def init_measure_identity(qctn: QCTN, backend) -> QCTN:
    """Fill each measure core with an identity matrix placeholder."""
    for core_info in qctn.adjacency_table:
        core_name = core_info['core_name']
        input_shape = core_info['input_shape']
        output_shape = core_info['output_shape']
        input_dim = core_info['input_dim']
        output_dim = core_info['output_dim']
        if input_dim != output_dim:
            raise ValueError(
                f"Measure core {core_name!r} must be square, got {input_dim} and {output_dim}."
            )
        core = backend.eye(input_dim)
        qctn.cores_weights[core_name] = backend.reshape(core, input_shape + output_shape)
    return qctn


def main():
    backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
    engine  = EngineCommon(backend=backend, strategy_mode='full')
    data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

    # Build model
    print(f"Building Quadratic model: {N_QUBITS} qubits, bond_dim={BOND_DIM}")
    t0 = time.time()
    graph = QCTNHelper.mps(N_QUBITS, bond_dim=BOND_DIM, phys_dim=PHYS_DIM)
    model = Quadratic(graph, PHYS_DIM, backend=backend)
    model._submodules['mps'].auto_init(orthogonal=True)
    model._submodules['circuit'].auto_init(orthogonal=False)
    init_circuit_01(model._submodules['circuit'], backend)
    init_measure_identity(model._submodules['mx'], backend)
    model._submodules['mps'].requires_grad_(True)
    combined = model.build()
    t_init = time.time() - t0

    print(f"Init time: {t_init:.1f}s")
    print(f"Combined: {combined.ncores} cores, {len(combined.parameters())} trainable")

    # mx core names for data injection
    mx_names = model.mx_core_names

    # Data function: standard Gaussian
    def data_fn(step):
        x = np.random.randn(BATCH_SIZE, N_QUBITS).astype(np.float32)
        Mx_list, _ = data_gen.generate(x, K=PHYS_DIM, ret_type='TNTensor')
        for i, name in enumerate(mx_names):
            combined[name] = Mx_list[i]

    # Train
    optimizer = create_optimizer("sgdg", combined.parameters(), backend=backend, lr=LR)
    loss_history = []

    print(f"\nTraining: {N_STEPS} steps, batch={BATCH_SIZE}, lr={LR}")
    print(f"Data distribution: standard Gaussian N(0, I)")
    print("-" * 50)

    for step in range(1, N_STEPS + 1):
        t0 = time.time()
        data_fn(step)
        loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
        optimizer.step(list(grads))
        t_step = time.time() - t0

        lv = float(loss_val)
        loss_history.append(lv)
        if step % LOG_EVERY == 0 or step == 1:
            print(f"  Step {step:4d}/{N_STEPS}  loss={lv:.6f}  ({t_step:.2f}s)")

    print(f"\nDone. Initial={loss_history[0]:.6f}  Final={loss_history[-1]:.6f}  "
          f"Reduced={loss_history[0] - loss_history[-1]:.6f}")

    # Save model
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "mps_1024_gaussian.safetensors")
    model._submodules['mps'].save_cores(save_path, metadata={
        'n_qubits': str(N_QUBITS),
        'bond_dim': str(BOND_DIM),
        'n_steps': str(N_STEPS),
        'final_loss': f"{loss_history[-1]:.6f}",
        'data_distribution': 'gaussian',
    })
    print(f"Model saved: {save_path}")

    # Reload validation
    print("\n=== Reload Validation ===")
    model_val = Quadratic(graph, PHYS_DIM, backend=backend)
    model_val._submodules['circuit'].auto_init(orthogonal=False)
    init_circuit_01(model_val._submodules['circuit'], backend)
    init_measure_identity(model_val._submodules['mx'], backend)
    model_val._submodules['mps'].load_cores(save_path)
    combined_val = model_val.build()

    from tneq_qc.core.tn_tensor import TNTensor
    ident = TNTensor(backend.eye(PHYS_DIM))
    mx_names_val = model_val.mx_core_names
    for name in mx_names:
        combined[name] = ident
    for name in mx_names_val:
        combined_val[name] = ident

    r_orig = engine.contract(combined)
    r_load = engine.contract(combined_val)
    r_orig_np = backend.tensor_to_numpy(r_orig)
    r_load_np = backend.tensor_to_numpy(r_load)
    reload_err = np.max(np.abs(r_orig_np - r_load_np))
    print(f"  Max reload error: {reload_err:.2e}")


if __name__ == "__main__":
    main()
