"""Quadratic training: 1024-qubit MPS with correlated Gaussian data.

Structure: circuit + mps + mx + mps_h + circuit_bra
Data: MultivariateNormal with tridiagonal covariance (nearest-neighbor correlation 0.2)
dtype: complex64, CPU

Usage:
    python examples/train_mps_1024_correlated.py
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

N_QUBITS   = 1024
BOND_DIM   = 2
PHYS_DIM   = 2
BATCH_SIZE = 128
N_STEPS    = 200
LR         = 0.01
LOG_EVERY  = 10
SAVE_DIR   = "checkpoints"


def make_correlated_gaussian_sampler(ndim):
    """Create a sampler from a MultivariateNormal with nearest-neighbor correlation.

    Covariance matrix: diagonal=1, sub/super-diagonal=0.2.
    """
    cov_matrix = torch.eye(ndim, dtype=torch.float64)
    indices = torch.arange(ndim - 1)
    cov_matrix[indices + 1, indices] = 0.2
    cov_matrix[indices, indices + 1] = 0.2
    cov_matrix.diagonal().fill_(1.0)

    dist = torch.distributions.MultivariateNormal(
        loc=torch.zeros(ndim, dtype=torch.float64),
        covariance_matrix=cov_matrix,
    )

    def sample_fn(batch_size, num_qubits):
        samples = dist.sample((batch_size,))
        return samples.float().numpy()

    return sample_fn


def main():
    backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
    engine  = EngineCommon(backend=backend, strategy_mode='full')
    data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

    # Build model
    print(f"Building Quadratic model: {N_QUBITS} qubits, bond_dim={BOND_DIM}")
    t0 = time.time()
    model = Quadratic(nqubits=N_QUBITS, bond_dim=BOND_DIM, phys_dim=PHYS_DIM,
                      backend=backend).auto_init()
    model._submodules['mps'].requires_grad_(True)
    combined = model.build()
    t_init = time.time() - t0

    print(f"Init time: {t_init:.1f}s")
    print(f"Combined: {combined.ncores} cores, {len(combined.parameters())} trainable")

    # Correlated Gaussian sampler
    print("Constructing correlated Gaussian sampler (Cholesky decomposition)...")
    t0 = time.time()
    sample_fn = make_correlated_gaussian_sampler(N_QUBITS)
    print(f"  Done in {time.time() - t0:.1f}s")

    # mx core names for data injection
    mx_names = model.mx_core_names

    # Data function: correlated Gaussian
    def data_fn(step):
        x = sample_fn(BATCH_SIZE, N_QUBITS)
        Mx_list, _ = data_gen.generate(x, K=PHYS_DIM, ret_type='TNTensor')
        for i, name in enumerate(mx_names):
            combined[name] = Mx_list[i]

    # Train
    optimizer = create_optimizer("sgdg", combined.parameters(), backend=backend, lr=LR)
    loss_history = []

    print(f"\nTraining: {N_STEPS} steps, batch={BATCH_SIZE}, lr={LR}")
    print(f"Data distribution: correlated Gaussian (cov diag=1, off-diag=0.2)")
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
    save_path = os.path.join(SAVE_DIR, "mps_1024_correlated.safetensors")
    model._submodules['mps'].save_cores(save_path, metadata={
        'n_qubits': str(N_QUBITS),
        'bond_dim': str(BOND_DIM),
        'n_steps': str(N_STEPS),
        'final_loss': f"{loss_history[-1]:.6f}",
        'data_distribution': 'correlated_gaussian',
    })
    print(f"Model saved: {save_path}")

    # Reload validation
    print("\n=== Reload Validation ===")
    model_val = Quadratic(nqubits=N_QUBITS, bond_dim=BOND_DIM, phys_dim=PHYS_DIM,
                          backend=backend).auto_init()
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
