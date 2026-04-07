"""Quadratic form training example.

Structure: circuit + mps + mx + mps_h + circuit_bra
Loss: NLL -mean(log(P) + log_scale) on batch expectation value.

After training: saves model, generates heatmap + scatter plot,
computes KL divergence between training distribution and sampled data.

Usage:
    python examples/train_quadratic.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tneq_qc import (
    QCTN, TNTensor, EngineCommon, BackendFactory, Quadratic,
    DataGenerator, make_data_fn, create_optimizer,
)

N_QUBITS   = 4
BOND_DIM   = 2
PHYS_DIM   = 2
BATCH_SIZE = 128
N_STEPS    = 1000
LR         = 0.01
LOG_EVERY  = 10
N_SAMPLES  = 2000
SAVE_DIR   = "checkpoints"
ASSETS_DIR = "assets/quadratic"


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


def compute_marginal_heatmap(engine, combined, data_gen, mx_core_names,
                             grid_size=100, bounds=(-3, 3)):
    """Compute marginal probability P(x_i) on a 1D grid for each qubit."""
    K = PHYS_DIM
    x_min, x_max = bounds
    grid = np.linspace(x_min, x_max, grid_size).astype(np.float32)

    # Identity for unsampled cores
    backend = engine.backend
    ident = backend.eye(K)
    if isinstance(ident, TNTensor):
        ident = TNTensor(ident)
    else:
        ident = TNTensor(ident)

    n_dims = len(mx_core_names)
    heatmap = np.zeros((n_dims, grid_size))

    for dim_idx, core_name in enumerate(mx_core_names):
        # Reset all mx cores to identity
        for name in mx_core_names:
            combined[name] = ident

        # Evaluate probability on grid for this dimension
        for gi, x_val in enumerate(grid):
            x_input = np.array([[x_val]], dtype=np.float32)
            Mx_list, _ = data_gen.generate(x_input, K=K, ret_type='TNTensor')
            combined[core_name] = Mx_list[0]

            prob = engine.calculate_probability(combined, {})
            heatmap[dim_idx, gi] = max(prob, 0.0)

        # Restore identity
        combined[core_name] = ident

    return heatmap, grid


def estimate_kl_divergence(train_samples, model_samples, n_bins=50, bounds=(-3, 3)):
    """Estimate KL divergence D_KL(P_train || P_model) per dimension.

    Uses histogram-based density estimation.
    """
    n_dims = train_samples.shape[1]
    kl_per_dim = []

    for d in range(n_dims):
        p_hist, edges = np.histogram(train_samples[:, d], bins=n_bins,
                                     range=bounds, density=True)
        q_hist, _ = np.histogram(model_samples[:, d], bins=n_bins,
                                 range=bounds, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p_hist = p_hist + eps
        q_hist = q_hist + eps

        # Normalize
        dx = edges[1] - edges[0]
        p_hist = p_hist / (p_hist.sum() * dx)
        q_hist = q_hist / (q_hist.sum() * dx)

        kl = np.sum(p_hist * np.log(p_hist / q_hist) * dx)
        kl_per_dim.append(kl)

    return kl_per_dim


def main():
    device = os.environ.get('TNEQ_DEVICE', 'cpu')
    backend  = BackendFactory.create_backend('pytorch', device=device, dtype='complex64')
    engine   = EngineCommon(backend=backend, strategy_mode="full")
    data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

    # Build model
    model = Quadratic(nqubits=N_QUBITS, bond_dim=BOND_DIM, phys_dim=PHYS_DIM,
                      backend=backend).auto_init()
    init_circuit_01(model._submodules['circuit'], backend)
    model._submodules['mps'].requires_grad_(True)
    combined = model.build()

    print(f"Device: {device}")
    print(f"Combined: {combined.ncores} cores, "
          f"{len(combined.parameters())} trainable")

    # Verify forward pass
    result = engine.contract(combined)
    if isinstance(result, TNTensor):
        print(f"Forward check: shape={tuple(result.shape)} eff={result.tensor * result.scale}")
    
    # Train
    optimizer = create_optimizer("sgdg", combined.parameters(), backend=backend, lr=LR)
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

    # Save model (save the mps submodule which has the trainable params)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "quadratic_mps.safetensors")
    model._submodules['mps'].save_cores(save_path, metadata={
        'n_qubits': str(N_QUBITS),
        'bond_dim': str(BOND_DIM),
        'n_steps': str(N_STEPS),
        'final_loss': f"{loss_history[-1]:.6f}",
    })
    print(f"Model saved: {save_path}")

    # === Validation ===
    print("\n=== Validation ===")

    mx_core_names = model.mx_core_names
    print(f"mx_core_names: {mx_core_names}")

    # 1. Marginal probability heatmap
    print("Computing marginal heatmap...")
    heatmap, grid = compute_marginal_heatmap(
        engine, combined, data_gen, mx_core_names,
        grid_size=100, bounds=(-3, 3)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(heatmap, aspect='auto', origin='lower',
                   extent=[grid[0], grid[-1], 0, len(mx_core_names)],
                   cmap='hot')
    ax.set_xlabel('x')
    ax.set_ylabel('Qubit index')
    ax.set_title(f'Marginal Probability Heatmap ({N_QUBITS} qubits)')
    plt.colorbar(im, ax=ax, label='P(x)')
    heatmap_path = os.path.join(ASSETS_DIR, "marginal_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"  Heatmap saved: {heatmap_path}")

    # 2. Sample from the model
    print(f"Sampling {N_SAMPLES} points...")
    with torch.no_grad():
        samples = engine.sample(
            combined, data_gen, mx_core_names,
            num_samples=N_SAMPLES, bounds=(-3, 3), grid_size=500
        )
    if isinstance(samples, TNTensor):
        samples = samples.tensor
    samples_np = samples.detach().cpu().numpy() if hasattr(samples, 'detach') else np.array(samples)

    # Scatter plot (first 2 dimensions if available)
    n_plot_dims = min(samples_np.shape[1], 2)
    fig, axes = plt.subplots(1, n_plot_dims, figsize=(6 * n_plot_dims, 5))
    if n_plot_dims == 1:
        axes = [axes]
    for d in range(n_plot_dims):
        axes[d].hist(samples_np[:, d], bins=50, density=True, alpha=0.7, label='Samples')
        axes[d].set_xlabel(f'x_{d}')
        axes[d].set_ylabel('Density')
        axes[d].set_title(f'Sampled distribution (dim {d})')
        axes[d].legend()
    scatter_path = os.path.join(ASSETS_DIR, "samples_hist.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"  Sample histogram saved: {scatter_path}")

    # 2D scatter if >= 2 dims
    if samples_np.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(samples_np[:, 0], samples_np[:, 1], s=2, alpha=0.5)
        ax.set_xlabel('x_0')
        ax.set_ylabel('x_1')
        ax.set_title(f'2D Scatter ({N_SAMPLES} samples)')
        scatter2d_path = os.path.join(ASSETS_DIR, "samples_scatter_2d.png")
        plt.tight_layout()
        plt.savefig(scatter2d_path, dpi=150)
        plt.close()
        print(f"  2D scatter saved: {scatter2d_path}")

    # 3. KL divergence: training distribution vs sampled distribution
    print("Computing KL divergence...")
    # Generate training-like samples (uniform in [-1, 1])
    train_samples = np.random.uniform(-1.0, 1.0, size=(N_SAMPLES, N_QUBITS)).astype(np.float32)

    kl_per_dim = estimate_kl_divergence(train_samples, samples_np, n_bins=50, bounds=(-3, 3))
    kl_mean = np.mean(kl_per_dim)
    print(f"  KL divergence per dim: {[f'{kl:.4f}' for kl in kl_per_dim]}")
    print(f"  KL divergence mean   : {kl_mean:.4f}")

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history)
    ax.set_xlabel('Step')
    ax.set_ylabel('NLL Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    loss_path = os.path.join(ASSETS_DIR, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"  Loss curve saved: {loss_path}")


if __name__ == "__main__":
    main()
