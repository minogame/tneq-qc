"""Quadratic training: 32-qubit Brickwall with correlated Gaussian data.

Structure: circuit + brickwall + mx + brickwall_h + circuit_bra
Data: MultivariateNormal with tridiagonal covariance (nearest-neighbor correlation 0.2)
dtype: complex64, CPU

Since Quadratic class hardcodes MPS, we manually build the 5-segment
structure using QCTN.concat.

Usage:
    python examples/train_brickwall_32.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import numpy as np

from tneq_qc import (
    QCTN, BackendFactory, EngineCommon, QCTNHelper,
    DataGenerator, SGDG,
)
from tneq_qc.modules.small import CircuitState, MeasureMatrix

N_QUBITS   = 32
N_LAYERS   = 4        # brickwall layers (m = n_layers // 2 cores per row-pair)
PHYS_DIM   = 4
BATCH_SIZE = 1024
N_STEPS    = 100
LR         = 0.01
LOG_EVERY  = 10
SAVE_DIR   = "checkpoints"


def make_correlated_gaussian_sampler(ndim):
    """Create a sampler from a MultivariateNormal with nearest-neighbor correlation."""
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
    # backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')
    engine  = EngineCommon(backend=backend, strategy_mode='full')
    data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

    # --- Build 5-segment Quadratic structure manually ---
    print(f"Building Brickwall Quadratic: {N_QUBITS} qubits, {N_LAYERS} layers")
    t0 = time.time()

    # 1) CircuitState (ket)
    circuit = CircuitState(N_QUBITS, PHYS_DIM, backend).auto_init()

    # 2) Brickwall (trainable)
    bw_graph = QCTNHelper.brickwall(N_QUBITS, n_layers=N_LAYERS, phys_dim=PHYS_DIM)
    # bw_graph = QCTNHelper.mps(N_QUBITS, n_layers=N_LAYERS, phys_dim=PHYS_DIM)
    # bw_graph = QCTNHelper.mps(N_QUBITS, bond_dim=PHYS_DIM, phys_dim=PHYS_DIM)
    brickwall = QCTN(bw_graph, backend=backend).auto_init()
    
    for name in brickwall.core_names:
        brickwall[name].auto_scale()

    brickwall.requires_grad_(True)

    # 3) MeasureMatrix (data injection)
    mx = MeasureMatrix(N_QUBITS, PHYS_DIM, backend).auto_init()

    # 4) Hermitian conjugate of brickwall
    bw_hermit = brickwall.hermit()

    # 5) CircuitState bra
    circuit_bra = circuit.bra()

    # Concatenate
    combined = QCTN.concat([
        ('cs',   circuit),
        ('bw',   brickwall),
        ('mx',   mx),
        ('bw_h', bw_hermit),
        ('cs_t', circuit_bra),
    ])
    t_init = time.time() - t0

    print(f"Init time: {t_init:.1f}s")
    print(f"Brickwall cores: {brickwall.ncores}")
    print(f"Combined: {combined.ncores} cores, {len(combined.parameters())} trainable")

    # Detect mx core names
    names_map = combined.core_names
    mx_core_names = [
        names_map[sym] for sym in combined.cores
        if names_map.get(sym, '').startswith('mx.')
    ]

    # Correlated Gaussian sampler
    sample_fn = make_correlated_gaussian_sampler(N_QUBITS)

    # Data function
    def data_fn(step):
        x = sample_fn(BATCH_SIZE, N_QUBITS)
        Mx_list, _ = data_gen.generate(x, K=PHYS_DIM, ret_type='TNTensor')
        for i, name in enumerate(mx_core_names):
            combined[name] = Mx_list[i]

    # Train
    optimizer = SGDG(combined.parameters(), backend, lr=LR)
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
    save_path = os.path.join(SAVE_DIR, "brickwall_32.safetensors")
    brickwall.save_cores(save_path, metadata={
        'n_qubits': str(N_QUBITS),
        'n_layers': str(N_LAYERS),
        'n_steps': str(N_STEPS),
        'final_loss': f"{loss_history[-1]:.6f}",
        'data_distribution': 'correlated_gaussian',
    })
    print(f"Model saved: {save_path}")



    import matplotlib.pyplot as plt
    from tneq_qc.core.tn_tensor import TNTensor
    
    OUTPUT_DIR = "assets"
    HEATMAP_EDGE = 80        # grid resolution per axis
    HEATMAP_RANGE = (-5, 5)
    NUM_SAMPLES = 2000
    SAMPLE_BOUNDS = (-5, 5)
    SAMPLE_GRID = 200

    mx_names = [
        combined.core_names[sym]
        for sym in combined.cores
        if combined.core_names[sym].startswith('mx.')
    ]
    print(f"  mx cores: {mx_names}")

    # ------------------------------------------------------------------
    # 3. Heatmap: marginal P(q0, q1) on a uniform grid
    # ------------------------------------------------------------------
    print(f"\nComputing heatmap ({HEATMAP_EDGE}x{HEATMAP_EDGE})...")

    lo, hi = HEATMAP_RANGE
    edge = HEATMAP_EDGE
    B = edge * edge

    # Build grid: (edge*edge, N_QUBITS), only first 2 dims vary.
    xs = torch.linspace(lo, hi, edge)
    ys = torch.linspace(lo, hi, edge)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
    x_flat = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (B, 2)

    # Pad remaining qubits with 0 (they'll get identity mx → traced out).
    if N_QUBITS > 2:
        pad = torch.zeros(B, N_QUBITS - 2)
        x_full = torch.cat([x_flat, pad], dim=1)  # (B, N_QUBITS)
    else:
        x_full = x_flat

    Mx_list, _ = data_gen.generate(x_full, K=PHYS_DIM, ret_type='TNTensor')

    # Set only first 2 mx cores (marginal over q0, q1).
    # Reset all mx cores to identity first.
    ident = TNTensor(backend.eye(PHYS_DIM))
    for name in mx_names:
        combined[name] = ident

    # Update q0 and q1 mx cores.
    combined[mx_names[0]] = Mx_list[0]
    combined[mx_names[1]] = Mx_list[1]

    result = engine.contract_with_compiled_strategy(combined)
    if isinstance(result, TNTensor):
        result.scale_to(1.0)
        heatmap_vals = result.tensor.detach()
    else:
        heatmap_vals = result.detach()

    # Born rule: complex → real probability |W|².
    if heatmap_vals.is_complex():
        heatmap_vals = (heatmap_vals * heatmap_vals.conj()).real

    heatmap = heatmap_vals.reshape(edge, edge).cpu().numpy()

    # ------------------------------------------------------------------
    # 4. Sample
    # ------------------------------------------------------------------
    print(f"Sampling {NUM_SAMPLES} points...")

    # Reset mx cores to identity before sampling.
    for name in mx_names:
        combined[name] = ident

    # samples = engine.sample(
    #     combined, data_gen, mx_names,
    #     num_samples=NUM_SAMPLES,
    #     bounds=SAMPLE_BOUNDS,
    #     grid_size=SAMPLE_GRID,
    # )
    # samples_np = samples.cpu().numpy()
    # print(f"  samples shape: {samples.shape}")
    # print(f"  range: [{samples.min():.2f}, {samples.max():.2f}]")
    # print(f"  mean:  {samples.mean(0).tolist()}")
    # print(f"  std:   {samples.std(0).tolist()}")

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap
    ax = axes[0]
    im = ax.imshow(
        heatmap.T, origin='lower', cmap='hot', interpolation='bilinear',
        extent=[lo, hi, lo, hi], aspect='equal',
    )
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Marginal P(q0, q1)\n{HEATMAP_EDGE}x{HEATMAP_EDGE} grid")
    ax.set_xlabel("q0")
    ax.set_ylabel("q1")

    # Scatter + heatmap overlay
    ax = axes[1]
    ax.imshow(
        heatmap.T, origin='lower', cmap='hot', interpolation='bilinear',
        extent=[lo, hi, lo, hi], aspect='equal', alpha=0.6,
    )
    # ax.scatter(
    #     samples_np[:, 0], samples_np[:, 1],
    #     s=4, c='cyan', alpha=0.5, edgecolors='none',
    # )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(f"Samples (N={NUM_SAMPLES}) on heatmap")
    ax.set_xlabel("q0")
    ax.set_ylabel("q1")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "quadratic_heatmap_samples.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

if __name__ == "__main__":
    main()
