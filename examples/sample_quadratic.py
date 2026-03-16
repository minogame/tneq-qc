"""
Sample and visualize a trained quadratic-form QCTN model.

Loads a checkpoint saved by ``train_quadratic.py``, reconstructs the
same ``cs | mps | mx | mps_h | cs_t`` combined QCTN, then:

1. Computes a 2D heatmap of marginal probability P(q0, q1) on a grid.
2. Draws samples via inverse-CDF and overlays them on the heatmap.

Usage::

    python examples/sample_quadratic.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.core.engine_common import EngineCommon
from tneq_qc.utils.graph_generators import QCTNHelper
from tneq_qc.utils.data_generator import DataGenerator

# ---------------------------------------------------------------------------
# Config — must match train_quadratic.py
# ---------------------------------------------------------------------------
N_QUBITS = 4
BOND_DIM = 2
PHYS_DIM = 2

CHECKPOINT = "checkpoints/quadratic_4q_K2_B2.safetensors"
OUTPUT_DIR = "assets"

HEATMAP_EDGE = 80        # grid resolution per axis
HEATMAP_RANGE = (-5, 5)
NUM_SAMPLES = 2000
SAMPLE_BOUNDS = (-5, 5)
SAMPLE_GRID = 200


def _raw(t):
    return t.tensor if isinstance(t, TNTensor) else t


def init_circuit_01(qctn, backend):
    """Fill each circuit core with alternating 0/1 pattern (same as training)."""
    for c in qctn.cores:
        core = qctn.cores_weights[c]
        shape = tuple(_raw(core).shape)
        dtype = _raw(core).dtype
        n = 1
        for d in shape:
            n *= d
        flat = torch.zeros(n, dtype=dtype)
        for i in range(n):
            flat[i] = float(i % 2)
        qctn.cores_weights[c] = backend.convert_to_tensor(flat.reshape(shape))
    return qctn


def main():
    backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
    engine = EngineCommon(backend=backend, strategy_mode="full")
    data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

    # ------------------------------------------------------------------
    # 1. Rebuild the combined QCTN structure (same as train_quadratic.py)
    # ------------------------------------------------------------------
    circuit = QCTN(QCTNHelper.circuit_state(N_QUBITS, phys_dim=PHYS_DIM), backend=backend).auto_init()
    circuit = init_circuit_01(circuit, backend)

    mps = QCTN(QCTNHelper.mps(N_QUBITS, bond_dim=BOND_DIM, phys_dim=PHYS_DIM), backend=backend).auto_init()

    mx = QCTN(QCTNHelper.measure_matrix(N_QUBITS, phys_dim=PHYS_DIM), backend=backend).auto_init()

    mps_h = mps.hermit()

    circ_bra = QCTN(QCTNHelper.circuit_bra(N_QUBITS, phys_dim=PHYS_DIM), backend=backend).auto_init()
    for c in circuit.cores:
        t = circuit.cores_weights[c]
        circ_bra.cores_weights[c] = TNTensor(
            t.tensor.conj() if isinstance(t, TNTensor) else t.conj(),
            scale=t.scale if isinstance(t, TNTensor) else 1.0,
        )

    combined = QCTN.concat([
        ('cs', circuit),
        ('mps', mps),
        ('mx', mx),
        ('mps_h', mps_h),
        ('cs_t', circ_bra),
    ])

    # ------------------------------------------------------------------
    # 2. Load trained weights
    # ------------------------------------------------------------------
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}")
        print("Run `python examples/train_quadratic.py` first.")
        return

    meta = combined.load_cores(CHECKPOINT)
    print(f"Loaded checkpoint: {CHECKPOINT}")
    print(f"  metadata: { {k: v for k, v in meta.items() if not k.startswith('_')} }")
    print(f"  {repr(combined)}")

    # Collect mx core readable names.
    mx_names = [
        combined.core_names[sym]
        for sym in combined.cores
        if combined.core_names[sym].startswith('mx.')
    ]
    print(f"  mx cores: {mx_names}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    samples = engine.sample(
        combined, data_gen, mx_names,
        num_samples=NUM_SAMPLES,
        bounds=SAMPLE_BOUNDS,
        grid_size=SAMPLE_GRID,
    )
    samples_np = samples.cpu().numpy()
    print(f"  samples shape: {samples.shape}")
    print(f"  range: [{samples.min():.2f}, {samples.max():.2f}]")
    print(f"  mean:  {samples.mean(0).tolist()}")
    print(f"  std:   {samples.std(0).tolist()}")

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
    ax.scatter(
        samples_np[:, 0], samples_np[:, 1],
        s=4, c='cyan', alpha=0.5, edgecolors='none',
    )
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
