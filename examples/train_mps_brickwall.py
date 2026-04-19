"""Quadratic training: custom MPS-of-Brickwall structure with standard Gaussian.

The TN structure is an MPS-like chain where each "block" is a 2-layer brickwall.
Adjacent blocks overlap by 3 qubits.

Example (7 qubits, block_qubits=5, overlap=3, 2 blocks):

    -2-a-------------2-
    -2-a-2-c---------2-
    -2-b-2-c-2-e-----2-     (overlap region starts)
    -2-b-2-d-2-e-2-g-2-
    -2-----d-2-f-2-g-2-     (overlap region ends)
    -2---------f-2-h-2-
    -2-------------h-2-

    Block 1 (a,b,c,d): brickwall on qubits 0-4
    Block 2 (e,f,g,h): brickwall on qubits 2-6

Structure: circuit + custom_tn + mx + custom_tn_h + circuit_bra
Data: standard Gaussian N(0, I)
dtype: complex64, CPU

Usage:
    python examples/train_mps_brickwall.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import math
import torch
import numpy as np
import opt_einsum

from tneq_qc import (
    QCTN, BackendFactory, EngineCommon,
    DataGenerator, create_optimizer,
)
from tneq_qc.modules.small import CircuitState, MeasureMatrix


# =====================================================================
# MPS-of-Brickwall graph generator
# =====================================================================

def generate_mps_brickwall_graph(total_qubits, block_qubits, overlap=3, phys_dim=2):
    """Generate an MPS-of-Brickwall graph string.

    Each block is a 2-layer brickwall:
    - Even layer: pairs (0,1), (2,3), (4,5), ...
    - Odd layer:  pairs (1,2), (3,4), (5,6), ...

    Adjacent blocks overlap by `overlap` qubits.

    Args:
        total_qubits: Minimum number of qubits (actual may be slightly more).
        block_qubits: Number of qubits per brickwall block.
        overlap: Number of overlapping qubits between adjacent blocks.
        phys_dim: Physical dimension (default 2).

    Returns:
        tuple: (graph_string, actual_qubits)
    """
    assert block_qubits > overlap, "block_qubits must be > overlap"
    assert block_qubits >= 3, "block_qubits must be >= 3"

    dim = str(phys_dim)
    stride = block_qubits - overlap

    # Number of blocks needed to cover total_qubits
    if total_qubits <= block_qubits:
        n_blocks = 1
    else:
        n_blocks = 1 + math.ceil((total_qubits - block_qubits) / stride)

    actual_qubits = block_qubits + (n_blocks - 1) * stride

    # Each block uses 2 column-slots (even layer + odd layer)
    total_slots = n_blocks * 2
    width = total_slots * 4  # 4 chars per slot

    # Initialize char grid
    line_list = [['-'] * width for _ in range(actual_qubits)]

    # Right boundary dim
    for i in range(actual_qubits):
        line_list[i][-2] = dim

    # Place cores and collect per-row core columns
    sym_idx = 0
    row_core_cols = [[] for _ in range(actual_qubits)]  # track which cols have cores

    for b in range(n_blocks):
        start = b * stride

        # Even layer: pairs (start+0, start+1), (start+2, start+3), ...
        even_col = (b * 2) * 4
        for p in range(block_qubits // 2):
            q1 = start + 2 * p
            q2 = start + 2 * p + 1
            sym = opt_einsum.get_symbol(sym_idx)
            sym_idx += 1

            line_list[q1][even_col] = sym
            line_list[q2][even_col] = sym
            row_core_cols[q1].append(even_col)
            row_core_cols[q2].append(even_col)

        # Odd layer: pairs (start+1, start+2), (start+3, start+4), ...
        odd_col = (b * 2 + 1) * 4
        for p in range((block_qubits - 1) // 2):
            q1 = start + 2 * p + 1
            q2 = start + 2 * p + 2
            sym = opt_einsum.get_symbol(sym_idx)
            sym_idx += 1

            line_list[q1][odd_col] = sym
            line_list[q2][odd_col] = sym
            row_core_cols[q1].append(odd_col)
            row_core_cols[q2].append(odd_col)

    # Place dim chars between consecutive cores on each row.
    # Do NOT place dim after the last core (unless it coincides with the
    # right boundary position), to avoid parsing two adjacent numbers as
    # a single multi-digit value after dash removal.
    right_boundary_col = width - 2
    for q in range(actual_qubits):
        cols = sorted(row_core_cols[q])
        for i, col in enumerate(cols):
            if i < len(cols) - 1:
                # Not the last core → always place dim
                line_list[q][col + 2] = dim
            else:
                # Last core → place dim only if it coincides with right boundary
                if col + 2 == right_boundary_col:
                    pass  # already set
                # else: skip (right boundary handles the edge)

    # Build graph string
    lines = ['-' + dim + '-' + ''.join(line_list[i]) for i in range(actual_qubits)]
    return '\n'.join(lines), actual_qubits


def print_graph_preview(graph_str, max_rows=20):
    """Print a preview of the graph (first and last rows if too many)."""
    lines = graph_str.split('\n')
    if len(lines) <= max_rows:
        for line in lines:
            print(f"  {line}")
    else:
        half = max_rows // 2
        for line in lines[:half]:
            print(f"  {line}")
        print(f"  ... ({len(lines) - max_rows} rows omitted) ...")
        for line in lines[-half:]:
            print(f"  {line}")


# =====================================================================
# Training
# =====================================================================

TOTAL_QUBITS = 49
BLOCK_QUBITS = 7      # qubits per brickwall block
OVERLAP      = 1       # overlap between adjacent blocks
PHYS_DIM     = 8
BATCH_SIZE   = 1024
N_STEPS      = 500
LR           = 0.01
LOG_EVERY    = 10
SAVE_DIR     = "checkpoints"


def main():
    backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
    engine  = EngineCommon(backend=backend, strategy_mode='full')
    data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

    # --- Generate custom TN graph ---
    print(f"Generating MPS-of-Brickwall graph:")
    print(f"  total_qubits={TOTAL_QUBITS}, block_qubits={BLOCK_QUBITS}, overlap={OVERLAP}")

    tn_graph, actual_qubits = generate_mps_brickwall_graph(
        TOTAL_QUBITS, BLOCK_QUBITS, OVERLAP, PHYS_DIM)

    stride = BLOCK_QUBITS - OVERLAP
    n_blocks = 1 + math.ceil(max(0, TOTAL_QUBITS - BLOCK_QUBITS) / stride)

    print(f"  n_blocks={n_blocks}, actual_qubits={actual_qubits}")
    print(f"\nGraph preview:")
    print_graph_preview(tn_graph)

    # --- Build 5-segment Quadratic structure ---
    print(f"\nBuilding Quadratic structure...")
    t0 = time.time()

    # 1) Custom TN (trainable)
    custom_tn = QCTN(tn_graph, backend=backend).auto_init()
    custom_tn.requires_grad_(True)

    # print('custom_tn', len(custom_tn.cores))
    # exit()

    # 2) CircuitState (ket)
    circuit = CircuitState(actual_qubits, PHYS_DIM, backend).auto_init()

    # 3) MeasureMatrix (data injection)
    mx = MeasureMatrix(actual_qubits, PHYS_DIM, backend).auto_init()

    # 4) Hermitian conjugate
    tn_hermit = custom_tn.hermit()

    # 5) CircuitState bra
    circuit_bra = circuit.bra()

    # Concatenate
    combined = QCTN.concat([
        ('cs',   circuit),
        ('tn',   custom_tn),
        ('mx',   mx),
        ('tn_h', tn_hermit),
        ('cs_t', circuit_bra),
    ])
    t_init = time.time() - t0

    print(f"Init time: {t_init:.1f}s")
    print(f"Custom TN cores: {custom_tn.ncores}")
    print(f"Combined: {combined.ncores} cores, {len(combined.parameters())} trainable")

    # Detect mx core names
    names_map = combined.core_names
    mx_core_names = [
        names_map[sym] for sym in combined.cores
        if names_map.get(sym, '').startswith('mx.')
    ]

    # Data function: standard Gaussian
    def data_fn(step):
        x = np.random.randn(BATCH_SIZE, actual_qubits).astype(np.float32)
        Mx_list, _ = data_gen.generate(x, K=PHYS_DIM, ret_type='TNTensor')
        for i, name in enumerate(mx_core_names):
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
    save_path = os.path.join(SAVE_DIR, "mps_brickwall.safetensors")
    custom_tn.save_cores(save_path, metadata={
        'actual_qubits': str(actual_qubits),
        'block_qubits': str(BLOCK_QUBITS),
        'overlap': str(OVERLAP),
        'n_blocks': str(n_blocks),
        'n_steps': str(N_STEPS),
        'final_loss': f"{loss_history[-1]:.6f}",
        'data_distribution': 'gaussian',
    })
    print(f"Model saved: {save_path}")

    # Reload validation
    print("\n=== Reload Validation ===")
    custom_tn_val = QCTN(tn_graph, backend=backend).auto_init()
    custom_tn_val.load_cores(save_path)
    tn_hermit_val = custom_tn_val.hermit()
    combined_val = QCTN.concat([
        ('cs',   circuit),
        ('tn',   custom_tn_val),
        ('mx',   mx),
        ('tn_h', tn_hermit_val),
        ('cs_t', circuit_bra),
    ])

    from tneq_qc.core.tn_tensor import TNTensor
    ident = TNTensor(backend.eye(PHYS_DIM))
    mx_names_val = [
        combined_val.core_names[sym] for sym in combined_val.cores
        if combined_val.core_names.get(sym, '').startswith('mx.')
    ]
    for name in mx_core_names:
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
