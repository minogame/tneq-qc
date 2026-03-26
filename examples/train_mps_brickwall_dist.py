"""Distributed Quadratic training: MPS-of-Brickwall with standard Gaussian.

Same model as train_mps_brickwall.py but using EngineDistributed for
multi-process training via torchrun.

Usage:
    # Single process (fallback)
    python examples/train_mps_brickwall_dist.py

    # 2 processes
    torchrun --nproc_per_node=2 examples/train_mps_brickwall_dist.py

    # 4 processes
    torchrun --nproc_per_node=4 examples/train_mps_brickwall_dist.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import math
import torch
import torch.distributed as dist
import numpy as np
import opt_einsum

from tneq_qc import (
    QCTN, BackendFactory,
    DataGenerator, SGDG,
)
from tneq_qc.modules.small import CircuitState, MeasureMatrix
from tneq_qc.distributed import EngineDistributed
from tneq_qc.distributed.engine.distributed_engine import PartitionConfig
from tneq_qc.distributed.comm import get_comm_backend


# =====================================================================
# MPS-of-Brickwall graph generator (same as train_mps_brickwall.py)
# =====================================================================

def generate_mps_brickwall_graph(total_qubits, block_qubits, overlap=3, phys_dim=2):
    """Generate an MPS-of-Brickwall graph string.

    Each block is a 2-layer brickwall:
    - Even layer: pairs (0,1), (2,3), (4,5), ...
    - Odd layer:  pairs (1,2), (3,4), (5,6), ...

    Adjacent blocks overlap by `overlap` qubits.

    Returns:
        tuple: (graph_string, actual_qubits)
    """
    assert block_qubits > overlap
    assert block_qubits >= 3

    dim = str(phys_dim)
    stride = block_qubits - overlap

    if total_qubits <= block_qubits:
        n_blocks = 1
    else:
        n_blocks = 1 + math.ceil((total_qubits - block_qubits) / stride)

    actual_qubits = block_qubits + (n_blocks - 1) * stride
    total_slots = n_blocks * 2
    width = total_slots * 4

    line_list = [['-'] * width for _ in range(actual_qubits)]
    for i in range(actual_qubits):
        line_list[i][-2] = dim

    sym_idx = 0
    row_core_cols = [[] for _ in range(actual_qubits)]

    for b in range(n_blocks):
        start = b * stride

        even_col = (b * 2) * 4
        for p in range(block_qubits // 2):
            q1 = start + 2 * p
            q2 = start + 2 * p + 1
            sym = opt_einsum.get_symbol(sym_idx); sym_idx += 1
            line_list[q1][even_col] = sym; line_list[q2][even_col] = sym
            row_core_cols[q1].append(even_col); row_core_cols[q2].append(even_col)

        odd_col = (b * 2 + 1) * 4
        for p in range((block_qubits - 1) // 2):
            q1 = start + 2 * p + 1; q2 = start + 2 * p + 2
            sym = opt_einsum.get_symbol(sym_idx); sym_idx += 1
            line_list[q1][odd_col] = sym; line_list[q2][odd_col] = sym
            row_core_cols[q1].append(odd_col); row_core_cols[q2].append(odd_col)

    for q in range(actual_qubits):
        cols = sorted(row_core_cols[q])
        for i, col in enumerate(cols):
            if i < len(cols) - 1:
                line_list[q][col + 2] = dim

    lines = ['-' + dim + '-' + ''.join(line_list[i]) for i in range(actual_qubits)]
    return '\n'.join(lines), actual_qubits


# =====================================================================
# Training
# =====================================================================

TOTAL_QUBITS = 49
BLOCK_QUBITS = 7
OVERLAP      = 1
PHYS_DIM     = 8
BATCH_SIZE   = 1024
N_STEPS      = 500
LR           = 0.01
LOG_EVERY    = 2
SAVE_DIR     = "checkpoints"

# TOTAL_QUBITS = 13
# BLOCK_QUBITS = 7
# OVERLAP      = 1
# PHYS_DIM     = 2
# BATCH_SIZE   = 1024
# N_STEPS      = 500
# LR           = 0.01
# LOG_EVERY    = 10
# SAVE_DIR     = "checkpoints"

def main():
    # --- Distributed setup ---
    if 'RANK' in os.environ:
        dist.init_process_group(backend='gloo')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        print("WARNING: Not launched with torchrun. Running single-process.")

    torch.manual_seed(42)
    np.random.seed(42)

    backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
    data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

    # --- Generate custom TN graph ---
    tn_graph, actual_qubits = generate_mps_brickwall_graph(
        TOTAL_QUBITS, BLOCK_QUBITS, OVERLAP, PHYS_DIM)
    print(f"tn_graph: {tn_graph}")

    stride = BLOCK_QUBITS - OVERLAP
    n_blocks = 1 + math.ceil(max(0, TOTAL_QUBITS - BLOCK_QUBITS) / stride)

    if rank == 0:
        print("=" * 60)
        print("Distributed MPS-of-Brickwall Training")
        print(f"  world_size={world_size}  qubits={actual_qubits}  "
              f"blocks={n_blocks}  phys_dim={PHYS_DIM}")
        print("=" * 60)

    # --- Build 5-segment Quadratic structure ---
    t0 = time.time()

    custom_tn = QCTN(tn_graph, backend=backend).auto_init()
    custom_tn.requires_grad_(True)

    circuit     = CircuitState(actual_qubits, PHYS_DIM, backend).auto_init()
    mx          = MeasureMatrix(actual_qubits, PHYS_DIM, backend).auto_init()
    tn_hermit   = custom_tn.hermit()
    circuit_bra = circuit.bra()

    combined = QCTN.concat([
        ('cs',   circuit),
        ('tn',   custom_tn),
        ('mx',   mx),
        ('tn_h', tn_hermit),
        ('cs_t', circuit_bra),
    ])
    t_init = time.time() - t0

    if rank == 0:
        print(f"Init time: {t_init:.1f}s")
        print(f"Combined: {combined.ncores} cores, {len(combined.parameters())} trainable")
        print(f"combined: {combined}")

    # --- Build qubit-based partition (minimize cross-partition edges) ---
    # Group cores by segment prefix
    tn_cores = []       # (symbol, index_within_segment)
    tn_h_cores = []
    cs_cores = []       # index = qubit index
    cs_t_cores = []
    mx_cores_list = []

    tn_i = tn_h_i = cs_i = cs_t_i = mx_i = 0
    for sym in combined.cores:
        name = combined.core_names.get(sym, sym)
        if name.startswith('tn_h.'):
            tn_h_cores.append((sym, tn_h_i)); tn_h_i += 1
        elif name.startswith('tn.'):
            tn_cores.append((sym, tn_i)); tn_i += 1
        elif name.startswith('cs_t.'):
            cs_t_cores.append((sym, cs_t_i)); cs_t_i += 1
        elif name.startswith('cs.'):
            cs_cores.append((sym, cs_i)); cs_i += 1
        elif name.startswith('mx.'):
            mx_cores_list.append((sym, mx_i)); mx_i += 1

    p0, p1 = [], []

    # tn: first 24 cores → P0, last 24 → P1
    tn_split = custom_tn.ncores // 2
    qb_split = actual_qubits // 2
    print(f"tn_split: {tn_split}  qb_split: {qb_split}")

    for sym, idx in tn_cores:
        (p0 if idx < tn_split else p1).append(sym)

    # tn_h: same partition as the corresponding tn core
    for sym, idx in tn_h_cores:
        (p0 if idx < tn_split else p1).append(sym)

    # cs: qubits 0-24 → P0, qubits 25-48 → P1
    for sym, qi in cs_cores:
        (p0 if qi <= qb_split else p1).append(sym)

    # cs_t: same rule as cs
    for sym, qi in cs_t_cores:
        (p0 if qi <= qb_split else p1).append(sym)

    # mx: qubits 0-23 → P0, qubits 24-48 → P1
    for sym, qi in mx_cores_list:
        (p0 if qi <= qb_split-1 else p1).append(sym)

    partitions = [p0, p1]

    if rank == 0:
        print(f"\nQubit-based partition: P0={len(p0)} cores, P1={len(p1)} cores")
        print(f"  p0: {p0}")
        print(f"  p1: {p1}")
        print(f"  tn split at core {tn_split} (of {custom_tn.ncores})")

    # --- Distributed engine ---
    comm = get_comm_backend(
        backend='torch' if world_size > 1 else 'auto',
        rank=rank, world_size=world_size,
    )
    engine = EngineDistributed(
        backend=backend,
        strategy_mode='full',
        comm=comm,
        partition_config=PartitionConfig(strategy='layer', num_partitions=world_size),
    )
    engine.init_distributed(combined, partitions=partitions)

    # --- Reverse qubit contraction order for the second half of partitions ---
    if world_size >= 2 and rank >= world_size // 2:
        combined.qubit_indices = list(reversed(combined.qubit_indices))

    if world_size > 1:
        dist.barrier()

    # --- Data function ---
    names_map = combined.core_names
    mx_core_names = [
        names_map[sym] for sym in combined.cores
        if names_map.get(sym, '').startswith('mx.')
    ]

    def data_fn(step):
        x = np.random.randn(BATCH_SIZE, actual_qubits).astype(np.float32)
        Mx_list, _ = data_gen.generate(x, K=PHYS_DIM, ret_type='TNTensor')
        for i, name in enumerate(mx_core_names):
            combined[name] = Mx_list[i]

    # --- Train ---
    # Optimizer only covers local partition's parameters
    local_params = engine._local_qctn.parameters() if engine._local_qctn else []
    print(f"local_params: {len(local_params)}")
    # exit()
    optimizer = SGDG(local_params, backend, lr=LR / world_size)
    loss_history = []

    if rank == 0:
        print(f"\nTraining: {N_STEPS} steps, batch={BATCH_SIZE}, lr={LR}")
        print(f"Data distribution: standard Gaussian N(0, I)")
        print("-" * 50)

    for step in range(1, N_STEPS + 1):
        t0 = time.time()
        data_fn(step)

        if world_size > 1:
            dist.barrier()

        loss_val, grads = engine.contract_distributed_with_gradient(target=1, loss='nll')
        optimizer.step(grads)

        if world_size > 1:
            dist.barrier()

        t_step = time.time() - t0
        lv = float(loss_val)
        loss_history.append(lv)

        if rank == 0 and (step % LOG_EVERY == 0 or step == 1):
            print(f"  Step {step:4d}/{N_STEPS}  loss={lv:.6f}  ({t_step:.2f}s)")

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        print(f"\nDone. Initial={loss_history[0]:.6f}  Final={loss_history[-1]:.6f}  "
              f"Reduced={loss_history[0] - loss_history[-1]:.6f}")

        # Save model (rank 0 only)
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, "mps_brickwall_dist.safetensors")
        custom_tn.save_cores(save_path, metadata={
            'actual_qubits': str(actual_qubits),
            'block_qubits': str(BLOCK_QUBITS),
            'overlap': str(OVERLAP),
            'n_blocks': str(n_blocks),
            'world_size': str(world_size),
            'n_steps': str(N_STEPS),
            'final_loss': f"{loss_history[-1]:.6f}",
        })
        print(f"Model saved: {save_path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
