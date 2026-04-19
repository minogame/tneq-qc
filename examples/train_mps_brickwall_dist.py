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
    DataGenerator, create_optimizer,
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

TOTAL_QUBITS = int(os.environ.get('TOTAL_QUBITS', '49'))
BLOCK_QUBITS = 7
OVERLAP      = 1
PHYS_DIM     = int(os.environ.get('PHYS_DIM', '8'))
BATCH_SIZE   = int(os.environ.get('BATCH_SIZE', '1024'))
N_STEPS      = int(os.environ.get('N_STEPS', '10'))
LR           = 0.01
LOG_EVERY    = 1
SAVE_DIR     = "checkpoints"

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

    # --- N-way qubit-aligned partition ---
    # Build a mapping: tn core index → set of qubits it touches
    tn_core_qubits = {}
    for entry in custom_tn.adjacency_table:
        cn = entry['core_name']
        ci = custom_tn.cores.index(cn) if cn in custom_tn.cores else -1
        if ci >= 0:
            qs = set()
            for d in ['in_edge_list', 'out_edge_list']:
                for e in entry.get(d, []):
                    qi = e.get('qubit_idx', -1)
                    if qi >= 0:
                        qs.add(qi)
            tn_core_qubits[ci] = qs

    n_tn = custom_tn.ncores
    # Split tn cores into world_size roughly equal chunks
    chunk = n_tn // world_size
    tn_splits = []  # (start_idx, end_idx) per partition
    for p in range(world_size):
        s = p * chunk
        e = (p + 1) * chunk if p < world_size - 1 else n_tn
        tn_splits.append((s, e))

    # Determine qubit range covered by each partition's tn cores
    partition_qubit_ranges = []  # (min_qubit, max_qubit) per partition
    for s, e in tn_splits:
        qs = set()
        for ci in range(s, e):
            qs.update(tn_core_qubits.get(ci, set()))
        partition_qubit_ranges.append((min(qs), max(qs)))

    if rank == 0:
        for p, ((s, e), (qmin, qmax)) in enumerate(zip(tn_splits, partition_qubit_ranges)):
            print(f"  P{p}: tn cores [{s},{e})  qubits {qmin}-{qmax}")

    # Assign each core to a partition
    partitions = [[] for _ in range(world_size)]

    for sym, idx in tn_cores:
        for p, (s, e) in enumerate(tn_splits):
            if s <= idx < e:
                partitions[p].append(sym)
                break

    for sym, idx in tn_h_cores:
        for p, (s, e) in enumerate(tn_splits):
            if s <= idx < e:
                partitions[p].append(sym)
                break

    # cs, mx, cs_t: assign by qubit to the partition whose range covers it.
    # If a qubit is in multiple partitions' ranges (overlap), pick the lower one.
    def qubit_to_partition(qi):
        for p, (qmin, qmax) in enumerate(partition_qubit_ranges):
            if qmin <= qi <= qmax:
                return p
        return world_size - 1  # fallback: last partition

    for sym, qi in cs_cores:
        partitions[qubit_to_partition(qi)].append(sym)
    for sym, qi in cs_t_cores:
        partitions[qubit_to_partition(qi)].append(sym)
    for sym, qi in mx_cores_list:
        partitions[qubit_to_partition(qi)].append(sym)

    if rank == 0:
        for p in range(world_size):
            print(f"  P{p}: {len(partitions[p])} cores")
        # Count cross-partition edges
        core_to_part = {}
        for p, cores in enumerate(partitions):
            for sym in cores:
                core_to_part[sym] = p
        cross_edges = 0
        for entry in combined.adjacency_table:
            core_sym = entry['core_name']
            cp = core_to_part.get(core_sym, -1)
            for edge in entry.get('in_edge_list', []) + entry.get('out_edge_list', []):
                neighbor = edge.get('neighbor_name', '')
                np_ = core_to_part.get(neighbor, -1)
                if neighbor and cp >= 0 and np_ >= 0 and cp != np_:
                    cross_edges += 1
        print(f"  Cross-partition edges (all directions): {cross_edges}")

    # --- Engine setup ---
    if world_size > 1:
        comm = get_comm_backend(
            backend='torch', rank=rank, world_size=world_size,
        )
        engine = EngineDistributed(
            backend=backend,
            strategy_mode='full',
            comm=comm,
            partition_config=PartitionConfig(strategy='layer', num_partitions=world_size),
        )
        engine.init_distributed(combined, partitions=partitions)

        # Reverse qubit contraction order for the second half of partitions
        if rank >= world_size // 2:
            combined.qubit_indices = list(reversed(combined.qubit_indices))

        # Free non-local cores to reduce per-rank memory
        local_core_set = set(partitions[rank])
        freed = 0
        for sym in list(combined.cores_weights.keys()):
            if sym not in local_core_set:
                combined.cores_weights[sym] = None
                freed += 1
        if rank == 0:
            print(f"  Freed {freed} non-local cores from combined")

        dist.barrier()

        local_params = engine._local_qctn.parameters() if engine._local_qctn else []
    else:
        from tneq_qc import EngineCommon
        engine = EngineCommon(backend=backend, strategy_mode='full')
        local_params = combined.parameters()

    if rank == 0:
        print(f"  trainable params: {len(local_params)}")

    # --- Data function ---
    names_map = combined.core_names
    mx_core_names = [
        names_map[sym] for sym in combined.cores
        if names_map.get(sym, '').startswith('mx.')
    ]
    # In distributed mode, only inject mx cores belonging to this rank's partition
    if world_size > 1:
        local_mx_indices = []
        for i, name in enumerate(mx_core_names):
            sym = [s for s, n in names_map.items() if n == name][0]
            if sym in local_core_set:
                local_mx_indices.append(i)

    def data_fn(step):
        x = np.random.randn(BATCH_SIZE, actual_qubits).astype(np.float32)
        Mx_list, _ = data_gen.generate(x, K=PHYS_DIM, ret_type='TNTensor')
        if world_size > 1:
            for i in local_mx_indices:
                combined[mx_core_names[i]] = Mx_list[i]
        else:
            for i, name in enumerate(mx_core_names):
                combined[name] = Mx_list[i]

    # --- Train ---
    optimizer = create_optimizer("sgdg", local_params, backend=backend, lr=LR / max(world_size, 1))
    loss_history = []
    step_times = []

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
            dist.barrier()
        else:
            loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
            optimizer.step(list(grads))

        t_step = time.time() - t0
        lv = float(loss_val)
        loss_history.append(lv)
        step_times.append(t_step)

        if rank == 0 and (step % LOG_EVERY == 0 or step == 1):
            print(f"  Step {step:4d}/{N_STEPS}  loss={lv:.6f}  ({t_step:.2f}s)")

    if world_size > 1:
        dist.barrier()

    # Report peak memory per rank
    def _peak_rss_mb():
        """Read VmHWM (RSS high water mark) from /proc — actual physical memory peak."""
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmHWM:'):
                        return int(line.split()[1]) / 1024  # kB → MB
        except Exception:
            pass
        return 0.0

    my_peak = _peak_rss_mb()
    print(f"  [Rank {rank}] Peak RSS: {my_peak:.0f} MB")

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        # Step time stats (skip first 2 warmup steps)
        warm = step_times[2:] if len(step_times) > 2 else step_times
        avg_t = np.mean(warm) if warm else 0
        std_t = np.std(warm) if warm else 0
        print(f"\nDone. Initial={loss_history[0]:.6f}  Final={loss_history[-1]:.6f}  "
              f"Reduced={loss_history[0] - loss_history[-1]:.6f}")
        print(f"  Avg step time (skip first 2): {avg_t:.3f} ± {std_t:.3f} s")

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
