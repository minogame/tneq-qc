"""
Distributed quadratic form training example.

Structure: circuit + mps + mx + mps† + circuit†  (same as train_quadratic.py)

Uses DistributedTrainer with torchrun for single-node multi-process training.
Each process gets a partition of the QCTN graph and performs local contraction,
then hierarchical reduction combines partial results across processes.

Usage:
    # Single node, 2 processes:
    torchrun --nproc_per_node=2 examples/train_dist.py

    # Or with explicit settings:
    torchrun --standalone --nnodes=1 --nproc_per_node=2 examples/train_dist.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.core.engine_common import EngineCommon
from tneq_qc.utils.graph_generators import QCTNHelper
from tneq_qc.utils.data_generator import DataGenerator

from tneq_qc.distributed import DistributedTrainer, DistributedConfig
from tneq_qc.distributed.optim import DistributedSGDG, LRScheduler

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_QUBITS   = 4
BOND_DIM   = 2
PHYS_DIM   = 2       # K for DataGenerator (== physical dim of mx core)
BATCH_SIZE = 128
N_STEPS    = 200
LR         = 0.01
LOG_EVERY  = 10

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw(t):
    """Return the underlying backend tensor regardless of TNTensor wrapping."""
    return t.tensor if isinstance(t, TNTensor) else t


def print_qctn_info(qctn: QCTN, label: str = "QCTN", rank: int = 0):
    if rank != 0:
        return
    print(f"\n[{label}]  nqubits={qctn.nqubits}  ncores={qctn.ncores}")
    print("  Core shapes:")
    for c in qctn.cores:
        t = qctn.cores_weights[c]
        raw = _raw(t)
        shape = tuple(raw.shape)
        print(f"    '{c}': {shape}")


# ---------------------------------------------------------------------------
# Build the QCTN graph string for the full quadratic structure
# ---------------------------------------------------------------------------

def build_quadratic_graph(n_qubits, bond_dim, phys_dim):
    """Build the combined graph: circuit + mps + mx + mps† + circuit†.

    Returns the graph string and a list of (prefix, sub_qctn) tuples
    for QCTN.concat.
    """
    backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')

    # Circuit state (fixed)
    graph_circuit = QCTNHelper.circuit_state(n_qubits, phys_dim=phys_dim)
    circuit = QCTN(graph_circuit, backend=backend).auto_init()
    # Fill with alternating 0/1
    for c in circuit.cores:
        core = circuit.cores_weights[c]
        shape = tuple(_raw(core).shape)
        dtype = _raw(core).dtype
        n = 1
        for d in shape:
            n *= d
        flat = torch.zeros(n, dtype=dtype)
        for i in range(n):
            flat[i] = float(i % 2)
        circuit.cores_weights[c] = backend.convert_to_tensor(flat.reshape(shape))

    # MPS (trainable, QR init)
    graph_mps = QCTNHelper.mps(n_qubits, bond_dim=bond_dim, phys_dim=phys_dim)
    mps = QCTN(graph_mps, backend=backend).auto_init()
    for c in mps.cores:
        _raw(mps.cores_weights[c]).requires_grad_(True)

    # Measurement matrix Mx (data-driven, replaced each step)
    graph_mx = QCTNHelper.measure_matrix(n_qubits, phys_dim=phys_dim)
    mx = QCTN(graph_mx, backend=backend).auto_init()

    # mps†
    mps_h = mps.hermit()

    # circuit† (bra)
    graph_circ_bra = QCTNHelper.circuit_bra(n_qubits, phys_dim=phys_dim)
    circ_bra = QCTN(graph_circ_bra, backend=backend).auto_init()
    for c in circuit.cores:
        t = circuit.cores_weights[c]
        circ_bra.cores_weights[c] = TNTensor(
            t.tensor.conj() if isinstance(t, TNTensor) else t.conj(),
            scale=t.scale if isinstance(t, TNTensor) else 1.0,
        )

    # Concat all parts
    combined = QCTN.concat([
        ('cs', circuit),
        ('mps', mps),
        ('mx', mx),
        ('mps_h', mps_h),
        ('cs_t', circ_bra),
    ])

    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Initialize PyTorch distributed
    if 'RANK' in os.environ:
        dist.init_process_group(backend='gloo')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        print("WARNING: Not launched with torchrun. Running single-process.")
        print("Use: torchrun --nproc_per_node=2 examples/train_dist.py")

    num_nodes = int(os.environ.get('NNODES', 1))
    node_rank = int(os.environ.get('NODE_RANK', 0))

    if rank == 0:
        print("=" * 60)
        print("Distributed Quadratic Form Training")
        print("=" * 60)
        print(f"  World size    : {world_size}")
        print(f"  Nodes         : {num_nodes}")
        print(f"  N_QUBITS      : {N_QUBITS}")
        print(f"  BOND_DIM      : {BOND_DIM}")
        print(f"  PHYS_DIM      : {PHYS_DIM}")
        print(f"  BATCH_SIZE    : {BATCH_SIZE}")
        print(f"  N_STEPS       : {N_STEPS}")
        print(f"  LR            : {LR}")

    # ------------------------------------------------------------------
    # 1. Build QCTN graph (same on every rank)
    # ------------------------------------------------------------------
    combined = build_quadratic_graph(N_QUBITS, BOND_DIM, PHYS_DIM)
    graph_str = combined.graph
    print_qctn_info(combined, "Combined QCTN", rank)

    if rank == 0:
        print(f"  core_names: {combined.core_names}")
        print(f"  graph:\n{graph_str}")

    # ------------------------------------------------------------------
    # 2. Create DistributedConfig + DistributedTrainer
    # ------------------------------------------------------------------
    config = DistributedConfig(
        backend_type='pytorch',
        device='cpu',
        strategy_mode='balanced',
        mx_K=PHYS_DIM,

        qctn_graph=graph_str,

        comm_backend='torch' if world_size > 1 else 'auto',
        use_distributed=world_size > 1,
        rank=rank,
        world_size=world_size,
        node_rank=node_rank,
        num_nodes=num_nodes,

        partition_strategy='layer',

        max_steps=N_STEPS,
        log_interval=LOG_EVERY,
        learning_rate=LR,
        optimizer='sgdg',
        momentum=0.9,
        stiefel=True,
    )

    trainer = DistributedTrainer(config)

    print(f"[Rank {rank}] Trainer initialized. "
          f"QCTN cores: {len(trainer.qctn.cores)}, "
          f"nqubits: {trainer.qctn.nqubits}")

    if world_size > 1:
        dist.barrier()

    # ------------------------------------------------------------------
    # 3. Generate training data (same data on all ranks for now)
    # ------------------------------------------------------------------
    if rank == 0:
        print()
        print("=" * 60)
        print("Generating training data ...")
        print("=" * 60)

    N_batches = 50
    K = PHYS_DIM
    num_qubits = trainer.qctn.nqubits

    train_data_list = []
    for i in range(N_batches):
        x_train = torch.empty(BATCH_SIZE, num_qubits).normal_(mean=0.0, std=1.0)
        Mx_train, _ = trainer.engine.generate_data(x_train, K=K, ret_type='TNTensor')
        train_data_list.append({'measure_input_list': Mx_train})

    if rank == 0:
        print(f"  Generated {N_batches} batches, "
              f"each Mx shape: {train_data_list[0]['measure_input_list'][0].shape}")

    # Circuit states
    def generate_circuit_states_list(nq, k, device='cpu'):
        states = [torch.zeros(k, device=device) for _ in range(nq)]
        for s in states:
            s[-1] = 1.0
        return states

    circuit_states_list = generate_circuit_states_list(num_qubits, K)

    if world_size > 1:
        dist.barrier()

    # ------------------------------------------------------------------
    # 4. Create optimizer
    # ------------------------------------------------------------------
    optimizer = DistributedSGDG(
        lr=LR / world_size,
        momentum=0.9,
        stiefel=True,
    )

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    if rank == 0:
        print()
        print("=" * 60)
        print(f"Training  steps={N_STEPS}  batch={BATCH_SIZE}  "
              f"lr={LR}  world_size={world_size}")
        print("=" * 60)

    loss_history = []

    for step in tqdm(range(1, N_STEPS + 1),
                     desc=f"Rank {rank}",
                     disable=(rank != 0)):
        data = train_data_list[(step - 1) % N_batches]

        loss = trainer.engine.train_step(
            circuit_states_list=circuit_states_list,
            measure_input_list=data['measure_input_list'],
            optimizer=optimizer,
            measure_is_matrix=True,
        )

        loss_history.append(loss)

        if rank == 0 and (step % LOG_EVERY == 0 or step == 1):
            print(f"  Step {step:4d}/{N_STEPS}  loss={loss:.6f}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        print()
        print("=" * 60)
        print("Training complete")
        print(f"  Initial loss : {loss_history[0]:.6f}")
        print(f"  Final   loss : {loss_history[-1]:.6f}")
        print(f"  Loss reduced : {loss_history[0] - loss_history[-1]:.6f}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 7. Save (rank 0 only)
    # ------------------------------------------------------------------
    save_path = f"checkpoints/dist_quadratic_{N_QUBITS}q_K{PHYS_DIM}_B{BOND_DIM}_w{world_size}.safetensors"
    trainer.engine.save_cores_distributed(
        file_path=save_path,
        metadata={
            'n_qubits': str(N_QUBITS),
            'bond_dim': str(BOND_DIM),
            'phys_dim': str(PHYS_DIM),
            'n_steps': str(N_STEPS),
            'world_size': str(world_size),
            'final_loss': f"{loss_history[-1]:.6f}" if loss_history else "N/A",
        },
    )
    if rank == 0:
        print(f"  Saved to: {save_path}")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
