"""Distributed quadratic form training example.

Structure: circuit + mps + mx + mps_h + circuit_bra

Uses DistributedTrainer with torchrun for single-node multi-process training.

Usage:
    torchrun --nproc_per_node=2 examples/train_dist.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import numpy as np

from tneq_qc import (
    QCTN, BackendFactory, Quadratic,
    DataGenerator, make_data_fn,
)
from tneq_qc.distributed import DistributedTrainer, DistributedConfig

N_QUBITS   = 4
BOND_DIM   = 2
PHYS_DIM   = 2
BATCH_SIZE = 128
N_STEPS    = 200
LR         = 0.01
LOG_EVERY  = 10

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
    if 'RANK' in os.environ:
        dist.init_process_group(backend='gloo')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        print("WARNING: Not launched with torchrun. Running single-process.")

    num_nodes = int(os.environ.get('NNODES', 1))
    node_rank = int(os.environ.get('NODE_RANK', 0))

    if rank == 0:
        print("=" * 60)
        print("Distributed Quadratic Form Training")
        print(f"  world_size={world_size}  N={N_QUBITS}  B={BOND_DIM}  K={PHYS_DIM}")
        print("=" * 60)

    backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')
    data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

    # Build model using Quadratic
    model = Quadratic(nqubits=N_QUBITS, bond_dim=BOND_DIM, phys_dim=PHYS_DIM,
                      backend=backend).auto_init()
    init_circuit_01(model._submodules['circuit'], backend)
    model._submodules['mps'].requires_grad_(True)
    combined = model.build()

    # Distributed trainer
    config = DistributedConfig(
        backend_type='pytorch',
        device='cpu',
        strategy_mode='full',

        qctn=combined,

        comm_backend='torch' if world_size > 1 else 'auto',
        use_distributed=world_size > 1,
        rank=rank,
        world_size=world_size,
        node_rank=node_rank,
        num_nodes=num_nodes,

        partition_strategy='layer',
        max_steps=N_STEPS,
        log_interval=LOG_EVERY,
        learning_rate=LR / world_size,
        optimizer='sgdg',
        momentum=0.9,
        stiefel=True,
    )

    trainer = DistributedTrainer(config)

    if world_size > 1:
        dist.barrier()

    data_fn = make_data_fn(data_gen, combined, batch_size=BATCH_SIZE, K=PHYS_DIM)

    if rank == 0:
        print(f"\nTraining  steps={N_STEPS}  batch={BATCH_SIZE}  lr={LR}")

    loss_history = trainer.fit(target=1, loss='nll', data_fn=data_fn)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        print(f"\nDone. Initial={loss_history[0]:.6f}  Final={loss_history[-1]:.6f}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
