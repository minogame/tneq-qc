"""Distributed quadratic form training example.

Structure: circuit + mps + mx + mps_h + circuit_bra

Uses EngineDistributed with torchrun for single-node multi-process training.

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
    QCTN, BackendFactory, BornMachine,
    DataGenerator, make_data_fn, create_optimizer,
)
from tneq_qc.distributed import EngineDistributed
from tneq_qc.distributed.engine.distributed_engine import PartitionConfig
from tneq_qc.utils.graph_generators import QCTNHelper

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
    if 'RANK' in os.environ:
        dist.init_process_group(backend='gloo')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        print("WARNING: Not launched with torchrun. Running single-process.")

    if rank == 0:
        print("=" * 60)
        print("Distributed Born Machine Training")
        print(f"  world_size={world_size}  N={N_QUBITS}  B={BOND_DIM}  K={PHYS_DIM}")
        print("=" * 60)

    backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')
    data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

    # Build model
    graph = QCTNHelper.mps(N_QUBITS, bond_dim=BOND_DIM, phys_dim=PHYS_DIM)
    model = BornMachine(graph, PHYS_DIM, backend=backend).auto_init(orthogonal=True)
    model._submodules['tn'].requires_grad_(True)
    combined = model.build()

    # Create engine
    from tneq_qc.distributed.comm import get_comm_backend
    comm = get_comm_backend(
        backend='torch' if world_size > 1 else 'auto',
        rank=rank, world_size=world_size,
    )
    engine = EngineDistributed(
        backend=backend,
        strategy='row_priority',
        comm=comm,
        partition_config=PartitionConfig(strategy='layer', num_partitions=world_size),
    )
    engine.init_distributed(combined)

    if world_size > 1:
        dist.barrier()

    # Train
    optimizer = create_optimizer(
        "sgdg", combined.parameters(), backend=backend, lr=LR / world_size
    )
    data_fn = make_data_fn(data_gen, combined, batch_size=BATCH_SIZE, K=PHYS_DIM)
    loss_history = []

    if rank == 0:
        print(f"\nTraining  steps={N_STEPS}  batch={BATCH_SIZE}  lr={LR}")

    for step in range(1, N_STEPS + 1):
        data_fn(step)
        loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
        optimizer.step(list(grads))
        lv = float(loss_val)
        loss_history.append(lv)
        if rank == 0 and (step % LOG_EVERY == 0 or step == 1):
            print(f"  Step {step:4d}/{N_STEPS}  loss={lv:.6f}")

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        print(f"\nDone. Initial={loss_history[0]:.6f}  Final={loss_history[-1]:.6f}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
