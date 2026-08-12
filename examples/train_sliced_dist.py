"""Distributed BornMachine training via cotengra index slicing.

Uses ``EngineSliced`` — the slicing-based, data-parallel paradigm:
the full network is replicated on every rank, cotengra cuts the contraction
into independent slices, each rank evaluates a disjoint slice subset, and the
partial sums (and gradients) are all-reduced.  No model/graph partitioning.

Run:
    torchrun --nproc_per_node=2 examples/train_sliced_dist.py

Single-process (falls back to all slices on one rank):
    python examples/train_sliced_dist.py
    python -m examples.train_sliced_dist
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.distributed as dist

from tneq_qc import QCTN, BackendFactory, BornMachine, DataGenerator, create_optimizer
from tneq_qc.distributed import EngineSliced
from tneq_qc.utils.graph_generators import QCTNHelper

N_QUBITS = 4
BOND_DIM = 4
PHYS_DIM = 2
BATCH_SIZE = 128
N_STEPS = 100
LR = 0.01
LOG_EVERY = 10

torch.manual_seed(42)
np.random.seed(42)


def main():
    if "RANK" in os.environ:
        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        print("WARNING: not launched with torchrun; running single-process.")

    backend = BackendFactory.create_backend("pytorch", device="cpu", dtype="float32")
    data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

    graph = QCTNHelper.mps(N_QUBITS, bond_dim=BOND_DIM, phys_dim=PHYS_DIM)
    model = BornMachine(graph, PHYS_DIM, backend=backend).auto_init(orthogonal=True)
    model._submodules["tn"].requires_grad_(True)
    combined = model.build()

    mx_names = [n for n in combined.cores
                if combined.core_names.get(n, "").startswith("mx.")]

    # EngineSliced: no partitioning, no init_distributed — just slice + reduce.
    engine = EngineSliced(backend=backend)

    if rank == 0:
        print("=" * 60)
        print("Sliced (cotengra) Born Machine Training")
        print(f"  world_size={world_size}  N={N_QUBITS}  B={BOND_DIM}  K={PHYS_DIM}")
        print("=" * 60)

    # Gradients are SUMMED across ranks (not averaged) → no lr/world_size scaling.
    optimizer = create_optimizer("sgdg", combined.parameters(), backend=backend, lr=LR)
    loss_history = []

    for step in range(1, N_STEPS + 1):
        # Rank 0 generates data, broadcast so every rank holds identical Mx.
        x = np.random.randn(BATCH_SIZE, N_QUBITS).astype(np.float32)
        if world_size > 1:
            xt = torch.from_numpy(x)
            dist.broadcast(xt, src=0)
            x = xt.numpy()
        mx_list, _ = data_gen.generate(x, K=PHYS_DIM, ret_type="TNTensor")
        for name, mx in zip(mx_names, mx_list):
            combined[name] = mx

        loss_val, grads = engine.contract_for_gradient(combined, target=1, loss="nll")
        optimizer.step(list(grads))
        lv = float(loss_val)
        loss_history.append(lv)
        if rank == 0 and (step % LOG_EVERY == 0 or step == 1):
            planner = getattr(combined, "_cotengra_planner", None)
            ns = planner.nslices if planner else "?"
            print(f"  Step {step:4d}/{N_STEPS}  loss={lv:.6f}  nslices={ns}")

    if rank == 0:
        print(f"\nDone. Initial={loss_history[0]:.6f}  Final={loss_history[-1]:.6f}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
