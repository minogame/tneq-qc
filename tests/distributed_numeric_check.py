"""Numerical check for EngineDistributed.

Run with torchrun, for example:

    torchrun --nproc_per_node=2 tests/distributed_numeric_check.py
    torchrun --nproc_per_node=4 tests/distributed_numeric_check.py

The check compares a distributed contraction against the single-process
RowPriority contraction for the same initialized model and the same Mx batch.
"""

from __future__ import annotations

import math
import os
import sys
import importlib.util

import numpy as np
import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from tneq_qc import BackendFactory, DataGenerator, EngineCommon, QCTN  # noqa: E402
from tneq_qc.core.tn_tensor import TNTensor  # noqa: E402
from tneq_qc.distributed import EngineDistributed  # noqa: E402
from tneq_qc.distributed.comm import get_comm_backend  # noqa: E402
from tneq_qc.distributed.engine.distributed_engine import PartitionConfig  # noqa: E402
from tneq_qc.modules.small import MeasureMatrix, State  # noqa: E402


def _load_dist_example_module():
    module_path = os.path.join(REPO_ROOT, "examples", "train_mps_brickwall_dist.py")
    spec = importlib.util.spec_from_file_location(
        "train_mps_brickwall_dist_for_numeric_check",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load distributed example from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_dist_example = _load_dist_example_module()
generate_mps_brickwall_graph = _dist_example.generate_mps_brickwall_graph
init_measure_identity = _dist_example.init_measure_identity


TOTAL_QUBITS = int(os.environ.get("TOTAL_QUBITS", "7"))
BLOCK_QUBITS = int(os.environ.get("BLOCK_QUBITS", "7"))
OVERLAP = int(os.environ.get("OVERLAP", "1"))
PHYS_DIM = int(os.environ.get("PHYS_DIM", "2"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "3"))
ATOL = float(os.environ.get("ATOL", "1e-4"))
RTOL = float(os.environ.get("RTOL", "1e-4"))


def _build_combined(backend):
    tn_graph, actual_qubits = generate_mps_brickwall_graph(
        TOTAL_QUBITS, BLOCK_QUBITS, OVERLAP, PHYS_DIM
    )
    custom_tn = QCTN(tn_graph, backend=backend).auto_init(orthogonal=True)
    state = State(actual_qubits, PHYS_DIM, backend).auto_init()
    mx = init_measure_identity(MeasureMatrix(actual_qubits, PHYS_DIM, backend), backend)
    combined = QCTN.concat([
        ("state", state),
        ("tn", custom_tn),
        ("mx", mx),
        ("tn_h", custom_tn.hermit()),
        ("state_t", state.bra()),
    ])
    return combined, custom_tn, actual_qubits


def _mx_core_names(combined):
    return [
        combined.core_names[sym]
        for sym in combined.cores
        if combined.core_names.get(sym, "").startswith("mx.")
    ]


def _inject_deterministic_mx(combined, backend, actual_qubits):
    data_gen = DataGenerator(backend, mx_K=PHYS_DIM)
    values = np.linspace(
        -0.75,
        0.75,
        num=BATCH_SIZE * actual_qubits,
        dtype=np.float32,
    ).reshape(BATCH_SIZE, actual_qubits)
    mx_list, _ = data_gen.generate(values, K=PHYS_DIM, ret_type="TNTensor")
    for name, mx in zip(_mx_core_names(combined), mx_list):
        combined[name] = mx


def _build_partitions(combined, custom_tn, world_size):
    tn_cores = []
    tn_h_cores = []
    state_cores = []
    state_t_cores = []
    mx_cores = []

    tn_i = tn_h_i = state_i = state_t_i = mx_i = 0
    for sym in combined.cores:
        name = combined.core_names.get(sym, sym)
        if name.startswith("tn_h."):
            tn_h_cores.append((sym, tn_h_i))
            tn_h_i += 1
        elif name.startswith("tn."):
            tn_cores.append((sym, tn_i))
            tn_i += 1
        elif name.startswith("state_t."):
            state_t_cores.append((sym, state_t_i))
            state_t_i += 1
        elif name.startswith("state."):
            state_cores.append((sym, state_i))
            state_i += 1
        elif name.startswith("mx."):
            mx_cores.append((sym, mx_i))
            mx_i += 1

    tn_core_qubits = {}
    for entry in custom_tn.adjacency_table:
        core_name = entry["core_name"]
        if core_name not in custom_tn.cores:
            continue
        core_idx = custom_tn.cores.index(core_name)
        qubits = set()
        for key in ("in_edge_list", "out_edge_list"):
            for edge in entry.get(key, []):
                qubit_idx = edge.get("qubit_idx", -1)
                if qubit_idx >= 0:
                    qubits.add(qubit_idx)
        tn_core_qubits[core_idx] = qubits

    split_arrays = np.array_split(np.arange(custom_tn.ncores), world_size)
    tn_splits = []
    for arr in split_arrays:
        if len(arr) == 0:
            raise RuntimeError(
                f"world_size={world_size} is larger than tn cores={custom_tn.ncores}"
            )
        tn_splits.append((int(arr[0]), int(arr[-1]) + 1))

    partition_qubit_ranges = []
    for start, end in tn_splits:
        qubits = set()
        for core_idx in range(start, end):
            qubits.update(tn_core_qubits.get(core_idx, set()))
        partition_qubit_ranges.append((min(qubits), max(qubits)))

    partitions = [[] for _ in range(world_size)]

    def add_by_tn_index(items):
        for sym, idx in items:
            for part_idx, (start, end) in enumerate(tn_splits):
                if start <= idx < end:
                    partitions[part_idx].append(sym)
                    break

    def qubit_to_partition(qubit_idx):
        for part_idx, (qmin, qmax) in enumerate(partition_qubit_ranges):
            if qmin <= qubit_idx <= qmax:
                return part_idx
        return world_size - 1

    add_by_tn_index(tn_cores)
    add_by_tn_index(tn_h_cores)
    for sym, qubit_idx in state_cores + state_t_cores + mx_cores:
        partitions[qubit_to_partition(qubit_idx)].append(sym)
    return partitions


def _effective_tensor(value):
    if isinstance(value, TNTensor):
        return value.tensor * value.scale
    return value


def main():
    if "RANK" not in os.environ:
        raise RuntimeError("Run this check with torchrun.")

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(1234)
    np.random.seed(1234)

    backend = BackendFactory.create_backend("pytorch", device="cpu", dtype="complex64")
    combined, custom_tn, actual_qubits = _build_combined(backend)
    _inject_deterministic_mx(combined, backend, actual_qubits)

    single_engine = EngineCommon(backend=backend, strategy="row_priority")
    single_result = _effective_tensor(single_engine.contract(combined)).detach()

    partitions = _build_partitions(combined, custom_tn, world_size)
    comm = get_comm_backend(backend="torch", rank=rank, world_size=world_size)
    dist_engine = EngineDistributed(
        backend=backend,
        strategy="row_priority",
        comm=comm,
        partition_config=PartitionConfig(strategy="layer", num_partitions=world_size),
    )
    dist_engine.init_distributed(combined, partitions=partitions)
    distributed_result = _effective_tensor(dist_engine.contract_distributed()).detach()

    if single_result.shape != distributed_result.shape:
        raise RuntimeError(
            f"shape mismatch: single={tuple(single_result.shape)} "
            f"distributed={tuple(distributed_result.shape)}"
        )

    abs_diff = torch.max(torch.abs(single_result - distributed_result)).real
    ref_norm = torch.max(torch.abs(single_result)).real
    allowed = ATOL + RTOL * ref_norm
    payload = torch.stack([abs_diff, allowed, ref_norm]).to(torch.float64)
    dist.all_reduce(payload, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(
            "distributed numeric check: "
            f"world_size={world_size} shape={tuple(single_result.shape)} "
            f"max_abs_diff={payload[0].item():.6e} "
            f"allowed={payload[1].item():.6e} "
            f"ref_norm={payload[2].item():.6e}"
        )

    if payload[0] > payload[1]:
        raise AssertionError(
            f"distributed result differs from single process: "
            f"diff={payload[0].item()} allowed={payload[1].item()}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
