# Distributed Training Module Documentation

> Distributed — Multi-process parallel training support

---

## 1. Overview

When the tensor network scale (number of qubits x bond dimension) exceeds single-node capacity, the `tneq_qc.distributed` module provides multi-process distributed training support. Core components:

| Component | Location | Responsibility |
|---|---|---|
| **CommBase** | `distributed/comm/` | Communication backend abstraction (MPI / torch.distributed) |
| **EngineDistributed** | `distributed/engine/` | Inherits EngineCommon, adds partitioning and AllReduce |
| **PartitionConfig** | `distributed/engine/` | Partition strategy configuration |
| **DistributedSGDG** | `distributed/optim/` | Distributed optimizer |

---

## 2. Communication Backend

### 2.1 Backend Selection

The framework provides two communication backends:

| Backend | Class | Use Case |
|---|---|---|
| **MPI** | `CommMPI` | HPC clusters (Fugaku, etc.), cross-node |
| **torch.distributed** | `CommTorch` | GPU clusters, launched via torchrun |

```python
from tneq_qc.distributed.comm import get_comm_backend

# Auto-detect
comm = get_comm_backend(backend='auto', rank=rank, world_size=world_size)

# Specify torch backend
comm = get_comm_backend(backend='torch', rank=rank, world_size=world_size)

# Specify MPI backend
comm = get_comm_backend(backend='mpi')
```

### 2.2 CommBase Interface

All communication backends implement the following interface:

```python
class CommBase:
    @property
    def rank(self) -> int: ...          # Current process rank
    @property
    def world_size(self) -> int: ...    # Total number of processes

    def broadcast(self, data, root=0): ...
    def allreduce(self, data, op=ReduceOp.SUM): ...
    def barrier(self): ...
    def send(self, data, dest, tag=0): ...
    def recv(self, data, source, tag=0): ...
```

### 2.3 Mock Backend

For testing and single-process debugging:

```python
from tneq_qc.distributed.comm import MockCommMPI, MockCommTorch

comm = MockCommMPI()     # rank=0, world_size=1
comm = MockCommTorch()   # rank=0, world_size=1
```

---

## 3. EngineDistributed

### 3.1 Creation

```python
from tneq_qc.distributed import EngineDistributed
from tneq_qc.distributed.engine.distributed_engine import PartitionConfig

engine = EngineDistributed(
    backend=backend,
    strategy_mode='full',
    comm=comm,
    partition_config=PartitionConfig(
        strategy='layer',           # Partition strategy
        num_partitions=world_size,  # Number of partitions (usually = number of processes)
    ),
)
```

### 3.2 Initialization

```python
engine.init_distributed(combined)
```

`init_distributed` performs the following:
1. Distributes QCTN cores across processes according to `PartitionConfig`
2. Builds a partition plan (`DistributedContractPlan`)
3. Sets up the communication topology

### 3.3 Training

After initialization, the training loop is nearly identical to single-machine `EngineCommon`:

```python
optimizer = SGDG(combined.parameters(), backend, lr=LR / world_size)
data_fn = make_data_fn(data_gen, combined, batch_size=BATCH_SIZE, K=PHYS_DIM)

for step in range(1, N_STEPS + 1):
    data_fn(step)
    loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
    optimizer.step(list(grads))
```

**Note**: In distributed training, the learning rate typically needs to be divided by `world_size`.

---

## 4. PartitionConfig

Configures the partition strategy:

```python
PartitionConfig(
    strategy='layer',       # 'layer' = partition by layer (recommended)
    num_partitions=4,       # Number of partitions
)
```

| Parameter | Description |
|---|---|
| `strategy` | `'layer'`: Partition by segments of the quadratic structure (cs/tn/mx/tn_h/cs_t) |
| `num_partitions` | Number of partitions, usually equal to `world_size` |

---

## 5. Distributed Optimization

### 5.1 AllReduceGrad

Gradient aggregation utility:

```python
from tneq_qc.distributed import AllReduceGrad

allreduce = AllReduceGrad(comm)
allreduce.reduce(gradients)  # Sum and average gradients across all processes
```

### 5.2 DistributedSGDG

Distributed SGDG optimizer (with built-in AllReduce):

```python
from tneq_qc.distributed import DistributedSGDG

optimizer = DistributedSGDG(params, backend, comm, lr=0.01)
```

---

## 6. Complete Example

```python
"""Distributed quadratic training"""
import torch
import torch.distributed as dist
from tneq_qc import QCTN, BackendFactory, Quadratic, DataGenerator, make_data_fn, SGDG
from tneq_qc.distributed import EngineDistributed
from tneq_qc.distributed.engine.distributed_engine import PartitionConfig
from tneq_qc.distributed.comm import get_comm_backend

# 1. Initialize distributed environment
dist.init_process_group(backend='gloo')
rank = dist.get_rank()
world_size = dist.get_world_size()

# 2. Create backend and model
backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')
data_gen = DataGenerator(backend, mx_K=2)

model = Quadratic(nqubits=4, bond_dim=2, phys_dim=2, backend=backend).auto_init()
model._submodules['mps'].requires_grad_(True)
combined = model.build()

# 3. Create distributed engine
comm = get_comm_backend(backend='torch', rank=rank, world_size=world_size)
engine = EngineDistributed(
    backend=backend,
    strategy_mode='full',
    comm=comm,
    partition_config=PartitionConfig(strategy='layer', num_partitions=world_size),
)
engine.init_distributed(combined)

# 4. Train
optimizer = SGDG(combined.parameters(), backend, lr=0.01 / world_size)
data_fn = make_data_fn(data_gen, combined, batch_size=128, K=2)

for step in range(1, 201):
    data_fn(step)
    loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
    optimizer.step(list(grads))

    if rank == 0 and step % 10 == 0:
        print(f"Step {step}  loss={float(loss_val):.6f}")

# 5. Cleanup
dist.barrier()
dist.destroy_process_group()
```

Launch command:

```bash
torchrun --nproc_per_node=2 train_dist.py
```

---

## 7. Important Notes

1. **Process Synchronization**: Use `dist.barrier()` or `comm.barrier()` at critical points (after initialization, before training ends) to ensure synchronization
2. **Learning Rate Scaling**: In distributed training, `lr` typically needs to be divided by `world_size`
3. **Logging Control**: Only print logs when `rank == 0` to avoid duplicate output
4. **Model Saving**: Only save checkpoints when `rank == 0`
5. **Random Seeds**: Set the same random seed to ensure consistent initialization

```python
torch.manual_seed(42)
np.random.seed(42)
```
