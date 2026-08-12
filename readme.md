# tneq-qc

Tensor Network Engine toward Quantum (Quantum Circuit) — a lightweight library for building, contracting, and training tensor network models on quantum circuit topologies.

## Overview

**tneq-qc** represents quantum circuits as tensor networks using an ASCII graph notation. Cores (tensors) are arranged on qubit rows with explicit bond dimensions, forming structures like Matrix Product States (MPS), brickwall circuits, and tree networks. The library provides:

- **Graph-driven construction**: Define tensor network topology via ASCII strings; shapes and contraction paths are inferred automatically.
- **Scaled tensors**: `TNTensor` stores `tensor × scale` to prevent under/overflow during deep contractions.
- **Composable modules**: Small modules (`MPS`, `CircuitState`, `MeasureMatrix`) compose into application-level models (`Quadratic`, `TNEQ`, `Encoding`) via `register_module` / `QCTN.concat`.
- **Automatic differentiation**: `EngineCommon.contract_for_gradient` computes forward contraction and parameter gradients in one call.
- **Multi-backend**: PyTorch and JAX backends with a unified `BackendFactory` interface.
- **Distributed training**: `EngineDistributed` partitions the tensor network across processes using `torch.distributed` (gloo / nccl).

## Architecture

```bash
tneq_qc/
├── core/               # Fundamental building blocks
│   ├── qctn.py         # QCTN class — graph parsing, core management, concat
│   ├── tn_tensor.py    # TNTensor — scaled tensor with arithmetic & gradient proxy
│   ├── tn_graph.py     # TNGraph — adjacency parsing from ASCII graphs
│   └── engine_common.py # EngineCommon — contraction + gradient computation
├── modules/
│   ├── small.py        # MPS, CircuitState, MeasureMatrix (leaf modules)
│   └── app.py          # Quadratic, TNEQ, Encoding, PlainMPS (application modules)
├── contractor/         # Contraction strategies
│   ├── einsum_strategy.py      # Direct opt_einsum contraction
│   ├── row_priority_strategy.py # Row-wise contraction for large networks
│   └── compiler.py     # StrategyCompiler — auto-selects optimal strategy
├── losses/             # Extensible loss functions
│   ├── builtin.py      # MSE, MAE, NLL, Fidelity, DiagonalMSE
│   └── registry.py     # LossRegistry + @register_loss decorator
├── optim/              # Optimizers
│   ├── optimizers.py   # Adam, SGD, SGDG, Momentum, RMSProp
│   └── lr_scheduler.py # StepLRScheduler
├── backends/           # Compute backends
│   ├── backend_pytorch.py  # PyTorch backend
│   ├── backend_jax.py      # JAX backend
│   └── backend_factory.py  # BackendFactory.create_backend()
├── distributed/        # Distributed training
│   ├── comm/           # Communication backends (MPI, torch.distributed)
│   ├── engine/         # EngineDistributed, PartitionConfig
│   └── optim/          # DistributedSGDG, AllReduceGrad
└── utils/
    ├── graph_generators.py  # QCTNHelper — generate MPS/circuit/measure graphs
    └── data_generator.py    # DataGenerator — Hermite feature maps & Mx matrices
```

## Quick Start

### Define a Tensor Network via ASCII Graph

Each row is a qubit. Letters are cores (tensors), numbers are bond dimensions.

```python
from tneq_qc import QCTN, BackendFactory

backend = BackendFactory.create_backend('pytorch', device='cpu')

graph = """\
-2-A-4-B-4-C-2-
-2-A-4-B-4-C-2-
-2-A-4-B-4-C-2-"""

qctn = QCTN(graph, backend=backend).auto_init()
print(qctn)
# QCTN(nqubits=3, cores=[A(2,4,2,4,2,4), B(4,4,4,4,4,4), C(4,2,4,2,4,2)])
```

### Forward Contraction

```python
from tneq_qc import EngineCommon

engine = EngineCommon(backend)
result = engine.contract(qctn)
```

### Train a Quadratic Model

The quadratic form `⟨circuit | mps† · Mx · mps | circuit⟩` computes expectation values. `mps` is the trainable component; `Mx` carries data.

```python
from tneq_qc import (
    Quadratic, EngineCommon, BackendFactory,
    DataGenerator, make_data_fn, SGDG,
)

backend = BackendFactory.create_backend('pytorch', device='cpu')
data_gen = DataGenerator(backend, mx_K=2)

# Build model
graph = QCTNHelper.mps(4, bond_dim=2, phys_dim=2)
model = Quadratic(graph, 2, backend=backend).auto_init(orthogonal=True)
model._submodules['mps'].requires_grad_(True)
combined = model.build()   # returns ready-to-contract QCTN

# Prepare data injection function
data_fn = make_data_fn(data_gen, combined, batch_size=128, K=2)

# Train
engine = EngineCommon(backend)
optimizer = SGDG(combined.parameters(), backend, lr=0.01)

for step in range(1, 201):
    data_fn(step)  # injects Mx matrices into combined
    loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
    optimizer.step(list(grads))
```

### Train a TNEQ (Student-Teacher)

Two MPS networks connected via trace — train one to approximate the other.

```python
from tneq_qc import QCTN, EngineCommon, BackendFactory, SGDG

backend = BackendFactory.create_backend('pytorch', device='cpu')

# Teacher (frozen)
teacher = QCTN(graph_teacher, backend=backend).auto_init()

# Student (trainable)
student = QCTN(graph_student, backend=backend).auto_init()
student.requires_grad_(True)

# Combine and trace all qubits
combined = QCTN.concat([('s', student), ('t', teacher.hermit())])
combined.set_trace('all')

engine = EngineCommon(backend)
optimizer = SGDG(combined.parameters(), backend, lr=0.01)

for step in range(500):
    loss_val, grads = engine.contract_for_gradient(combined, target=1.0, loss='mse')
    optimizer.step(list(grads))
```

## Core Concepts

### ASCII Graph Format

The tensor network topology is defined by an ASCII string where each line represents a qubit:

```
-2-A-4-B-4-C-2-
-2-A-4-B-4-C-2-
-2-A-4-B-4-C-2-
```

- **Letters** (`A`, `B`, `C`, ...): Core tensors. A core appearing on multiple rows spans those qubits.
- **Numbers** (`2`, `4`, ...): Bond dimensions between adjacent cores, or boundary dimensions at the edges.
- **Dashes** (`-`): Separators (no core on this segment of the qubit).

The parser extracts each core's shape from the bond dimensions on all rows it touches, building the full adjacency table automatically.

### TNTensor

`TNTensor(tensor, scale)` represents the value `tensor × scale`. During contraction, the scale factors are accumulated separately to avoid floating-point under/overflow. Key features:

- **`auto_scale()`**: Normalizes the tensor so `max|tensor| ≈ 1`.
- **`hermit()`** / **`conj_transpose()`**: Conjugate-transpose view (zero-copy, shares parameters).
- **Arithmetic**: `+`, `-`, `*`, `/`, `@` with correct scale propagation.
- **Gradient proxy**: `requires_grad`, `grad` delegate to the underlying tensor.
- **Batch support**: `has_batch=True` flags the first dimension as a batch axis.

### Modules

**Leaf modules** wrap a single graph-based QCTN:

| Module | Description | Graph Example |
|--------|-------------|---------------|
| `MPS(n, bond, phys)` | Matrix Product State chain | `-2-A-4-B-4-C-2-` |
| `CircuitState(n, phys)` | Product-state ket vector | `-A-2-` |
| `MeasureMatrix(n, phys)` | Per-qubit observable | `-2-A-2-` |

**Application modules** compose leaf modules:

| Module | Structure | Use Case |
|--------|-----------|----------|
| `Quadratic` | `⟨circuit \| mps† · Mx · mps \| circuit⟩` | Density estimation, quantum expectation values |
| `TNEQ` | Inner product of two MPS | Student-teacher trace matching |
| `Encoding` | CircuitState + MPS | Quantum state encoding |
| `PlainMPS` | Single MPS wrapper | Standalone MPS model |

### EngineCommon

The unified contraction engine:

```python
engine = EngineCommon(backend, strategy_mode='balanced')
```

- **`contract(qctn)`**: Forward contraction, returns the result tensor.
- **`contract_for_gradient(qctn, target, loss)`**: Forward + backward, returns `(loss_value, gradients)`.

Strategy modes: `'fast'` (einsum only), `'balanced'` (einsum + MPS chain), `'full'` (all strategies including row-priority).

### Loss Functions

Built-in losses, usable by name:

| Name | Description |
| :--- | :--- |
| `'diagonal_mse'` | Reshape → diagonal → MSE (default) |
| `'mse'` | Mean Squared Error |
| `'mae'` | Mean Absolute Error |
| `'nll'` | Negative Log-Likelihood |
| `'fidelity'` | Quantum fidelity $- \vert \langle \text{result} \vert \text{target} \rangle \vert^2$ |

Custom losses via decorator:

```python
from tneq_qc.losses import register_loss, BaseLoss

@register_loss('my_loss')
class MyLoss(BaseLoss):
    def compute(self, result, target, backend):
        return backend.mean((result.tensor * result.scale) ** 2)
```

### Optimizers

All optimizers follow a unified API: `optimizer = Optimizer(params, backend, lr=...)`, then `optimizer.step(grads)`.

| Optimizer | Description |
|-----------|-------------|
| `SGD` | Stochastic Gradient Descent |
| `SGDG` | Gradient descent on Stiefel manifold (for unitary tensors) |
| `Adam` | Adaptive moment estimation |
| `Momentum` | SGD with momentum |
| `RMSProp` | Root mean square propagation |

Learning rate scheduling: `StepLRScheduler(optimizer, step_size, gamma)`.

### Data Generation

`DataGenerator` computes Hermite-polynomial feature maps for embedding continuous data into tensor networks:

```python
from tneq_qc import DataGenerator, make_data_fn

gen = DataGenerator(backend, mx_K=2)
data_fn = make_data_fn(gen, combined, batch_size=128, K=2)

# Each call generates new random data and injects Mx matrices into the QCTN
data_fn(step)
```

### Save & Load

Core tensors can be saved/loaded in safetensors format:

```python
# Save (stores tensor * scale as raw values)
qctn.save_cores("model.safetensors", metadata={'n_qubits': '4'})

# Load (restores values and calls auto_scale)
qctn.load_cores("model.safetensors")
```

## Distributed Training

Multi-process training with `torch.distributed`:

```python
from tneq_qc.distributed import EngineDistributed
from tneq_qc.distributed.engine.distributed_engine import PartitionConfig
from tneq_qc.distributed.comm import get_comm_backend

comm = get_comm_backend(backend='auto')
engine = EngineDistributed(
    backend=backend,
    strategy_mode='full',
    comm=comm,
    partition_config=PartitionConfig(strategy='layer', num_partitions=world_size),
)
engine.init_distributed(combined)
loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
```

Launch with torchrun:

```bash
torchrun --nproc_per_node=2 examples/train_dist.py
```

## Examples

| Script | Description |
|--------|-------------|
| [`examples/train_quadratic.py`](examples/train_quadratic.py) | Quadratic form training with NLL loss, heatmap visualization, sampling, KL divergence |
| [`examples/train_tneq.py`](examples/train_tneq.py) | Student-teacher MPS training via trace matching |
| [`examples/train_mnist.py`](examples/train_mnist.py) | Train a TN to approximate MNIST images |
| [`examples/train_dist.py`](examples/train_dist.py) | Distributed quadratic training with torchrun |

Jupyter notebooks (single-process, for exploration):

| Notebook | Description |
|----------|-------------|
| [`notebooks/train_dist.ipynb`](notebooks/train_dist.ipynb) | Distributed training demo (single-process mode) |
| [`notebooks/train_tneq.ipynb`](notebooks/train_tneq.ipynb) | Student-teacher trace matching |
| [`notebooks/train_mnist.ipynb`](notebooks/train_mnist.ipynb) | MNIST image training |

## Testing

```bash
pytest tests/
```

Test coverage includes: QCTN construction, graph parsing, TNTensor operations, contraction strategies (einsum, row-priority), module composition (concat, hermit, clone), output symbol ordering, and probability computation.

## Project Structure

```bash
tneq-qc/
├── tneq_qc/            # Library source
├── examples/           # Training scripts (runnable)
├── notebooks/          # Jupyter notebooks (exploration)
├── tests/              # pytest test suite
├── scripts/            # Shell scripts (distributed launch, CMG binding)
├── docs/               # Internal refactoring documentation
└── README.md
```

## Requirements

See `requirements.txt` for the full dependency list.

Core dependencies:

- Python ≥ 3.10
- PyTorch ≥ 2.0
- opt-einsum
- numpy
- safetensors
- tqdm
- cotengra

Optional:

- JAX (for JAX backend)
- torchvision (for MNIST example)
- matplotlib (for visualization)

## License

MIT License

## Citation

```bibtex
@software{wang2025tneq,
  title  = {tneq_qc: Tensor Network Engine toward Quantum: Yet Another Software But More Flexiable, Learnable, and Distributabl},
  author = {Wang, Yifeng and Li, Chao and Sun, Zhun},
  year   = {2025},
  url    = {https://github.com/minogame/tneq-qc}
}
```
