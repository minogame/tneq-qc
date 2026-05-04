# tneq-qc Project Overview

> Quantum Circuit Tensor Network (QCTN) Machine Learning Framework

---

## 1. Research Background and Motivation

Quantum-Inspired Machine Learning borrows the mathematical structures of quantum mechanics — Tensor Networks — to build machine learning models that can run efficiently on classical computers. The core ideas are:

- **Quantum states** can be represented as products of multiple local tensors (tensor networks)
- **Expectation values** (observables) can be efficiently computed via tensor contraction
- **Core tensors** serve as trainable parameters, optimized through gradient descent

The tneq-qc framework engineers these ideas into practice, providing a complete pipeline from **network topology definition** to **gradient-based training**, supporting MPS, tree, brickwall, and other topologies, with compatibility for both PyTorch and JAX backends.

---

## 2. Core Computational Models

The framework provides two computational objectives.

### 2.1 Objective One: TNEQ (Tensor Network Inner Product)

TNEQ computes the inner product between two independent tensor networks $A$ and $B$:

$$\mathcal{L} = \langle A | B \rangle = \mathrm{tr}(A^\dagger \cdot B)$$

When $A = B$, this degenerates to the norm $\|A\|^2$.

- **Corresponding modules**: `TNEQ` (`MPS_L` + `MPS_R` with independent parameters), `MPS_with_Ref` ($A = B$, parameter sharing)
- **Learning objective**: Approximate the Frobenius inner product structure of a target matrix $M$ using the inner product of two small TNs
- **Characteristics**: No external data input, pure parameter optimization; suitable for unsupervised density matrix learning

### 2.2 Objective Two: BornMachine Form

The BornMachine introduces a data-dependent measurement operator $M_x$ on top of TNEQ:

$$\mathcal{L}(x) = \langle \psi(x) | A^\dagger \cdot M_x \cdot A | \psi(x) \rangle = \mathrm{tr}\!\left(A \cdot M_x \cdot A^\dagger\right)$$

Where:
- $|\psi(x)\rangle$: Input state (State), encoding sample features $x$ into a quantum state
- $A$: Small tensor network parameters (MPS or other structures), trainable; $A^\dagger$ is its zero-copy conjugate transpose
- $M_x$: Measurement matrix generated from input $x$ (e.g., Hermite polynomial basis expansion), representing the data-dependent part of the "large matrix"

- **Corresponding module**: `BornMachine` (CS + MPS + Mx + MPS† + CS†)
- **Learning objective**: Use a small TN $A$ to learn a low-rank factorization of a high-dimensional operator $M$, so that $\mathcal{L}(x)$ fits the supervised signal
- **Characteristics**: Data-driven measurement matrix; suitable for supervised/semi-supervised quantum-inspired classification and regression

### 2.3 Tensor Network Topologies

Tensor network topologies are defined via **ASCII diagram strings**, where each line corresponds to a qubit, letters represent core tensors, and numbers represent bond dimensions:

```
-2-A-5-----C-3-----E-2-      ← qubit 0
-2-----B----4------E-2-      ← qubit 1
-2-A-4-B-7-C-2-D-4-E-2-      ← qubit 2
-2-----B-6-----D-----2-      ← qubit 3
-2-A-3-----C-8-D-----2-      ← qubit 4
```

The above example is a 5-qubit TensorNetwork with 5 core tensors A, B, C, D, E. Each core appears on a subset of qubit lines, forming a non-uniform connectivity graph.

---

## 3. System Architecture

### 3.1 Layered Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Application Layer                                │
│                                                                             │
│            ┌────────────────────────┐        ┌────────────────┐             │
│            │  Train  /  Distribute  │        │   Inference    │             │
│            └──────┬────────┬────────┘        └───────┬────────┘             │
└───────────────────┼────────┼─────────────────────────┼──────────────────────┘
                    │        │                         │
          ┌─────────┘        └──────────┐              │
          ▼                            ▼               ▼
┌──────────────────────┐    ┌──────────────────────────────────────────────────┐
│      Train Utils     │    │                     Engine                       │
│                      │    │                                                  │
│  ┌────────────────┐  │    │   ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │   Optimizer    │  │───▶│   │ TNEQ_QC  │    │ BornMachine│    │  Trace   │   │
│  ├────────────────┤  │    │   └─────┬────┘    └─────┬────┘    └─────┬────┘   │
│  │     Loss       │  │    │         └───────────────┼───────────────┘        │
│  └───────┬────────┘  │    │                (defined via QCTN)                │
└──────────┼───────────┘    └─────────────────────────┼────────────────────────┘
           │                                          │
           │                                          ▼
           │              ┌────────────────────────────────────────────────────┐
           │              │                     Model Layer                    │
           │              │                                                    │
           │              │   TNGraph ──(init)──▶ ┌──────────────────────┐     │
           │              │                       │        QCTN          │     │
           │              │                       │  ┌────────────────┐  │     │
           │              │                       │  │  State  │  │     │
           │              │                       │  │  MeasureMx     │  │     │
           │              │                       │  └────────────────┘  │     │
           │              │                       └────┬────────────┬────┘     │
           │              └────────────────────────────┼────────────┼──────────┘
           │                                           │            │
           │                              core tensor  │   forward  │
           │                                           │            │
           │              ┌────────────────────────────┼            │
           │              │                            │            │
           ▼              ▼                            ▼            ▼
┌──────────────────────────────┐    ┌───────────────────────────────────────┐
│                              │    │           Strategy Layer              │
│          TNTensor            │◀───┤                                       │
│       (core tensor)          │    │   ┌──────────────────────────┐        │
│                              │    │   │    StrategyCompiler      │        │
│                              │    │   └─────┬────────┬───────┬───┘        │
│                              │    │         ▼        ▼       ▼            │
│                              │    │   ┌────────┐┌────────┐┌───────────┐   │
│                              │    │   │Einsum  ││Greedy  ││RowPriority│   │
│                              │    │   │Strategy││Strategy││Strategy   │   │
│                              │    │   └────────┘└────────┘└───────────┘   │
└────────────┬─────────────────┘    └──────────────────┬────────────────────┘
             │                                         │
             └──────────────────┬──────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Backend Layer                                   │
│                                                                              │
│     ┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐       │
│     │Torch Backend │    │ JAX Backend  │    │ Distributed Primitives │       │
│     └──────────────┘    └──────────────┘    └────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Core Component Responsibilities

| Component | Responsibility |
|---|---|
| **QCTN** | Holds topology (adjacency_table) and parameters (cores_weights); supports concat/chunk composition; parameters() collects trainable parameters |
| **TNGraph** | Bidirectional conversion between ASCII diagram strings and structured adjacency tables |
| **TNTensor** | Tensor wrapper with independent scale factor, preventing numerical underflow/overflow during contraction; supports reference semantics, gradient proxy, and transparent arithmetic |
| **EngineCommon** | Unified contraction entry point; integrated contract / contract_for_gradient interface; probability computation and sampling |
| **StrategyCompiler** | Evaluates cost of each strategy, automatically selects the optimal contraction strategy (EinsumStrategy / RowPriorityStrategy) |
| **ComputeBackend** | Abstracts away PyTorch/JAX differences, providing unified tensor operations, JIT, gradient, and optimizer interfaces |
| **Optimizer** | PyTorch-style optimizers: Adam / SGD / SGDG / Momentum / RMSProp + LR scheduling |
| **LossRegistry** | Extensible loss function registry: MSE / NLL / MAE / Fidelity, supporting both string names and custom functions |
| **EngineDistributed** | Inherits from EngineCommon, adds partition strategies and AllReduce communication, supporting multi-process distributed training |

---

## 4. Module System

### 4.1 Leaf Modules

Each leaf module is a QCTN subclass with a clear physical meaning, holding a segment of the tensor network graph and the corresponding core tensor parameters.

```
MPS(nqubits=3, bond_dim=4, phys_dim=2)
    Graph: -2-A-4-B-4-C-2-   (3 identical rows)
    Physical meaning: Matrix Product State, bond dimension controls entanglement capacity

State(nqubits=3, phys_dim=2)
    Graph: -A-2- / -B-2- / -C-2-
    Physical meaning: Input quantum state |ψ⟩, auto-initialized as [1, 0, ..., 0]

MeasureMatrix(nqubits=3, phys_dim=2)
    Graph: -2-A-2- / -2-B-2- / -2-C-2-
    Physical meaning: Measurement operator, one matrix per qubit
```

### 4.2 Application Modules

Application modules combine leaf modules into complete computational graphs using the **composite pattern** (graph=None):

| Module | Structure | Computation |
|---|---|---|
| `PlainMPS` | Single MPS | $\langle A \rangle$ (core norm) |
| `TransposeMPS` | Zero-copy conjugate transpose view of MPS | $A^\dagger$ (parameter sharing) |
| `MPS_with_Ref` | left MPS + right = left† | $\|A\|^2$ (symmetric normalization) |
| `Encoding` | State + MPS | $A|\psi\rangle$ (feature encoding) |
| `TNEQ` | MPS_L + MPS_R (independent parameters) | $\langle\phi\|\psi\rangle$ (inner product) |
| `BornMachine` | State + TN + Mx + TN† + State† | $\langle\psi\|A^\dagger M_x A\|\psi\rangle$ (Born probability) |

`BornMachine(graph, dim, backend=..., mx_graph=None)` uses one local `dim x dim`
Mx core per qubit by default. To make an Mx core span multiple qubits, provide a
custom `mx_graph` whose core symbol appears on multiple qubit rows, or build the
Mx QCTN manually and compose the five segments with `QCTN.concat()`.

### 4.3 Horizontal Composition: concat and chunk

QCTN supports **horizontal concatenation** (concat) and **splitting** (chunk), enabling flexible module assembly:

```
concat(State[3], MPS[3, bond=4], MeasureMatrix[3]):

CS  (3 qubits, 1 core/qubit):   MPS (3 qubits, 3 cores):   MX (3 qubits, 1 core/qubit):
  -a-2-                           -2-a-4-b-4-c-2-              -2-a-2-
  -b-2-                           -2-a-4-b-4-c-2-              -2-b-2-
  -c-2-                           -2-a-4-b-4-c-2-              -2-c-2-

Merged result (3 qubits, 5 cores/qubit, cores renamed to a..i):
  -a-2-d-4-e-4-f-2-g-2-
  -b-2-d-4-e-4-f-2-h-2-
  -c-2-d-4-e-4-f-2-i-2-
```

After concat, core names are automatically renumbered (opt_einsum symbols) and weights are copied according to the mapping. `chunk()` is the inverse of concat, splitting into two sub-QCTNs by core index.

---

## 5. Contraction Strategies

### 5.1 Three Strategies

Tensor network contraction is the process of combining multiple tensors via Einstein summation into a scalar or lower-order tensor. The framework provides three strategies, automatically selected by `StrategyCompiler`:

| Strategy | Principle | Applicable Scenario |
|---|---|---|
| **EinsumStrategy** | Uses opt_einsum to compute the entire model in one pass | Small-scale networks, fast compilation |
| **GreedyStrategy** | Greedily selects the optimal contraction pair step by step | Medium-scale, fine-grained control |
| **RowPriorityStrategy** | Contracts qubit-by-qubit (row by row), controlling intermediate state memory | Large models with many qubits but relatively sparse |

### 5.2 Symmetric Expansion

When computing the BornMachine $\langle \psi | A^\dagger M_x A | \psi \rangle$, the tensor network expands into 5 columns (left-middle-right):

```
CIRCUIT  LEFT (A)    MIDDLE (Mx)    RIGHT (A_T)   CIRCUIT
C0        a              Mx₀           a_T          C0
C1        b              Mx₁           b_T          C1
C2        c              Mx₂           c_T          C2
```

### 5.3 Contraction Flow

```
QCTN + shapes_info
      │
      ▼
StrategyCompiler.compile()
  ├── check_compatibility()   Check if each strategy is applicable
  ├── estimate_cost()         Estimate FLOPs
  └── Select optimal strategy
      │
      ▼
strategy.get_compute_function()  → compute_fn
      │
      ▼
EngineCommon.contract(qctn)
      │
      ▼
  Scalar / tensor result
```

---

## 6. Numerical Stability: TNTensor

Deep tensor network contraction is prone to numerical underflow or overflow. `TNTensor` solves this problem by separating the scale factor:

$$\text{true value} = \text{tensor} \times \text{scale}$$

- `auto_scale()`: Normalizes the tensor to $\max|t|=1$, absorbing the ratio into scale
- `log_scale`: Handles extreme values using logarithmic scale
- `conj_transpose()`: Automatically computes conjugate for complex transpose
- **Reference semantics**: Zero-copy views with `is_ref=True`, no underlying data duplication
- **Gradient proxy**: `requires_grad`, `grad`, `detach()`, `backward()` delegate directly to the underlying tensor
- **Transparent arithmetic**: Supports `+`, `-`, `*`, `/`, `@`, `**` and other operators, automatically managing scale propagation

Contraction-time auto scaling is backend-controlled and disabled by default:

```python
backend = BackendFactory.create_backend(
    "pytorch",
    device="cpu",
    dtype="complex64",
    enable_auto_scale=True,
)
```

When enabled, supported contraction paths normalize TNTensor intermediate results while preserving `tensor * scale`.

---

## 7. Backend Abstraction

The framework abstracts away the differences between PyTorch and JAX through a unified `ComputeBackend` interface:

```python
class ComputeBackend:
    execute_expression(expr, *tensors)         # Execute contraction
    compute_value_and_grad(loss_fn, argnums)   # Value + gradient
    jit_compile(func)                          # JIT compilation
    optimizer_update(params, grads, state, …)  # Optimization step
    init_random_core(shape)                    # Orthogonal random initialization
```

`BackendFactory` uses a factory + singleton pattern to manage backend instances. Training code is backend-transparent.

---

## 8. Training Pipeline

### 8.1 Typical Training Loop

```python
from tneq_qc import (
    QCTN, QCTNHelper, EngineCommon, BackendFactory, BornMachine,
    DataGenerator, SGDG,
)

# 1. Create backend and engine
backend  = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
engine   = EngineCommon(backend=backend, strategy='row_priority')
data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

# 2. Build model
graph = QCTNHelper.mps(4, bond_dim=2, phys_dim=2)
model = BornMachine(graph, 2, backend=backend).auto_init(orthogonal=True)
model._submodules['tn'].requires_grad_(True)
combined = model.build()

# 3. Create optimizer and data function
optimizer = SGDG(combined.parameters(), backend, lr=0.01)
data_fn = make_data_fn(data_gen, combined, batch_size=128, K=2)

# 4. Training loop
for step in range(1, N_STEPS + 1):
    data_fn(step)                                                   # Inject data
    loss_val, grads = engine.contract_for_gradient(combined,        # Contraction + gradient
                                                    target=1, loss='nll')
    optimizer.step(list(grads))                                     # Parameter update
```

### 8.2 Loss Functions

The framework provides the following built-in loss functions, accessible by string name or `BaseLoss` subclass:

| Name | Description |
|---|---|
| `'mse'` | Mean Squared Error |
| `'mae'` | Mean Absolute Error |
| `'nll'` | Negative Log-Likelihood (commonly used for density estimation) |
| `'fidelity'` | Quantum fidelity $-\|\langle result \| target \rangle\|^2$ |
| `'diagonal_mse'` | Diagonal MSE (backward compatible) |

Custom loss:

```python
from tneq_qc.losses import register_loss, BaseLoss

@register_loss('custom')
class CustomLoss(BaseLoss):
    def compute(self, result, target, backend):
        return backend.sum((result - target) ** 2)
```

### 8.3 Optimizers

| Optimizer | Description |
|---|---|
| `Adam` | Adaptive Moment Estimation (default β₁=0.9, β₂=0.999) |
| `SGD` | Stochastic Gradient Descent |
| `SGDG` | Stiefel Manifold Gradient Descent (Cayley transform preserves orthogonality) |
| `Momentum` | SGD with momentum |
| `RMSProp` | Root Mean Square Propagation |

Learning rate scheduling:

```python
from tneq_qc import StepLRScheduler
scheduler = StepLRScheduler(optimizer, step_size=100, gamma=0.5)
```

---

## 9. Distributed Parallel Training

When the tensor network scale (number of qubits x bond dimension) exceeds single-node memory/compute capacity, the framework adopts a two-stage distributed strategy of **model parallelism + tensor parallelism**.

### 9.1 Overall Architecture

```
  Input data x_batch (generated by Rank 0, broadcast to all nodes)
                              │
              ┌───────────────▼───────────────┐
              │         Data Broadcast         │
              │   Rank 0 generates Mx_list,    │
              │        broadcasts              │
              └──┬──────────────┬──────────────┘
                 │              │              │
    ┌────────────▼──┐  ┌────────▼──┐  ┌───────▼───────┐
    │   Worker 0    │  │ Worker 1  │  │   Worker N    │   ← Model parallelism
    │  QCTN chunk 0 │  │ QCTN chunk│  │ QCTN chunk N  │
    │  cores: a,b,c │  │ cores:d,e │  │ cores: ...    │
    └───────┬───────┘  └─────┬─────┘  └──────┬────────┘
            │                │                │
            │   local forward + backward      │
            └────────────────┼────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     Weight Sync (AllReduce)  │
              └─────────────────────────────┘
```

### 9.2 Usage

```python
from tneq_qc.distributed import EngineDistributed
from tneq_qc.distributed.engine.distributed_engine import PartitionConfig

engine = EngineDistributed(
    backend=backend,
    strategy="row_priority",
    comm=comm,
    partition_config=PartitionConfig(strategy='layer', num_partitions=world_size),
)
engine.init_distributed(combined)

# Training loop is the same as single-machine
loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
```

Launch command:

```bash
torchrun --nproc_per_node=2 examples/train_dist.py
```

---

## 10. Key Design Principles

1. **ASCII diagram as interface**: Topology definition is fully decoupled from parameters; modifying network structure requires no changes to code logic
2. **Composition over inheritance**: concat/chunk allows arbitrary horizontal module assembly; composite pattern enables hierarchical nesting
3. **Strategy-structure separation**: Contraction strategies do not parse graph strings; they only consume adjacency_table (the parsed structure)
4. **Zero-copy parameter sharing**: TNTensor reference semantics ensure siamese networks ($A$ and $A^\dagger$) incur no memory redundancy
5. **Backend transparency**: Training logic and strategy logic are independent of any specific backend, supporting cross-platform migration
6. **Distributed training**: Supports distributed training, combining model parallelism and tensor parallelism

---

## 11. Package Structure

```
tneq_qc/
├── core/                      # Core abstractions
│   ├── qctn.py                # QCTN main class
│   ├── _qctn_graph.py         # Graph parsing Mixin
│   ├── _qctn_io.py            # IO / initialization Mixin
│   ├── _qctn_contractor.py    # Contraction interface Mixin
│   ├── tn_tensor.py           # TNTensor tensor wrapper
│   ├── tn_graph.py            # ASCII graph parser
│   └── engine_common.py       # Unified contraction engine
├── contractor/                # Contraction strategies
│   ├── base.py                # ContractionStrategy ABC
│   ├── einsum_strategy.py     # opt_einsum one-pass contraction
│   ├── greedy_strategy.py     # Greedy contraction
│   ├── row_priority_strategy.py  # Row-by-row contraction
│   └── compiler.py            # StrategyCompiler auto-selection
├── backends/                  # Backend abstraction
│   ├── backend_interface.py   # ComputeBackend ABC
│   ├── backend_factory.py     # Factory + singleton
│   ├── backend_pytorch.py     # PyTorch implementation
│   └── backend_jax.py         # JAX implementation
├── modules/                   # Predefined modules
│   ├── small.py               # MPS / State / MeasureMatrix
│   └── app.py                 # BornMachine / TNEQ / Encoding / PlainMPS
├── optim/                     # Optimizers
│   ├── base.py                # OptimizerBase
│   ├── optimizers.py          # Adam / SGD / SGDG / Momentum / RMSProp
│   └── lr_scheduler.py        # StepLRScheduler
├── losses/                    # Loss functions
│   ├── base.py                # BaseLoss ABC
│   ├── builtin.py             # Built-in losses (MSE / NLL / MAE / Fidelity)
│   ├── registry.py            # LossRegistry
│   └── target.py              # TargetResolver
├── distributed/               # Distributed training
│   ├── comm/                  # Communication backend (MPI / torch.distributed)
│   ├── engine/                # EngineDistributed
│   └── optim/                 # Distributed optimizers
└── utils/                     # Utilities
    ├── data_generator.py      # DataGenerator / make_data_fn
    └── graph_generators.py    # QCTNHelper graph generator
```
