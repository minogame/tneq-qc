# tneq-qc Project Overview (Phase 2.6)

> Quantum Circuit Tensor Network (QCTN) Machine Learning Framework

---

## 1. Background and Motivation

Quantum-Inspired Machine Learning borrows the mathematical structure of quantum mechanics — tensor networks — to build machine learning models that run efficiently on classical computers. The core ideas are:

- **Quantum states** can be represented as products of local tensors (tensor networks)
- **Expectation values** (observables) can be computed efficiently via tensor contraction
- **Core tensors** serve as trainable parameters optimized by gradient descent

tneq-qc engineers these ideas into a complete pipeline from **network topology definition** to **gradient-based training**, supporting MPS, tree, brick-wall and other topologies, with PyTorch and JAX backends.

---

## 2. Core Computation Models

The framework provides two distinct learning objectives.

### 2.1 Objective 1: TNEQ (Tensor Network Inner Product)

TNEQ computes the inner product between two independent tensor networks $A$ and $B$:

$$\mathcal{L} = \langle A | B \rangle = \mathrm{tr}(A^\dagger \cdot B)$$

When $A = B$ this reduces to the norm $\|A\|^2$.

- **Modules**: `TNEQ` (independent `MPS_L` + `MPS_R`), `MPS_with_Ref` ($A = B$, shared parameters)
- **Learning goal**: approximate the Frobenius inner-product structure of a target matrix $M$ using two compact tensor networks
- **Characteristics**: no external data input; pure parameter optimization; suited for unsupervised density-matrix learning

### 2.2 Objective 2: Quadratic Form

The quadratic objective extends TNEQ by introducing a data-dependent measurement operator $M_x$:

$$\mathcal{L}(x) = \langle \psi(x) | A^\dagger \cdot M_x \cdot A | \psi(x) \rangle = \mathrm{tr}\!\left(A \cdot M_x \cdot A^\dagger\right)$$

where:
- $|\psi(x)\rangle$: input state (CircuitState) encoding sample features $x$ as a quantum state
- $A$: small tensor network (MPS or other topology), trainable; $A^\dagger$ is its zero-copy conjugate transpose
- $M_x$: measurement matrix generated from input $x$ (e.g., Hermite polynomial basis expansion), representing the data-dependent part of the "large matrix"

- **Module**: `Quadratic` (CS + MPS + Mx + MPS† + CS†)
- **Learning goal**: learn a low-rank tensor network decomposition of a high-dimensional operator $M$, fitting $\mathcal{L}(x)$ to a supervision signal
- **Characteristics**: data-driven measurement matrices; suited for supervised / semi-supervised quantum-inspired classification and regression

### 2.3 Tensor Network Topology

Network topology is specified via an **ASCII graph string** where each row is a qubit, letters denote core tensors, and numbers denote bond dimensions:

```
-2-A-5-----C-3-----E-2-      ← qubit 0
-2-----B----4------E-2-      ← qubit 1
-2-A-4-B-7-C-2-D-4-E-2-      ← qubit 2
-2-----B-6-----D-----2-      ← qubit 3
-2-A-3-----C-8-D-----2-      ← qubit 4
```

This is a 5-qubit tensor network with 5 core tensors A, B, C, D, E. Each core appears on a subset of qubit rows, forming a non-uniform connectivity graph.

Supported topologies:

| Topology | Structure | Typical use |
|---|---|---|
| **MPS** (Matrix Product State) | linear chain, all qubits share all cores | 1D quantum state, variational optimization |
| **Tree** | hierarchical binary tree | hierarchical feature extraction |
| **Brick-wall** | alternating two-qubit gate layers | shallow quantum circuit simulation |
| **CircuitState** | one core per qubit, right edge only | input ket $|\psi\rangle$ |
| **MeasureMatrix** | one core per qubit, left and right edges | observable / quantum channel |

---

## 3. System Architecture

### 3.1 Layered Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Application Layer                                │
│                                                                             │
│            ┌────────────────────────┐        ┌────────────────┐             │
│            │  Trainer / Distribute  │        │   Inference    │             │
│            └──────┬────────┬────────┘        └───────┬────────┘             │
└───────────────────┼────────┼─────────────────────────┼──────────────────────┘
                    │        │                         │
          ┌─────────┘        └──────────┐              │
          ▼                            ▼               ▼
┌──────────────────────┐    ┌──────────────────────────────────────────────────┐
│      Training        │    │                     Engine                       │
│                      │    │                                                  │
│  ┌────────────────┐  │    │   ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │   Optimizer    │  │───▶│   │ TNEQ_QC  │    │ Quadratic│    │  Trace   │   │
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
           │              │                       │  │  CircuitState  │  │     │
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

### 3.2 Component Responsibilities

| Component | Responsibility |
|---|---|
| **QCTN** | Holds topology (adjacency_table) and parameters (cores_weights); supports concat/chunk composition |
| **TNGraph** | Bidirectional conversion: ASCII graph string ↔ structured adjacency list |
| **TNTensor** | Tensor wrapper with separate scale factor to prevent underflow/overflow during contraction |
| **EngineCommon** | Unified contraction entry point; manages per-qubit operation configuration (measurement, input state, trace) |
| **StrategyCompiler** | Evaluates cost of each strategy and selects the optimal one automatically |
| **ComputeBackend** | Abstracts PyTorch/JAX differences; unified interface for tensor ops, JIT, gradients, optimizer |

---

## 4. Module System

### 4.1 Leaf Modules

Each leaf module is a QCTN subclass with a clear physical meaning, holding a graph and its corresponding core tensor parameters.

```
MPS(nqubits=3, bond_dim=4, phys_dim=2)
    graph: -2-A-4-B-4-C-2-   (same for all 3 rows)
    meaning: Matrix Product State; bond_dim controls entanglement capacity

CircuitState(nqubits=3, phys_dim=2)
    graph: -A-2- / -B-2- / -C-2-
    meaning: input quantum state |ψ⟩, one independent core per qubit

A-2-
B-2-
C-2-

MeasureMatrix(nqubits=3, phys_dim=2)
    graph: -2-A-2- / -2-B-2- / -2-C-2-
    meaning: measurement operator, one matrix per qubit


myQCTN
  _CircuitState
  _MPS
  _Mx



concat(_MPS_1, _MPS_1.clone())

4 cores



0, 5
1, 6

cores_list 8, 4 tensor_source='core',  tensor_souce='transpose', 


4

get_tensor()

CircuitState + MPS + Mx + MPS_T + CircuitState_T

```

### 4.2 Application Modules

Application modules use **composite mode** (graph=None) to wire leaf modules into complete computation graphs:

| Module | Composition | Computation |
|---|---|---|
| `PlainMPS` | single MPS | $\langle A \rangle$ (core norm) |
| `TransposeMPS` | zero-copy conjugate-transpose view of MPS | $A^\dagger$ (shared parameters) |
| `MPS_with_Ref` | left MPS + right = left† | $\|A\|^2$ (symmetric normalization) |
| `Encoding` | CircuitState + MPS | $A|\psi\rangle$ (feature encoding) |
| `TNEQ` | MPS_L + MPS_R (independent) | $\langle\phi|\psi\rangle$ (inner product) |
| `Quadratic` | CS + MPS + Mx + MPS† + CS† | $\langle\psi|A^\dagger M_x A|\psi\rangle$ (quadratic form) |

### 4.3 Horizontal Composition: concat and chunk

QCTN supports **horizontal concatenation** (concat) and **splitting** (chunk) for flexible module assembly:

```
concat(CircuitState[3], MPS[3, bond=4], MeasureMatrix[3]):

CS  (3 qubits, 1 core/qubit):   MPS (3 qubits, 3 cores):   MX (3 qubits, 1 core/qubit):
  -a-2-                           -2-a-4-b-4-c-2-              -2-a-2-
  -b-2-                           -2-a-4-b-4-c-2-              -2-b-2-
  -c-2-                           -2-a-4-b-4-c-2-              -2-c-2-

Merged result (3 qubits, 5 cores/row, cores renamed a..i):
  -a-2-d-4-e-4-f-2-g-2-
  -b-2-d-4-e-4-f-2-h-2-
  -c-2-d-4-e-4-f-2-i-2-
```

After concat, core names are renumbered automatically (opt_einsum symbols) and weights are copied according to the name mapping. `chunk()` is the inverse operation, splitting by core index into two sub-QCTNs.

---

## 5. Contraction Strategies

Tensor network contraction reduces a set of tensors via Einstein summation into a scalar or low-rank tensor. The framework provides three extensible strategies, selected automatically by `StrategyCompiler`:

| Strategy | Description | Best suited for |
|---|---|---|
| **EinsumStrategy** | Contracts the entire model via a single opt_einsum expression | Small networks, fast compilation |
| **GreedyStrategy** | Greedy per-qubit contraction with fine-grained control | Medium-scale networks |
| **RowPriorityStrategy** | Row-by-row contraction; bounds peak memory usage | Large, sparse networks (many qubits) |

### 5.1 Symmetric Expansion

When computing the quadratic form $\langle \psi | A^\dagger M_x A | \psi \rangle$, the tensor network is expanded into five columns: circuit input, LEFT ($A$), MIDDLE ($M_x$), RIGHT ($A^\dagger$), circuit input†:

```
CIRCUIT   LEFT (A)   MIDDLE (Mx)   RIGHT (A†)   CIRCUIT†
  C0         a           Mx0           a†            C0†
  C1         b           Mx1           b†            C1†
  C2         c           Mx2           c†            C2†
```

`TensorSide` enum (LEFT / MIDDLE / RIGHT) labels each core's role. `RowPriorityStrategy` contracts row by row to control memory.

### 5.2 Contraction Pipeline

```
QCTN + shapes_info
      │
      ▼
StrategyCompiler.compile()
  ├── check_compatibility()   each strategy checks applicability
  ├── estimate_cost()         estimate FLOPs
  └── select best strategy
      │
      ▼
strategy.get_compute_function()  →  compute_fn
      │
      ▼
backend.execute_expression(compute_fn, *tensors)
      │
      ▼
  scalar / tensor result
```

---

## 6. Numerical Stability: TNTensor

Deep tensor network contractions are prone to numerical underflow or overflow. `TNTensor` addresses this by separating a scalar scale factor:

$$\text{true value} = \text{tensor} \times \text{scale}$$

- `auto_scale()`: normalizes the tensor to $\max|t|=1$, absorbing the ratio into scale
- `log_scale`: logarithmic scale for extreme values
- `conj_transpose()`: computes the conjugate transpose for complex tensors
- **Reference semantics**: `conj_transpose()` returns an `is_ref=True` zero-copy view; the underlying data is not duplicated

This allows siamese networks where $A$ and $A^\dagger$ share the same memory buffer with no redundant copies.

---

## 7. Backend Abstraction

A unified `ComputeBackend` interface hides the differences between PyTorch and JAX:

```python
class ComputeBackend:
    execute_expression(expr, *tensors)         # execute contraction
    compute_value_and_grad(loss_fn, argnums)   # value + gradients
    jit_compile(func)                          # JIT compilation
    optimizer_update(params, grads, state, …)  # optimizer step
    init_random_core(shape)                    # orthogonal random initialization
```

`BackendFactory` manages backend instances via a factory + singleton pattern. Training code is fully backend-agnostic.

---

## 8. Training Pipeline

Typical training loop:

```python
# 1. Build model
backend = BackendFactory.create_backend("pytorch", device="cuda", dtype="complex64")
model = Quadratic(nqubits=D, bond_dim=chi, phys_dim=K, backend=backend)
model.auto_init()

# 2. Build engine
engine = EngineCommon(backend=backend, strategy_mode="balanced")

# 3. Train
for x_batch in dataloader:
    Mx_list = engine.generate_data(x_batch, K=K)   # data → measurement matrices
    loss, grads = engine.contract_with_compiled_strategy_for_gradient(
        model, measure_input_list=Mx_list
    )
    optimizer.step(model, grads)
```

`generate_data` expands input vector $x$ into measurement matrix $M_x$ via Hermite polynomial basis. `contract_with_compiled_strategy_for_gradient` performs contraction and backpropagation in a single call.

---

## 9. Distributed Parallel Training

When the tensor network scale (qubits × bond dimension) exceeds the capacity of a single node, the framework adopts a two-stage strategy combining **model parallelism** and **tensor parallelism**.

### 9.1 Overall Architecture

```
  Input x_batch (generated by Rank 0, MPI broadcast to all workers)
                              │
              ┌───────────────▼───────────────┐
              │          Data Broadcast        │
              │  Rank 0 generates Mx_list      │
              │  and broadcasts to all ranks   │
              └──┬──────────────┬─────────────┘
                 │              │              │
    ┌────────────▼──┐  ┌────────▼──┐  ┌───────▼───────┐
    │   Worker 0    │  │ Worker 1  │  │   Worker N    │  ← Model Parallel
    │               │  │           │  │               │    (MPI Ranks)
    │  QCTN chunk 0 │  │QCTN chunk1│  │ QCTN chunk N  │
    │  cores: a,b,c │  │cores: d,e │  │ cores: ...    │
    │               │  │           │  │               │
    │  ┌──────────┐ │  │┌─────────┐│  │ ┌──────────┐  │
    │  │ GPU 0    │ │  ││ GPU 0   ││  │ │ GPU 0    │  │  ← Tensor Parallel
    │  │ GPU 1    │ │  ││ GPU 1   ││  │ │ GPU 1    │  │    (intra-node)
    │  └──────────┘ │  │└─────────┘│  │ └──────────┘  │
    └───────┬───────┘  └─────┬─────┘  └───────┬───────┘
            │                │                │
            │   local forward + backward pass │
            │   (each worker owns local cores)│
            └────────────────┼────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │      Weight Sync            │
              │  sync_weights_after_update  │
              │  each worker broadcasts its │
              │  locally updated weights    │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │       Optimizer Step        │
              │  each worker updates its    │
              │  local cores independently  │
              └─────────────────────────────┘
```

### 9.2 Stage 1: Model Parallelism

Core tensors are partitioned evenly across MPI workers by index:

```
QCTN (16 cores: a,b,c,...,p)  →  partitioned via chunk()
                                        │
         ┌──────────────┬───────────────┼───────────────┐
         ▼              ▼               ▼               ▼
    Rank 0          Rank 1          Rank 2          Rank 3
  cores: a,b,c,d  cores: e,f,g,h  cores: i,j,k,l  cores: m,n,o,p
```

- **Forward pass**: each rank runs the full contraction (fetching missing cores from other ranks via MPI), producing a local loss value
- **Backward pass**: each rank computes and retains gradients only for its local cores
- **Weight update**: local update → `sync_weights_after_update()` broadcasts updated weights across all ranks
- **Implementation**: `ModelParallelManager`, `ModelParallelTrainer`, `DistributedEngineSiamese`

### 9.3 Stage 2: Tensor Parallelism

On top of model parallelism, large tensor contractions within each rank are further sharded across multiple GPUs:

```
Inside a single Worker (tensor parallelism):

QCTN chunk (local cores)
       │
  ┌────▼───────────────────────────────────┐
  │  Large core tensor (e.g. bond_dim=1024)│
  │  Sharded along bond dimension:         │
  │                                        │
  │  GPU 0: shard[0:256, :]                │
  │  GPU 1: shard[256:512, :]              │
  │  GPU 2: shard[512:768, :]              │
  │  GPU 3: shard[768:1024, :]             │
  │                                        │
  │  Each GPU executes local partial einsum│
  │  → all-reduce to assemble full result  │
  └────────────────────────────────────────┘
```

- **Contraction sharding**: large matrix multiplications are split along the bond dimension across GPUs; results are aggregated via all-reduce
- **Use case**: large bond_dim (≥ 512) with moderate qubit count
- **Backend support**: `BackendPyTorch` provides inter-GPU communication primitives via `Distributed Primitives`

### 9.4 Combining Both Strategies

```
Large-scale QCTN training
        │
        ├─ Many cores (large ncores)   →  Model parallel: distribute cores across nodes
        │
        └─ Large tensors (large bond)  →  Tensor parallel: shard one core across GPUs

Combined:
  inter-node  →  model parallelism  (MPI,  process-level communication)
  intra-node  →  tensor parallelism (NCCL/Gloo, GPU-level communication)
```

---

## 10. Comparison with Related Work

TODO:

---

## 11. Key Design Principles

1. **ASCII graph as interface**: topology and parameters are fully decoupled; changing network structure requires no changes to computation logic
2. **Composition over inheritance**: concat/chunk enables arbitrary horizontal module assembly; composite mode enables hierarchical nesting
3. **Strategy-structure separation**: contraction strategies consume `adjacency_table` (parsed structure), never raw graph strings
4. **Zero-copy parameter sharing**: TNTensor reference semantics allow siamese networks where $A$ and $A^\dagger$ share the same memory with no redundancy
5. **Backend transparency**: training logic and strategy logic are independent of the specific backend, enabling cross-platform migration
6. **Distributed training**: supports joint model parallelism and tensor parallelism via MPI
