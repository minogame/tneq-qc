# tneq-qc Architecture (Phase 2.6)

> Quantum Circuit Tensor Network (QCTN) ML Framework
> Suitable for paper figures, architecture diagrams, and developer reference.

---

## 1. Package Structure

```
tneq_qc/
├── core/                   # Core data structures
│   ├── qctn.py             # QCTN (main class, ~480L)
│   ├── _qctn_graph.py      # QCTNGraphMixin (~230L)
│   ├── _qctn_io.py         # QCTNIOMixin (~290L)
│   ├── _qctn_contractor.py # QCTNContractorMixin + TensorSide (~650L)
│   ├── tn_tensor.py        # TNTensor (tensor + scale wrapper)
│   ├── tn_graph.py         # TNGraph (ASCII graph parser)
│   ├── engine_common.py    # EngineCommon (unified contraction engine)
│   └── engine_siamese.py   # EngineSiamese (deprecated)
├── contractor/             # Contraction strategies
│   ├── base.py             # ContractionStrategy (abstract)
│   ├── einsum_strategy.py  # EinsumStrategy
│   ├── greedy_strategy.py  # GreedyStrategy
│   ├── row_priority_strategy.py  # RowPriorityStrategy
│   └── compiler.py         # StrategyCompiler
├── backends/               # Compute backends
│   ├── backend_interface.py  # ComputeBackend (abstract)
│   ├── backend_factory.py    # BackendFactory
│   ├── backend_pytorch.py    # BackendPyTorch
│   └── backend_jax.py        # BackendJAX
├── modules/                # High-level modules
│   ├── small.py            # MPS, CircuitState, MeasureMatrix
│   ├── app.py              # PlainMPS, TransposeMPS, MPS_with_Ref, Encoding, TNEQ, Quadratic
│   └── __init__.py
└── utils/
    └── graph_generators.py # QCTNHelper
```

---

## 2. Class Hierarchy

```
QCTN  (core/qctn.py)
├── Mixins (multiple inheritance):
│   ├── QCTNGraphMixin    (_qctn_graph.py)   — graph → adjacency_table
│   ├── QCTNIOMixin       (_qctn_io.py)      — init, save/load, set_cores
│   └── QCTNContractorMixin (_qctn_contractor.py) — einsum info, core list
│
└── Subclasses:
    ├── Leaf modules  (modules/small.py)
    │   ├── MPS              — uniform MPS: all rows see all cores
    │   ├── CircuitState     — ket: one core per qubit, right edge only
    │   └── MeasureMatrix    — operator: one core per qubit, both edges
    │
    └── Application modules  (modules/app.py)
        ├── PlainMPS         — single MPS container
        ├── TransposeMPS     — zero-copy conj-transpose view of an MPS
        ├── MPS_with_Ref     — left MPS + right MPS sharing weights
        ├── Encoding         — CircuitState → MPS
        ├── TNEQ             — inner product of two independent MPS
        └── Quadratic        — <cs | mps† · mx · mps | cs>

TNTensor  (core/tn_tensor.py)
└── Wraps any backend tensor with a scalar scale factor

TNGraph  (core/tn_graph.py)
└── ASCII graph string ↔ adjacency list of (name, left_bond, right_bond) tuples

ComputeBackend  (backends/backend_interface.py)
├── BackendPyTorch
└── BackendJAX

ContractionStrategy  (contractor/base.py)
├── EinsumStrategy        — opt_einsum, general
├── GreedyStrategy        — per-qubit greedy
└── RowPriorityStrategy   — per-qubit, graph wiring delegated to QCTN
```

---

## 3. QCTN: Key Attributes

| Attribute | Type | Description |
|---|---|---|
| `graph` | `str \| None` | Raw ASCII graph string. `None` in composite mode. |
| `tn_graph` | `TNGraph` | Parsed graph structure. |
| `qubits` | `list[str]` | Raw qubit lines (rows of ASCII graph). |
| `nqubits` | `int` | Number of qubit rows. |
| `cores` | `list[str]` | Ordered list of core names (opt_einsum symbols). |
| `ncores` | `int` | Number of core tensors. |
| `adjacency_table` | `list[dict]` | Sole topology source of truth (see §6). |
| `cores_weights` | `dict[str, TNTensor]` | Trainable parameters. |
| `backend` | `ComputeBackend \| None` | Attached compute backend. |
| `_submodules` | `dict[str, QCTN]` | Named child modules (composite mode). |

### Two Modes

| Mode | `graph` | `cores` | `_submodules` | Use case |
|---|---|---|---|---|
| **Graph-based** | ASCII string | non-empty | usually empty | leaf modules, raw QCTN |
| **Composite** | `None` | `[]` | non-empty | PlainMPS, TNEQ, Quadratic |

---

## 4. ASCII Graph Format

### Syntax

```
-[d]-[C]-[d]-[C]-[d]-
```

- `-` — separator / bond connector
- `[d]` — integer bond dimension
- `[C]` — single character (opt_einsum symbol): core tensor
- One line per qubit; uppercase or opt_einsum lowercase symbols

### Parsing Rules

1. Each row is parsed left-to-right.
2. Each core gets a `(left_bond, right_bond)` pair.
3. If a core is absent on a qubit row, that qubit does not connect to it.
4. Leading/trailing `-d-` segments are the physical (input/output) edge dimensions.

### Concrete Examples

```
# MPS (3 qubits, 3 cores, bond_dim=4, phys_dim=2)
-2-A-4-B-4-C-2-
-2-A-4-B-4-C-2-
-2-A-4-B-4-C-2-

# Tree (4 qubits, 3 cores)
-3-----a-3-
-3-b-3-a-3-
-3-b-3-c-3-
-3-----c-3-

# Brick-wall circuit (5 qubits, 2 layers)
-3-A-3---3-
-3-A-3-C-3-
-3-B-3-C-3-
-3-B-3-D-3-
-3-----D-3-

# CircuitState (ket, 3 qubits)
-A-2-
-B-2-
-C-2-

# MeasureMatrix (operator, 3 qubits)
-2-A-2-
-2-B-2-
-2-C-2-
```

---

## 5. TNTensor: Scale-Separated Tensor

$$\text{TNTensor} = \text{tensor} \times \text{scale}$$

| Attribute | Description |
|---|---|
| `_tensor` | Backend tensor (torch.Tensor / jnp.ndarray) |
| `scale` | Float scalar multiplier (prevents overflow/underflow) |
| `is_ref` | True → this is a view, not an owner |
| `is_transposed` | True → this is a conjugate-transpose view |
| `source` | Original TNTensor (for reference views) |

Reference semantics allow zero-copy siamese networks: right-side cores share memory with left-side cores via `conj_transpose()` views.

---

## 6. Adjacency Table Structure

Built by `QCTNGraphMixin._circuit_to_adjacency()` from the ASCII graph. One entry per core.

```python
adjacency_table[i] = {
    'core_idx':    int,         # index into self.cores
    'core_name':   str,         # e.g. 'A'
    'in_edge_list': [           # edges coming into this core
        {
            'neighbor_idx':  int,  # -1 = physical input (no core neighbor)
            'neighbor_name': str,
            'edge_rank':     int,  # bond dimension
            'qubit_idx':     int,  # which qubit row
        }, ...
    ],
    'out_edge_list': [...],     # same structure, edges going out
    'input_shape':  [int, ...], # per-qubit input bond dims
    'output_shape': [int, ...], # per-qubit output bond dims
    'input_dim':    int,        # product of input_shape
    'output_dim':   int,        # product of output_shape
}
```

**Invariant**: An internal edge between cores i and j appears in both `out_edge_list[i]` and `in_edge_list[j]` with matching `edge_rank`.

---

## 7. Contraction Pipeline

### Full Workflow

```
                  ┌─────────────┐
  ASCII graph ──► │    QCTN     │ ──► adjacency_table
                  │  (topology) │ ──► cores_weights (trainable)
                  └──────┬──────┘
                         │
              ┌──────────▼──────────┐
              │    EngineCommon     │   (shapes_info, per-qubit ops)
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  StrategyCompiler   │   estimate cost, select strategy
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   EinsumStrategy  GreedyStrategy  RowPriorityStrategy
   (opt_einsum)    (per-qubit)     (per-qubit + QCTN graph)
          │              │              │
          └──────────────┴──────────────┘
                         │
              ┌──────────▼──────────┐
              │   ComputeBackend    │   execute, grad, jit
              └──────────┬──────────┘
                         │
                      result (scalar / tensor)
```

### Siamese Pattern (Expectation Value)

For computing $\langle \psi | \hat{O} | \psi \rangle$:

```
CircuitState ──► MPS ──► MeasureMatrix ──► MPS† ──► CircuitState†
    cs              A          Mx              A†         cs†
```

Implemented as: `QCTN.concat(cs, mps).concat(mx).concat(mpsT).concat(csT)`

or as a single `Quadratic` application module.

---

## 8. Strategy Comparison

| Strategy | Graph Logic | Compilation | Best for |
|---|---|---|---|
| **EinsumStrategy** | `QCTN.get_einsum_info()` | opt_einsum expression | Small–medium networks |
| **GreedyStrategy** | Internal, manual | Per-qubit loop | Medium networks, fine control |
| **RowPriorityStrategy** | `QCTN.build_symmetric_expansion_graph()` | Per-qubit loop | Large symmetric networks |

`StrategyCompiler` selects the strategy by calling `estimate_cost()` on each compatible strategy.

---

## 9. Modules: Graph Patterns

### Leaf Modules

| Module | Graph pattern (3 qubits) | ncores | Physical meaning |
|---|---|---|---|
| `MPS(3, 4, 2)` | `-2-A-4-B-4-C-2-` × 3 rows | 3 | Matrix product state |
| `CircuitState(3, 2)` | `-A-2-` / `-B-2-` / `-C-2-` | 3 | Ket (product state input) |
| `MeasureMatrix(3, 2)` | `-2-A-2-` / `-2-B-2-` / `-2-C-2-` | 3 | Observable / quantum channel |

### Application Modules

| Module | Composition | Physical meaning |
|---|---|---|
| `PlainMPS` | 1× MPS | Single variational MPS |
| `TransposeMPS` | zero-copy view of MPS | Bra (conjugate-transpose) |
| `MPS_with_Ref` | left MPS + right = left† | Symmetric norm computation |
| `Encoding` | CircuitState + MPS | Feature embedding |
| `TNEQ` | MPS_left + MPS_right | Inner product $\langle\phi\|\psi\rangle$ |
| `Quadratic` | CS + MPS + Mx + MPS† + CS† | Quadratic form (expectation value) |

### Concat / Chunk

- `QCTN.concat(q1, q2)` — horizontal merge (left-right), renames cores contiguously
- `QCTN.chunk(split_idx)` — split by core index into two QCTNs

```
concat(CS, MPS, MX):
cs:   -a-2-                      (3 qubits, 1 core each)
mps:  -2-a-4-b-4-c-2-            (3 qubits, 3 cores)
mx:   -2-a-2-                    (3 qubits, 1 core each)

merged: -a-2-d-4-e-4-f-2-g-2-   (3 qubits, 5 cores a,d,e,f,g per row)
```

---

## 10. Backend Interface

```python
class ComputeBackend:
    def execute_expression(expression, *tensors) -> Tensor
    def compute_value_and_grad(loss_fn, argnums) -> (value, grads)
    def jit_compile(func) -> func
    def convert_to_tensor(array) -> Tensor
    def optimizer_update(params, grads, state, method, hyperparams)
    def get_backend_name() -> str
```

`BackendFactory.create_backend("pytorch", device="cpu", dtype="complex64")` returns a singleton backend instance.

---

## 11. Key Design Decisions

| Decision | Rationale |
|---|---|
| ASCII graph → adjacency_table | Human-readable topology spec; decouples structure from parameters |
| TNTensor scale factor | Numerical stability for deep networks; avoids overflow/underflow |
| Reference semantics (is_ref) | Zero-copy siamese networks; right side = conj(left) in memory |
| Mixin decomposition of QCTN | Each file < 700L; clear separation of graph / IO / contraction concerns |
| Graph parsing in QCTN, not contractor | Contraction strategies receive adjacency_table, not raw strings (clean interface) |
| Duck typing in `_qctn_contractor.py` | `hasattr(obj, 'adjacency_table')` avoids circular import with `isinstance(obj, QCTN)` |
| `_concat_impl` stays in `qctn.py` | `chunk()` and `_concat_impl` create new `QCTN(...)` instances; putting them in a separate mixin would cause circular imports |
| `cores` is a LIST | Order determines einsum index assignment; never rename |
| Composite mode (`graph=None`) | Enables hierarchical modules without requiring a flat graph |

---

## 12. Typical Usage (Code Sketch)

```python
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.modules import MPS, CircuitState, MeasureMatrix, Quadratic

backend = BackendFactory.create_backend("pytorch", device="cpu", dtype="complex64")

# Build a quadratic model: <cs | mps† · mx · mps | cs>
model = Quadratic(nqubits=4, bond_dim=8, phys_dim=2, backend=backend)
model.auto_init()

# Or build manually via concat:
cs  = CircuitState(4, 2, backend).auto_init()
mps = MPS(4, 8, 2, backend).auto_init()
mx  = MeasureMatrix(4, 2, backend).auto_init()
full = cs.concat_with(mps).concat_with(mx)  # → QCTN with all cores

# Contract:
from tneq_qc.core.engine_common import EngineCommon
engine = EngineCommon(backend=backend)
result = engine.contract_with_compiled_strategy(full, shapes_info={...})
```
