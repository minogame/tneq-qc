# QCTN Module Documentation

> Quantum Circuit Tensor Network — The core data structure for tensor networks

---

## 1. Overview

`QCTN` is the core class of the tneq-qc framework, responsible for the following:

- **Holding network topology**: Parses ASCII diagram strings to generate `adjacency_table`
- **Managing parameter tensors**: `cores_weights` dictionary stores all core tensors (`TNTensor`)
- **Module composition**: Implements horizontal concatenation and splitting between modules via `concat` / `chunk`
- **Parameter collection**: `parameters()` / `named_parameters()` collects trainable parameters for use by optimizers
- **Reference operations**: `hermit()` / `bra()` creates zero-copy conjugate transpose views
- **Trace operations**: `set_trace()` / `clear_trace()` marks qubits for which a trace should be taken

The QCTN class is split into four parts via Mixins:

| Mixin | File | Responsibility |
|---|---|---|
| `QCTN` (main class) | `core/qctn.py` | Properties, submodule registration, parameter collection |
| `QCTNGraphMixin` | `core/_qctn_graph.py` | ASCII diagram parsing → adjacency_table |
| `QCTNIOMixin` | `core/_qctn_io.py` | Core initialization, save/load (safetensors) |
| `QCTNContractorMixin` | `core/_qctn_contractor.py` | Contraction interface: build_graph / get_einsum_info / concat |

---

## 2. Creating a QCTN

### 2.1 Creating from an ASCII Diagram

```python
from tneq_qc import QCTN, BackendFactory

backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')

# Define a 3-qubit MPS diagram
graph = """\
-2-A-4-B-4-C-2-
-2-A-4-B-4-C-2-
-2-A-4-B-4-C-2-"""

qctn = QCTN(graph, backend=backend).auto_init()

print(qctn.nqubits)    # 3
print(qctn.ncores)     # 3
print(qctn.cores)      # ['A', 'B', 'C']
```

`auto_init()` performs lazy initialization: parse diagram → create adjacency_table → randomly initialize core tensors (QR orthogonalization for square tensors, Gaussian for non-square), returns `self` for method chaining.

### 2.2 Using Predefined Modules

```python
from tneq_qc.modules.small import MPS, State, MeasureMatrix

mps = MPS(nqubits=4, bond_dim=8, phys_dim=2, backend=backend).auto_init()
state = State(nqubits=4, phys_dim=2, backend=backend).auto_init()
mx = MeasureMatrix(nqubits=4, phys_dim=2, backend=backend).auto_init()
```

### 2.3 Using Application Modules

```python
from tneq_qc import BornMachine
from tneq_qc import QCTNHelper

graph = QCTNHelper.mps(4, bond_dim=2, phys_dim=2)
model = BornMachine(graph, 2, backend=backend).auto_init(orthogonal=True)
combined = model.build()  # Returns the complete QCTN after concat
```

---

## 3. Core Attributes

| Attribute | Type | Description |
|---|---|---|
| `graph` | `str` | ASCII diagram string |
| `nqubits` | `int` | Number of qubits (number of rows) |
| `ncores` | `int` | Number of core tensors |
| `cores` | `list[str]` | List of core symbols, e.g., `['A', 'B', 'C']` |
| `cores_weights` | `dict[str, TNTensor]` | Symbol → core tensor |
| `core_names` | `dict[str, str]` | Symbol → readable name, e.g., `{'A': 'mps.a'}` |
| `adjacency_table` | `list` | Connection information for each core (edges, shapes, dims) |
| `trace_qubits` | `set` | Set of qubit indices to be traced |
| `_submodules` | `dict` | Submodule registry (composite mode) |

---

## 4. Core Operations

### 4.1 Parameter Management

```python
# Enable training
qctn.requires_grad_(True)

# Collect trainable parameters (for use by optimizers)
params = qctn.parameters()          # list[torch.Tensor]
named  = qctn.named_parameters()    # list[(name, tensor)]

# Iterate over all cores (including submodules)
for name, core in qctn.named_cores():
    print(f"{name}: shape={tuple(core.shape)}")
```

### 4.2 concat (Horizontal Concatenation)

Horizontally concatenate multiple QCTNs into a single complete network:

```python
# Named concatenation (recommended)
combined = QCTN.concat([
    ('state', state),
    ('tn', tn),
    ('mx', mx),
    ('tn_h', tn.hermit()),
    ('state_t', state.bra()),
])

# After concatenation, access cores via readable names
combined['tn.a']  # Access the first core of the MPS
combined['mx.a']  # Access the first core of the measure matrix
```

### 4.3 hermit / bra (Conjugate Transpose)

```python
tn_h = tn.hermit()        # Zero-copy conjugate transpose view
state_bra = state.bra()   # Bra state used to close the right boundary

# The hermit view shares parameters with the original QCTN
# Modifying the original parameters is automatically reflected in the hermit view
```

### 4.4 set_trace / clear_trace (Trace Operations)

```python
# Mark all qubits for trace
combined.set_trace('all')

# Mark specific qubits for trace
combined.set_trace({0, 2})

# Clear trace markings
combined.clear_trace()
```

### 4.5 Core Access and Modification

```python
# Access by symbol
core_A = qctn['A']                  # TNTensor
core_A = qctn.cores_weights['A']    # Equivalent

# Access by readable name (after concat)
mx_core = combined['mx.a']

# Modify a core (e.g., inject data)
combined['mx.a'] = new_tensor       # Automatically uses TNTensor.set()
```

### 4.6 Save and Load

```python
# Save (safetensors format)
qctn.save_cores("model.safetensors", metadata={
    'n_qubits': '4',
    'bond_dim': '2',
})

# Load
qctn_loaded = QCTN(graph, backend=backend).auto_init()
qctn_loaded.load_cores("model.safetensors")
```

---

## 5. TNTensor

`TNTensor` is the wrapper class for all core tensors, providing numerical stability and reference semantics.

### 5.1 Scale Mechanism

```
actual value = tensor × scale
```

```python
t = TNTensor(raw_tensor)
t.auto_scale()        # Normalize tensor so that max|·| = 1
t.scale_to(target)    # Adjust scale so that tensor * scale = target * old_tensor
```

Automatic scaling during contraction is disabled by default. Enable it on the backend when needed:

```python
backend = BackendFactory.create_backend(
    "pytorch",
    device="cpu",
    dtype="complex64",
    enable_auto_scale=True,
)
```

### 5.2 Reference Semantics

```python
t_ref = t.hermit()    # Conjugate transpose reference, no data copy
t_clone = t.clone()   # Independent copy

t_ref.is_ref          # True
t_ref.source          # Points to the original TNTensor
```

### 5.3 Transparent Proxy

TNTensor behaves like a regular tensor externally; all operations automatically manage scale:

```python
c = a + b       # Scale propagation
d = a * 2.0     # Scalar multiplication
e = a @ b       # matmul, scales are multiplied
f = a[0, :]     # Slicing, scale preserved

a.requires_grad   # Proxied to the underlying tensor
a.grad            # Proxied to the underlying tensor
a.shape           # Proxied to the underlying tensor
```

---

## 6. Typical Usage Patterns

### 6.1 BornMachine Model (Complete Workflow)

```python
from tneq_qc import QCTNHelper, BackendFactory, BornMachine, EngineCommon, DataGenerator, make_data_fn, SGDG

backend  = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
engine   = EngineCommon(backend=backend, strategy="row_priority")
data_gen = DataGenerator(backend, mx_K=2)

# Build model
graph = QCTNHelper.mps(4, bond_dim=2, phys_dim=2)
model = BornMachine(graph, 2, backend=backend).auto_init(orthogonal=True)
model._submodules['tn'].requires_grad_(True)
combined = model.build()

# Training
optimizer = SGDG(combined.parameters(), backend, lr=0.01)
data_fn = make_data_fn(data_gen, combined, batch_size=128, K=2)

for step in range(500):
    data_fn(step)
    loss, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
    optimizer.step(list(grads))
```

### 6.2 TNEQ Inner Product (Parameter Matching)

```python
# Create teacher and student
teacher = QCTN(graph, backend=backend).auto_init()
student = QCTN(graph, backend=backend).auto_init()
student.requires_grad_(True)

# Concatenate + full trace
combined = QCTN.concat([('u', student), ('t', teacher)])
combined.set_trace('all')

# Train so that Tr(student * teacher) → 1.0
loss, grads = engine.contract_for_gradient(combined, target=1.0, loss='mse')
```
