# Tutorial: TNEQ Inner Product Training

> Train a student tensor network to approximate a teacher tensor network

---

## 1. Task Description

TNEQ (Tensor Network EQuality) is the most basic training scenario: given a fixed teacher tensor network, train a student tensor network so that the trace inner product $\mathrm{tr}(\text{student} \cdot \text{teacher}^\dagger)$ approaches 1.0.

**Mathematical objective**:

$$\min_{\theta} \left| \mathrm{tr}(S(\theta) \cdot T^\dagger) - 1.0 \right|^2$$

where $S$ is the student and $T$ is the teacher.

**Characteristics**:
- No external data required
- Pure parameter optimization
- Uses MSE loss
- Uses trace operations

---

## 2. Complete Code

```python
import torch
from tneq_qc import QCTN, EngineCommon, BackendFactory, SGDG
from tneq_qc.core.tn_tensor import TNTensor

# ====================== Configuration ======================
N_QUBITS = 4       # Number of qubits
PHYS_DIM = 2       # Physical dimension
N_STEPS  = 500     # Training steps
LR       = 0.01    # Learning rate

# ====================== Initialization ======================

# 1. Create backend and engine
backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')
engine  = EngineCommon(backend=backend, strategy_mode='full')

# 2. Define graph (one independent core per qubit)
graph = "\n".join(
    f"-{PHYS_DIM}-{chr(ord('A') + i)}-{PHYS_DIM}-"
    for i in range(N_QUBITS)
)
# Generated graph looks like:
# -2-A-2-
# -2-B-2-
# -2-C-2-
# -2-D-2-

# 3. Create teacher (fixed) and student (trainable)
teacher = QCTN(graph, backend=backend).auto_init()
student = QCTN(graph, backend=backend).auto_init()
student.requires_grad_(True)

# 4. Concatenate and set full trace
combined = QCTN.concat([('u', student), ('t', teacher)])
combined.set_trace('all')   # Trace over all qubits → scalar result

# ====================== Training ======================

optimizer = SGDG(combined.parameters(), backend, lr=LR)

for step in range(1, N_STEPS + 1):
    # Contract + compute MSE(Tr(combined), 1.0) + gradients
    loss_val, grads = engine.contract_for_gradient(
        combined,
        target=1.0,      # Target: trace = 1.0
        loss='mse',       # MSE loss
    )
    optimizer.step(list(grads))

    if step % 50 == 0:
        print(f"Step {step:4d}/{N_STEPS}  loss={float(loss_val):.6f}")

# ====================== Verification ======================

result = engine.contract(combined)
if isinstance(result, TNTensor):
    result.scale_to(1.0)
    trace_val = result.tensor
else:
    trace_val = result

print(f"Final trace: {float(trace_val.real):.6f}")  # Should be close to 1.0
```

---

## 3. Code Walkthrough

### 3.1 Graph Definition

```python
graph = "-2-A-2-\n-2-B-2-\n-2-C-2-\n-2-D-2-"
```

Each qubit has an independent 2x2 core. This is the simplest tensor network structure -- each core has no bond connections, only physical dimensions.

For more complex scenarios, an MPS structure can be used:

```python
graph = "-2-A-4-B-4-C-4-D-2-\n-2-A-4-B-4-C-4-D-2-\n..."
```

### 3.2 concat + set_trace

```python
combined = QCTN.concat([('u', student), ('t', teacher)])
combined.set_trace('all')
```

`concat` horizontally concatenates the student and teacher, so each qubit row becomes `student_core - teacher_core`.

`set_trace('all')` marks all qubits for trace mode -- during contraction these dimensions are summed over, yielding the scalar $\mathrm{tr}(S \cdot T)$.

### 3.3 target=1.0

We want $\mathrm{tr}(S \cdot T^\dagger) = 1.0$, meaning the student perfectly matches the teacher. Setting target to 1.0 with MSE loss gives the loss $(\mathrm{tr}(\cdot) - 1.0)^2$.

### 3.4 Gradient Propagation

`contract_for_gradient` only computes gradients for tensors with `requires_grad=True` and `is_leaf=True`. Since the teacher's cores do not have `requires_grad` set, gradients only flow to the student's parameters.

---

## 4. Saving and Loading

```python
import os

# Save student
os.makedirs("checkpoints", exist_ok=True)
student.save_cores("checkpoints/tneq_student.safetensors", metadata={
    'n_qubits': str(N_QUBITS),
    'final_loss': f"{float(loss_val):.6f}",
})

# Load and verify
student_loaded = QCTN(graph, backend=backend).auto_init()
student_loaded.load_cores("checkpoints/tneq_student.safetensors")

# Rebuild combined for verification
combined_val = QCTN.concat([('u', student_loaded), ('t', teacher)])
combined_val.set_trace('all')
result = engine.contract(combined_val)
```

---

## 5. Variants

### 5.1 Using MPS Structure

```python
from tneq_qc.modules.small import MPS

teacher = MPS(nqubits=4, bond_dim=4, phys_dim=2, backend=backend).auto_init()
student = MPS(nqubits=4, bond_dim=4, phys_dim=2, backend=backend).auto_init()
student.requires_grad_(True)

combined = QCTN.concat([('u', student), ('t', teacher)])
combined.set_trace('all')
```

### 5.2 Partial Trace

Trace over only some qubits, keeping the remaining qubits as output dimensions:

```python
combined.set_trace({0, 1})  # Trace over qubits 0 and 1 only
# Contraction result is a matrix rather than a scalar
```

### 5.3 Using the TNEQ Application Module

```python
from tneq_qc import TNEQ

model = TNEQ(nqubits=4, bond_dim=4, phys_dim=2, backend=backend).auto_init()
# TNEQ internally creates two independent MPS and sets full trace automatically
```
