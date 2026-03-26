# Tutorial: Quadratic Form Density Estimation Training

> Learning the probability density function of data distributions using quadratic-form tensor networks

---

## 1. Task Description

Quadratic (quadratic form) is the most fundamental training scenario in the tneq-qc framework. By training a tensor network $A$, we approximate the probability density of data $x$ via the quadratic form $\langle\psi(x)|A^\dagger M_x A|\psi(x)\rangle$.

**Mathematical Model**:

$$P(x) = \langle\psi(x)|A^\dagger \cdot M_x \cdot A|\psi(x)\rangle$$

Where:
- $|\psi(x)\rangle$: CircuitState, encoding data $x$ as a quantum state
- $A$: MPS parameters (trainable)
- $M_x$: Measurement matrix generated from Hermite polynomial basis
- $A^\dagger$: Zero-copy conjugate transpose of $A$

**Five-segment Structure**:

```
circuit ─── mps ─── mx ─── mps_hermit ─── circuit_bra
  (CS)      (TN)   (Mx)    (TN†)           (CS†)
```

**Loss Function**: NLL (Negative Log-Likelihood)

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log P(x_i)$$

---

## 2. Complete Code

```python
import torch
import numpy as np
from tneq_qc import (
    QCTN, EngineCommon, BackendFactory, Quadratic,
    DataGenerator, make_data_fn, SGDG,
)

# ====================== Configuration ======================
N_QUBITS   = 4         # Number of qubits (= data dimension)
BOND_DIM   = 2         # MPS bond dimension
PHYS_DIM   = 2         # Physical dimension (Hermite polynomial order)
BATCH_SIZE = 128        # Batch size
N_STEPS    = 1000       # Number of training steps
LR         = 0.01       # Learning rate

# ====================== Initialization ======================

# 1. Create backend, engine, and data generator
backend  = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
engine   = EngineCommon(backend=backend, strategy_mode='full')
data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

# 2. Build the Quadratic model
model = Quadratic(
    nqubits=N_QUBITS,
    bond_dim=BOND_DIM,
    phys_dim=PHYS_DIM,
    backend=backend,
).auto_init()

# 3. Initialize circuit cores (optional: alternating 0-1 pattern)
def init_circuit_01(qctn, backend):
    for c in qctn.cores:
        core = qctn.cores_weights[c]
        shape = tuple(core.shape)
        n = 1
        for d in shape:
            n *= d
        flat = torch.zeros(n, dtype=core.dtype)
        for i in range(n):
            flat[i] = float(i % 2)
        qctn.cores_weights[c] = backend.convert_to_tensor(flat.reshape(shape))
    return qctn

init_circuit_01(model._submodules['circuit'], backend)

# 4. Set MPS as trainable and build the full network
model._submodules['mps'].requires_grad_(True)
combined = model.build()

print(f"Combined: {combined.ncores} cores, {len(combined.parameters())} trainable")

# ====================== Training ======================

# 5. Create optimizer and data injection function
optimizer = SGDG(combined.parameters(), backend, lr=LR)
data_fn   = make_data_fn(data_gen, combined, batch_size=BATCH_SIZE, K=PHYS_DIM)

loss_history = []

for step in range(1, N_STEPS + 1):
    # a. Inject data: generate random x → Mx → set into combined's mx cores
    data_fn(step)

    # b. Contraction + gradient
    loss_val, grads = engine.contract_for_gradient(
        combined,
        target=1,          # Probability normalization target
        loss='nll',         # Negative log-likelihood
    )

    # c. Parameter update
    optimizer.step(list(grads))

    lv = float(loss_val)
    loss_history.append(lv)
    if step % 100 == 0:
        print(f"Step {step:4d}/{N_STEPS}  loss={lv:.6f}")

print(f"\nInitial: {loss_history[0]:.6f}  Final: {loss_history[-1]:.6f}")
```

---

## 3. Code Walkthrough

### 3.1 Quadratic Model Construction

```python
model = Quadratic(nqubits=4, bond_dim=2, phys_dim=2, backend=backend).auto_init()
combined = model.build()
```

`Quadratic` internally creates:
- `circuit`: CircuitState (4 cores, one per qubit)
- `mps`: MPS (chain structure, bond_dim=2)
- `mx`: MeasureMatrix (4 cores, one per qubit)

`build()` performs concat, generating the five-segment structure:

```
combined = QCTN.concat([
    ('cs',   circuit),
    ('tn',   mps),
    ('mx',   measure_matrix),
    ('tn_h', mps.hermit()),        # Zero-copy conjugate transpose
    ('cs_t', circuit.bra()),       # Zero-copy conjugate transpose
])
```

### 3.2 DataGenerator and make_data_fn

```python
data_gen = DataGenerator(backend, mx_K=PHYS_DIM)
data_fn  = make_data_fn(data_gen, combined, batch_size=128, K=2)
```

`DataGenerator` uses a Hermite polynomial basis to convert a scalar $x$ into a $K \times K$ measurement matrix $M_x$.

`make_data_fn` returns a closure that, when called:
1. Generates `batch_size` random samples $x \sim \text{Uniform}(-1, 1)$
2. For each sample, calls `data_gen.generate()` to produce the Mx list
3. Injects the Mx values into the corresponding mx cores of `combined`

```python
data_fn(step)   # Injects new random data at each step
```

### 3.3 NLL Loss

```python
loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
```

`target=1` combined with `loss='nll'`:
- The contraction result $P(x)$ represents probability
- NLL = $-\log P(x)$
- Minimizing NLL is equivalent to maximizing likelihood

### 3.4 SGDG Optimizer

```python
optimizer = SGDG(combined.parameters(), backend, lr=0.01)
```

SGDG (Stiefel Gradient Descent) preserves parameter orthogonality via the Cayley transform, and is the recommended optimizer for density estimation tasks. The orthogonality constraint helps:
- Maintain probability normalization
- Prevent parameter degeneracy
- Improve numerical stability

---

## 4. Model Saving and Loading

```python
import os

# Save MPS parameters (only the trainable part)
os.makedirs("checkpoints", exist_ok=True)
model._submodules['mps'].save_cores("checkpoints/quadratic_mps.safetensors", metadata={
    'n_qubits': str(N_QUBITS),
    'bond_dim': str(BOND_DIM),
    'n_steps': str(N_STEPS),
    'final_loss': f"{loss_history[-1]:.6f}",
})
```

---

## 5. Inference: Probability Computation and Sampling

### 5.1 Marginal Probability Heatmap

After training, we can compute the marginal probability $P(x_i)$ for each dimension:

```python
mx_core_names = model.mx_core_names   # ['mx.a', 'mx.b', ...]

# Identity tensor (used to trace out unqueried dimensions)
ident = backend.eye(PHYS_DIM)
from tneq_qc.core.tn_tensor import TNTensor
ident = TNTensor(ident)

grid = np.linspace(-3, 3, 100).astype(np.float32)
heatmap = np.zeros((len(mx_core_names), len(grid)))

for dim_idx, core_name in enumerate(mx_core_names):
    # Set all mx cores to identity
    for name in mx_core_names:
        combined[name] = ident

    # Iterate over the grid for this dimension
    for gi, x_val in enumerate(grid):
        Mx_list, _ = data_gen.generate(
            np.array([[x_val]], dtype=np.float32), K=PHYS_DIM, ret_type='TNTensor'
        )
        combined[core_name] = Mx_list[0]
        prob = engine.calculate_probability(combined, {})
        heatmap[dim_idx, gi] = max(prob, 0.0)
```

### 5.2 Sampling

Using inverse CDF autoregressive sampling:

```python
with torch.no_grad():
    samples = engine.sample(
        combined,
        data_gen,
        mx_core_names,           # Dimensions to sample in order
        num_samples=2000,
        bounds=(-3, 3),
        grid_size=500,
    )
# samples: Tensor of shape (2000, N_QUBITS)
```

### 5.3 KL Divergence Evaluation

```python
def estimate_kl(train_samples, model_samples, n_bins=50, bounds=(-3, 3)):
    """Estimate KL divergence using histograms."""
    n_dims = train_samples.shape[1]
    kl_per_dim = []
    for d in range(n_dims):
        p_hist, edges = np.histogram(train_samples[:, d], bins=n_bins,
                                      range=bounds, density=True)
        q_hist, _ = np.histogram(model_samples[:, d], bins=n_bins,
                                  range=bounds, density=True)
        eps = 1e-10
        p_hist, q_hist = p_hist + eps, q_hist + eps
        dx = edges[1] - edges[0]
        p_hist = p_hist / (p_hist.sum() * dx)
        q_hist = q_hist / (q_hist.sum() * dx)
        kl_per_dim.append(np.sum(p_hist * np.log(p_hist / q_hist) * dx))
    return kl_per_dim
```

---

## 6. Variants

### 6.1 Large-Scale MPS (1024 qubits)

```python
model = Quadratic(nqubits=1024, bond_dim=2, phys_dim=2, backend=backend).auto_init()
engine = EngineCommon(backend=backend, strategy_mode='balanced')  # RowPriority
```

### 6.2 Brickwall Topology

Instead of the Quadratic module, build manually:

```python
from tneq_qc.utils.graph_generators import QCTNHelper

brickwall_graph = QCTNHelper.brickwall(nqubits=32, phys_dim=4)
brickwall = QCTN(brickwall_graph, backend=backend).auto_init()

# Manual five-segment concatenation
combined = QCTN.concat([
    ('cs',   circuit),
    ('tn',   brickwall),
    ('mx',   measure),
    ('tn_h', brickwall.hermit()),
    ('cs_t', circuit.bra()),
])
```

### 6.3 Distributed Training

See [MODULE_DISTRIBUTE.md](MODULE_DISTRIBUTE.md). Key difference:

```python
from tneq_qc.distributed import EngineDistributed
from tneq_qc.distributed.engine.distributed_engine import PartitionConfig

engine = EngineDistributed(
    backend=backend, strategy_mode='full', comm=comm,
    partition_config=PartitionConfig(strategy='layer', num_partitions=world_size),
)
engine.init_distributed(combined)
```

---

## 7. Summary

| Element | Choice |
|---|---|
| Model Structure | Quadratic = CS + MPS + Mx + MPS† + CS† |
| Loss Function | NLL (standard choice for density estimation) |
| Optimizer | SGDG (preserves orthogonality) |
| Data | DataGenerator + make_data_fn for automatic injection |
| dtype | complex64 (quantum states are complex-valued) |
| target | 1 (probability normalization) |
| strategy_mode | `'full'` (small-scale) or `'balanced'` (large-scale) |
