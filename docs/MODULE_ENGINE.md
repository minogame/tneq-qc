# Engine Module Documentation

> EngineCommon — Unified Contraction and Training Engine

---

## 1. Overview

`EngineCommon` is the execution core of tneq-qc, integrating **strategy compilation** (StrategyCompiler) and **backend execution** (ComputeBackend) into a unified interface. Users do not need to worry about the underlying contraction path optimization details — simply call `contract` or `contract_for_gradient`.

```python
from tneq_qc import EngineCommon, BackendFactory

backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
engine  = EngineCommon(backend=backend, strategy='row_priority')
```

---

## 2. Constructor Parameters

```python
EngineCommon(
    backend: str | ComputeBackend = None,   # Backend, defaults to framework global default
    strategy: str | list[str] = 'row_priority',
    nqubits: int = None,                     # Number of qubits, None to infer from QCTN
)
```

| Parameter | Description |
|---|---|
| `backend` | `'pytorch'` / `'jax'` or an already created `ComputeBackend` instance |
| `strategy` | Strategy name, e.g. `'row_priority'`, `'einsum_default'`, or a registered custom strategy |
| `nqubits` | Usually does not need to be specified manually, automatically inferred from QCTN |

---

## 3. Core API

### 3.1 contract — Forward Contraction

```python
result = engine.contract(qctn)
```

Execution flow:
1. On the first call, selects the optimal strategy via `StrategyCompiler.compile()` and caches it
2. Collects all core tensors from `qctn.cores_weights`
3. Calls the compiled `compute_fn(cores_dict, None, None)` to perform contraction
4. Returns the contraction result (scalar or tensor, typically a `TNTensor`)

```python
result = engine.contract(combined)
# result is a TNTensor, actual value = result.tensor * result.scale
```

### 3.2 contract_for_gradient — Contraction + Gradient

```python
loss_val, grads = engine.contract_for_gradient(qctn, target=1.0, loss='nll')
```

Execution flow:
1. Compile strategy (same as contract, with caching)
2. Collect trainable leaf tensors (`requires_grad=True` and `is_leaf=True`)
3. Build loss closure: contraction → resolve target → compute loss
4. Call `backend.compute_value_and_grad()` to obtain value and gradients
5. Return `(loss_value, gradients)` tuple

**Parameter Description**:

| Parameter | Type | Description |
|---|---|---|
| `qctn` | `QCTN` | The tensor network to contract |
| `target` | `None / float / list / TNTensor / QCTN` | Learning target |
| `loss` | `None / str / callable / BaseLoss` | Loss function |

**Target Resolution Rules** (handled by `TargetResolver`):
- `None` → zero tensor
- `float` → scalar tensor
- `list` → converted to tensor
- `TNTensor` → used directly
- `QCTN` → contracted first, then the result is used

**Loss Resolution Rules** (handled by `LossRegistry`):
- `None` → default `DiagonalMSELoss`
- `'mse'` / `'nll'` / `'mae'` / `'fidelity'` → corresponding built-in Loss
- `callable` → wrapped as `FunctionalLoss`
- `BaseLoss` instance → used directly

### 3.3 calculate_probability — Probability Calculation

```python
prob = engine.calculate_probability(qctn, mx_dict)
```

Temporarily updates mx cores and contracts, returning a scalar probability value. The updated cores are restored before the method returns unless `restore=False` is passed.

For BornMachine-style networks the contraction already represents `<state|tn† · Mx · tn|state>`, so complex outputs are converted by taking the real part. The probability API does **not** apply an extra `|.|²`.

Preferred semantic wrappers:

```python
full = engine.full_probability(combined, full_mx_dict)
marginal = engine.marginal_probability(combined, {'mx.a': Mx_tensor})
```

```python
# Calculate marginal probability for a certain dimension.
# Unspecified mx cores are temporarily set to identity.
prob = engine.marginal_probability(combined, {'mx.a': Mx_tensor})
```

**Full/Marginal Probability**:
- `full_probability`: `mx_dict` must cover all mx cores.
- `marginal_probability`: unspecified mx cores are temporarily set to identity, equivalent to tracing out.
- `calculate_probability`: low-level compatibility API; it only applies the provided `mx_dict`.

### 3.4 sample — Sampling

```python
samples = engine.sample(
    qctn,
    data_generator,
    sample_core_names,    # List of human-readable names for mx cores
    num_samples=1000,
    bounds=(-3, 3),
    grid_size=500,
    use_marginal=False,
)
# samples: Tensor of shape (num_samples, len(sample_core_names))
```

Uses **inverse CDF autoregressive sampling**:

1. For the 1st dimension: compute marginal probability on the grid → CDF → inverse CDF sampling
2. Fix the sampled value for the 1st dimension (set as the corresponding Mx)
3. For the 2nd dimension: compute conditional probability P(x₂|x₁) → sample
4. Continue similarly for subsequent dimensions...

For independent marginal sampling, pass `use_marginal=True`. For discrete values, use `sample_discrete(...)` with a discrete generator.

```python
from tneq_qc import DiscreteDataGenerator

gen = DiscreteDataGenerator(backend, values=(0, 1), mx_K=4)
samples = engine.sample_discrete(
    combined,
    gen,
    model.mx_core_names,
    num_samples=1000,
    use_marginal=True,
)
```

---

## 4. Usage Patterns

### 4.1 Basic Training

```python
engine = EngineCommon(backend=backend, strategy='row_priority')
optimizer = SGDG(combined.parameters(), backend, lr=0.01)

for step in range(N_STEPS):
    # Generate one batch, convert it to Mx, then assign Mx cores directly.
    loss_val, grads = engine.contract_for_gradient(combined,         # Contraction + gradient
                                                    target=1, loss='nll')
    optimizer.step(list(grads))                                      # Update parameters
```

### 4.2 TNEQ Inner Product Training

```python
# No data injection, contract directly
loss_val, grads = engine.contract_for_gradient(combined, target=1.0, loss='mse')
optimizer.step(list(grads))
```

### 4.3 Inference (No Gradient)

```python
import torch

with torch.no_grad():
    result = engine.contract(combined)
    prob   = engine.full_probability(combined, mx_dict)
```

### 4.4 Probability Heatmap

```python
# Calculate marginal probability for each qubit dimension
for dim_idx, core_name in enumerate(mx_core_names):
    # Set all other mx cores to identity
    for name in mx_core_names:
        combined[name] = identity_tensor

    # Iterate over the grid for this dimension
    for gi, x_val in enumerate(grid):
        Mx_list, _ = data_gen.generate([[x_val]], K=K, ret_type='TNTensor')
        prob = engine.marginal_probability(combined, {core_name: Mx_list[0]})
        heatmap[dim_idx, gi] = prob
```

---

## 5. Strategy Caching

`contract` and `contract_for_gradient` cache the compiled strategy on the QCTN instance's attributes (`_compiled_strategy_{strategy}`). Therefore:

- First call: compile + execute (slower)
- Subsequent calls: execute directly (fast)
- If the QCTN's structure (graph/number of cores) changes, the QCTN instance needs to be recreated

---

## 6. QubitOp Enum

`EngineCommon` internally uses the `QubitOp` enum to describe the operation type for each qubit:

| Enum Value | Description |
|---|---|
| `TRACE` | Trace (sum out this dimension) |
| `CIRCUIT_LEFT` | Left-multiply by state vector |
| `CIRCUIT_RIGHT` | Right-multiply by state vector |
| `CIRCUIT_BOTH` | Multiply by state on both left and right |
| `MEASURE` | Apply measurement matrix Mx |
| `IDENTITY` | Keep unchanged (explicit identity) |

These operations are automatically determined by the QCTN's structure during the `build_graph()` phase; users typically do not need to set them manually.
