# Optimizer Module Documentation

> Optimizer, registry, and learning-rate scheduling

---

## 1. Overview

tneq-qc now uses a backend-decoupled optimizer design.

- `EngineCommon` computes `loss, grads`
- optimizers own update rules and optimizer state
- backend participation is reduced to low-level tensor operations via `TensorOps`
- custom optimizers can be registered outside the framework

Public entry points:

```python
from tneq_qc import (
    Adam, SGD, SGDG, Momentum, RMSProp,
    StepLRScheduler,
    create_optimizer,
    register_optimizer,
    get_registered_optimizers,
)
```

---

## 2. Core Pieces

### 2.1 OptimizerBase

**Location**: `tneq_qc/optim/base.py`

All modern optimizers inherit from `OptimizerBase`.

Core interface:

```python
class OptimizerBase:
    def __init__(self, params, backend=None, ops=None, lr=0.01, **kwargs):
        ...

    def step(self, grads):
        ...

    def zero_grad(self):
        ...

    @property
    def state(self):
        ...

    def state_dict(self):
        ...

    def load_state_dict(self, state_dict):
        ...
```

Constructor notes:

| Parameter | Type | Description |
|---|---|---|
| `params` | `list[TNTensor]` | Trainable parameters, typically from `qctn.parameters()` |
| `backend` | `ComputeBackend \| None` | Compatibility shortcut; used to construct `BackendTensorOps` |
| `ops` | `TensorOps \| None` | Explicit tensor-ops adapter |
| `lr` | `float` | Learning rate |
| `**kwargs` | | Optimizer-specific hyperparameters |

Typical usage:

```python
loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
optimizer.step(list(grads))
```

### 2.2 TensorOps

**Location**: `tneq_qc/optim/ops.py`

Optimizers no longer depend on backend-owned optimizer algorithms. They only require a minimal tensor-op interface:

- `zeros_like`
- `sqrt`
- `conj`
- `abs_square`
- `copy_into`
- `replace`

The default adapter is `BackendTensorOps`, which wraps the existing backend.

### 2.3 Registry / Factory

**Location**: `tneq_qc/optim/registry.py`

Custom and built-in optimizers are instantiated through:

- `register_optimizer(name, optimizer_cls)`
- `get_registered_optimizers()`
- `create_optimizer(name, params, *, backend=None, ops=None, **kwargs)`

---

## 3. Built-in Optimizers

### 3.1 Adam

```python
optimizer = create_optimizer(
    "adam",
    params,
    backend=backend,
    lr=0.01,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
)
```

Use case: default choice for many unconstrained training tasks.

### 3.2 SGD

```python
optimizer = create_optimizer("sgd", params, backend=backend, lr=0.01)
```

Use case: simple baselines and debugging.

### 3.3 SGDG

`SGDG` is now supported by the new optimizer path and no longer relies on `backend.optimizer_update(...)`.

```python
optimizer = create_optimizer(
    "sgdg",
    params,
    backend=backend,
    lr=0.01,
    momentum=0.9,
    stiefel=True,
)
```

Behavior:

- supports Stiefel-manifold updates via Cayley transform
- handles complex tensors using conjugate transpose where required
- reshapes higher-order cores into matrices during the update, then restores the original shape

Use case: tensor-network training where orthogonality matters, especially NLL / density-estimation setups.

### 3.4 Momentum

```python
optimizer = create_optimizer(
    "momentum",
    params,
    backend=backend,
    lr=0.01,
    momentum=0.9,
)
```

### 3.5 RMSProp

```python
optimizer = create_optimizer(
    "rmsprop",
    params,
    backend=backend,
    lr=0.01,
    alpha=0.99,
    epsilon=1e-8,
)
```

---

## 4. Creating Optimizers

### 4.1 Recommended Pattern

Prefer the registry/factory API:

```python
from tneq_qc import create_optimizer

optimizer = create_optimizer(
    "sgdg",
    combined.parameters(),
    backend=backend,
    lr=0.01,
)
```

This is the pattern used by current `examples/train_*.py`.

### 4.2 Direct Class Construction

Direct construction is still supported:

```python
from tneq_qc import Adam

optimizer = Adam(combined.parameters(), backend=backend, lr=1e-3)
```

Use this when you already know the concrete optimizer class.

---

## 5. Custom Optimizers

You can define an optimizer outside the framework and register it.

```python
from tneq_qc import OptimizerBase, register_optimizer, create_optimizer

class MyOptimizer(OptimizerBase):
    method = "my_optimizer"

    def update_raw_params(self, params, grads, state, hyperparams):
        lr = hyperparams.get("learning_rate", 0.01)
        new_params = [p - lr * g for p, g in zip(params, grads)]
        return new_params, state

register_optimizer("my_optimizer", MyOptimizer)

optimizer = create_optimizer(
    "my_optimizer",
    qctn.parameters(),
    backend=backend,
    lr=1e-2,
)
```

Reference example:

- `examples/example_custom_optimizer.py`

Inspect registry contents:

```python
from tneq_qc import get_registered_optimizers

print(sorted(get_registered_optimizers().keys()))
```

---

## 6. Learning Rate Scheduling

### StepLRScheduler

**Location**: `tneq_qc/optim/lr_scheduler.py`

```python
from tneq_qc import StepLRScheduler

scheduler = StepLRScheduler(optimizer, [
    (0, 1e-2),
    (200, 1e-3),
    (800, 1e-4),
])

for step in range(1000):
    loss, grads = engine.contract_for_gradient(...)
    optimizer.step(list(grads))
    scheduler.step()
```

Note: `StepLRScheduler` updates `optimizer.lr` directly.

---

## 7. Parameter Retrieval

Typical parameter sources:

```python
params = combined.parameters()
params = model._submodules["mps"].parameters()
named = combined.named_parameters()
```

Ensure gradient tracking is enabled before training:

```python
model._submodules["mps"].requires_grad_(True)
```

---

## 8. Complete Example

```python
from tneq_qc import EngineCommon, StepLRScheduler, create_optimizer

engine = EngineCommon(backend=backend, strategy_mode="full")
optimizer = create_optimizer(
    "sgdg",
    combined.parameters(),
    backend=backend,
    lr=0.01,
    momentum=0.9,
    stiefel=True,
)
scheduler = StepLRScheduler(optimizer, [
    (0, 1e-2),
    (200, 5e-3),
    (500, 1e-3),
])

for step in range(1, 1001):
    data_fn(step)
    loss_val, grads = engine.contract_for_gradient(combined, target=1, loss="nll")
    optimizer.step(list(grads))
    scheduler.step()

    if step % 100 == 0:
        print(f"Step {step} loss={float(loss_val):.6f} lr={optimizer.lr:.6f}")
```

---

## 9. Legacy Optimizer Wrapper

**Location**: `tneq_qc/optim/optimizer.py`

The old `Optimizer` class still exists for backward compatibility, but it is now a legacy trainer-style wrapper around `create_optimizer(...)`.

Important notes:

- it emits `DeprecationWarning`
- new code should not use it
- it no longer owns the optimizer algorithm itself

Prefer:

```python
optimizer = create_optimizer("adam", params, backend=backend, lr=1e-3)
```

instead of:

```python
from tneq_qc.optim.optimizer import Optimizer
```

---

## 10. Recommendations

| Task Type | Recommended Optimizer | Reason |
|---|---|---|
| Density estimation (Quadratic + NLL) | `sgdg` | Orthogonality constraints preserve normalization |
| TNEQ inner-product matching | `sgdg` | Helps control scale and geometry |
| MNIST approximation | `adam` | Good default for unconstrained fitting |
| Debugging / sanity checks | `sgd` | Simplest update rule |

---

## 11. Related Docs

- `docs/PLAN_OPTIMIZER_DECOUPLING.md`
- `docs/MODULE_ENGINE.md`
- `docs/MODULE_OPTIMIZER.md`
