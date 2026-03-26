# Optimizer Module Documentation

> Optimizer and learning rate scheduling — PyTorch-style parameter update interface

---

## 1. Overview

tneq-qc provides a PyTorch-style optimizer framework located in `tneq_qc/optim/`. All optimizers inherit from `OptimizerBase` and internally delegate to `ComputeBackend.optimizer_update()` for actual parameter updates, ensuring backend transparency.

```python
from tneq_qc import Adam, SGD, SGDG, Momentum, RMSProp, StepLRScheduler
```

---

## 2. OptimizerBase

**Location**: `tneq_qc/optim/base.py`

Base class for all optimizers:

```python
class OptimizerBase:
    def __init__(self, params, backend, lr=0.01, **kwargs):
        ...

    def step(self, grads):
        """Update parameters with gradients."""
        ...

    def zero_grad(self):
        """Clear gradients."""
        ...

    @property
    def state(self):
        """Current optimizer state dictionary."""
        ...
```

**Constructor parameters**:

| Parameter | Type | Description |
|---|---|---|
| `params` | `list[Tensor]` | List of trainable parameters, typically obtained from `qctn.parameters()` |
| `backend` | `ComputeBackend` | Compute backend instance |
| `lr` | `float` | Learning rate, default 0.01 |
| `**kwargs` | | Additional hyperparameters passed to specific optimizers |

**Core methods**:

```python
# Standard training loop
loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
optimizer.step(list(grads))   # grads is a list of gradient tensors
```

---

## 3. Built-in Optimizers

### 3.1 Adam

Adaptive moment estimation optimizer.

```python
optimizer = Adam(params, backend, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)
```

| Parameter | Default | Description |
|---|---|---|
| `lr` | 0.01 | Learning rate |
| `beta1` | 0.9 | First moment decay rate |
| `beta2` | 0.999 | Second moment decay rate |
| `eps` | 1e-8 | Numerical stability constant |

**Use case**: Default choice for most training tasks.

### 3.2 SGD

Stochastic gradient descent.

```python
optimizer = SGD(params, backend, lr=0.01)
```

**Use case**: Simple tasks, debugging.

### 3.3 SGDG

Gradient descent on the Stiefel manifold (Cayley transform), guaranteeing that parameters remain orthogonal after updates.

```python
optimizer = SGDG(params, backend, lr=0.01)
```

**Use case**: Recommended choice for tensor network training. Orthogonality constraints help maintain numerical stability and prevent parameter degeneration. Particularly effective for density estimation (NLL loss) tasks.

### 3.4 Momentum

SGD with momentum.

```python
optimizer = Momentum(params, backend, lr=0.01, momentum=0.9)
```

| Parameter | Default | Description |
|---|---|---|
| `momentum` | 0.9 | Momentum coefficient |

### 3.5 RMSProp

Root mean square propagation.

```python
optimizer = RMSProp(params, backend, lr=0.01, alpha=0.99, eps=1e-8)
```

| Parameter | Default | Description |
|---|---|---|
| `alpha` | 0.99 | Moving average decay rate |
| `eps` | 1e-8 | Numerical stability constant |

---

## 4. Learning Rate Scheduling

### StepLRScheduler

Step-wise learning rate decay at fixed intervals:

```python
scheduler = StepLRScheduler(optimizer, step_size=100, gamma=0.5)

for step in range(1000):
    loss, grads = engine.contract_for_gradient(...)
    optimizer.step(list(grads))
    scheduler.step()   # Every 100 steps, multiply lr by 0.5
```

| Parameter | Description |
|---|---|
| `optimizer` | The optimizer instance to schedule |
| `step_size` | Number of steps between each decay |
| `gamma` | Decay factor (new lr = old lr × gamma) |

---

## 5. Usage Recommendations

### 5.1 Optimizer Selection

| Task Type | Recommended Optimizer | Reason |
|---|---|---|
| Density estimation (Quadratic + NLL) | `SGDG` | Orthogonality constraints preserve normalization |
| TNEQ inner product matching | `SGDG` | Orthogonality constraints prevent scale explosion |
| MNIST approximation | `Adam` | Adaptive learning rate suits non-convex optimization |
| Large-scale training | `SGDG` + `StepLRScheduler` | Stable convergence |

### 5.2 Parameter Retrieval

```python
# Retrieve directly from QCTN
params = combined.parameters()

# Retrieve from submodules (train only the MPS part)
params = model._submodules['mps'].parameters()

# Ensure requires_grad is set
model._submodules['mps'].requires_grad_(True)
```

### 5.3 Complete Example

```python
from tneq_qc import SGDG, StepLRScheduler, EngineCommon

engine    = EngineCommon(backend=backend, strategy_mode='full')
optimizer = SGDG(combined.parameters(), backend, lr=0.01)
scheduler = StepLRScheduler(optimizer, step_size=200, gamma=0.5)

for step in range(1, 1001):
    data_fn(step)
    loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
    optimizer.step(list(grads))
    scheduler.step()

    if step % 100 == 0:
        print(f"Step {step}  loss={float(loss_val):.6f}  lr={optimizer.lr:.6f}")
```

---

## 6. Loss Functions

Optimizers work in conjunction with loss functions. Loss functions are managed through `LossRegistry`:

### 6.1 Built-in Losses

| Name | Class | Description |
|---|---|---|
| `'mse'` | `MSELoss` | Mean squared error $(y - \hat{y})^2$ |
| `'mae'` | `MAELoss` | Mean absolute error $\|y - \hat{y}\|$ |
| `'nll'` | `NLLLoss` | Negative log-likelihood $-\log P$ |
| `'fidelity'` | `FidelityLoss` | Quantum fidelity $-\|\langle y \| \hat{y} \rangle\|^2$ |
| `'diagonal_mse'` | `DiagonalMSELoss` | Diagonal MSE (backward-compatible default) |

### 6.2 Usage

```python
# String name
loss_val, grads = engine.contract_for_gradient(qctn, target=1.0, loss='mse')

# Custom function
def my_loss(result, target, backend):
    return backend.sum((result - target) ** 2)

loss_val, grads = engine.contract_for_gradient(qctn, target=1.0, loss=my_loss)

# Custom class
from tneq_qc.losses import register_loss, BaseLoss

@register_loss('weighted_mse')
class WeightedMSE(BaseLoss):
    def __init__(self, weight=2.0):
        self.weight = weight
    def compute(self, result, target, backend):
        return self.weight * backend.mean((result - target) ** 2)
```
