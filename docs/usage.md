# Usage Notes

This document uses the current public interfaces:

- `State` for product-state modules.
- `BornMachine(graph, dim, backend=..., mx_graph=None)` for density models.
- `EngineCommon(strategy="row_priority")` or another registered strategy name.

## BornMachine Training Pattern

```python
import numpy as np

from tneq_qc import (
    BackendFactory,
    BornMachine,
    DataGenerator,
    EngineCommon,
    QCTNHelper,
    create_optimizer,
)

backend = BackendFactory.create_backend("pytorch", device="cpu", dtype="complex64")
engine = EngineCommon(backend=backend, strategy="row_priority")
data_gen = DataGenerator(backend, mx_K=2)

graph = QCTNHelper.mps(nqubits=32, bond_dim=2, phys_dim=2)
model = BornMachine(graph, 2, backend=backend).auto_init(orthogonal=True)
model._submodules["tn"].requires_grad_(True)
combined = model.build()

optimizer = create_optimizer("sgdg", combined.parameters(), backend=backend, lr=0.01)
mx_core_names = model.mx_core_names

for step in range(100):
    x = np.random.randn(128, 32).astype(np.float32)
    mx_list, _ = data_gen.generate(x, K=2, ret_type="TNTensor")
    for name, mx in zip(mx_core_names, mx_list):
        combined[name] = mx

    loss, grads = engine.contract_for_gradient(combined, target=1, loss="nll")
    optimizer.step(list(grads))
```

## Manual Five-Segment Composition

Use manual `QCTN.concat` when the trainable `tn` is not a plain MPS:

```python
from tneq_qc import QCTN
from tneq_qc.modules.small import State, MeasureMatrix

state = State(nqubits=32, phys_dim=2, backend=backend).auto_init()
tn = QCTN(QCTNHelper.brickwall(32, n_layers=4, phys_dim=2), backend=backend)
tn.auto_init(orthogonal=True).requires_grad_(True)
mx = MeasureMatrix(nqubits=32, phys_dim=2, backend=backend).auto_init()

combined = QCTN.concat([
    ("state", state),
    ("tn", tn),
    ("mx", mx),
    ("tn_h", tn.hermit()),
    ("state_t", state.bra()),
])
```
