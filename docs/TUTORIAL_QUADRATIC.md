# Tutorial: BornMachine Density Estimation

`BornMachine` represents the five-segment contraction

```text
state -> tn -> mx -> tn_h -> state_t
```

and evaluates

```text
P(x) = <state| tn_h · Mx(x) · tn |state>
```

## Minimal Training Loop

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

N_QUBITS = 4
BOND_DIM = 2
PHYS_DIM = 2
BATCH_SIZE = 128

backend = BackendFactory.create_backend("pytorch", device="cpu", dtype="complex64")
engine = EngineCommon(backend=backend, strategy="row_priority")
data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

graph = QCTNHelper.mps(N_QUBITS, bond_dim=BOND_DIM, phys_dim=PHYS_DIM)
model = BornMachine(graph, PHYS_DIM, backend=backend).auto_init(orthogonal=True)
model._submodules["tn"].requires_grad_(True)
combined = model.build()

mx_core_names = model.mx_core_names
optimizer = create_optimizer("sgdg", combined.parameters(), backend=backend, lr=0.01)

for step in range(100):
    x = np.random.randn(BATCH_SIZE, N_QUBITS).astype(np.float32)
    mx_list, _ = data_gen.generate(x, K=PHYS_DIM, ret_type="TNTensor")
    for name, mx in zip(mx_core_names, mx_list):
        combined[name] = mx

    loss, grads = engine.contract_for_gradient(combined, target=1, loss="nll")
    optimizer.step(list(grads))
```

## Custom Mx Topology

By default, `BornMachine(graph, dim)` creates one local `dim x dim` mx core per
qubit. To make one mx core span multiple qubits, pass `mx_graph`:

```python
mx_graph = "-2-a-2-\n-2-a-2-"
model = BornMachine(graph, PHYS_DIM, backend=backend, mx_graph=mx_graph)
```
