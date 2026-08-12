"""Student-teacher MPS training on TPU via the JAX backend.

Device note: the JAX backend's gpu auto-detect would crash on a TPU-only box,
so we pass device='cpu' (skips the explicit gpu device_put); JAX then places
every array on its default backend, which here is the TPU.
"""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import jax
from tneq_qc import QCTN, EngineCommon, BackendFactory, create_optimizer
from tneq_qc.modules.small import MPS
from tneq_qc.core.tn_tensor import TNTensor

print("JAX devices:", jax.devices())

b = BackendFactory.create_backend("jax", device="cpu", dtype="float32")
np.random.seed(0)

NQ, BOND = 6, 4
teacher = MPS(nqubits=NQ, bond_dim=BOND, phys_dim=2, backend=b).auto_init(orthogonal=True)
student = MPS(nqubits=NQ, bond_dim=BOND, phys_dim=2, backend=b).auto_init(orthogonal=True)

# Perturb student away from teacher so there's a clear gradient signal.
for n in student.cores:
    tw = np.asarray(b.tensor_to_numpy(teacher.cores_weights[n]))
    sw = tw + 0.3 * np.random.randn(*tw.shape).astype(np.float32)
    t = b.convert_to_tensor(sw)
    t = t if isinstance(t, TNTensor) else TNTensor(t)
    t.requires_grad_(True)
    student.cores_weights[n] = t

combined = QCTN.concat([("u", student), ("t", teacher)])
combined.set_trace("all")

eng = EngineCommon(backend=b, strategy="cotengra")
opt = create_optimizer("adam", combined.parameters(), backend=b, lr=0.02)

pdev = list(combined.parameters()[0].tensor.devices())
print("param device:", pdev)

losses = []
t0 = time.time()
for step in range(200):
    loss, grads = eng.contract_for_gradient(combined, target=1.0, loss="mse")
    opt.step(list(grads))
    lv = float(np.asarray(b.tensor_to_numpy(loss)))
    losses.append(lv)
    if step % 20 == 0 or step == 199:
        print(f"step {step:3d}  loss={lv:.6e}")
dt = time.time() - t0

print(f"\nfinished {len(losses)} steps in {dt:.2f}s ({dt/len(losses)*1000:.1f} ms/step)")
print(f"loss {losses[0]:.6e} -> {losses[-1]:.6e}  decreased={losses[-1] < losses[0]*0.5}")
