"""Cross-backend TNEQ gradient check: jax cotengra vs torch cotengra (same weights)."""
import sys; sys.path.insert(0, "/home/wangyifeng/work/code/tneq-qc")
import numpy as np
from tneq_qc import QCTN, EngineCommon, BackendFactory
from tneq_qc.core.tn_tensor import TNTensor

N, K = 4, 2
g = "\n".join(f"-{K}-{chr(65+i)}-{K}-" for i in range(N))


def to_np(backend, x):
    if isinstance(x, TNTensor):
        x = x.tensor * x.scale
    return np.asarray(backend.tensor_to_numpy(x))


def build(backend, weights=None, train_names=None):
    teacher = QCTN(g, backend=backend).auto_init(orthogonal=True)
    student = QCTN(g, backend=backend).auto_init(orthogonal=True)
    student.requires_grad_(True)
    combined = QCTN.concat([("u", student), ("t", teacher)])
    combined.set_trace("all")
    if weights is None:
        weights = {n: to_np(backend, combined.cores_weights[n]) for n in combined.cores}
        train_names = [n for n in combined.cores if combined.cores_weights[n].requires_grad]
    else:
        for n in combined.cores:
            raw = backend.convert_to_tensor(weights[n].astype(np.float32))
            t = raw if isinstance(raw, TNTensor) else TNTensor(raw)
            if n in train_names:
                t.requires_grad_(True)
            combined.cores_weights[n] = t
    return combined, weights, train_names


# Reference weights from torch.
tb = BackendFactory.create_backend("pytorch", device="cpu", dtype="float32")
ct_ref, weights, train_names = build(tb)
l_t, g_t = EngineCommon(backend=tb, strategy="cotengra").contract_for_gradient(
    ct_ref, target=1.0, loss="mse")
g_t = [to_np(tb, x) for x in g_t]

# Same weights on jax.
jb = BackendFactory.create_backend("jax", device="cpu", dtype="float32")
ct_jax, _, _ = build(jb, weights, train_names)
l_j, g_j = EngineCommon(backend=jb, strategy="cotengra").contract_for_gradient(
    ct_jax, target=1.0, loss="mse")
g_j = [to_np(jb, x) for x in g_j]

print(f"torch loss={float(to_np(tb,l_t)):.6f}  jax loss={float(to_np(jb,l_j)):.6f}")
print(f"n grads: torch={len(g_t)} jax={len(g_j)}")
gmax = max(float(np.max(np.abs(x))) for x in g_j) if g_j else 0.0
print(f"jax grad max-abs (nonzero?): {gmax:.4e}")
if g_t and g_j and len(g_t) == len(g_j):
    gerr = max(float(np.max(np.abs(a - b))) for a, b in zip(g_t, g_j))
    dl = abs(float(to_np(tb, l_t)) - float(to_np(jb, l_j)))
    print(f"|loss diff|={dl:.2e}  max|grad diff|={gerr:.2e}")
    print("RESULT:", "PASS" if (dl < 1e-4 and gerr < 1e-3 and gmax > 1e-8) else "FAIL")
else:
    print("RESULT: FAIL (grad count mismatch)")
