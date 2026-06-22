"""Cross-backend BornMachine grad check: jax cotengra vs torch cotengra (same weights+data)."""
import sys; sys.path.insert(0, "/home/wangyifeng/work/code/tneq-qc")
import numpy as np
from tneq_qc import EngineCommon, BackendFactory, BornMachine, DataGenerator
from tneq_qc.utils.graph_generators import QCTNHelper
from tneq_qc.core.tn_tensor import TNTensor

N, B, K, BATCH = 3, 2, 2, 8


def to_np(backend, x):
    if isinstance(x, TNTensor):
        x = x.tensor * x.scale
    return np.asarray(backend.tensor_to_numpy(x))


def build(backend, x, weights=None, train_names=None):
    import torch
    torch.manual_seed(3)
    g = QCTNHelper.mps(N, bond_dim=B, phys_dim=K)
    m = BornMachine(g, K, backend=backend).auto_init(orthogonal=True)
    m._submodules["tn"].requires_grad_(True)
    c = m.build()
    dg = DataGenerator(backend, mx_K=K)
    mx, _ = dg.generate(x, K=K, ret_type="TNTensor")
    mx_names = [n for n in c.cores if c.core_names.get(n, "").startswith("mx.")]
    if weights is not None:
        for n in c.cores:
            if n in [nm for nm, _ in zip(mx_names, mx)]:
                continue
            raw = backend.convert_to_tensor(weights[n])
            t = raw if isinstance(raw, TNTensor) else TNTensor(raw)
            if n in train_names:
                t.requires_grad_(True)
            c.cores_weights[n] = t
    for nm, t in zip(mx_names, mx):
        c[nm] = t
    if weights is None:
        weights = {n: to_np(backend, c.cores_weights[n]) for n in c.cores
                   if n not in mx_names}
        train_names = [n for n in c.cores if c.cores_weights[n].requires_grad]
    return c, weights, train_names


x = np.linspace(-1, 1, BATCH * N).reshape(BATCH, N).astype(np.float32)

tb = BackendFactory.create_backend("pytorch", device="cpu", dtype="float32")
c_t, weights, train_names = build(tb, x)
l_t, g_t = EngineCommon(backend=tb, strategy="cotengra").contract_for_gradient(c_t, target=1, loss="nll")
g_t = [to_np(tb, v) for v in g_t]

jb = BackendFactory.create_backend("jax", device="cpu", dtype="float32")
c_j, _, _ = build(jb, x, weights, train_names)
l_j, g_j = EngineCommon(backend=jb, strategy="cotengra").contract_for_gradient(c_j, target=1, loss="nll")
g_j = [to_np(jb, v) for v in g_j]

print(f"train_names={train_names}")
print(f"torch loss={float(to_np(tb,l_t)):.6f}  jax loss={float(to_np(jb,l_j)):.6f}  |dl|={abs(float(to_np(tb,l_t))-float(to_np(jb,l_j))):.2e}")
gerr = max(float(np.max(np.abs(a - b))) for a, b in zip(g_t, g_j))
gmag = max(float(np.max(np.abs(a))) for a in g_t)
print(f"grad: torch max-abs={gmag:.4e}  max|jax-torch diff|={gerr:.4e}  ratio={gerr/gmag:.3f}")
print("RESULT:", "MATCH" if gerr < 1e-3 else "MISMATCH (hermit branch likely missing)")
