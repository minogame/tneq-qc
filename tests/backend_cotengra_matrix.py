"""Single-process multi-backend test for the cotengra sliced strategy.

For a given (backend, device) it checks, using that backend's own row_priority
as the oracle:
  1. cotengra forward (1 slice)  == row_priority
  2. cotengra forward (8 slices) == row_priority   (slicing correctness)
  3. gradient via backend.compute_value_and_grad matches row_priority

Usage:
  python tests/backend_cotengra_matrix.py pytorch cpu
  python tests/backend_cotengra_matrix.py pytorch cuda
  python tests/backend_cotengra_matrix.py jax cpu
  python tests/backend_cotengra_matrix.py jax cuda
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from tneq_qc import EngineCommon, BackendFactory, BornMachine, DataGenerator
from tneq_qc.utils.graph_generators import QCTNHelper
from tneq_qc.core.tn_tensor import TNTensor

N, B, K, BATCH = 4, 3, 2, 16


def to_np(backend, x):
    if isinstance(x, TNTensor):
        x = x.tensor * x.scale
    return np.asarray(backend.tensor_to_numpy(x))


def build(backend, dg, seed=1):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = QCTNHelper.mps(N, bond_dim=B, phys_dim=K)
    m = BornMachine(g, K, backend=backend).auto_init(orthogonal=True)
    m._submodules["tn"].requires_grad_(True)
    c = m.build()
    x = np.linspace(-1, 1, BATCH * N).reshape(BATCH, N).astype(np.float32)
    mx, _ = dg.generate(x, K=K, ret_type="TNTensor")
    names = [n for n in c.cores if c.core_names.get(n, "").startswith("mx.")]
    for nm, t in zip(names, mx):
        c[nm] = t
    return c


def run(backend_name, device):
    dtype = "float32"
    backend = BackendFactory.create_backend(backend_name, device=device, dtype=dtype)
    dg = DataGenerator(backend, mx_K=K)
    real_dev = backend.backend_info.device
    print(f"\n[{backend_name}/{device}] resolved device = {real_dev}")

    def loss_and_grad(strategy, target_slices=1):
        c = build(backend, dg)
        if target_slices > 1:
            c._cotengra_target_slices = target_slices
        eng = EngineCommon(backend=backend, strategy=strategy)
        loss, grads = eng.contract_for_gradient(c, target=1, loss="nll")
        return float(to_np(backend, loss)), [to_np(backend, g) for g in grads], c

    l_rp, g_rp, _ = loss_and_grad("row_priority")
    l_c1, g_c1, c1 = loss_and_grad("cotengra", 1)
    l_c8, g_c8, c8 = loss_and_grad("cotengra", 8)

    ns1 = getattr(c1, "_cotengra_planner").nslices
    ns8 = getattr(c8, "_cotengra_planner").nslices
    dl1, dl8 = abs(l_rp - l_c1), abs(l_rp - l_c8)
    ge1 = max(float(np.max(np.abs(a - b))) for a, b in zip(g_rp, g_c1))
    ge8 = max(float(np.max(np.abs(a - b))) for a, b in zip(g_rp, g_c8))

    print(f"  row_priority loss = {l_rp:.6f}")
    print(f"  cotengra(1)  loss = {l_c1:.6f}  nslices={ns1}  |dloss|={dl1:.2e}  |dgrad|={ge1:.2e}")
    print(f"  cotengra(8)  loss = {l_c8:.6f}  nslices={ns8}  |dloss|={dl8:.2e}  |dgrad|={ge8:.2e}")

    tol_l, tol_g = 1e-4, 1e-3   # float32 tolerance
    ok = dl1 < tol_l and dl8 < tol_l and ge1 < tol_g and ge8 < tol_g
    print(f"  RESULT[{backend_name}/{real_dev}]:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    bk = sys.argv[1] if len(sys.argv) > 1 else "pytorch"
    dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    sys.exit(0 if run(bk, dev) else 1)
