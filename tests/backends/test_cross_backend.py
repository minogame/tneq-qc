"""Cross-backend parity: JAX cotengra == PyTorch cotengra (value + gradient).

Validates the JAX functional-autodiff path, including hermit re-derivation for
the BornMachine.  Needs an environment with *both* torch and a working JAX
(the project's ``jax`` conda env has no torch, and some torch envs have a broken
JAX cuda plugin) — skipped gracefully otherwise.
"""
import os
# Force JAX onto CPU before it is imported: avoids broken cuda-plugin envs and
# keeps this a pure numerical CPU parity check.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from tneq_qc import QCTN, EngineCommon, BackendFactory, BornMachine, DataGenerator
from tneq_qc.utils.graph_generators import QCTNHelper
from tneq_qc.core.tn_tensor import TNTensor

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")


def _jax_works():
    try:
        BackendFactory.create_backend("jax", device="cpu", dtype="float32")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _jax_works(), reason="no working JAX backend")


def to_np(backend, x):
    if isinstance(x, TNTensor):
        x = x.tensor * x.scale
    return np.asarray(backend.tensor_to_numpy(x))


@pytest.fixture(scope="module")
def tb():
    return BackendFactory.create_backend("pytorch", device="cpu", dtype="float32")


@pytest.fixture(scope="module")
def jb():
    return BackendFactory.create_backend("jax", device="cpu", dtype="float32")


def test_tneq_grad_parity(tb, jb):
    """Student/teacher trace TNEQ: identical weights -> identical loss + grad."""
    N = 4
    g = "\n".join(f"-2-{chr(65 + i)}-2-" for i in range(N))

    def build(backend, weights=None, train=None):
        s = QCTN(g, backend=backend).auto_init(orthogonal=True)
        t = QCTN(g, backend=backend).auto_init(orthogonal=True)
        s.requires_grad_(True)
        c = QCTN.concat([("u", s), ("t", t)])
        c.set_trace("all")
        if weights is None:
            weights = {n: to_np(backend, c.cores_weights[n]) for n in c.cores}
            train = [n for n in c.cores if c.cores_weights[n].requires_grad]
        else:
            for n in c.cores:
                raw = backend.convert_to_tensor(weights[n].astype(np.float32))
                tt = raw if isinstance(raw, TNTensor) else TNTensor(raw)
                if n in train:
                    tt.requires_grad_(True)
                c.cores_weights[n] = tt
        return c, weights, train

    c_t, w, tr = build(tb)
    l_t, g_t = EngineCommon(backend=tb, strategy="cotengra").contract_for_gradient(
        c_t, target=1.0, loss="mse")
    c_j, _, _ = build(jb, w, tr)
    l_j, g_j = EngineCommon(backend=jb, strategy="cotengra").contract_for_gradient(
        c_j, target=1.0, loss="mse")

    assert float(to_np(tb, l_t)) == pytest.approx(float(to_np(jb, l_j)), abs=1e-4)
    assert len(g_t) == len(g_j) > 0
    for a, b in zip(g_t, g_j):
        np.testing.assert_allclose(to_np(tb, a), to_np(jb, b), atol=1e-3)


def test_bornmachine_grad_parity_hermit(tb, jb):
    """BornMachine: hermit branch must contribute on JAX too (no 0.5x gap).

    Weights are injected at the tn submodule before build() so build() rebuilds
    the tn_h = tn.hermit() linkage on both backends.
    """
    N, B, K, BATCH = 3, 2, 2, 8
    x = np.linspace(-1, 1, BATCH * N).reshape(BATCH, N).astype(np.float32)

    def build(backend, tn_w=None):
        torch.manual_seed(3)
        g = QCTNHelper.mps(N, bond_dim=B, phys_dim=K)
        m = BornMachine(g, K, backend=backend).auto_init(orthogonal=True)
        tn = m._submodules["tn"]
        if tn_w is None:
            tn_w = {n: to_np(backend, tn.cores_weights[n]) for n in tn.cores}
        else:
            for n in tn.cores:
                raw = backend.convert_to_tensor(tn_w[n])
                tn.cores_weights[n] = raw if isinstance(raw, TNTensor) else TNTensor(raw)
        tn.requires_grad_(True)
        c = m.build()
        dg = DataGenerator(backend, mx_K=K)
        mx, _ = dg.generate(x, K=K, ret_type="TNTensor")
        for nm, t in zip([k for k in c.cores if c.core_names.get(k, "").startswith("mx.")], mx):
            c[nm] = t
        return c, tn_w

    c_t, tn_w = build(tb)
    l_t, g_t = EngineCommon(backend=tb, strategy="cotengra").contract_for_gradient(
        c_t, target=1, loss="nll")
    c_j, _ = build(jb, tn_w)
    l_j, g_j = EngineCommon(backend=jb, strategy="cotengra").contract_for_gradient(
        c_j, target=1, loss="nll")

    assert float(to_np(tb, l_t)) == pytest.approx(float(to_np(jb, l_j)), abs=1e-4)
    gmag = max(float(np.max(np.abs(to_np(tb, a)))) for a in g_t)
    for a, b in zip(g_t, g_j):
        np.testing.assert_allclose(to_np(tb, a), to_np(jb, b), atol=1e-3 * max(gmag, 1.0))
