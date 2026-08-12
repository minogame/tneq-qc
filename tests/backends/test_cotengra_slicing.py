"""Cotengra strategy: slicing correctness and device parity (PyTorch)."""
import numpy as np
import pytest

from tneq_qc import QCTN, EngineCommon, BackendFactory
from tneq_qc.core.tn_tensor import TNTensor

torch = pytest.importorskip("torch")


def _born(backend, n=4, bond=3, phys=2, batch=16, target_slices=1):
    from tneq_qc import BornMachine, DataGenerator
    from tneq_qc.utils.graph_generators import QCTNHelper
    torch.manual_seed(1)
    g = QCTNHelper.mps(n, bond_dim=bond, phys_dim=phys)
    m = BornMachine(g, phys, backend=backend).auto_init(orthogonal=True)
    m._submodules["tn"].requires_grad_(True)
    c = m.build()
    if target_slices > 1:
        c._cotengra_target_slices = target_slices
    dg = DataGenerator(backend, mx_K=phys)
    x = np.linspace(-1, 1, batch * n).reshape(batch, n).astype(np.float32)
    mx, _ = dg.generate(x, K=phys, ret_type="TNTensor")
    for nm, t in zip([k for k in c.cores if c.core_names.get(k, "").startswith("mx.")], mx):
        c[nm] = t
    return c


def _loss_grad(backend, c, strategy):
    loss, grads = EngineCommon(backend=backend, strategy=strategy).contract_for_gradient(
        c, target=1, loss="nll")
    gp = [np.asarray(backend.tensor_to_numpy(g)) for g in grads]
    return float(np.asarray(backend.tensor_to_numpy(loss))), gp


@pytest.mark.parametrize("target_slices", [1, 8])
def test_slicing_matches_row_priority(target_slices):
    """cotengra (sliced or not) reproduces row_priority's loss and gradients."""
    b = BackendFactory.create_backend("pytorch", device="cpu", dtype="float32")
    l_rp, g_rp = _loss_grad(b, _born(b), "row_priority")
    l_ct, g_ct = _loss_grad(b, _born(b, target_slices=target_slices), "cotengra")
    assert l_ct == pytest.approx(l_rp, abs=1e-4)
    for a, c in zip(g_rp, g_ct):
        np.testing.assert_allclose(a, c, atol=1e-3)


def test_slicing_actually_slices():
    """target_slices>1 produces nslices>1 on a network with contractable bonds."""
    b = BackendFactory.create_backend("pytorch", device="cpu", dtype="float32")
    c = _born(b, target_slices=8)
    _loss_grad(b, c, "cotengra")
    assert getattr(c, "_cotengra_planner").nslices > 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_cotengra_runs_on_gpu_and_matches_row_priority():
    """On the GPU, cotengra reproduces row_priority's loss + gradients."""
    b = BackendFactory.create_backend("pytorch", device="cuda", dtype="float32")
    assert "cuda" in b.backend_info.device
    l_rp, g_rp = _loss_grad(b, _born(b), "row_priority")
    l_ct, g_ct = _loss_grad(b, _born(b), "cotengra")
    assert l_ct == pytest.approx(l_rp, abs=1e-3)
    for a, c in zip(g_rp, g_ct):
        np.testing.assert_allclose(a, c, atol=1e-3)
