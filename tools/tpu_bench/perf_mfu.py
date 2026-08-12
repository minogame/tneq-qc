"""Estimate TPU inference MFU (model FLOPs utilization).

Steps:
  1) Calibrate cotengra's contraction_cost() convention against a known matmul
     (true HW FLOPs = 2*M*N*K) so we know whether to multiply by 2.
  2) Measure the achievable matmul peak on THIS TPU (empirical ceiling).
  3) For each compute-heavy model config, time the JIT-fused contraction
     precisely and compute achieved TFLOP/s and MFU vs both the theoretical
     v6e bf16 peak and the empirical matmul peak.
"""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, jax, jax.numpy as jnp
from tneq_qc import QCTN, EngineCommon, BackendFactory
from tneq_qc.modules.small import MPS
from tneq_qc.modules.app import BornMachine
from tneq_qc.utils.graph_generators import QCTNHelper
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.contractor.cotengra_strategy import assemble_global_einsum
from tneq_qc.contractor.cotengra_planner import CotengraPlanner

raw = lambda t: t.tensor if isinstance(t, TNTensor) else t

# TPU v6e (Trillium) per-chip published peak.
PEAK_BF16 = 918e12   # 918 BF16 TFLOP/s

# ---- builders (scalar-output nets) ----
def build_mps(b, nq, bond):
    m = MPS(nqubits=nq, bond_dim=bond, phys_dim=2, backend=b).auto_init(orthogonal=True)
    c = QCTN.concat([("u", m), ("t", m.hermit())]); c.set_trace("all"); return c
def build_tneq(b, nq, bond):
    a = MPS(nqubits=nq, bond_dim=bond, phys_dim=2, backend=b).auto_init(orthogonal=True)
    z = MPS(nqubits=nq, bond_dim=bond, phys_dim=2, backend=b).auto_init(orthogonal=True)
    c = QCTN.concat([("u", a), ("t", z.hermit())]); c.set_trace("all"); return c
def build_born(b, nq, bond):
    g = QCTNHelper.mps(nq, bond_dim=bond, phys_dim=2)
    return BornMachine(g, 2, backend=b).auto_init(orthogonal=True).build()
def build_tree(b, nq, dim):
    g = QCTNHelper.generate_example_graph(nq, graph_type="tree", dim_char=str(dim))
    t = QCTN(g, backend=b).auto_init(orthogonal=True)
    c = QCTN.concat([("u", t), ("t", t.hermit())]); c.set_trace("all"); return c

def time_jit(f, arg, it=100):
    raw(f(arg)).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(it):
        r = f(arg)
    raw(r).block_until_ready()
    return (time.perf_counter() - t0) / it


def calibrate():
    """Single matmul: true FLOPs = 2 N^3. See what cotengra cost reports."""
    N = 2048
    pl = CotengraPlanner("ab,bc->ac", [(N, N), (N, N)], target_slices=1, target_size=None)
    cost = pl.contraction_cost()
    ratio = cost / (N**3)
    conv = "MAC (=N^3)" if abs(ratio - 1) < 0.2 else ("FLOP (=2N^3)" if abs(ratio - 2) < 0.4 else f"?({ratio:.2f}N^3)")
    return cost, ratio, conv


def matmul_peak():
    b = BackendFactory.create_backend("jax", device="tpu", dtype="float32")
    d = jax.devices("tpu")[0]
    best = 0.0
    for N in (2048, 4096):
        A = jax.device_put(jnp.ones((N, N), jnp.float32), d)
        B = jax.device_put(jnp.ones((N, N), jnp.float32), d)
        f = jax.jit(lambda x, y: x @ y)
        dt = time_jit(lambda _: f(A, B), None, it=50)
        tf = 2 * N**3 / dt / 1e12
        best = max(best, tf)
    return best


CONFIGS = [
    ("MPS  nq16 bond512",  build_mps,  dict(nq=16, bond=512)),
    ("MPS  nq32 bond512",  build_mps,  dict(nq=32, bond=512)),
    ("MPS  nq16 bond2048", build_mps,  dict(nq=16, bond=2048)),
    ("TNEQ nq16 bond2048", build_tneq, dict(nq=16, bond=2048)),
    ("Born nq16 bond512",  build_born, dict(nq=16, bond=512)),
    ("Tree nq16 dim64",    build_tree, dict(nq=16, dim=64)),
]


def main():
    print("JAX devices:", jax.devices())
    cost, ratio, conv = calibrate()
    print(f"\ncotengra cost calibration (2048^3 matmul): cost={cost:.3e}, "
          f"cost/N^3={ratio:.3f} -> convention = {conv}")
    flop_factor = 1.0 if ratio > 1.5 else 2.0   # convert cost -> HW FLOPs
    print(f"  => HW FLOPs = contraction_cost x {flop_factor:.0f}")

    peak_emp = matmul_peak()
    print(f"\nTPU v6e peak: theoretical bf16 = {PEAK_BF16/1e12:.0f} TFLOP/s | "
          f"empirical matmul (this chip, fp32-default) = {peak_emp:.0f} TFLOP/s "
          f"({peak_emp*1e12/PEAK_BF16*100:.0f}% of peak)")

    b = BackendFactory.create_backend("jax", device="tpu", dtype="float32")
    eng = EngineCommon(backend=b, strategy="cotengra")
    print(f"\n{'config':<20}{'GFLOP':>9}{'lat(ms)':>9}{'TFLOP/s':>9}"
          f"{'MFU_th':>8}{'MFU_emp':>8}")
    for label, fn, kw in CONFIGS:
        c = fn(b, **kw)
        raw(eng.contract(c)).block_until_ready()
        eq, rt, *_ = assemble_global_einsum(c)
        arrays = [jnp.asarray(raw(t)) for t in rt]
        pl = c._cotengra_planner
        gflops_hw = pl.contraction_cost() * flop_factor / 1e9
        f = jax.jit(lambda a: pl.contract(a))
        lat = time_jit(f, arrays, it=100)
        achieved = gflops_hw / lat / 1e3    # TFLOP/s
        mfu_th = achieved * 1e12 / PEAK_BF16 * 100
        mfu_emp = achieved / peak_emp * 100
        print(f"{label:<20}{gflops_hw:>9.1f}{lat*1e3:>9.4f}{achieved:>9.1f}"
              f"{mfu_th:>7.1f}%{mfu_emp:>7.1f}%")


if __name__ == "__main__":
    main()
