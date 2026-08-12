"""Diagnose where TPU inference is bound: roofline via XLA cost_analysis.

For each contraction we compile the JIT-fused program and read XLA's own
flops + bytes-accessed, giving the true arithmetic intensity. Comparing AI to
the TPU's ridge point (peak FLOP/s / HBM BW) tells us compute- vs memory-bound.
We also split measured latency into a fixed launch floor + compute.
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

raw = lambda t: t.tensor if isinstance(t, TNTensor) else t

# TPU v6e (Trillium) per-chip peaks.
PEAK_BF16 = 918e12          # FLOP/s
HBM_BW    = 1640e9          # bytes/s (~1.64 TB/s)
RIDGE     = PEAK_BF16 / HBM_BW   # FLOP/byte ridge point

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

CONFIGS = [
    ("MPS  nq16 bond512",  build_mps,  dict(nq=16, bond=512)),
    ("MPS  nq32 bond512",  build_mps,  dict(nq=32, bond=512)),
    ("MPS  nq16 bond2048", build_mps,  dict(nq=16, bond=2048)),
    ("Born nq16 bond512",  build_born, dict(nq=16, bond=512)),
    ("Tree nq16 dim64",    build_tree, dict(nq=16, dim=64)),
]

def cost_of(compiled):
    ca = compiled.cost_analysis()
    if isinstance(ca, (list, tuple)):
        ca = ca[0]
    flops = float(ca.get("flops", 0.0))
    # XLA exposes bytes either as one 'bytes accessed' or per-operand keys.
    bytes_acc = ca.get("bytes accessed")
    if bytes_acc is None:
        bytes_acc = sum(v for k, v in ca.items() if k.startswith("bytes accessed"))
    return flops, float(bytes_acc or 0.0)

def time_jit(f, arg, it=100):
    raw(f(arg)).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(it):
        r = f(arg)
    raw(r).block_until_ready()
    return (time.perf_counter() - t0) / it

def main():
    print("JAX devices:", jax.devices())
    print(f"v6e roofline: peak={PEAK_BF16/1e12:.0f} TFLOP/s, HBM={HBM_BW/1e9:.0f} GB/s, "
          f"ridge={RIDGE:.0f} FLOP/byte\n")

    b = BackendFactory.create_backend("jax", device="tpu", dtype="float32")
    eng = EngineCommon(backend=b, strategy="cotengra")

    # Fixed launch floor: time a trivial jitted contraction (tiny network).
    tiny = build_mps(b, 8, 2); raw(eng.contract(tiny)).block_until_ready()
    eq, rt, *_ = assemble_global_einsum(tiny)
    arrs = [jnp.asarray(raw(t)) for t in rt]
    pl = tiny._cotengra_planner
    floor = time_jit(jax.jit(lambda a: pl.contract(a)), arrs)
    print(f"fixed launch floor (tiny net): {floor*1e6:.1f} us\n")

    hdr = (f"{'config':<20}{'XLAflops':>10}{'XLAbytes':>10}{'AI':>7}{'bound':>8}"
           f"{'lat(ms)':>9}{'-floor':>8}{'cmpTF/s':>9}{'MFU':>7}")
    print(hdr)
    for label, fn, kw in CONFIGS:
        c = fn(b, **kw); raw(eng.contract(c)).block_until_ready()
        eq, rt, *_ = assemble_global_einsum(c)
        arrays = [jnp.asarray(raw(t)) for t in rt]
        pl = c._cotengra_planner
        f = jax.jit(lambda a: pl.contract(a))
        compiled = f.lower(arrays).compile()
        flops, byts = cost_of(compiled)
        ai = flops / byts if byts else float("nan")
        bound = min(PEAK_BF16, ai * HBM_BW) / 1e12   # roofline TFLOP/s
        lat = time_jit(f, arrays)
        lat_c = max(lat - floor, 1e-9)               # compute-only (floor removed)
        cmp_tf = flops / lat_c / 1e12
        mfu = cmp_tf * 1e12 / PEAK_BF16 * 100
        print(f"{label:<20}{flops/1e9:>9.1f}G{byts/1e9:>9.2f}G{ai:>7.0f}"
              f"{bound:>7.0f}T{lat*1e3:>9.4f}{lat_c*1e3:>8.4f}{cmp_tf:>9.1f}{mfu:>6.1f}%")
    print("\nAI=arithmetic intensity (FLOP/byte). bound=roofline ceiling = "
          "min(peak, AI*HBM). cmpTF/s & MFU use latency MINUS launch floor.")

if __name__ == "__main__":
    main()
