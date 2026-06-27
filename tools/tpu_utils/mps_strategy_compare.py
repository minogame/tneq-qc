"""Single-chip comparison of FOUR contraction methods on ONE model.

Model: repo MPS norm <psi|psi> with nqubits=3 (= 3 core tensors),
bond_dim=512, phys_dim=512 (override via CLI). All four methods contract the
*identical* tensors (extracted once from the QCTN), so latency / FLOP / MFU are
directly comparable.

Methods:
  1. row_priority             — repo RowPriorityStrategy (fixed row order)
  2. cotengra                 — repo SlicedCotengraStrategy (optimized tree)
  3. einsum optimize=optimal  — jnp.einsum, opt_einsum-chosen order
  4. einsum optimize=False    — jnp.einsum, naive left-to-right order

All are JIT-fused over array inputs (no constant folding) and timed identically.

Run:
    python tools/tpu_utils/mps_strategy_compare.py
    python tools/tpu_utils/mps_strategy_compare.py --nq 3 --bond 512 --phys 512
"""
import argparse, time, io, contextlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import jax, jax.numpy as jnp
from tneq_qc import QCTN, EngineCommon, BackendFactory
from tneq_qc.modules.small import MPS
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.contractor.cotengra_strategy import assemble_global_einsum

raw = lambda t: t.tensor if isinstance(t, TNTensor) else t
PEAK_TFLOPS = 197.0    # v5e bf16 theoretical / chip
PEAK_BW_GBPS = 819.0   # v5e HBM bandwidth / chip


def build_norm(backend, nq, bond, phys):
    # orthogonal=False: orthogonal QR init needs a (flat_dim, flat_dim) matrix
    # which OOMs at phys=512 (flat_dim=phys*bond). Gaussian init is fine for a
    # latency/FLOP/MFU benchmark (values are irrelevant).
    m = MPS(nqubits=nq, bond_dim=bond, phys_dim=phys, backend=backend).auto_init(orthogonal=False)
    c = QCTN.concat([("u", m), ("t", m.hermit())])
    c.set_trace("all")
    return c


def time_fn(f, args, iters):
    r = f(args); jax.block_until_ready(r)
    t0 = time.perf_counter()
    for _ in range(iters):
        r = f(args)
    jax.block_until_ready(r)
    return (time.perf_counter() - t0) / iters, float(r if jnp.ndim(r) == 0 else r)


def cost_of(f, args):
    try:
        ca = f.lower(args).compile().cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0]
        return float(ca.get("flops", 0.0)), float(ca.get("bytes accessed", 0.0))
    except Exception:
        return 0.0, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=3)
    ap.add_argument("--bond", type=int, default=512)
    ap.add_argument("--phys", type=int, default=512)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    dev = jax.devices()[0]
    print(f"device: {dev} ({dev.device_kind})")
    print(f"MPS norm: nq={args.nq} (={args.nq} cores) bond={args.bond} phys={args.phys}")

    b = BackendFactory.create_backend("jax", device="tpu", dtype="float32")
    c = build_norm(b, args.nq, args.bond, args.phys)
    eq, rt, *_ = assemble_global_einsum(c)
    arrs = [jnp.asarray(raw(t)) for t in rt]
    for i, a in enumerate(arrs):
        print(f"  tensor{i} shape={tuple(a.shape)}")

    # cotengra planner (built by the engine)
    with contextlib.redirect_stdout(io.StringIO()):
        EngineCommon(backend=b, strategy="cotengra").contract(c)
    pl = c._cotengra_planner

    # row_priority via cores_weights injection (jittable over array inputs)
    with contextlib.redirect_stdout(io.StringIO()):
        eng_row = EngineCommon(backend=b, strategy="row_priority")
        eng_row.contract(c)
    core_order = list(c.cores)
    row_base = [jnp.asarray(raw(c.cores_weights[n])) for n in core_order]
    def row_fn(aa):
        for n, a in zip(core_order, aa):
            c.cores_weights[n] = a
        return raw(eng_row.contract(c))

    methods = [
        ("row_priority",            jax.jit(row_fn),                                              row_base),
        ("cotengra",                jax.jit(lambda a: pl.contract(a)),                            arrs),
        ("einsum optimize=optimal", jax.jit(lambda a: jnp.einsum(eq, *a, optimize="optimal")),    arrs),
        ("einsum optimize=False",   jax.jit(lambda a: jnp.einsum(eq, *a, optimize=False)),        arrs),
    ]

    print(f"\n{'method':<26}{'val':>11}{'lat(ms)':>11}{'GFLOP':>11}"
          f"{'GBytes':>9}{'TF/s':>8}{'MFU':>8}{'AI':>9}{'peakGB':>8}")
    for name, f, a in methods:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                lat, val = time_fn(f, a, args.iters)
                flops, bytes_ = cost_of(f, a)
            tflops = flops / lat / 1e12
            ai = flops / bytes_ if bytes_ else 0.0
            peak = dev.memory_stats().get("peak_bytes_in_use", 0) / 1e9
            print(f"{name:<26}{val:>11.3f}{lat*1e3:>11.4f}{flops/1e9:>11.2f}"
                  f"{bytes_/1e9:>9.2f}{tflops:>8.1f}{tflops/PEAK_TFLOPS*100:>7.1f}%{ai:>9.1f}{peak:>8.2f}")
        except Exception as e:
            print(f"{name:<26}  ERROR: {str(e)[:64]}")


if __name__ == "__main__":
    main()
