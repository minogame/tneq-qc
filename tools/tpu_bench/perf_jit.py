"""TPU inference perf, JIT-fused vs eager.

The eager EngineCommon.contract() path re-assembles the einsum and dispatches the
cotengra tree op-by-op from Python every call, so it is host-bound. Here we JIT
the contraction into a single XLA program (the cotengra tree contracts via autoray
-> jnp ops, which are traceable) to measure the TPU's real fused throughput and to
quantify the per-call host overhead.
"""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import jax
import jax.numpy as jnp

from tneq_qc import QCTN, EngineCommon, BackendFactory
from tneq_qc.modules.small import MPS
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.contractor.cotengra_strategy import assemble_global_einsum


def raw(t):
    return t.tensor if isinstance(t, TNTensor) else t


def build(backend, nq, bond):
    np.random.seed(0)
    teacher = MPS(nqubits=nq, bond_dim=bond, phys_dim=2, backend=backend).auto_init(orthogonal=True)
    student = MPS(nqubits=nq, bond_dim=bond, phys_dim=2, backend=backend).auto_init(orthogonal=True)
    c = QCTN.concat([("u", student), ("t", teacher)])
    c.set_trace("all")
    return c


def time_fn(fn, arg, iters):
    raw(fn(arg)).block_until_ready()                 # warmup / compile
    t0 = time.perf_counter()
    for _ in range(iters):
        r = fn(arg)
    raw(r).block_until_ready()
    return (time.perf_counter() - t0) / iters


def bench(device, configs, iters=100):
    b = BackendFactory.create_backend("jax", device=device, dtype="float32")
    eng = EngineCommon(backend=b, strategy="cotengra")
    rows = []
    for nq, bond in configs:
        c = build(b, nq, bond)

        # Prime eager path (builds planner) and time it.
        raw(eng.contract(c)).block_until_ready()
        eager = time_fn(lambda _c: eng.contract(_c), c, iters)

        # Build a jitted fused contraction over the raw arrays.
        eq, raw_tensors, scale, logscale, _ = assemble_global_einsum(c)
        arrays = [jnp.asarray(raw(t)) for t in raw_tensors]
        planner = c._cotengra_planner
        jit_contract = jax.jit(lambda arrs: planner.contract(arrs))
        jitted = time_fn(jit_contract, arrays, iters)

        rows.append((nq, bond, eager, jitted))
    return rows


def main():
    print("JAX devices:", jax.devices())
    configs = [(16, 8), (16, 16), (16, 32), (16, 64), (32, 16), (32, 32), (32, 64)]

    res = {}
    for device in ("tpu", "cpu"):
        print(f"\n=== device={device} ===")
        res[device] = bench(device, configs)
        print(f"{'nq':>4} {'bond':>5} {'eager(ms)':>11} {'jit(ms)':>10} {'jit speedup':>12}")
        for nq, bond, eager, jitted in res[device]:
            print(f"{nq:>4} {bond:>5} {eager*1e3:>11.3f} {jitted*1e3:>10.4f} {eager/jitted:>11.1f}x")

    print(f"\n=== JIT-fused: TPU vs CPU (real compute) ===")
    print(f"{'nq':>4} {'bond':>5} {'tpu(ms)':>10} {'cpu(ms)':>10} {'speedup':>9}")
    for (nq, bond, _, tj), (_, _, _, cj) in zip(res["tpu"], res["cpu"]):
        print(f"{nq:>4} {bond:>5} {tj*1e3:>10.4f} {cj*1e3:>10.4f} {cj/tj:>8.1f}x")


if __name__ == "__main__":
    main()
