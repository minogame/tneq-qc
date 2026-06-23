"""Inference (forward-contraction) performance benchmark for tneq-qc on TPU.

Inference here = EngineCommon.contract(qctn): a forward contraction of a
student-teacher MPS trace network to a scalar. We warm up the JIT (first call
compiles), then time steady-state latency. JAX dispatch is async, so every timed
result is forced to completion with .block_until_ready().
"""
import sys, time, argparse
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import jax

from tneq_qc import QCTN, EngineCommon, BackendFactory
from tneq_qc.modules.small import MPS
from tneq_qc.core.tn_tensor import TNTensor


def sync(result):
    """Block until a contraction result is materialized on device."""
    arr = result.tensor if isinstance(result, TNTensor) else result
    return arr.block_until_ready()


def build(backend, nq, bond):
    np.random.seed(0)
    teacher = MPS(nqubits=nq, bond_dim=bond, phys_dim=2, backend=backend).auto_init(orthogonal=True)
    student = MPS(nqubits=nq, bond_dim=bond, phys_dim=2, backend=backend).auto_init(orthogonal=True)
    combined = QCTN.concat([("u", student), ("t", teacher)])
    combined.set_trace("all")
    return combined


def bench(device, configs, warmup=5, iters=50):
    backend = BackendFactory.create_backend("jax", device=device, dtype="float32")
    eng = EngineCommon(backend=backend, strategy="cotengra")
    rows = []
    for nq, bond in configs:
        combined = build(backend, nq, bond)

        # Warmup: triggers strategy compile + XLA compile for this shape.
        for _ in range(warmup):
            sync(eng.contract(combined))

        # Latency: block on each call.
        t0 = time.perf_counter()
        for _ in range(iters):
            sync(eng.contract(combined))
        lat = (time.perf_counter() - t0) / iters

        # Throughput: dispatch all, block once on the last (overlaps dispatch).
        t0 = time.perf_counter()
        last = None
        for _ in range(iters):
            last = eng.contract(combined)
        sync(last)
        thr_dt = time.perf_counter() - t0
        rows.append((nq, bond, lat, iters / thr_dt))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    print("JAX devices:", jax.devices())
    configs = [(8, 4), (8, 16), (16, 8), (16, 16), (16, 32), (32, 16), (32, 32)]

    results = {}
    for device in ("tpu", "cpu"):
        print(f"\n=== device={device} ===")
        results[device] = bench(device, configs, iters=args.iters)
        print(f"{'nq':>4} {'bond':>5} {'latency(ms)':>12} {'throughput(/s)':>15}")
        for nq, bond, lat, thr in results[device]:
            print(f"{nq:>4} {bond:>5} {lat*1e3:>12.3f} {thr:>15.1f}")

    # Speedup summary
    print(f"\n=== TPU vs CPU latency speedup ===")
    print(f"{'nq':>4} {'bond':>5} {'tpu(ms)':>10} {'cpu(ms)':>10} {'speedup':>9}")
    for (nq, bond, tlat, _), (_, _, clat, _) in zip(results["tpu"], results["cpu"]):
        print(f"{nq:>4} {bond:>5} {tlat*1e3:>10.3f} {clat*1e3:>10.3f} {clat/tlat:>8.1f}x")


if __name__ == "__main__":
    main()
