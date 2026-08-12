"""Optimization: batched inference via jax.vmap on a single accelerator.

vmap turns the per-op skinny matmuls of an overhead-bound contraction into
batched (fatter) matmuls -> amortizes the kernel-launch floor and fills the MXU,
raising throughput and effective MFU.  Caveat: vmap scales peak memory ~xB, so
only small/overhead-bound configs can be batched (large ones OOM and gain little
anyway, being bandwidth-bound).

    python benchmark/bench_batched.py
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
import common

CASES = [("MPS", 16, 16), ("MPS", 16, 32), ("Tree", 16, 16)]   # (structure, nq, D)
BATCHES = [1, 2, 4, 8, 16, 32, 64]


def main():
    backend, _ = common.make_backend()
    peak_tflops, _, kind = common.get_device_peak()
    print(f"DEVICE: {kind}  (peak {peak_tflops:.0f} TF/s)")
    dev = jax.devices()[0]

    for structure, nq, P in CASES:
        c = common.make_born(backend, structure, nq, P)
        eq, arrs = common.global_arrays(c)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            common.EngineCommon(backend=backend, strategy="cotengra").contract(c)
        pl = c._cotengra_planner
        hw_flops = 2.0 * float(pl.contraction_cost())
        print(f"\n=== {structure} nq{nq} D{P}  ({hw_flops/1e9:.4f} GFLOP/sample) ===")
        print(f"{'batch':>6}{'total(ms)':>11}{'per-sample(ms)':>16}{'thrpt(/s)':>12}{'TF/s':>8}{'MFU':>8}{'peakGB':>9}")
        L1 = None
        for B in BATCHES:
            try:
                batched = [jnp.stack([a * (1.0 + 1e-3 * k) for k in range(B)]) for a in arrs]
                fv = jax.jit(jax.vmap(lambda a: pl.contract(a)))
                fv(batched).block_until_ready()
                L = common.timed(lambda: fv(batched))
            except Exception as e:
                print(f"{B:>6}   OOM/err: {str(e)[:48]}")
                break
            per = L / B
            tflops = hw_flops * B / L / 1e12
            peak = dev.memory_stats().get("peak_bytes_in_use", 0) / 1e9
            if B == 1:
                L1 = per
            spd = (L1 / per) if L1 else 1.0
            print(f"{B:>6}{L*1e3:>11.4f}{per*1e3:>16.4f}{B/L:>12.0f}{tflops:>8.3f}"
                  f"{tflops/peak_tflops*100:>7.2f}%{peak:>9.3f}" + (f"  ({spd:.1f}x/sample)" if B > 1 else ""))


if __name__ == "__main__":
    main()
