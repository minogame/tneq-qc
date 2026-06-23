# tools/tpu_bench — Cloud TPU benchmarks (JAX backend)

Benchmark + diagnostic scripts used to profile tneq-qc inference/training on a
Google Cloud TPU. Full write-up and results: [`docs/tpu_raw.md`](../../docs/tpu_raw.md).

Each script derives the repo root from `__file__`, so run them from anywhere with
a Python env that has `jax[tpu]` and the library deps installed (see
`docs/tpu_raw.md` §1):

```bash
python tools/tpu_bench/<script>.py
```

| Script | What it does |
|--------|--------------|
| `perf_models.py`   | Main sweep over MPS/TNEQ/BornMachine/tree/brickwall. Pre-flight **memory estimate** (peak intermediate vs HBM budget), then **inference + training**, TPU vs CPU. `--estimate-only` stops after the memory table (no execution). |
| `perf_mfu.py`      | **MFU**: achieved TFLOP/s and % of the v6e peak for inference; calibrates the cotengra cost convention and measures the empirical matmul ceiling. |
| `perf_roofline.py` | **Where is the bound**: XLA `cost_analysis` (true FLOPs + bytes) → arithmetic intensity + roofline; splits latency into launch floor vs compute. |
| `perf_jit.py`      | Eager `contract()` vs JIT-fused contraction (quantifies host/dispatch overhead). |
| `perf_infer.py`    | Eager inference latency/throughput sweep, TPU vs CPU. |
| `perf_bigbond.py`  | Large-bond JIT-fused sweep (crossover where TPU pulls ahead). |
| `perf_matmul.py`   | Dense `NxN` matmul control (TPU peak sanity check; no tneq-qc deps). |
| `train_tpu.py`     | Minimal end-to-end training smoke test on the TPU. |

Notes:
- The JAX backend auto-detects the TPU (`BackendFactory.create_backend("jax")`);
  pass `device="cpu"` to pin a run to the host CPU for comparison.
- Memory is checked **before** execution — large bond / tree dim is gated against
  an HBM budget so runs don't OOM the chip.
