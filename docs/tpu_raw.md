# Running tneq-qc on Cloud TPU (JAX backend) — setup & raw benchmarks

> Raw notes + measurements from bringing tneq-qc up on a Google Cloud TPU and
> profiling inference/training across the common model families.
> Date: 2026-06-23. Hardware: **TPU v6e-1 (Trillium), single chip**.

---

## 1. Environment

| Resource | Value |
|----------|-------|
| Accelerator | TPU **v6e-1** (Trillium), `ct6e-standard-1t`, single chip |
| TPU HBM | **33.6 GB** |
| Host vCPU | 44 |
| Host RAM | 172 GB (≈174 GB available) |
| Boot disk | 10 GB (≈6 GB free — keep installs lean, skip CUDA torch) |
| Python | 3.11 (conda env `tneq`) |
| JAX | `jax[tpu]` 0.10.2 + libtpu 0.0.42 |

### Install (Miniconda + JAX TPU)

```bash
# Miniconda
curl -fsSL -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p $HOME/miniconda3
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -y -n tneq python=3.11

# JAX for TPU + library deps (torch not needed for the JAX path)
$HOME/miniconda3/envs/tneq/bin/pip install "jax[tpu]" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    numpy opt_einsum cotengra safetensors tqdm
```

Verify:

```python
import jax
jax.default_backend()           # 'tpu'
jax.devices()                   # [TpuDevice(id=0, ...)]
```

The repo is **not** pip-installable (no `setup.py`/`pyproject.toml`), so scripts
prepend the repo to `sys.path`:

```python
import sys; sys.path.insert(0, "/path/to/tneq-qc")
```

---

## 2. Code changes for TPU support

### 2.1 Device auto-detection (torch-style)

`tneq_qc/backends/backend_jax.py` gained module-level helpers, exported from
`tneq_qc.backends`:

```python
from tneq_qc.backends import detect_device, is_tpu_available, is_gpu_available

detect_device()       # 'tpu' | 'gpu' | 'cpu', preferring TPU > GPU > CPU
is_tpu_available()     # like torch.cuda.is_available(), for TPU
is_gpu_available()
```

- `BackendJAX.__init__` now auto-detects via `detect_device()` when `device=None`.
- `jax.devices('gpu')` is wrapped in `_safe_jax_devices()` — it previously
  **crashed** on a TPU-only host (raises `RuntimeError` when the GPU backend is
  absent). Now it returns `[]`.
- `_gpu_put` → `_device_put`: handles `tpu`/`gpu` placement, and **pins `cpu`
  to the host CPU** (so `device='cpu'` really means CPU, as in torch).
- `BackendFactory.get_default_backend()` default changed from `'gpu'` → `None`
  (auto-detect).

Net effect: `BackendFactory.create_backend("jax")` now just works on TPU.

| `device=` | `backend_info.device` | array lands on |
|-----------|----------------------|----------------|
| `None` (auto) | `'tpu'` | `TPU_0` |
| `'tpu'` | `'tpu'` | `TPU_0` |
| `'cpu'` | `'cpu'` | `cpu:0` |

### 2.2 Bug fix: `init_random_core` ignored the device

`init_random_core` built cores with `jax.random.normal`, which lands on JAX's
**default** device (the TPU) regardless of the backend's `device`. So
`device='cpu'` still allocated random cores on the TPU — tainting CPU benchmarks
and consuming HBM. Fixed by routing both return paths (orthogonal and
non-orthogonal) through `_device_put`. Verified: cpu→`cpu:0`, tpu→`TPU_0`.

---

## 3. Memory pre-flight (avoid OOM before running)

Before executing anything we estimate the contraction's **peak intermediate
size** from the cotengra tree *without executing it*:

```python
eq, raw_tensors, *_ = assemble_global_einsum(combined)
shapes  = [tuple(t.shape) for t in raw_tensors]
planner = CotengraPlanner(eq, shapes, target_slices=1, target_size=None)
peak    = planner.tree.peak_size()        # peak simultaneous elements
cost    = planner.contraction_cost()      # FLOPs
width   = planner.tree.contraction_width()  # log2 of largest tensor
```

Budget: peak `× 4 bytes × 3` (autodiff/live-intermediate safety) must stay under
~24 GB (of 33.6 GB HBM). Note `cotengra_target_size = None` by default → **no
slicing**, so large bond must be checked.

**Estimate table** (all SAFE; largest peak ≈6 GB):

```
config               params(MB)   peakInter  peak(GB)     GFLOP  width  verdict
MPS  nq32 bond2            0.00   9.960e+02     0.000      0.00    4.0  SAFE
MPS  nq32 bond512        243.34   6.085e+07     0.730     25.67   20.0  SAFE
MPS  nq16 bond512        109.12   2.730e+07     0.328      8.49   20.0  SAFE
MPS  nq16 bond2048      1745.09   4.363e+08     5.236    406.42   24.0  SAFE
TNEQ nq32 bond2            0.00   9.960e+02     0.000      0.00    4.0  SAFE
TNEQ nq16 bond512        109.12   2.730e+07     0.328      8.49   20.0  SAFE
TNEQ nq16 bond2048      1745.09   4.363e+08     5.236    406.42   24.0  SAFE
Born nq16 bond2            0.00   6.160e+02     0.000      0.00    3.0  SAFE
Born nq16 bond512        109.12   2.780e+07     0.334      0.64   19.0  SAFE
Tree nq16 dim2             0.00   4.840e+02     0.000      0.00    4.0  SAFE
Tree nq16 dim64         2013.27   5.033e+08     6.040     30.06   24.0  SAFE
Brick nq12 L4              0.00   7.840e+02     0.000      0.00    6.0  SAFE
Brick nq16 L4              0.00   1.040e+03     0.000      0.00    6.0  SAFE
Brick nq16 L8              0.01   3.427e+04     0.000      0.00   14.0  SAFE
```

### Model-specific memory notes

- **MPS / TNEQ / BornMachine**: bond `D` is a real axis; cores `~D²`, contraction
  peak small. Scales cleanly to bond 2048 (~5 GB peak, ~1.7 GB params).
- **tree**: bond is a uniform `dim`; max core degree is **4**, so a single core is
  `dim⁴`. `dim=128` already OOMs at init (QR on a `dim²×dim²` matrix). Cap at
  **dim ≤ 64**.
- **brickwall**: has **no bond axis** — gates are `phys⁴` and cost is
  ~exponential in circuit width (`2^width`). Keep `phys=2`, `≤16` qubits, and
  vary `n_layers`. The `bond=2/512` split does not apply.

---

## 4. Benchmark methodology

- **Inference** = `EngineCommon.contract(qctn)` → scalar. Timed **JIT-fused**:
  `jax.jit(lambda a: qctn._cotengra_planner.contract(a))`. The eager
  `contract()` path re-assembles the einsum and dispatches the cotengra tree
  op-by-op from Python every call (host-bound; ~50–90× slower — see §6), so the
  fused number is the meaningful device measurement.
- **Training** = `EngineCommon.contract_for_gradient(qctn, target=1.0,
  loss="mse")`, **eager** (the real per-step cost today; not JIT-fused).
- Each model forms a scalar network: a single net concatenated with its
  `hermit()` and `set_trace("all")` (squared norm / overlap); BornMachine uses
  its native 5-segment `build()`.
- JAX dispatch is async — every timed result is forced with
  `.block_until_ready()`. Warmup precedes each timed loop.
- Models: **MPS, TNEQ, BornMachine, tree, brickwall**. Bond groups: small
  (`bond=2`) and large (`bond=512/2048`, tree `dim=64`).

Script: `scratchpad/perf_models.py` (estimate + run in one file).

---

## 5. Results — TPU vs CPU

### 5.1 Inference (JIT-fused contraction), ms per call

```
config                 TPU(ms)   CPU(ms)  speedup
MPS  nq32 bond2         0.0640    0.0582     0.9x
MPS  nq32 bond512       0.2005   76.9001   383.5x
MPS  nq16 bond512       0.1011   30.9787   306.3x
MPS  nq16 bond2048      2.6007  634.3733   243.9x
TNEQ nq32 bond2         0.0642    0.0514     0.8x
TNEQ nq16 bond512       0.0978   35.0335   358.1x
TNEQ nq16 bond2048      2.5885  863.8046   333.7x
Born nq16 bond2         0.0781    0.0625     0.8x
Born nq16 bond512       0.9903   26.3047    26.6x
Tree nq16 dim2          0.0528    0.0263     0.5x
Tree nq16 dim64         4.9415  588.6557   119.1x
Brick nq12 L4           0.0558    0.0457     0.8x
Brick nq16 L4           0.0657    0.0609     0.9x
Brick nq16 L8           0.3363    2.0661     6.1x
```

### 5.2 Training (eager fwd+bwd), ms per step

```
config                 TPU(ms)   CPU(ms)  speedup
MPS  nq32 bond2         98.772    77.880     0.8x
MPS  nq32 bond512      100.066   277.196     2.8x
MPS  nq16 bond512       50.397    99.849     2.0x
MPS  nq16 bond2048      49.624  2817.795    56.8x
TNEQ nq32 bond2         96.085    78.077     0.8x
TNEQ nq16 bond512       47.905    95.184     2.0x
TNEQ nq16 bond2048      48.457  3008.519    62.1x
Born nq16 bond2        119.500    96.641     0.8x
Born nq16 bond512      122.522   147.484     1.2x
Tree nq16 dim2          47.471    38.402     0.8x
Tree nq16 dim64         57.532  4317.402    75.0x
Brick nq12 L4           69.454    57.007     0.8x
Brick nq16 L4           96.761    78.893     0.8x
Brick nq16 L8          196.126   157.837     0.8x
```

---

## 6. Interpretation

1. **Inference at large bond is the TPU's main win.** bond 512/2048 → **100–380×**
   over CPU (MPS/TNEQ ~250–380×, tree dim64 ~120×). BornMachine is only ~27×
   because its contraction is much cheaper (~0.64 GFLOP — the measure matrices
   are identity and the path simplifies).

2. **Small bond / small models: TPU ≈ CPU, sometimes slower (0.5–0.9×).** The
   contractions are tiny (µs-scale) and dominated by the TPU's fixed ~50 µs
   kernel-launch floor; the host CPU is just as fast.

3. **Training is host-dispatch bound today.** TPU step time is essentially
   **flat ~48–120 ms regardless of bond** (MPS bond 512→2048: 50.4→49.6 ms) —
   that floor is the per-step Python work in eager `contract_for_gradient`
   (re-assemble einsum + autodiff trace), not device compute. TPU only wins big
   (57–75×) where the CPU is catastrophically slow (bond2048: 2.8–4.3 s/step);
   at bond512 it is only 2–3×, at bond2 it loses.

4. **The earlier eager-vs-fused finding** (single MPS): JIT-fusing the
   contraction is **~50–90× faster** than the eager `contract()` path, which is
   pure Python/dispatch overhead. Control: a jitted dense `N×N` matmul hits
   **~800 TFLOP/s fp32** on this TPU (4096³) vs ~2 on CPU (40–450×) — the TPU is
   healthy; small TN contractions just don't exercise it.

### Takeaways

- Use the TPU for **large-bond inference** — that is where it pays off massively.
- For **small bond / small networks**, the host CPU is equal or better; no reason
  to pay TPU launch overhead.
- **brickwall** is width-limited (`2^width`); only deeper/wider circuits (e.g.
  L8) show meaningful TPU benefit (~6×).
- **Biggest opportunity:** JIT-fuse the **training** step the way inference is
  fused. That would remove the ~50 ms host floor, speed up small-model training,
  and make the large-bond advantage even more decisive.

---

## 7. MFU — how much of the TPU's peak FLOPs does inference reach?

Method: achieved FLOP/s = `2 × cotengra.contraction_cost() / latency`. The factor
2 was **calibrated** — a known `2048³` matmul gives `contraction_cost()/N³ =
1.000`, i.e. cotengra reports **MACs**, and HW FLOPs = 2 × MACs.

Reference peaks for **TPU v6e**:
- Theoretical **bf16 = 918 TFLOP/s** per chip.
- **Empirical** ceiling on this chip: a jitted `4096³` matmul reaches **803
  TFLOP/s = 87 % of peak** — the realistic achievable maximum.

> Precision note: JAX's default matmul precision on TPU feeds **bf16** into the
> MXU (single pass), so even "float32" arrays run at ~bf16 rates. That is why we
> compare against the bf16 peak. Forcing true fp32 (3-pass bf16) would cut the
> peak ~3× and make MFU look ~3× higher, at a large speed cost.

MFU of JIT-fused inference (latency, 100 iters):

```
config                  GFLOP  lat(ms)  TFLOP/s  MFU_theo  MFU_emp
MPS  nq16 bond512        17.0   0.0952    178.4    19.4%    22.2%
MPS  nq32 bond512        51.3   0.2248    228.4    24.9%    28.4%
MPS  nq16 bond2048      812.8   2.6435    307.5    33.5%    38.3%
TNEQ nq16 bond2048      812.8   2.6871    302.5    33.0%    37.7%
Born nq16 bond512         1.3   0.2415      5.4     0.6%     0.7%
Tree nq16 dim64          60.1   4.8938     12.3     1.3%     1.5%
```

(`MFU_theo` = vs 918 bf16 peak; `MFU_emp` = vs the 803 TFLOP/s matmul ceiling.
GFLOP here is HW FLOPs = 2 × MACs.)

### Reading the MFU

- **Best case (MPS/TNEQ bond2048): ~33 % of theoretical bf16 peak**, ~38 % of the
  achievable matmul ceiling — i.e. ~300 of 918 TFLOP/s. That is a *healthy* MFU
  for a chained tensor-network contraction.
- bond512 lands at **~20–25 %** (theoretical); smaller `K`/fewer FLOPs to amortize
  the ~50 µs launch floor, so MFU is lower.
- **Why not higher:** an MPS-norm is a *sequence* of ~30 moderate matmuls with
  transposes/reshapes between them, and the `phys=2` legs make some contractions
  skinny (poor MXU fill). Even a single dense matmul only hits 87 %, so ~38 % of
  that for a chained contraction is reasonable.
- **Tree dim64 (1.3 %)** and **Born bond512 (0.6 %)** are far off-peak: the tree
  path is dominated by transposes/small ops (not MXU-bound), and Born has too few
  FLOPs (1.3 G) to hide the launch overhead. These are latency/overhead-bound,
  not compute-bound.

**Bottom line:** for compute-heavy, large-bond inference the TPU runs at **roughly
one-third of its theoretical FLOPs (≈300 TFLOP/s)** — most of the realistically
achievable budget. Low-FLOP or transpose-heavy contractions sit near a few
percent and are bound by launch overhead, not the MXU.

### Where is the bound? (roofline, XLA cost analysis)

Ridge point = peak / HBM BW = 918 TFLOP/s / 1.64 TB/s ≈ **560 FLOP/byte**.
Fixed launch floor (tiny jitted contraction) = **55 µs/op**. XLA `cost_analysis`
gives true FLOPs + bytes; latency is split into the floor + compute:

```
config                XLAflops  XLAbytes   AI(F/B)  lat(ms)  -floor  cmpTF/s   MFU
MPS  nq16 bond512        17.0G     0.51G      33     0.0982  0.0431    394.3  43.0%
MPS  nq32 bond512        51.3G     1.18G      44     0.2173  0.1621    316.7  34.5%
MPS  nq16 bond2048      812.8G     3.55G     229     2.7203  2.6651    305.0  33.2%
Born nq16 bond512         1.3G     0.62G       2     0.2433  0.1881      6.7   0.7%
Tree nq16 dim64          60.1G     6.98G       9     4.5902  4.5350     13.3   1.4%
```

**The bound is different at every scale** — this is the key finding:

- **bond=2048 → HBM-bandwidth bound.** XLA bytes = 3.55 GB; at 1.64 TB/s that is
  **2.16 ms of unavoidable HBM traffic ≈ 80 % of the 2.72 ms**. Intermediates are
  `D²·4 ≈ 16 MB` each — too big for on-chip VMEM, so every op in the ~30-op
  cotengra chain round-trips through HBM. The contraction runs as **separate
  einsum ops, not one fused kernel**, so intermediates can't stay on-chip. Pure
  compute is only 33 % of peak because most of the time is moving bytes, not
  multiplying. (Sanity: 2.16 ms < 2.72 ms, so memory-bound is consistent.)

- **bond=512 → launch / on-chip-pipeline bound, NOT memory.** Sanity check: 0.51 GB
  at 1.64 TB/s would need 311 µs, but total latency is only 98 µs — impossible if
  it hit HBM, so the `D²·4 = 1 MB` intermediates **stay on-chip** and XLA's "bytes
  accessed" is logical, not HBM. The compute kernel itself runs at **394 TFLOP/s
  (43 %!)** — quite efficient. Its low *end-to-end* MFU (22 %) is because the fixed
  **55 µs launch floor is ~half** of the 98 µs latency.

- **Born bond512 → pure overhead bound.** Only **1.3 GFLOP** of work — even at full
  speed that is <2 µs, dwarfed by the 55 µs floor → 0.7 %. Nothing to optimize on
  the MXU; it is latency-bound.

- **Tree dim64 → transpose / memory bound.** Lowest AI (**9 FLOP/byte**, 7 GB of
  byte traffic for 60 GFLOP). High-degree `dim⁴` cores force large transposes /
  reshapes (zero-FLOP HBM traffic) between contractions → 13 TFLOP/s, 1.4 %.

**Summary of the bottleneck hierarchy:**

1. The cotengra tree executes as **~30 separate einsum ops, each materializing to
   HBM** (it is not one fused XLA kernel even under `jit`). This is the root cause.
2. At **large bond** that makes it **HBM-bandwidth bound** (intermediates spill).
3. At **medium bond** intermediates fit on-chip, so it is **launch-floor bound**
   (~55 µs/call dominates).
4. Throughout, **`phys=2` skinny legs underfill the 128×128 MXU** and **transposes
   add zero-FLOP traffic** — worst for the tree.
5. **Born / tiny nets** are **latency bound** — too few FLOPs to amortize anything.

The one lever that helps all of these: make the contraction **one fused kernel**
that keeps intermediates on-chip (raise arithmetic intensity) instead of a chain
of HBM-materializing einsums — and batch small/medium-bond inference to hide the
55 µs launch floor.

## 8. Reproduce

All scripts live in [`tools/tpu_bench/`](../tools/tpu_bench/) (see its README) and
derive the repo root from `__file__`, so they run from anywhere. Use the env from
§1 (`$HOME/miniconda3/envs/tneq/bin/python`).

```bash
# memory estimate only (no execution, safe)
python tools/tpu_bench/perf_models.py --estimate-only

# full sweep (inference + training, TPU vs CPU)
python tools/tpu_bench/perf_models.py

# MFU: achieved TFLOP/s and % of v6e peak for inference
python tools/tpu_bench/perf_mfu.py

# roofline / where-is-the-bound (XLA cost analysis)
python tools/tpu_bench/perf_roofline.py
```

Other benchmarks: `perf_infer.py` (eager latency/throughput), `perf_jit.py`
(eager vs JIT-fused), `perf_bigbond.py` (large-bond sweep), `perf_matmul.py`
(dense-matmul control), `train_tpu.py` (training smoke test).
