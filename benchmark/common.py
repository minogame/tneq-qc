"""Shared benchmark utilities — device-portable across TPU and NVIDIA GPU.

Everything device-specific (peak FLOP/s, HBM bandwidth, backend selection) is
funnelled through here so the same bench scripts run on a Cloud TPU and on an
NV GPU node without edits.  Override the peak constants with env vars when the
auto-detected values are wrong:

    BENCH_PEAK_TFLOPS=312  BENCH_PEAK_BW_GBPS=2039  python bench_methods.py ...

Precision note: on TPU, float32 arrays are fed to the MXU as bf16 (single pass),
so we compare against the bf16 peak.  On GPU, float32 matmuls use FP32/TF32 cores
by default — for an apples-to-apples bf16 comparison set
    BENCH_MATMUL_PRECISION=bfloat16
which lowers jax_default_matmul_precision (and then compare vs the bf16 peak).
"""
import os
import jax
import jax.numpy as jnp

# --------------------------------------------------------------------------- #
# Repo imports
# --------------------------------------------------------------------------- #
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tneq_qc import QCTN, EngineCommon, BackendFactory          # noqa: E402
from tneq_qc.modules.small import MPS                            # noqa: E402
from tneq_qc.modules.app import BornMachine                     # noqa: E402
from tneq_qc.utils.graph_generators import QCTNHelper           # noqa: E402
from tneq_qc.core.tn_tensor import TNTensor                     # noqa: E402
from tneq_qc.contractor.cotengra_strategy import assemble_global_einsum  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpsbrick_graph import generate_mps_brickwall_graph         # noqa: E402

raw = lambda t: t.tensor if isinstance(t, TNTensor) else t

# --------------------------------------------------------------------------- #
# Device peak table  (bf16 dense TFLOP/s, HBM GB/s) — per chip/GPU.
# Matched by substring against jax device_kind.  Extend as needed.
# --------------------------------------------------------------------------- #
DEVICE_PEAKS = {
    # TPU
    "TPU v5 lite": (197.0, 819.0),     # v5e
    "TPU v5":      (197.0, 819.0),
    "TPU v6":      (918.0, 1640.0),    # v6e / Trillium
    # NVIDIA GPU (bf16 tensor-core dense, no sparsity)
    "A100":        (312.0, 2039.0),    # A100-80GB SXM (40GB: BW 1555)
    "H100":        (989.0, 3350.0),    # H100 SXM
    "H200":        (989.0, 4800.0),
    "L4":          (121.0, 300.0),
    "L40":         (181.0, 864.0),
    "V100":        (125.0, 900.0),     # fp16 (no bf16 tensor cores)
    "4090":        (165.0, 1008.0),    # RTX 4090
    "A10":         (125.0, 600.0),
}


def get_device_peak(dev=None):
    """Return (peak_tflops, peak_bw_gbps, label).  Env overrides win."""
    if dev is None:
        dev = jax.devices()[0]
    kind = getattr(dev, "device_kind", str(dev))
    tflops = bw = None
    for key, (tf, b) in DEVICE_PEAKS.items():
        if key in kind:
            tflops, bw = tf, b
            break
    if os.environ.get("BENCH_PEAK_TFLOPS"):
        tflops = float(os.environ["BENCH_PEAK_TFLOPS"])
    if os.environ.get("BENCH_PEAK_BW_GBPS"):
        bw = float(os.environ["BENCH_PEAK_BW_GBPS"])
    if tflops is None:
        raise RuntimeError(
            f"Unknown device '{kind}'. Set BENCH_PEAK_TFLOPS and "
            f"BENCH_PEAK_BW_GBPS env vars to its bf16 peak / HBM bandwidth.")
    return tflops, (bw or float("nan")), kind


def make_backend(dtype="float32"):
    """Create a JAX backend on the best accelerator (TPU>GPU>CPU)."""
    prec = os.environ.get("BENCH_MATMUL_PRECISION")
    if prec:
        jax.config.update("jax_default_matmul_precision", prec)
    # detect_device() in the repo prefers tpu>gpu>cpu
    from tneq_qc.backends import detect_device
    dev = detect_device()
    return BackendFactory.create_backend("jax", device=dev, dtype=dtype), dev


# --------------------------------------------------------------------------- #
# Model builders.  Unified D=phys for MPS/Tree; phys=2 + depth knob for the
# brick families (their natural physical setting).
# --------------------------------------------------------------------------- #
def make_born(backend, structure, nq, P):
    """BornMachine over one of the 4 structures.  P = D (MPS/Tree),
    n_layers (BrickWall) or block_qubits (MPSBrickWall)."""
    if structure == "MPS":
        g = QCTNHelper.mps(nq, bond_dim=P, phys_dim=P); dim = P
    elif structure == "Tree":
        g = QCTNHelper.generate_example_graph(nq, graph_type="tree", dim_char=str(P)); dim = P
    elif structure == "BrickWall":
        g = QCTNHelper.brickwall(nq, n_layers=P, phys_dim=2); dim = 2
    elif structure == "MPSBrickWall":
        g, _ = generate_mps_brickwall_graph(nq, block_qubits=P, overlap=3, phys_dim=2); dim = 2
    else:
        raise ValueError(structure)
    return BornMachine(g, dim, backend=backend).auto_init(orthogonal=True).build()


def make_mps_norm(backend, nq, bond, phys=2, orthogonal=True):
    """Bare MPS norm <psi|psi> (no state projection) — the high-MFU reference."""
    m = MPS(nqubits=nq, bond_dim=bond, phys_dim=phys, backend=backend).auto_init(orthogonal=orthogonal)
    c = QCTN.concat([("u", m), ("t", m.hermit())]); c.set_trace("all")
    return c


def global_arrays(combined):
    """eq + the JAX arrays of every operand in the global einsum, in order."""
    eq, rt, *_ = assemble_global_einsum(combined)
    return eq, [jnp.asarray(raw(t)) for t in rt]


# --------------------------------------------------------------------------- #
# Timing + cost analysis
# --------------------------------------------------------------------------- #
import time


def timed(fn, iters=50):
    r = fn(); jax.block_until_ready(r)
    t0 = time.perf_counter()
    for _ in range(iters):
        r = fn()
    jax.block_until_ready(r)
    return (time.perf_counter() - t0) / iters


def xla_cost(jit_fn, args):
    """Real FLOPs + bytes accessed from XLA's compiled cost analysis."""
    try:
        ca = jit_fn.lower(args).compile().cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0]
        return float(ca.get("flops", 0.0)), float(ca.get("bytes accessed", 0.0))
    except Exception:
        return 0.0, 0.0


def peak_hbm_gb():
    try:
        return jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 1e9
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Contraction-method dispatch.  Returns (eager_fn, jit_fn, jit_args) for one of:
#   row | cotengra | einsum_greedy | einsum_false
# All four contract the *same* network; row/cotengra are the repo strategies,
# einsum_* call opt_einsum directly.  The repo strategies are made jittable over
# array inputs by injecting tracer arrays into qctn.cores_weights (the strategy
# compute_fn re-reads them via build_graph / cores_override each call).
# --------------------------------------------------------------------------- #
import io, contextlib            # noqa: E402
import opt_einsum               # noqa: E402


def build_method(backend, combined, method):
    eq, arrs = global_arrays(combined)
    if method == "cotengra":
        with contextlib.redirect_stdout(io.StringIO()):
            EngineCommon(backend=backend, strategy="cotengra").contract(combined)
        pl = combined._cotengra_planner
        return (lambda: raw(pl.contract(arrs)),
                jax.jit(lambda a: pl.contract(a)), arrs)
    if method == "row":
        with contextlib.redirect_stdout(io.StringIO()):
            eng = EngineCommon(backend=backend, strategy="row_priority")
            eng.contract(combined)
        order = list(combined.cores)
        base = [jnp.asarray(raw(combined.cores_weights[n])) for n in order]
        def f(aa):
            for n, a in zip(order, aa):
                combined.cores_weights[n] = a
            return raw(eng.contract(combined))
        return (lambda: raw(eng.contract(combined)), jax.jit(f), base)
    if method in ("einsum_greedy", "einsum_false"):
        optimize = "greedy" if method == "einsum_greedy" else False
        return (lambda: opt_einsum.contract(eq, *arrs, optimize=optimize),
                jax.jit(lambda a: opt_einsum.contract(eq, *a, optimize=optimize)), arrs)
    raise ValueError(method)


def measure_method(backend, combined, method, iters=50):
    """Full measurement of one method on one model. Returns a metrics dict."""
    peak_tflops, peak_bw, _ = get_device_peak()
    with contextlib.redirect_stdout(io.StringIO()):
        eager, fjit, jit_args = build_method(backend, combined, method)
        r = eager()
        if hasattr(raw(r), "block_until_ready"):
            raw(r).block_until_ready()
        t_eager = timed(eager, iters)
        fjit(jit_args); jax.block_until_ready(fjit(jit_args))
        t_jit = timed(lambda: fjit(jit_args), iters)
        flops, bytes_ = xla_cost(fjit, jit_args)
    tflops = flops / t_jit / 1e12 if t_jit else 0.0
    return dict(method=method, ncores=len(combined.cores),
                t_eager_ms=t_eager * 1e3, t_jit_ms=t_jit * 1e3,
                gflop=flops / 1e9, gbytes=bytes_ / 1e9,
                tflops=tflops, mfu=tflops / peak_tflops,
                ai=(flops / bytes_ if bytes_ else 0.0),
                bw_util=(bytes_ / t_jit / 1e9 / peak_bw if (t_jit and peak_bw == peak_bw) else 0.0),
                peak_gb=peak_hbm_gb())

