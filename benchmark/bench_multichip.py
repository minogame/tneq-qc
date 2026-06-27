"""Multi-chip inference benchmark (4-chip TPU pod slice, or multi-GPU node).

Two parts, both using the cotengra strategy (JIT-fused), one independent
contraction pinned per device, dispatched concurrently (JAX is async; we
block on all of them at the end so wall-clock reflects true parallelism):

  A. Data-parallel throughput scaling.  The SAME model replicated across
     1..N devices.  Ideal scaling -> ~Nx throughput (each device does a full,
     independent contraction).  This is the robust distributed-inference win.

  B. Heterogeneous fan-out.  The 4 model structures, one per device, run
     concurrently.  Wall-clock -> max(per-case latency) (not the sum), proving
     real parallelism; speedup vs sequential is bounded by load imbalance.

Portable: device pinning is by input placement (jax.device_put), so the same
code runs on a 4-chip TPU and on a multi-GPU node.

    python benchmark/bench_multichip.py
"""
import sys, os, io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
import common

NQ = 16


def pin_model(backend, structure, P, device):
    """Build BornMachine(structure), pin its inputs to `device`, return a jitted
    cotengra contraction + the on-device arrays (runs where its inputs live)."""
    c = common.make_born(backend, structure, NQ, P)
    eq, arrs = common.global_arrays(c)
    with contextlib.redirect_stdout(io.StringIO()):
        common.EngineCommon(backend=backend, strategy="cotengra").contract(c)
    pl = c._cotengra_planner
    arrs_d = [jax.device_put(a, device) for a in arrs]
    f = jax.jit(lambda a: pl.contract(a))
    r = f(arrs_d); jax.block_until_ready(r)
    return f, arrs_d, float(r), list(r.devices())[0]


def part_a_data_parallel(backend, devices):
    N = len(devices)
    chip_counts = sorted({1, 2, N} & set(range(1, N + 1)))
    print(f"\n{'='*78}\nA. DATA-PARALLEL THROUGHPUT SCALING (same model on k devices)")
    for structure, P in [("MPS", 32), ("MPS", 48), ("Tree", 32)]:
        # build one pinned replica per device up front
        reps = [pin_model(backend, structure, P, devices[i]) for i in range(N)]
        fns = [r[0] for r in reps]; args = [r[1] for r in reps]
        print(f"\n  {structure} D{P}:")
        print(f"    {'#chips':>7}{'wall(ms)':>10}{'thrpt(/s)':>12}{'scaling':>9}")
        base_thr = None
        for k in chip_counts:
            def run_k():
                outs = [fns[i](args[i]) for i in range(k)]   # async on k chips
                jax.block_until_ready(outs); return outs
            L = common.timed(run_k)
            thr = k / L
            if k == 1:
                base_thr = thr
            print(f"    {k:>7}{L*1e3:>10.4f}{thr:>12.0f}{thr/base_thr:>8.2f}x")


def part_b_heterogeneous(backend, devices):
    N = len(devices)
    CASES = [("MPS", 52), ("Tree", 48), ("BrickWall", 12), ("MPSBrickWall", 16)]
    CASES = CASES[:N]
    print(f"\n{'='*78}\nB. HETEROGENEOUS FAN-OUT ({len(CASES)} structures, one per device)")
    reps, labels = [], []
    for i, (structure, P) in enumerate(CASES):
        f, a, val, rdev = pin_model(backend, structure, P, devices[i])
        reps.append((f, a)); labels.append(f"{structure} P{P}")
        print(f"    device {i}: {labels[i]:<22} -> {str(rdev)[:18]}  val={val:.3e}")

    singles = []
    for i, (f, a) in enumerate(reps):
        singles.append(common.timed(lambda: jax.block_until_ready(f(a))))
    def run_all():
        outs = [reps[i][0](reps[i][1]) for i in range(len(reps))]
        jax.block_until_ready(outs); return outs
    conc = common.timed(run_all)

    print(f"\n    {'case':<22}{'alone(ms)':>11}")
    for lab, s in zip(labels, singles):
        print(f"    {lab:<22}{s*1e3:>11.3f}")
    seq = sum(singles)
    print(f"\n    sequential (sum of {len(reps)}): {seq*1e3:8.3f} ms")
    print(f"    concurrent ({len(reps)} on {len(reps)} chips): {conc*1e3:8.3f} ms")
    print(f"    speedup: {seq/conc:.2f}x   (bounded by load imbalance; "
          f"balanced -> ~{len(reps)}x)")
    print(f"    throughput: {len(reps)/conc:.0f} contractions/s")


def part_c_model_parallel(backend, devices):
    """Model parallelism: split ONE contraction across chips.

    A real hypergraph partitioner (KaHyPar) establishes the partition — it cuts
    the einsum operands into K weakly-coupled blocks, minimizing the boundary.
    The *computation reuses the repo's contractor* (CotengraPlanner): each block
    is contracted locally on its own chip into a small boundary tensor (open legs
    = the cut bonds), then a final reduce contracts the K boundary tensors.  This
    is the BENCHMARK_DISTRIBUTED.md scheme (partition -> local contract -> reduce
    cross-partition edges), now library-partitioned and on the TPU.

    The win here is MEMORY / FEASIBILITY: each chip holds only ~1/K of the
    network, so a contraction can run across chips using a fraction of the per-
    chip HBM.  It is NOT a latency win — partitioning forces larger boundary
    tensors than the monolithic path (e.g. an MPS norm keeps a D^2 environment
    but a partition exposes D^4 boundaries), so wall-clock typically rises.  Use
    it to fit networks that don't fit on one chip, not to go faster.
    """
    try:
        import partition_contract as pc
    except ImportError as e:
        print(f"\n{'='*78}\nC. MODEL-PARALLEL (skipped: {e})"); return
    from tneq_qc.contractor.cotengra_planner import CotengraPlanner

    N = len(devices)
    K = min(N, 4)
    CASES = [("MPS", 48), ("Tree", 32), ("MPSBrickWall", 16)]
    print(f"\n{'='*78}\nC. MODEL-PARALLEL: one contraction split across {K} chips "
          f"(KaHyPar partition + reuse cotengra)")
    print(f"   win = memory/feasibility (per-chip ~1/K of the net); not latency.")
    dev0 = devices[0]
    for structure, P in CASES:
        c = common.make_born(backend, structure, NQ, P)
        eq, arrs = common.global_arrays(c)
        shapes = [tuple(a.shape) for a in arrs]

        # monolithic baseline on chip 0
        with contextlib.redirect_stdout(io.StringIO()):
            pl_full = CotengraPlanner(eq, shapes, seed=0)
        a0 = [jax.device_put(a, dev0) for a in arrs]
        f0 = jax.jit(lambda a: pl_full.contract(a))
        ref = float(f0(a0)); jax.block_until_ready(f0(a0))
        t_full = common.timed(lambda: f0(a0))
        full_peak = pl_full.tree.peak_size()

        # KaHyPar partition (eps=0.5 -> minimal boundary legs) + per-block planners
        plan = pc.make_partitioned(eq, shapes, K, epsilon=0.5, seed=0)
        block_fns = []
        for b in range(K):
            dev = devices[b]
            sub = [jax.device_put(arrs[n], dev) for n in plan["blocks"][b]]
            fb = jax.jit(lambda s, _b=b: plan["block_planners"][_b].contract(s))
            jax.block_until_ready(fb(sub))
            block_fns.append((fb, sub))

        def run_dist():
            bs = [fb(sub) for fb, sub in block_fns]      # concurrent on K chips
            jax.block_until_ready(bs)
            bs0 = [jax.device_put(x, dev0) for x in bs]  # gather + reduce on chip 0
            return plan["reduce_pl"].contract(bs0)
        val = float(run_dist()); jax.block_until_ready(run_dist())
        t_dist = common.timed(run_dist)
        err = abs(val - ref) / (abs(ref) + 1e-30)

        # Per-chip peak intermediate (theoretical, device-independent): the
        # largest tensor a single block's local contraction materializes, vs the
        # monolithic peak.  (Measured per-chip HBM is unreliable in one process:
        # JAX's peak_bytes_in_use is a high-water mark that never resets, so it
        # accumulates across configs — use the deterministic peak_size instead.)
        max_blk_peak = max(p.tree.peak_size() for p in plan["block_planners"])

        print(f"\n  {structure} P{P}:  blocks={plan['block_sizes']}  "
              f"cut={plan['cut_edges']}  open_legs={[len(o) for o in plan['block_open']]}"
              f"  err={err:.1e}")
        print(f"    latency:  full(1-chip)={t_full*1e3:8.3f} ms   "
              f"dist({K}-chip)={t_dist*1e3:8.3f} ms   ({t_full/t_dist:.2f}x — partition trades speed for memory)")
        print(f"    memory:   full peak={full_peak/1e6:8.2f}M elem   "
              f"per-block peak={max_blk_peak/1e6:8.2f}M elem   ({full_peak/max_blk_peak:.1f}x lower/chip)")


def main():
    backend, _ = common.make_backend()
    devices = jax.devices()
    _, _, kind = common.get_device_peak()
    print(f"DEVICE: {kind} x {len(devices)}")
    part_a_data_parallel(backend, devices)
    if len(devices) >= 2:
        part_b_heterogeneous(backend, devices)
        part_c_model_parallel(backend, devices)


if __name__ == "__main__":
    main()
