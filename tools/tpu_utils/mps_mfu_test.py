"""Standalone single-chip MFU testbed on a simple MPS.

A minimal, fully hand-built MPS (no QCTN/BornMachine machinery) so the exact
tensor shapes and contraction order are under our control.  Goal: study how to
raise the TPU MXU utilization (MFU) of a tensor-network contraction.

Default structure: 3 core tensors (= 3 physical sites), open boundary, all
bond dims = all phys dims = 512:

    A(p0, b0) --b0-- B(b0, p1, b1) --b1-- C(b1, p2)
       |p0              |p1                  |p2

The scalar workload is the squared norm  <psi|psi>  (contract the MPS with its
conjugate over every leg).  With bond=phys=512 every pairwise op is a fat
512x512-ish matmul, which is exactly what fills the 128x128 MXU.

We compare three ways to evaluate the SAME contraction and report MFU for each:
  1. global jnp.einsum, optimize=False  -> one naive (bad) order
  2. global jnp.einsum, optimize='optimal' -> opt_einsum-chosen order
  3. staged transfer-matrix tensordot   -> explicit, hand-optimal order

Run:
    python tools/tpu_utils/mps_mfu_test.py
    python tools/tpu_utils/mps_mfu_test.py --n-cores 4 --bond 512 --phys 512
"""
import argparse, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import jax, jax.numpy as jnp
import opt_einsum
from tneq_qc.contractor.cotengra_planner import CotengraPlanner

PEAK_TFLOPS = 197.0   # TPU v5e bf16 theoretical / chip
PEAK_BW_GBPS = 819.0  # v5e HBM bandwidth / chip


# --------------------------------------------------------------------------- #
# Build a simple open-boundary MPS as a list of raw jax arrays.
# --------------------------------------------------------------------------- #
def build_mps(n_cores=3, bond=512, phys=512, seed=0):
    key = jax.random.PRNGKey(seed)
    cores = []
    for i in range(n_cores):
        key, sk = jax.random.split(key)
        if i == 0:
            shape = (phys, bond)                # (p0, b0)
        elif i == n_cores - 1:
            shape = (bond, phys)                # (b_{n-2}, p_{n-1})
        else:
            shape = (bond, phys, bond)          # (b_{i-1}, p_i, b_i)
        # scale down so the norm stays O(1) and finite
        cores.append(jax.random.normal(sk, shape, jnp.float32) / (bond ** 0.5))
    return cores


# --------------------------------------------------------------------------- #
# Build the global einsum equation for <psi|psi> of an n-core MPS.
# --------------------------------------------------------------------------- #
def norm_equation(n_cores):
    sym = opt_einsum.get_symbol
    idx = 0
    def fresh():
        nonlocal idx
        s = sym(idx); idx += 1; return s
    phys = [fresh() for _ in range(n_cores)]          # shared ket/bra phys legs
    kb = [fresh() for _ in range(n_cores - 1)]        # ket bonds
    bb = [fresh() for _ in range(n_cores - 1)]        # bra bonds (independent)
    ket, bra = [], []
    for i in range(n_cores):
        if i == 0:
            ket.append(phys[i] + kb[0]);            bra.append(phys[i] + bb[0])
        elif i == n_cores - 1:
            ket.append(kb[-1] + phys[i]);           bra.append(bb[-1] + phys[i])
        else:
            ket.append(kb[i-1] + phys[i] + kb[i]);  bra.append(bb[i-1] + phys[i] + bb[i])
    eq = ",".join(ket + bra) + "->"
    return eq


def norm_global(cores, optimize):
    eq = norm_equation(len(cores))
    operands = list(cores) + [jnp.conjugate(c) for c in cores]
    return jnp.einsum(eq, *operands, optimize=optimize)


# --------------------------------------------------------------------------- #
# State-projected (BornMachine-like) contraction:  <s|psi> * conj(<s|psi>).
# Each physical leg is contracted with a per-site product state vector instead
# of being summed between ket and bra.  phys is taken = bond (unified D).
# --------------------------------------------------------------------------- #
def build_states(n_cores, phys, seed=1):
    key = jax.random.PRNGKey(seed)
    states = []
    for _ in range(n_cores):
        key, sk = jax.random.split(key)
        v = jax.random.normal(sk, (phys,), jnp.float32)
        states.append(v / jnp.linalg.norm(v))
    return states


def state_equation(n_cores):
    """Equation for <s|psi> * conj(<s|psi>): phys legs go to state vectors."""
    sym = opt_einsum.get_symbol
    idx = 0
    def fresh():
        nonlocal idx
        s = sym(idx); idx += 1; return s
    kp = [fresh() for _ in range(n_cores)]   # ket phys legs (-> state)
    bp = [fresh() for _ in range(n_cores)]   # bra phys legs (-> conj state)
    kb = [fresh() for _ in range(n_cores - 1)]
    bb = [fresh() for _ in range(n_cores - 1)]
    ket, bra = [], []
    for i in range(n_cores):
        if i == 0:
            ket.append(kp[i] + kb[0]);           bra.append(bp[i] + bb[0])
        elif i == n_cores - 1:
            ket.append(kb[-1] + kp[i]);          bra.append(bb[-1] + bp[i])
        else:
            ket.append(kb[i-1] + kp[i] + kb[i]); bra.append(bb[i-1] + bp[i] + bb[i])
    states = kp[:]            # one index each, the ket state vectors
    cstates = bp[:]          # the conj state vectors
    eq = ",".join(ket + bra + states + cstates) + "->"
    return eq


def norm_with_states(cores, states, optimize):
    eq = state_equation(len(cores))
    operands = (list(cores) + [jnp.conjugate(c) for c in cores]
                + list(states) + [jnp.conjugate(s) for s in states])
    return jnp.einsum(eq, *operands, optimize=optimize)


def make_cotengra_states(cores, states):
    """cotengra planner for the state-projected <s|psi>*conj(<s|psi>) contraction."""
    eq = state_equation(len(cores))
    shapes = ([tuple(c.shape) for c in cores] * 2
              + [tuple(s.shape) for s in states] * 2)
    planner = CotengraPlanner(eq, shapes, target_slices=1, target_size=None)
    def fn(c):
        ops = (list(c) + [jnp.conjugate(x) for x in c]
               + list(states) + [jnp.conjugate(s) for s in states])
        return planner.contract(ops)
    return fn


def make_cotengra(cores):
    """Build a cotengra planner for the norm contraction (repo cotengra engine)."""
    eq = norm_equation(len(cores))
    shapes = [tuple(c.shape) for c in cores] * 2          # cores + conj(cores)
    planner = CotengraPlanner(eq, shapes, target_slices=1, target_size=None)
    def fn(c):
        return planner.contract(list(c) + [jnp.conjugate(x) for x in c])
    return fn


def norm_transfer(cores):
    """Explicit transfer-matrix sweep == row-priority (sequential) order on an MPS."""
    n = len(cores)
    A = cores[0]                                       # (p0, b0)
    E = jnp.einsum("pa,pA->aA", A, jnp.conjugate(A))   # (b0, b0')
    for i in range(1, n - 1):
        B = cores[i]                                   # (a, p, b)
        E = jnp.einsum("aA,apb,ApB->bB", E, B, jnp.conjugate(B))
    C = cores[-1]                                      # (b, p)
    return jnp.einsum("bB,bp,Bp->", E, C, jnp.conjugate(C))


# --------------------------------------------------------------------------- #
# Measure one contraction fn: latency, FLOPs/bytes (XLA), TFLOP/s, MFU, AI.
# --------------------------------------------------------------------------- #
def measure(name, fn, cores, iters=50):
    f = jax.jit(fn)
    r = f(cores); r.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        r = f(cores)
    r.block_until_ready()
    lat = (time.perf_counter() - t0) / iters

    flops = bytes_ = 0.0
    try:
        ca = f.lower(cores).compile().cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0]
        flops = float(ca.get("flops", 0.0))
        bytes_ = float(ca.get("bytes accessed", 0.0))
    except Exception:
        pass

    dev = jax.devices()[0]
    peak_gb = dev.memory_stats().get("peak_bytes_in_use", 0) / 1e9
    tflops = flops / lat / 1e12
    return dict(name=name, value=float(r), lat_ms=lat * 1e3,
                gflop=flops / 1e9, gbytes=bytes_ / 1e9,
                tflops=tflops, mfu=tflops / PEAK_TFLOPS,
                ai=(flops / bytes_ if bytes_ else 0.0), peak_gb=peak_gb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cores", type=int, default=3)
    ap.add_argument("--bond", type=int, default=512)
    ap.add_argument("--phys", type=int, default=512)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    print(f"device: {jax.devices()[0]} ({jax.devices()[0].device_kind})")
    print(f"MPS: n_cores={args.n_cores} bond={args.bond} phys={args.phys} (float32)")
    cores = build_mps(args.n_cores, args.bond, args.phys)
    for i, c in enumerate(cores):
        print(f"  core{i} shape={tuple(c.shape)}  {c.nbytes/1e6:.1f} MB")

    states = build_states(args.n_cores, args.phys)
    print(f"  + {len(states)} product state vectors, dim={args.phys} (phys=bond)")

    methods = [
        ("norm: row-order",            norm_transfer),
        ("norm: cotengra",             make_cotengra(cores)),
        ("norm: einsum optimal",       lambda c: norm_global(c, "optimal")),
        ("norm: einsum opt=False",     lambda c: norm_global(c, False)),
        ("STATE-proj: cotengra",       make_cotengra_states(cores, states)),
        ("STATE-proj: einsum optimal", lambda c: norm_with_states(c, states, "optimal")),
        ("STATE-proj: einsum False",   lambda c: norm_with_states(c, states, False)),
    ]
    print(f"\n{'method':<28}{'val':>9}{'lat(ms)':>10}{'GFLOP':>10}"
          f"{'GBytes':>9}{'TF/s':>8}{'MFU':>8}{'AI':>8}{'peakGB':>8}")
    for name, fn in methods:
        try:
            m = measure(name, fn, cores, args.iters)
            print(f"{m['name']:<28}{m['value']:>9.4f}{m['lat_ms']:>10.4f}"
                  f"{m['gflop']:>10.2f}{m['gbytes']:>9.2f}{m['tflops']:>8.1f}"
                  f"{m['mfu']*100:>7.1f}%{m['ai']:>8.1f}{m['peak_gb']:>8.2f}")
        except Exception as e:
            print(f"{name:<28}  ERROR: {str(e)[:60]}")


if __name__ == "__main__":
    main()
