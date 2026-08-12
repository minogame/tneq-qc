"""Big table: BornMachine, 4 structures x sizes x 3 contraction methods.

Reports, per (structure, size, method): eager/JIT latency, GFLOP, TF/s, MFU, peak HBM.

Each (structure, size, method) runs in its OWN subprocess so that (a) peak HBM is a
clean per-config high-water mark, and (b) an OOM / pathological-order blowup on one
config can't take down the rest.  Run the driver:

    python benchmark/bench_methods.py                 # full sweep -> table
    python benchmark/bench_methods.py --worker MPS 32 cotengra   # one config (JSON)

Portable: uses benchmark/common.py for device peaks + backend, so the same file
runs on TPU and on an NV GPU (set BENCH_PEAK_TFLOPS/BW if the device is unknown).
"""
import sys, os, json, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# (structure, size).  size = D (MPS/Tree, unified phys=D) | n_layers (BrickWall) | block (MPSBrickWall)
CONFIGS = [
    ("MPS", 16), ("MPS", 32), ("MPS", 64),
    ("Tree", 16), ("Tree", 32),
    ("BrickWall", 4), ("BrickWall", 8),
    ("MPSBrickWall", 8), ("MPSBrickWall", 12),
]
METHODS = ["row", "cotengra", "einsum_greedy"]
PLABEL = {"MPS": "D", "Tree": "D", "BrickWall": "L", "MPSBrickWall": "blk"}
NQ = 16


def worker(structure, P, method):
    import common
    backend, _ = common.make_backend()
    c = common.make_born(backend, structure, NQ, P)
    m = common.measure_method(backend, c, method)
    m.update(structure=structure, P=P)
    print("JSON" + json.dumps(m))


def run(structure, P, method, timeout=220):
    try:
        p = subprocess.run([PY, __file__, "--worker", structure, str(P), method],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    for line in p.stdout.splitlines():
        if line.startswith("JSON"):
            return json.loads(line[4:])
    sys.stderr.write(f"[FAIL {structure} {P} {method}] {p.stderr[-300:]}\n")
    return None


def driver():
    rows = {}
    for s, P in CONFIGS:
        for m in METHODS:
            r = run(s, P, m)
            rows[(s, P, m)] = r
            print(f"  {s:13s} {PLABEL[s]}{P:<3d} {m:14s} " + ("ok" if r else "FAIL/timeout"), flush=True)

    import common
    _, _, kind = common.get_device_peak()
    pk, bw, _ = common.get_device_peak()
    print(f"\n{'='*106}\nDEVICE: {kind}  (peak {pk:.0f} TF/s bf16, {bw:.0f} GB/s)")
    print(f"{'model / size':<20}{'method':<15}{'ncores':>7}{'t_eager':>10}{'t_jit':>10}"
          f"{'GFLOP':>11}{'TF/s':>8}{'MFU':>8}{'peakGB':>8}")
    print("-" * 106)
    for s, P in CONFIGS:
        print(f"{s + ' ' + PLABEL[s] + str(P):<20}")
        for m in METHODS:
            r = rows[(s, P, m)]
            if not r:
                print(f"{'':<20}{m:<15} FAIL/timeout"); continue
            print(f"{'':<20}{m:<15}{r['ncores']:>7}{r['t_eager_ms']:>10.3f}{r['t_jit_ms']:>10.4f}"
                  f"{r['gflop']:>11.3f}{r['tflops']:>8.2f}{r['mfu']*100:>7.2f}%{r['peak_gb']:>8.3f}")

    print(f"\n{'='*50}\nJIT latency ratio  row / cotengra:")
    for s, P in CONFIGS:
        rr, rc = rows[(s, P, "row")], rows[(s, P, "cotengra")]
        if rr and rc:
            print(f"  {s + ' ' + PLABEL[s] + str(P):<20} {rr['t_jit_ms']/rc['t_jit_ms']:.2f}x")

    with open(os.path.join(HERE, "results_methods.json"), "w") as f:
        json.dump({f"{s}|{P}|{m}": rows[(s, P, m)] for s, P in CONFIGS for m in METHODS}, f, indent=1)
    print(f"\nsaved -> {os.path.join(HERE, 'results_methods.json')}")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=3, metavar=("STRUCT", "P", "METHOD"))
    args = ap.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), args.worker[2])
    else:
        driver()
