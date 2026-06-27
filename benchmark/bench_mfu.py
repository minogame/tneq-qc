"""MFU root-cause study: BornMachine vs bare MPS-norm (cotengra, JIT-fused).

Two sweeps that together explain why BornMachine MFU is stuck near zero:
  1. bond sweep  (phys=2):  large bond -> fat matmuls.  MPS-norm climbs to ~50%
     MFU; BornMachine stays ~1-2% because the product-state projection collapses
     the double-layer contraction.
  2. phys sweep  (bond=256): raising phys=dim helps the norm a lot but barely
     moves BornMachine -> the state projection (not phys=2) is the structural cap.

    python benchmark/bench_mfu.py
    python benchmark/bench_mfu.py --worker mpsnorm 16 512 2     # model nq bond phys
"""
import sys, os, json, subprocess, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

BOND_SWEEP = [("mpsnorm", 16, b, 2) for b in (512, 1024, 2048)] + \
             [("born",    16, b, 2) for b in (512, 1024, 2048)]
PHYS_SWEEP = [(m, 16, 256, p) for p in (2, 4, 8, 16) for m in ("mpsnorm", "born")]


def worker(model, nq, bond, phys):
    import common
    backend, _ = common.make_backend()
    if model == "mpsnorm":
        c = common.make_mps_norm(backend, nq, bond, phys)
    else:
        from tneq_qc.utils.graph_generators import QCTNHelper
        from tneq_qc.modules.app import BornMachine
        g = QCTNHelper.mps(nq, bond_dim=bond, phys_dim=phys)
        c = BornMachine(g, phys, backend=backend).auto_init(orthogonal=True).build()
    m = common.measure_method(backend, c, "cotengra")
    m.update(model=model, nq=nq, bond=bond, phys=phys)
    print("JSON" + json.dumps(m))


def run(model, nq, bond, phys, timeout=400):
    try:
        p = subprocess.run([PY, __file__, "--worker", model, str(nq), str(bond), str(phys)],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    for line in p.stdout.splitlines():
        if line.startswith("JSON"):
            return json.loads(line[4:])
    sys.stderr.write(f"[FAIL {model} {nq} {bond} {phys}] {p.stderr[-300:]}\n")
    return None


def show(title, rows):
    print(f"\n{'='*88}\n{title}")
    print(f"{'model':<10}{'phys':>5}{'bond':>6}{'GFLOP':>10}{'t_jit(ms)':>11}{'TF/s':>8}{'MFU':>8}{'AI':>8}{'peakGB':>9}")
    print("-" * 88)
    for r in rows:
        if not r:
            continue
        print(f"{r['model']:<10}{r['phys']:>5}{r['bond']:>6}{r['gflop']:>10.2f}{r['t_jit_ms']:>11.3f}"
              f"{r['tflops']:>8.1f}{r['mfu']*100:>7.1f}%{r['ai']:>8.1f}{r['peak_gb']:>9.3f}")


def driver():
    import common
    _, _, kind = common.get_device_peak()
    print(f"DEVICE: {kind}")
    bond = [run(*cfg) for cfg in BOND_SWEEP]
    phys = [run(*cfg) for cfg in PHYS_SWEEP]
    show("Sweep 1 - bond (phys=2): does large bond lift MFU?", bond)
    show("Sweep 2 - phys=dim (bond=256): phys=2 vs state-projection?", phys)
    out = {"bond_sweep": [r for r in bond if r], "phys_sweep": [r for r in phys if r]}
    with open(os.path.join(HERE, "results_mfu.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved -> {os.path.join(HERE, 'results_mfu.json')}")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=4, metavar=("MODEL", "NQ", "BOND", "PHYS"))
    a = ap.parse_args()
    if a.worker:
        worker(a.worker[0], int(a.worker[1]), int(a.worker[2]), int(a.worker[3]))
    else:
        driver()
