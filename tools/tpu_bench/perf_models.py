"""Perf sweep over common tneq-qc models: MPS, tree, brickwall, TNEQ, BornMachine.

Two phases:
  1) ESTIMATE — build each network on CPU, plan the cotengra contraction from
     shapes only (no execution), read peak intermediate size + FLOPs, and decide
     SAFE/SKIP against an HBM budget so we never OOM the 33.6 GB TPU.
  2) RUN — for SAFE configs, time inference (JIT-fused contraction) and training
     (eager contract_for_gradient) on TPU and CPU.

Bond groups: small (bond=2) and large (bond=512, tree 128) to expose any TPU
advantage. Brickwall has no bond axis (gates are phys^4, cost ~exp in width), so
it stays phys=2 / small qubits and we vary layers instead.
"""
import sys, time, traceback, gc
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import jax
import jax.numpy as jnp

from tneq_qc import QCTN, EngineCommon, BackendFactory
from tneq_qc.modules.small import MPS
from tneq_qc.modules.app import BornMachine
from tneq_qc.utils.graph_generators import QCTNHelper
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.contractor.cotengra_strategy import assemble_global_einsum
from tneq_qc.contractor.cotengra_planner import CotengraPlanner

DTYPE_BYTES = 4
HBM_BUDGET_GB = 24.0          # leave headroom under the 33.6 GB TPU HBM
SAFETY = 3.0                  # autodiff / multiple live intermediates

raw = lambda t: t.tensor if isinstance(t, TNTensor) else t


# ---------- model builders: each returns a scalar-output combined QCTN ----------
def build_mps(backend, nq, bond):
    m = MPS(nqubits=nq, bond_dim=bond, phys_dim=2, backend=backend).auto_init(orthogonal=True)
    c = QCTN.concat([("u", m), ("t", m.hermit())])
    c.set_trace("all")
    return c

def build_tneq(backend, nq, bond):
    a = MPS(nqubits=nq, bond_dim=bond, phys_dim=2, backend=backend).auto_init(orthogonal=True)
    b = MPS(nqubits=nq, bond_dim=bond, phys_dim=2, backend=backend).auto_init(orthogonal=True)
    c = QCTN.concat([("u", a), ("t", b.hermit())])
    c.set_trace("all")
    return c

def build_born(backend, nq, bond):
    graph = QCTNHelper.mps(nq, bond_dim=bond, phys_dim=2)
    model = BornMachine(graph, 2, backend=backend).auto_init(orthogonal=True)
    return model.build()

def build_tree(backend, nq, dim):
    graph = QCTNHelper.generate_example_graph(nq, graph_type="tree", dim_char=str(dim))
    t = QCTN(graph, backend=backend).auto_init(orthogonal=True)
    c = QCTN.concat([("u", t), ("t", t.hermit())])
    c.set_trace("all")
    return c

def build_brick(backend, nq, layers):
    graph = QCTNHelper.brickwall(nq, n_layers=layers, phys_dim=2)
    t = QCTN(graph, backend=backend).auto_init(orthogonal=True)
    c = QCTN.concat([("u", t), ("t", t.hermit())])
    c.set_trace("all")
    return c


# (label, builder, kwargs, bond_group)
CONFIGS = [
    ("MPS  nq32 bond2",    build_mps,  dict(nq=32, bond=2),    "small"),
    ("MPS  nq32 bond512",  build_mps,  dict(nq=32, bond=512),  "large"),
    ("MPS  nq16 bond512",  build_mps,  dict(nq=16, bond=512),  "large"),
    ("MPS  nq16 bond2048", build_mps,  dict(nq=16, bond=2048), "large"),
    ("TNEQ nq32 bond2",    build_tneq, dict(nq=32, bond=2),    "small"),
    ("TNEQ nq16 bond512",  build_tneq, dict(nq=16, bond=512),  "large"),
    ("TNEQ nq16 bond2048", build_tneq, dict(nq=16, bond=2048), "large"),
    ("Born nq16 bond2",    build_born, dict(nq=16, bond=2),    "small"),
    ("Born nq16 bond512",  build_born, dict(nq=16, bond=512),  "large"),
    ("Tree nq16 dim2",     build_tree, dict(nq=16, dim=2),     "small"),
    ("Tree nq16 dim64",    build_tree, dict(nq=16, dim=64),    "large"),
    ("Brick nq12 L4",      build_brick, dict(nq=12, layers=4), "small"),
    ("Brick nq16 L4",      build_brick, dict(nq=16, layers=4), "small"),
    ("Brick nq16 L8",      build_brick, dict(nq=16, layers=8), "small"),
]


def plan_only(combined):
    """Plan the contraction from shapes (no execution); return planner + metrics."""
    eq, raw_tensors, *_ = assemble_global_einsum(combined)
    shapes = [tuple(int(d) for d in raw(t).shape) for t in raw_tensors]
    n_param_elems = sum(int(np.prod(s)) for s in shapes)
    planner = CotengraPlanner(eq, shapes, target_slices=1, target_size=None)
    tree = planner.tree
    peak = float(tree.peak_size())
    cost = planner.contraction_cost()
    width = tree.contraction_width()    # log2 of largest tensor
    return planner, n_param_elems, peak, cost, width


def ram_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def estimate():
    print(f"TPU HBM budget: {HBM_BUDGET_GB} GB (of 33.6) | RAM available: {ram_gb():.0f} GB")
    cb = BackendFactory.create_backend("jax", device="cpu", dtype="float32")
    table = []
    print(f"\n{'config':<20}{'params(MB)':>11}{'peakInter':>12}{'peak(GB)':>10}{'GFLOP':>10}{'width':>7}  verdict")
    for label, fn, kw, group in CONFIGS:
        try:
            c = fn(cb, **kw)
            planner, npar, peak, cost, width = plan_only(c)
            par_mb = npar * DTYPE_BYTES / 1e6
            peak_gb = peak * DTYPE_BYTES * SAFETY / 1e9
            safe = peak_gb < HBM_BUDGET_GB
            verdict = "SAFE" if safe else "SKIP(OOM risk)"
            print(f"{label:<20}{par_mb:>11.2f}{peak:>12.3e}{peak_gb:>10.3f}{cost/1e9:>10.2f}{width:>7.1f}  {verdict}")
            table.append((label, fn, kw, group, safe, peak_gb))
        except Exception as e:
            print(f"{label:<20}  ERROR during estimate: {e}")
            table.append((label, fn, kw, group, False, None))
    return table


def time_inference(backend, eng, combined, iters=30):
    raw(eng.contract(combined)).block_until_ready()         # plan + warm
    eq, raw_tensors, *_ = assemble_global_einsum(combined)
    arrays = [jnp.asarray(raw(t)) for t in raw_tensors]
    planner = combined._cotengra_planner
    f = jax.jit(lambda a: planner.contract(a))
    raw(f(arrays)).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        r = f(arrays)
    raw(r).block_until_ready()
    return (time.perf_counter() - t0) / iters


def time_training(backend, eng, combined, steps=15):
    combined.requires_grad_(True)
    loss, grads = eng.contract_for_gradient(combined, target=1.0, loss="mse")
    raw(loss).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(steps):
        loss, grads = eng.contract_for_gradient(combined, target=1.0, loss="mse")
    raw(loss).block_until_ready()
    return (time.perf_counter() - t0) / steps


def run(table):
    results = []
    for label, fn, kw, group, safe, _ in table:
        if not safe:
            continue
        row = {"label": label}
        for device in ("tpu", "cpu"):
            try:
                b = BackendFactory.create_backend("jax", device=device, dtype="float32")
                eng = EngineCommon(backend=b, strategy="cotengra")
                c = fn(b, **kw)
                inf = time_inference(b, eng, c)
                c2 = fn(b, **kw)
                eng2 = EngineCommon(backend=b, strategy="cotengra")
                trn = time_training(b, eng2, c2)
                row[device] = (inf, trn)
            except Exception as e:
                row[device] = None
                print(f"  [{label} / {device}] FAILED: {e}")
                traceback.print_exc()
            finally:
                # Release device arrays + compiled caches between configs.
                for v in ("c", "c2", "eng", "eng2", "b"):
                    if v in dir():
                        pass
                gc.collect()
                jax.clear_caches()
        results.append(row)
    return results


def report(results):
    print(f"\n{'='*78}\nINFERENCE (JIT-fused contraction), ms per call")
    print(f"{'config':<20}{'TPU(ms)':>10}{'CPU(ms)':>10}{'speedup':>9}")
    for r in results:
        t = r.get("tpu"); c = r.get("cpu")
        if t and c:
            print(f"{r['label']:<20}{t[0]*1e3:>10.4f}{c[0]*1e3:>10.4f}{c[0]/t[0]:>8.1f}x")
    print(f"\nTRAINING (eager fwd+bwd), ms per step")
    print(f"{'config':<20}{'TPU(ms)':>10}{'CPU(ms)':>10}{'speedup':>9}")
    for r in results:
        t = r.get("tpu"); c = r.get("cpu")
        if t and c:
            print(f"{r['label']:<20}{t[1]*1e3:>10.3f}{c[1]*1e3:>10.3f}{c[1]/t[1]:>8.1f}x")


if __name__ == "__main__":
    print("JAX devices:", jax.devices())
    table = estimate()
    if "--estimate-only" not in sys.argv:
        results = run(table)
        report(results)
