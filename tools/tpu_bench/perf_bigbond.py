import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, jax, jax.numpy as jnp
from tneq_qc import QCTN, EngineCommon, BackendFactory
from tneq_qc.modules.small import MPS
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.contractor.cotengra_strategy import assemble_global_einsum
raw=lambda t:t.tensor if isinstance(t,TNTensor) else t
def build(b,nq,bond):
    np.random.seed(0)
    te=MPS(nqubits=nq,bond_dim=bond,phys_dim=2,backend=b).auto_init(orthogonal=True)
    st=MPS(nqubits=nq,bond_dim=bond,phys_dim=2,backend=b).auto_init(orthogonal=True)
    c=QCTN.concat([("u",st),("t",te)]); c.set_trace("all"); return c
def timed(fn,a,it):
    raw(fn(a)).block_until_ready(); t0=time.perf_counter()
    for _ in range(it): r=fn(a)
    raw(r).block_until_ready(); return (time.perf_counter()-t0)/it
def bench(dev,cfgs,it=50):
    b=BackendFactory.create_backend("jax",device=dev,dtype="float32")
    eng=EngineCommon(backend=b,strategy="cotengra"); out=[]
    for nq,bond in cfgs:
        c=build(b,nq,bond); raw(eng.contract(c)).block_until_ready()
        eq,rt,_,_,_=assemble_global_einsum(c); arrs=[jnp.asarray(raw(t)) for t in rt]
        pl=c._cotengra_planner; f=jax.jit(lambda a:pl.contract(a))
        out.append((nq,bond,timed(f,arrs,it),pl.contraction_cost())); 
    return out
print("devices:",jax.devices())
cfgs=[(8,128),(8,256),(8,512),(8,1024),(16,256),(16,512)]
res={}
for dev in ("tpu","cpu"):
    res[dev]=bench(dev,cfgs)
print(f"{'nq':>4}{'bond':>6}{'tpu(ms)':>10}{'cpu(ms)':>10}{'speedup':>9}{'GFLOP':>10}")
for (nq,bd,tj,cost),(_,_,cj,_) in zip(res["tpu"],res["cpu"]):
    print(f"{nq:>4}{bd:>6}{tj*1e3:>10.3f}{cj*1e3:>10.3f}{cj/tj:>8.1f}x{cost/1e9:>10.2f}")
