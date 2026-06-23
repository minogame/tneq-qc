import sys,time
import jax, jax.numpy as jnp
def bench(dev,N,it=50):
    d=jax.devices(dev)[0]
    A=jax.device_put(jnp.ones((N,N),jnp.float32),d); B=jax.device_put(jnp.ones((N,N),jnp.float32),d)
    f=jax.jit(lambda a,b:a@b)
    f(A,B).block_until_ready()
    t0=time.perf_counter()
    for _ in range(it): r=f(A,B)
    r.block_until_ready(); dt=(time.perf_counter()-t0)/it
    out_dev=str(r.devices()); flops=2*N**3
    return dt,flops/dt/1e12,out_dev
print("control: dense NxN matmul, jitted")
print(f"{'N':>6}{'tpu(ms)':>10}{'tpuTFLOPs':>11}{'cpu(ms)':>10}{'cpuTFLOPs':>11}{'speedup':>9}")
for N in (1024,2048,4096,8192):
    td,tf,tdev=bench("tpu",N); cd,cf,cdev=bench("cpu",N)
    print(f"{N:>6}{td*1e3:>10.3f}{tf:>11.1f}{cd*1e3:>10.3f}{cf:>11.2f}{cd/td:>8.0f}x")
print("out devices -> tpu:",tdev," cpu:",cdev)
