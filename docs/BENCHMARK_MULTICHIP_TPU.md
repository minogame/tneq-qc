# Multi-chip TPU Benchmark (4× TPU v5e)

> Distributed inference / contraction across a 4-chip TPU v5e host.
> Hardware: **TPU v5e × 4** (single host, `jax.device_count() == 4`), 16 GB HBM/chip.
> Script: [`benchmark/bench_multichip.py`](../benchmark/bench_multichip.py).
> Partitioner: [`benchmark/partition_contract.py`](../benchmark/partition_contract.py) (KaHyPar).

Run:

```bash
python benchmark/bench_multichip.py
```

Three parts, each a different way to use 4 chips. **A/B win on throughput/latency
(data parallelism); C wins on memory (model parallelism).** Device pinning is by
input placement (`jax.device_put`), so the same code runs on a 4-chip TPU and on a
multi-GPU node.

---

## A. Data-parallel throughput scaling

The same model replicated across k chips, each running a full independent
cotengra contraction (JIT-fused), all dispatched concurrently and blocked on
together. Ideal → ~k× throughput.

| model | 1 chip | 2 chips | 4 chips |
|-------|:------:|:-------:|:-------:|
| MPS D32  | 1.00× (1145/s) | 1.77× | **2.92×** |
| MPS D48  | 1.00× (404/s)  | 1.92× | **3.53×** |
| Tree D32 | 1.00× (1140/s) | 1.77× | **2.90×** |

Scaling is 2.9–3.5× on 4 chips. The gap from 4× is the single-host dispatch
thread + per-call launch floor (these contractions are sub-ms, so launch
overhead is a visible fraction). Bigger contractions (MPS D48) scale better
(3.53×) because compute amortizes the dispatch.

## B. Heterogeneous fan-out

The 4 model structures, one pinned per chip, run concurrently. Wall-clock →
max(per-case latency), not the sum.

```
case                  alone(ms)
MPS P52                  3.353
Tree P48                 2.482
BrickWall P12            2.662
MPSBrickWall P16         0.206
sequential (sum of 4):   8.703 ms
concurrent (4 chips):    3.371 ms   -> 2.58x   throughput 1187 contractions/s
```

Speedup (2.58×) is bounded by load imbalance — MPSBrickWall finishes in 0.2 ms
while MPS takes 3.4 ms, so the slowest case sets the wall-clock. A balanced
workload → ~4×.

## C. Model-parallel: one contraction split across 4 chips

Splits a **single** contraction across the chips. A real hypergraph partitioner
(**KaHyPar**) establishes the partition — it cuts the einsum operands into K
weakly-coupled blocks, minimizing the boundary. **The computation reuses the
repo's contractor** (`CotengraPlanner`): each block is contracted locally on its
own chip into a small boundary tensor whose open legs are exactly the cut bonds,
then a final reduce contracts the K boundary tensors into the scalar. This is the
[`BENCHMARK_DISTRIBUTED.md`](BENCHMARK_DISTRIBUTED.md) scheme (partition → local
contract → reduce cross-partition edges), now library-partitioned and on the TPU.

```
                              latency                 per-chip peak intermediate
model            blocks   full     dist(4)  ratio   full      per-block   reduction
MPS P48          [20,17,20,21]  2.30ms  4.04ms  0.57x  159.4M    42.6M       3.7x
Tree P32         [21,18,21,18]  0.69ms  1.47ms  0.47x   31.5M     8.4M       3.7x
MPSBrickWall P16 [20,17,20,21]  0.12ms  1.05ms  0.12x    tiny     tiny       3.6x
```
(cut = 6 edges, open legs ≤ 4 per block, correctness err ≤ 5e-4 vs the
monolithic contraction; float64 reproduces the result to 2e-15.)

**The win is MEMORY / FEASIBILITY, not latency:** each chip materializes a peak
intermediate **~K× smaller** (3.7× here) and holds only ~1/K of the cores, so a
contraction can run across chips using a fraction of the per-chip HBM — the way
to fit a network that does not fit on one chip.

**It is not a latency win** (0.5× = slower): partitioning forces larger boundary
tensors than the monolithic path. An MPS norm, contracted whole, keeps only a
`D²` transfer environment; partitioned, each interior block exposes a `D⁴`
boundary (2 ket + 2 bra bonds). More total work, spread over chips → wall-clock
rises. This matches `BENCHMARK_DISTRIBUTED.md`: model-parallel partition wins on
speed only when the local contraction dominates (small `phys` / brickwall) and on
memory increasingly as `phys` grows.

### Partition quality is decisive (and why a library is needed)

The cost driver is the **boundary tensor size** (∏ of a block's open-leg dims),
not the cut-edge count. The metric to minimize is the max per-block boundary:

```
MPS P48 K4, max per-block boundary (lower = better)
  naive contiguous split      legs=[16,32,34,18]   2^190     (catastrophic)
  KaHyPar eps=0.03            legs=[5,6,3,6]       2^34
  KaHyPar eps=0.1             legs=[2,5,2,5]       2^28
  KaHyPar eps=0.5             legs=[2,4,2,4]       2^22      (structural optimum)
```

- A **naive contiguous split is unusable** — the einsum operand order is not
  spatial (ket cores, bra cores, measure interleaved), so contiguous blocks get
  ~34 open legs → 2^190 boundary. A real partitioner is required.
- **`epsilon` (imbalance tolerance) is the key knob.** Tight balance (0.03–0.1)
  forces 5-leg blocks → boundary 48× larger → OOM / 30× slowdown. Loosening to
  `epsilon=0.5` lets KaHyPar respect locality and reach the 4-leg structural
  optimum (2 ket + 2 bra bonds per interior block). The benchmark uses 0.5.

### Why not cotengra/cuTensorNet slicing?

cotengra and cuTensorNet's only distributed primitive is **slicing** (split into
independent slices, round-robin across ranks, all-reduce the sum —
`CotengraPlanner.slice_ids_for_rank`, `ContractionTree.contract_mpi`). On this
TPU workload slicing helps neither: these contractions are launch-bound, so
shattering one fused kernel into many slices *raises* latency (≤1.0× on 4 chips),
and their peak intermediate has no sliceable inner index (`target_size` slicing
returns nslices=1; `target_slices` cuts peak only ~1.07×). Slicing is built for
compute-bound, memory-spilling circuit simulations on GPU clusters — not for
small launch-bound tensor-network norms. The KaHyPar **partition** (Part C) is
what actually distributes memory here.

---

## Takeaways

- **Throughput / latency on TPU → data parallelism (A/B).** Replicate the model,
  fan out independent contractions: 2.9–3.5× on 4 chips, robust and simple.
- **Memory / feasibility → model parallelism (C).** KaHyPar partition + reuse the
  cotengra contractor: per-chip peak ~K× smaller, at the cost of more total
  compute (bigger boundaries) — use it to fit networks that don't fit on one chip,
  not to go faster.
- **Splitting a single contraction does not make it faster on TPU** (neither by
  partition nor by slicing). The distributed speed win is from running *many*
  independent contractions, not from parallelizing *one*.
