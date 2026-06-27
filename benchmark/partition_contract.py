"""Library-partitioned, model-parallel tensor-network contraction.

KaHyPar establishes the partition (split the einsum operands into K weakly-coupled
blocks, minimizing the cut = cross-block bonds).  The *computation reuses the
repo's contraction* (CotengraPlanner): each block is contracted locally into a
small boundary tensor whose open legs are exactly the cut bonds, then a final
reduce contracts the K boundary tensors into the scalar.

This is the BENCHMARK_DISTRIBUTED.md scheme (partition -> local contract ->
reduce cross-partition edges), but the partition is found by a real hypergraph
partitioner and the locals/reduce run through the existing CotengraPlanner.  Map
the K blocks onto K TPU chips by placing each block's operands on its chip.
"""
import io, contextlib, os
import opt_einsum
import kahypar
from tneq_qc.contractor.cotengra_planner import CotengraPlanner

_INI = os.path.join(os.path.dirname(__import__("cotengra").__file__),
                    "pathfinders", "kahypar_profiles", "km1_kKaHyPar_sea20.ini")


# --------------------------------------------------------------------------- #
# Partition the einsum operands into K blocks with KaHyPar.
# --------------------------------------------------------------------------- #
def kahypar_partition(inputs, output, size_dict, K, epsilon=0.5, seed=0):
    """inputs: list of index-strings (one per operand). Returns block id per op."""
    # nets = indices; each net connects the operands (nodes) that carry it.
    idx_to_nodes = {}
    for n, term in enumerate(inputs):
        for ix in set(term):
            idx_to_nodes.setdefault(ix, []).append(n)
    nets = [ix for ix, nodes in idx_to_nodes.items() if len(nodes) >= 2]

    eptr, eind, ewts = [0], [], []
    for ix in nets:
        eind.extend(idx_to_nodes[ix])
        eptr.append(len(eind))
        # cut cost of a bond ~ log2(dim): cutting a fat bond costs more
        import math
        ewts.append(max(1, int(round(math.log2(size_dict[ix])))))

    # node weight ~ log2(operand size) so blocks are balanced by (log) work/mem
    import math
    nwts = []
    for term in inputs:
        sz = 1
        for ix in term:
            sz *= size_dict[ix]
        nwts.append(max(1, int(round(math.log2(max(sz, 2))))))

    hg = kahypar.Hypergraph(len(inputs), len(nets), eptr, eind, K, ewts, nwts)
    ctx = kahypar.Context()
    ctx.loadINIconfiguration(_INI)
    ctx.setK(K)
    ctx.setEpsilon(epsilon)
    ctx.setSeed(seed)
    ctx.suppressOutput(True)
    kahypar.partition(hg, ctx)
    return [hg.blockID(n) for n in range(len(inputs))]


# --------------------------------------------------------------------------- #
# Build the per-block sub-equations + the final reduce equation.
# --------------------------------------------------------------------------- #
def build_plan(inputs, output, size_dict, part, K):
    """Return (blocks, block_open, reduce_eq, reduce_open).

    blocks[b]      = operand ids in block b
    block_open[b]  = ordered open index-string of block b's boundary tensor
    reduce_eq      = einsum over the K boundary tensors -> output
    """
    out_set = set(output)
    # occurrences of each index across blocks
    idx_blocks = {}
    for n, term in enumerate(inputs):
        b = part[n]
        for ix in term:
            idx_blocks.setdefault(ix, set()).add(b)

    blocks = [[] for _ in range(K)]
    for n in range(len(inputs)):
        blocks[part[n]].append(n)

    block_open = []
    for b in range(K):
        present = set()
        for n in blocks[b]:
            present.update(inputs[n])
        # keep an index open if it leaves block b: spans >1 block, or is a global output
        openset = {ix for ix in present
                   if len(idx_blocks[ix]) > 1 or ix in out_set}
        block_open.append("".join(sorted(openset)))

    reduce_eq = ",".join(block_open) + "->" + output
    return blocks, block_open, reduce_eq


# --------------------------------------------------------------------------- #
# One-shot: build planners for each block + the reduce (shapes only).
# --------------------------------------------------------------------------- #
def make_partitioned(eq, shapes, K, epsilon=0.1, seed=0):
    inputs, output = eq.split("->")
    inputs = inputs.split(",")
    size_dict = {}
    for term, shp in zip(inputs, shapes):
        for ix, d in zip(term, shp):
            size_dict[ix] = d

    part = kahypar_partition(inputs, output, size_dict, K, epsilon, seed)
    blocks, block_open, reduce_eq = build_plan(inputs, output, size_dict, part, K)

    # per-block planner (reuse CotengraPlanner) over the block's operands
    block_planners = []
    for b in range(K):
        sub_inputs = [inputs[n] for n in blocks[b]]
        sub_shapes = [shapes[n] for n in blocks[b]]
        sub_eq = ",".join(sub_inputs) + "->" + block_open[b]
        with contextlib.redirect_stdout(io.StringIO()):
            pl = CotengraPlanner(sub_eq, sub_shapes, seed=seed)
        block_planners.append(pl)

    # reduce planner over the K boundary tensors
    reduce_shapes = [tuple(size_dict[ix] for ix in bo) for bo in block_open]
    with contextlib.redirect_stdout(io.StringIO()):
        reduce_pl = CotengraPlanner(reduce_eq, reduce_shapes, seed=seed)

    # cut metrics
    cut_edges = sum(1 for ix, bs in
                    {ix: {part[n] for n in range(len(inputs)) if ix in inputs[n]}
                     for ix in size_dict}.items() if len(bs) > 1)
    return dict(part=part, blocks=blocks, block_open=block_open,
                block_planners=block_planners, reduce_pl=reduce_pl,
                reduce_eq=reduce_eq, cut_edges=cut_edges,
                block_sizes=[len(b) for b in blocks])


def contract_partitioned(plan, arrs):
    """Contract using the plan. arrs in original operand order. Single-device."""
    out = []
    for b, pl in enumerate(plan["block_planners"]):
        sub = [arrs[n] for n in plan["blocks"][b]]
        out.append(pl.contract(sub))
    return plan["reduce_pl"].contract(out)
