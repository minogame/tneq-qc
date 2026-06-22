"""Sliced (data-parallel-over-slices) distributed engine.

This is the slicing-based distributed paradigm (cf. the cuQuantum / quimb +
cotengra pattern): cotengra cuts the contraction into ``nslices`` independent,
structurally-identical sub-contractions whose results **sum** to the full
result.  The full network is *replicated* on every rank (no model/graph
partitioning); each rank evaluates a disjoint subset of the slice indices and
the partial sums are all-reduced.

Because cotengra slices only contracted (inner) indices (``allow_outer=False``),
the batch index — an *output* index — is never sliced, so every rank produces
the **full-batch** partial result.  Loss is therefore computed exactly as in the
single-node case, on the all-reduced global result.

Gradient flow (replicated data parallelism):

    P       = sum_r P_r                      (P_r = this rank's slice sum)
    loss    = g(P)
    dloss/dθ = g'(P) · sum_r dP_r/dθ

We use the identity ``P = P_r + (allreduce(P_r.detach()) - P_r.detach())`` so the
forward value is global while the local autograd graph only sees ``P_r``; the
per-rank gradient contributions ``g'(P)·dP_r/dθ`` are then summed with a final
all-reduce.  No learning-rate rescaling is needed (gradients are summed, not
averaged).

Contrast with :class:`EngineDistributed` (layer/model partitioning), which is
kept available as a fallback.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ...core.engine_common import EngineCommon
from ...core.tn_tensor import TNTensor
from ...losses import LossRegistry
from ...losses.target import TargetResolver
from ...contractor.cotengra_strategy import assemble_global_einsum
from ...contractor.cotengra_planner import CotengraPlanner, cotengra_available
from ...config import Configuration


def _dist():
    """Return torch.distributed if usable in this process, else None."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist
    except Exception:  # pragma: no cover
        pass
    return None


class EngineSliced(EngineCommon):
    """Distributed engine using cotengra index slicing (data parallel).

    Usage::

        engine = EngineSliced(backend='pytorch', comm=comm)
        loss, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
        optimizer.step(list(grads))
    """

    def __init__(
        self,
        backend=None,
        comm=None,
        target_slices: Optional[int] = None,
        strategy: Optional[str] = None,
    ):
        if strategy is None:
            strategy = Configuration.distributed_default_strategy
        super().__init__(backend=backend, strategy=strategy)

        if not cotengra_available():
            raise ImportError(
                "EngineSliced requires cotengra. Install with `pip install cotengra`."
            )

        self.comm = comm
        d = _dist()
        if comm is not None:
            self.rank = comm.rank
            self.world_size = comm.world_size
        elif d is not None:
            self.rank = d.get_rank()
            self.world_size = d.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # Desired slice count; at least world_size so every rank gets work.
        self._target_slices = target_slices

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def _ensure_plan(self, qctn) -> List[int]:
        """Build the cotengra plan (once) and return this rank's slice ids."""
        requested = self._target_slices or self.world_size
        target = max(int(requested), self.world_size, 1)

        planner = getattr(qctn, "_cotengra_planner", None)
        if planner is None or getattr(qctn, "_cotengra_target_slices", None) != target:
            eq, raw_tensors, _s, _ls, _out = assemble_global_einsum(qctn)
            shapes = [tuple(int(d) for d in t.shape) for t in raw_tensors]

            # Build the sliced tree on rank 0, then broadcast it so every rank
            # shares the *identical* slicing (slice ids are only comparable
            # across ranks when the tree is the same).  This is the canonical
            # "plan once, broadcast" paradigm.
            d = _dist()
            if self.rank == 0 or self.world_size == 1:
                planner = CotengraPlanner(
                    eq,
                    shapes,
                    target_slices=target,
                    target_size=Configuration.cotengra_target_size,
                    seed=Configuration.cotengra_seed,
                )
                tree = planner.tree
            else:
                tree = None

            if self.world_size > 1 and d is not None:
                obj = [tree]
                d.broadcast_object_list(obj, src=0)
                tree = obj[0]
                if self.rank != 0:
                    planner = CotengraPlanner(eq, shapes, tree=tree)

            qctn._cotengra_planner = planner
            qctn._cotengra_target_slices = target

        slice_ids = planner.slice_ids_for_rank(self.rank, self.world_size)
        qctn._cotengra_slice_ids = slice_ids
        return slice_ids

    @property
    def nslices(self) -> Optional[int]:
        return None

    # ------------------------------------------------------------------
    # Collective helpers
    # ------------------------------------------------------------------
    def _allreduce_sum(self, tensor):
        """Plain (non-autograd) all-reduce SUM across all ranks."""
        d = _dist()
        if self.world_size > 1 and d is not None:
            out = tensor.clone().contiguous()
            d.all_reduce(out, op=d.ReduceOp.SUM)
            return out
        return tensor

    @staticmethod
    def _value_tensor(result):
        """Extract the underlying value tensor (tensor * scale) from a result."""
        if isinstance(result, TNTensor):
            return result.tensor * result.scale
        return result

    # ------------------------------------------------------------------
    # Engine interface
    # ------------------------------------------------------------------
    def contract(self, qctn) -> Any:
        """Forward-only distributed contraction (sum slices, all-reduce)."""
        self._ensure_plan(qctn)
        local = super().contract(qctn)          # sums this rank's slices
        value = self._value_tensor(local)
        if self.world_size > 1:
            value = self._allreduce_sum(value.detach())
        return TNTensor(value) if isinstance(local, TNTensor) else value

    def contract_for_gradient(self, qctn, target=None, loss=None) -> Tuple:
        """Distributed contraction + gradient (data-parallel over slices)."""
        import torch

        self._ensure_plan(qctn)

        cache_key = f"_compiled_strategy_{self.strategy}"
        if not hasattr(qctn, cache_key):
            compute_fn, name, cost = self.strategy_compiler.compile(qctn, {}, self.backend)
            setattr(qctn, cache_key, {"compute_fn": compute_fn, "strategy_name": name, "cost": cost})
        compute_fn = getattr(qctn, cache_key)["compute_fn"]

        # Collect trainable leaf tensors (replicated across ranks).
        leaves: List[Any] = []
        for c_name in qctn.cores:
            c = qctn.cores_weights[c_name]
            raw = c.tensor if isinstance(c, TNTensor) else c
            if not getattr(raw, "requires_grad", False):
                continue
            if hasattr(raw, "is_leaf") and not raw.is_leaf:
                continue
            leaves.append(raw)

        # Local partial result (this rank's slice subset), differentiable.
        local = compute_fn(None, None, None)
        v_local = self._value_tensor(local)

        # Globalise the value while keeping the local autograd graph:
        #   value = v_local + (allreduce(detach(v_local)) - detach(v_local))
        if self.world_size > 1:
            v_global_const = self._allreduce_sum(v_local.detach())
            value = v_local + (v_global_const - v_local.detach())
        else:
            value = v_local

        result = TNTensor(value) if isinstance(local, TNTensor) else value

        loss_obj = LossRegistry.resolve(loss)
        resolved_target = TargetResolver.resolve(
            target, result.shape, self.backend, engine=self
        )
        loss_val = loss_obj(result, resolved_target, self.backend)

        grads_local = torch.autograd.grad(
            outputs=loss_val,
            inputs=leaves,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )

        grads = []
        for leaf, g in zip(leaves, grads_local):
            if g is None:
                g = torch.zeros_like(leaf)
            grads.append(self._allreduce_sum(g.contiguous()))

        return loss_val, grads

    def train_step(self, optimizer, qctn, target=None, loss=None) -> float:
        loss_val, grads = self.contract_for_gradient(qctn, target=target, loss=loss)
        optimizer.step(list(grads))
        return loss_val.item() if hasattr(loss_val, "item") else float(loss_val)
