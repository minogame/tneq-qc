"""Cotengra-based contraction planner with index slicing.

This module wraps cotengra to turn an opt_einsum-style equation into an
explicit *contraction tree*, optionally **sliced** over a subset of contracted
indices.  Slicing rewrites

    out = einsum(eq, *arrays)

as a sum over independent, structurally-identical sub-contractions

    out = sum_i  contract_slice(arrays, i)        i = 0 .. nslices-1

Each slice fixes the sliced indices to one combination of values.  Because we
slice **only contracted (inner) indices** (``allow_outer=False``), the summation
is exact and every slice produces a tensor with the *full* output shape — so the
result can be reduced with a plain ``+`` / all-reduce.  This is the basis for the
distributed (data-parallel-over-slices) engine, where each rank evaluates a
disjoint subset of the slice indices and the partial sums are all-reduced.

cotengra dispatches array operations through autoray, so ``contract_slice`` runs
natively on PyTorch tensors and is differentiable via ``torch.autograd``.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

from ..config import Configuration

try:
    import cotengra as ctg  # noqa: F401
    _HAS_COTENGRA = True
except ImportError:  # pragma: no cover - import guard
    ctg = None
    _HAS_COTENGRA = False


def cotengra_available() -> bool:
    """Return whether cotengra is importable in this environment."""
    return _HAS_COTENGRA


def parse_einsum(einsum_eq: str, tensor_shapes: Sequence[Sequence[int]]):
    """Parse an opt_einsum-style equation into cotengra inputs.

    Args:
        einsum_eq: ``"abc,cd,...->out"`` style string (one char per index).
        tensor_shapes: shape tuples in the same order as the input terms.

    Returns:
        ``(inputs, output, size_dict)`` where *inputs* is a list of index-tuples,
        *output* is the output index-tuple, and *size_dict* maps index -> size.
    """
    if "->" not in einsum_eq:
        raise ValueError(f"einsum equation must contain '->': {einsum_eq!r}")
    lhs, rhs = einsum_eq.split("->")
    inputs = [tuple(term) for term in lhs.split(",")]
    output = tuple(rhs)

    if len(inputs) != len(tensor_shapes):
        raise ValueError(
            f"einsum has {len(inputs)} terms but {len(tensor_shapes)} shapes given."
        )

    size_dict: dict = {}
    for term, shape in zip(inputs, tensor_shapes):
        if len(term) != len(shape):
            raise ValueError(
                f"term {term!r} rank {len(term)} != shape {tuple(shape)} rank."
            )
        for ix, dim in zip(term, shape):
            prev = size_dict.get(ix)
            if prev is not None and prev != dim:
                raise ValueError(
                    f"inconsistent size for index {ix!r}: {prev} vs {dim}."
                )
            size_dict[ix] = int(dim)
    return inputs, output, size_dict


class CotengraPlanner:
    """Plan (and execute, slice-by-slice) a contraction with cotengra.

    The plan is built once from the equation + shapes and reused across calls
    (cores change values, not shapes).  ``contract`` accepts a subset of slice
    ids so a distributed caller can evaluate only the slices it owns.
    """

    def __init__(
        self,
        einsum_eq: str,
        tensor_shapes: Sequence[Sequence[int]],
        *,
        target_slices: int = 1,
        target_size: Optional[int] = None,
        methods: Optional[Sequence[str]] = None,
        max_repeats: Optional[int] = None,
        minimize: Optional[str] = None,
        seed: Optional[int] = None,
        tree=None,
    ):
        if not _HAS_COTENGRA:
            raise ImportError(
                "cotengra is required for CotengraPlanner. Install with "
                "`pip install cotengra`."
            )

        self.einsum_eq = einsum_eq
        self.inputs, self.output, self.size_dict = parse_einsum(einsum_eq, tensor_shapes)

        # Output shape lets us synthesise a correctly-shaped zero when a rank
        # owns no slices.
        self.output_shape: Tuple[int, ...] = tuple(
            self.size_dict[ix] for ix in self.output
        )

        # Injected tree (e.g. broadcast from rank 0): use it verbatim so every
        # rank shares the identical slicing.
        if tree is not None:
            self.tree = tree
            return

        opt = ctg.HyperOptimizer(
            methods=list(methods or Configuration.cotengra_methods),
            max_repeats=max_repeats or Configuration.cotengra_max_repeats,
            minimize=minimize or Configuration.cotengra_minimize,
            parallel=False,        # never spawn pools (clashes with MPI/torchrun)
            progbar=False,
            **({"seed": seed} if seed is not None else {}),
        )
        tree = ctg.array_contract_tree(
            self.inputs, self.output, self.size_dict, optimize=opt
        )

        # Slice only contracted (inner) indices so partial sums are additive.
        # The slice search is stochastic too, so it must be seeded as well —
        # distributed ranks rely on producing the identical sliced tree.
        if target_size is not None or (target_slices and target_slices > 1):
            tree = tree.slice(
                target_size=target_size,
                target_slices=target_slices if target_slices and target_slices > 1 else None,
                allow_outer=False,
                **({"seed": seed} if seed is not None else {}),
            )

        self.tree = tree

    @property
    def nslices(self) -> int:
        """Number of independent slices (product of sliced index sizes)."""
        return self.tree.nslices

    @property
    def sliced_inds(self) -> List[str]:
        return list(self.tree.sliced_inds)

    def slice_ids_for_rank(self, rank: int, world_size: int) -> List[int]:
        """Round-robin assignment of slice ids to a rank."""
        return [i for i in range(self.nslices) if i % world_size == rank]

    def contract(self, arrays: Sequence, slice_ids: Optional[Iterable[int]] = None):
        """Contract over ``slice_ids`` (default: all slices) and sum the result.

        Args:
            arrays: tensors in the canonical input order (matches the equation).
            slice_ids: subset of slice indices to evaluate; ``None`` means all.

        Returns:
            A tensor with shape ``output_shape``.  If ``slice_ids`` is empty,
            returns a zero tensor of the right shape/dtype/device (a rank that
            owns no slices contributes zero to the global sum).
        """
        ids = list(range(self.nslices)) if slice_ids is None else list(slice_ids)

        if not ids:
            return self._zeros_like_output(arrays)

        total = self.tree.contract_slice(arrays, ids[0])
        for i in ids[1:]:
            total = total + self.tree.contract_slice(arrays, i)
        return total

    def contraction_cost(self) -> float:
        """Estimated total FLOPs of the (sliced) contraction; inf on failure."""
        try:
            return float(self.tree.contraction_cost())
        except Exception:  # pragma: no cover - cotengra version drift
            return float("inf")

    # ------------------------------------------------------------------
    def _zeros_like_output(self, arrays: Sequence):
        """Build a zero tensor matching the contraction output."""
        ref = arrays[0]
        # Use autoray-friendly construction via the reference array's backend.
        try:
            import torch

            if isinstance(ref, torch.Tensor):
                return torch.zeros(
                    self.output_shape, dtype=ref.dtype, device=ref.device
                )
        except ImportError:  # pragma: no cover
            pass
        import numpy as np

        return np.zeros(self.output_shape, dtype=getattr(ref, "dtype", float))
