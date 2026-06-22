"""Sliced contraction strategy backed by cotengra.

``SlicedCotengraStrategy`` plans (and optionally **slices**) a contraction with
cotengra.  Crucially it derives its single global einsum equation from
``qctn.build_graph()`` — the *same* correctly-wired graph that
:class:`RowPriorityStrategy` uses (paired bond symbols, trace closure, batch
hyper-indices).  It deliberately does **not** use ``qctn.get_einsum_info`` (the
``EinsumStrategy`` path), which mis-wires the generic concat/trace case.

Like ``RowPriorityStrategy``, the returned ``compute_fn`` ignores the
``cores_dict`` argument and reads the live tensors from ``build_graph()``; those
tensors are the same leaf objects autograd differentiates, so gradients flow
unchanged.

Single-node use is a drop-in strategy (``strategy="cotengra"``).  With the
default ``target_slices == 1`` it produces the same result as ``row_priority``,
just planned by cotengra.  The sliced/distributed engine raises
``target_slices`` and assigns each rank a slice subset via
``qctn._cotengra_slice_ids``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from .einsum_strategy import EinsumStrategy
from .cotengra_planner import CotengraPlanner, cotengra_available
from ..config import Configuration
from ..core.tn_tensor import TNTensor
from ..core.qctn import TensorSide


def assemble_global_einsum(qctn, cores_override=None):
    """Build one global einsum from ``qctn.build_graph()``.

    Returns ``(eq, raw_tensors, total_scale, total_log_scale, output)`` where
    *raw_tensors* are the unwrapped (scale-stripped) tensors in equation order
    and *total_scale*/*total_log_scale* carry the combined TNTensor scale.

    The wiring (symbols, trace closure, batch indices) is exactly what
    ``RowPriorityStrategy`` consumes, so the contraction is numerically
    identical to ``row_priority``.

    Args:
        cores_override: optional ``{core_name: tensor}`` mapping.  When given,
            each entry's tensor is replaced by ``cores_override[core_name]`` if
            present.  This routes the *traced* parameters from the gradient
            ``loss_fn`` into the contraction so that backends with functional
            autodiff (JAX's ``value_and_grad``) see the dependency.  For PyTorch
            the overrides are the same leaf objects, so behaviour is unchanged.
    """
    entries, _ = qctn.build_graph()

    terms: List[str] = []
    raw_tensors: List[Any] = []
    batch_syms: set = set()
    total_scale = None
    total_log_scale = None

    for e in entries:
        tensor = e["tensor"]
        if cores_override is not None:
            name = e.get("core_name")
            if name in cores_override:
                tensor = cores_override[name]
        batch_sym = e.get("batch_symbol", "") or ""
        for ch in batch_sym:
            batch_syms.add(ch)

        ins = e["in_edge_list"]
        outs = e["out_edge_list"]
        raw = tensor.tensor if isinstance(tensor, TNTensor) else tensor

        if e["side"] == TensorSide.RIGHT:
            # RIGHT tensors store dims in a reversed layout (see _contract_group).
            oin = e.get("original_in_edge_count", len(outs))
            oout = e.get("original_out_edge_count", len(ins))
            dim_syms: List[Any] = [None] * (raw.ndim - len(batch_sym))
            for idx, edge in enumerate(outs):
                dim_syms[oin - 1 - idx] = edge["symbol"]
            for idx, edge in enumerate(ins):
                dim_syms[oin + oout - 1 - idx] = edge["symbol"]
            term = batch_sym + "".join(s for s in dim_syms if s is not None)
        else:
            term = (
                batch_sym
                + "".join(ed["symbol"] for ed in ins)
                + "".join(ed["symbol"] for ed in outs)
            )
        terms.append(term)

        raw_tensors.append(raw)
        if isinstance(tensor, TNTensor):
            total_scale = tensor.scale if total_scale is None else total_scale * tensor.scale
            total_log_scale = (
                tensor.log_scale if total_log_scale is None
                else total_log_scale + tensor.log_scale
            )

    # Output indices: batch hyper-indices ('a','b') first, then open legs
    # (symbols appearing exactly once).  Contracted/trace symbols appear twice
    # and are summed.
    counts: Dict[str, int] = {}
    for term in terms:
        for ch in term:
            counts[ch] = counts.get(ch, 0) + 1
    output: List[str] = []
    for b in ("a", "b"):
        if b in batch_syms:
            output.append(b)
    output += sorted(s for s, c in counts.items() if c == 1 and s not in batch_syms)

    eq = ",".join(terms) + "->" + "".join(output)
    return eq, raw_tensors, total_scale, total_log_scale, "".join(output)


class SlicedCotengraStrategy(EinsumStrategy):
    """cotengra-backed sliced contraction (wiring matches row_priority)."""

    @property
    def name(self) -> str:
        return "cotengra"

    def check_compatibility(self, qctn, shapes_info: Dict[str, Any]) -> bool:
        """Usable on any structure, provided cotengra is installed."""
        return cotengra_available()

    def get_compute_function(self, qctn, shapes_info: Dict[str, Any], backend, **_kwargs) -> Callable:
        def compute_fn(_cores_dict=None, _circuit_states=None, _measure_matrices=None, **_):
            # Wire from build_graph(); when a cores_dict is supplied (the
            # gradient loss_fn passes traced params here) route those tensors in
            # so functional-autodiff backends (JAX) see the dependency.  For
            # PyTorch the overrides are the same leaf objects → unchanged.
            eq, raw_tensors, total_scale, total_log_scale, _out = assemble_global_einsum(
                qctn, cores_override=_cores_dict
            )

            planner = getattr(qctn, "_cotengra_planner", None)
            if planner is None:
                target_slices = int(
                    getattr(
                        qctn,
                        "_cotengra_target_slices",
                        Configuration.cotengra_target_slices,
                    )
                )
                planner = CotengraPlanner(
                    eq,
                    [tuple(int(d) for d in t.shape) for t in raw_tensors],
                    target_slices=target_slices,
                    target_size=Configuration.cotengra_target_size,
                )
                qctn._cotengra_planner = planner

            slice_ids = getattr(qctn, "_cotengra_slice_ids", None)
            result = planner.contract(raw_tensors, slice_ids=slice_ids)

            if total_scale is not None:
                result = TNTensor(result, scale=total_scale, log_scale=total_log_scale)
                result = backend.maybe_auto_scale(result)
            return result

        return compute_fn

    def estimate_cost(self, qctn, shapes_info: Dict[str, Any]) -> float:
        if not cotengra_available():
            return float("inf")
        planner = getattr(qctn, "_cotengra_planner", None)
        if planner is not None:
            return planner.contraction_cost()
        return 1.0
