"""Application-level QCTN modules.

Each class composes small modules (MPS, State, MeasureMatrix) via
``register_module`` to form a complete computational graph.

Example::

    model = BornMachine(QCTNHelper.mps(3, bond_dim=2, phys_dim=2), 2, backend=backend).auto_init(orthogonal=True)
    combined = model.build()       # ready-to-use combined QCTN
    data_fn = make_data_fn(gen, combined, batch_size=128, K=2)
"""

from __future__ import annotations

from typing import List

from ..core.qctn import QCTN
from ..core.tn_tensor import TNTensor
from .small import MPS, State, MeasureMatrix


class PlainMPS(QCTN):
    """A single MPS as a self-contained application module."""

    def __init__(self, nqubits: int, bond_dim: int, phys_dim: int = 2, backend=None):
        super().__init__(graph=None, backend=backend, _defer_init=True)
        self.register_module("mps", MPS(nqubits, bond_dim, phys_dim, backend))


class TransposeMPS(QCTN):
    """Conjugate-transpose view of an existing MPS.

    Holds a reference to *source_mps*; ``named_cores`` yields live
    conjugate-transpose views of the source cores.
    """

    def __init__(self, source_mps: MPS):
        super().__init__(graph=None, backend=source_mps.backend, _defer_init=True)
        self._source_mps = source_mps

    def named_cores(self, prefix: str = ""):
        for name, tensor in self._source_mps.named_cores():
            full_name = f"{prefix}{name}" if prefix else name
            if isinstance(tensor, TNTensor):
                yield full_name, tensor.conj_transpose()
            else:
                yield full_name, TNTensor(tensor).conj_transpose()

    @property
    def all_cores(self) -> dict:
        return dict(self.named_cores())


class MPS_with_Ref(QCTN):
    """Two MPS modules where the right side shares parameters with the left.

    After ``auto_init``, the right MPS cores are conjugate-transpose
    references of the left MPS cores.
    """

    def __init__(self, nqubits: int, bond_dim: int, phys_dim: int = 2, backend=None):
        super().__init__(graph=None, backend=backend, _defer_init=True)
        self.register_module("left", MPS(nqubits, bond_dim, phys_dim, backend))
        self.register_module("right", MPS(nqubits, bond_dim, phys_dim, backend))

    def auto_init(
        self,
        dtype=None,
        device=None,
        distribution: str = "gaussian",
        orthogonal: bool = False,
    ) -> "MPS_with_Ref":
        """Initialize left, then wire right as conj-transpose references."""
        left = self._submodules["left"]
        right = self._submodules["right"]
        left.auto_init(
            dtype=dtype,
            device=device,
            distribution=distribution,
            orthogonal=orthogonal,
        )
        for name in left.cores:
            tensor = left.cores_weights[name]
            if isinstance(tensor, TNTensor):
                right.cores_weights[name] = tensor.conj_transpose()
            else:
                right.cores_weights[name] = TNTensor(tensor).conj_transpose()
        return self


class Encoding(QCTN):
    """Encoding network: State + MPS."""

    def __init__(self, nqubits: int, bond_dim: int, phys_dim: int = 2, backend=None):
        super().__init__(graph=None, backend=backend, _defer_init=True)
        self.register_module("state", State(nqubits, phys_dim, backend))
        self.register_module("mps", MPS(nqubits, bond_dim, phys_dim, backend))


class TNEQ(QCTN):
    """TNEQ model: inner product of two independent MPS."""

    def __init__(self, nqubits: int, bond_dim: int, phys_dim: int = 2, backend=None):
        super().__init__(graph=None, backend=backend, _defer_init=True)
        self.register_module("mps1", MPS(nqubits, bond_dim, phys_dim, backend))
        self.register_module("mps2", MPS(nqubits, bond_dim, phys_dim, backend))


def _init_measure_identity(qctn: QCTN, backend) -> QCTN:
    """Initialize measure-like cores as identity matrices."""
    for core_info in qctn.adjacency_table:
        core_name = core_info['core_name']
        input_shape = core_info['input_shape']
        output_shape = core_info['output_shape']
        input_dim = core_info['input_dim']
        output_dim = core_info['output_dim']
        if input_dim != output_dim:
            raise ValueError(
                f"Measure core {core_name!r} must be square, got {input_dim} and {output_dim}."
            )
        core = backend.eye(input_dim)
        raw = core.tensor if isinstance(core, TNTensor) else core
        qctn.cores_weights[core_name] = TNTensor(
            backend.reshape(raw, input_shape + output_shape)
        )
        qctn.cores_weights[core_name].requires_grad_(False)
    return qctn


class BornMachine(QCTN):
    """Born machine: <state | tn_h · mx · tn | state>.

    Composes state, tn (trainable), and mx (data-driven). Call
    ``build()`` to get the 5-segment combined QCTN ready for contraction.

    Example::

        graph = QCTNHelper.mps(4, bond_dim=2, phys_dim=2)
        model = BornMachine(graph, 2, backend=backend).auto_init(orthogonal=True)
        combined = model.build()
        # combined has: state + tn + mx + tn_h + state_t
    """

    def __init__(
        self,
        graph: str,
        dim: int,
        backend=None,
        mx_graph: str = None,
    ):
        if graph is None:
            raise ValueError("BornMachine requires a non-None graph string.")
        super().__init__(graph=None, backend=backend, _defer_init=True)
        tn_module = QCTN.from_graph(graph, backend=backend)
        mx_module = (
            QCTN.from_graph(mx_graph, backend=backend)
            if mx_graph is not None
            else MeasureMatrix(tn_module.nqubits, dim, backend)
        )
        self._graph = graph
        self._mx_graph = mx_graph
        self._dim = dim
        self._nqubits = tn_module.nqubits
        self.register_module("state", State(self._nqubits, dim, backend))
        self.register_module("tn", tn_module)
        self.register_module("mx", mx_module)

    def auto_init(
        self,
        dtype=None,
        device=None,
        distribution: str = "gaussian",
        orthogonal: bool = False,
    ) -> "BornMachine":
        self._submodules["state"].auto_init(dtype=dtype, device=device)
        self._submodules["tn"].auto_init(
            dtype=dtype,
            device=device,
            distribution=distribution,
            orthogonal=orthogonal,
        )
        _init_measure_identity(self._submodules["mx"], self.backend)
        return self

    def build(self) -> QCTN:
        """Return the 5-segment combined QCTN: state + tn + mx + tn_h + state_t.

        The tn submodule should have ``requires_grad_(True)`` set before
        calling this method. The returned QCTN is ready for training.

        Returns:
            Combined QCTN with all segments concatenated.
        """
        state = self._submodules['state']
        tn = self._submodules['tn']
        mx = self._submodules['mx']

        tn_h = tn.hermit()
        state_bra = state.bra()

        combined = QCTN.concat([
            ('state', state),
            ('tn', tn),
            ('mx', mx),
            ('tn_h', tn_h),
            ('state_t', state_bra),
        ])
        self._combined = combined
        return combined

    @property
    def mx_core_names(self) -> List[str]:
        """Readable names of mx cores in the combined QCTN.

        Must be called after ``build()``.
        """
        combined = getattr(self, '_combined', None)
        if combined is None:
            raise RuntimeError("Call build() before accessing mx_core_names")
        return [
            combined.core_names[sym] for sym in combined.cores
            if combined.core_names.get(sym, '').startswith('mx.')
        ]

