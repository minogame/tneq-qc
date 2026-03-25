"""Application-level QCTN modules.

Each class composes small modules (MPS, CircuitState, MeasureMatrix) via
``register_module`` to form a complete computational graph.

Example::

    model = Quadratic(nqubits=3, bond_dim=4, backend=backend).auto_init()
    combined = model.build()       # ready-to-use combined QCTN
    data_fn = make_data_fn(gen, combined, batch_size=128, K=2)
"""

from __future__ import annotations

from typing import List

from ..core.qctn import QCTN
from ..core.tn_tensor import TNTensor
from .small import MPS, CircuitState, MeasureMatrix


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

    def auto_init(self, dtype=None, device=None) -> "MPS_with_Ref":
        """Initialize left, then wire right as conj-transpose references."""
        left = self._submodules["left"]
        right = self._submodules["right"]
        left.auto_init(dtype=dtype, device=device)
        for name in left.cores:
            tensor = left.cores_weights[name]
            if isinstance(tensor, TNTensor):
                right.cores_weights[name] = tensor.conj_transpose()
            else:
                right.cores_weights[name] = TNTensor(tensor).conj_transpose()
        return self


class Encoding(QCTN):
    """Encoding network: CircuitState + MPS."""

    def __init__(self, nqubits: int, bond_dim: int, phys_dim: int = 2, backend=None):
        super().__init__(graph=None, backend=backend, _defer_init=True)
        self.register_module("circuit", CircuitState(nqubits, phys_dim, backend))
        self.register_module("mps", MPS(nqubits, bond_dim, phys_dim, backend))


class TNEQ(QCTN):
    """TNEQ model: inner product of two independent MPS."""

    def __init__(self, nqubits: int, bond_dim: int, phys_dim: int = 2, backend=None):
        super().__init__(graph=None, backend=backend, _defer_init=True)
        self.register_module("mps1", MPS(nqubits, bond_dim, phys_dim, backend))
        self.register_module("mps2", MPS(nqubits, bond_dim, phys_dim, backend))


class Quadratic(QCTN):
    """Quadratic form: <circuit | mps_h · mx · mps | circuit>.

    Composes circuit, mps (trainable), and mx (data-driven). Call
    ``build()`` to get the 5-segment combined QCTN ready for contraction.

    Example::

        model = Quadratic(nqubits=4, bond_dim=2, backend=backend).auto_init()
        combined = model.build()
        # combined has: cs + mps + mx + mps_h + cs_t
    """

    def __init__(self, nqubits: int, bond_dim: int, phys_dim: int = 2, backend=None):
        super().__init__(graph=None, backend=backend, _defer_init=True)
        self._nqubits = nqubits
        self._phys_dim = phys_dim
        self.register_module("circuit", CircuitState(nqubits, phys_dim, backend))
        self.register_module("mps", MPS(nqubits, bond_dim, phys_dim, backend))
        self.register_module("mx", MeasureMatrix(nqubits, phys_dim, backend))

    def build(self) -> QCTN:
        """Return the 5-segment combined QCTN: cs + mps + mx + mps_h + cs_t.

        The mps submodule should have ``requires_grad_(True)`` set before
        calling this method. The returned QCTN is ready for training.

        Returns:
            Combined QCTN with all segments concatenated.
        """
        circuit = self._submodules['circuit']
        mps = self._submodules['mps']
        mx = self._submodules['mx']

        mps_h = mps.hermit()
        circuit_bra = circuit.bra()

        combined = QCTN.concat([
            ('cs', circuit),
            ('mps', mps),
            ('mx', mx),
            ('mps_h', mps_h),
            ('cs_t', circuit_bra),
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
