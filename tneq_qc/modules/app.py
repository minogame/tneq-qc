"""Application-level QCTN modules.

Each class composes small modules (MPS, CircuitState, MeasureMatrix) via
``register_module`` to form a complete computational graph.  All
application modules use ``graph=None`` (composite mode) and register their
sub-networks as named submodules.

Initialization is deferred: call :meth:`auto_init` (or ``auto_init`` on
individual submodules) to populate core tensors.

Example::

    # TNEQ with two independent MPS
    model = TNEQ(nqubits=3, bond_dim=4).auto_init()

    # Quadratic form
    model = Quadratic(nqubits=3, bond_dim=4).auto_init()
    # model.mps and model.mx share no parameters with model.circuit
"""

from __future__ import annotations

from ..core.qctn import QCTN
from ..core.tn_tensor import TNTensor
from .small import MPS, CircuitState, MeasureMatrix


class PlainMPS(QCTN):
    """A single MPS as a self-contained application module.

    Wraps one :class:`MPS` sub-module and exposes its cores via the
    standard :meth:`~QCTN.named_cores` / :attr:`~QCTN.all_cores` API.

    Args:
        nqubits: Number of qubits.
        bond_dim: Bond dimension between adjacent cores.
        phys_dim: Physical boundary dimension.
        backend: Compute backend.
    """

    def __init__(self, nqubits: int, bond_dim: int, phys_dim: int = 2, backend=None):
        super().__init__(graph=None, backend=backend, _defer_init=True)
        self.register_module("mps", MPS(nqubits, bond_dim, phys_dim, backend))


class TransposeMPS(QCTN):
    """Conjugate-transpose view of an existing MPS.

    Holds a reference to *source_mps*; its :meth:`named_cores` method
    yields live conjugate-transpose views of the source cores.  Any
    in-place modification to *source_mps* tensors is automatically
    reflected here.

    No own cores are stored; :meth:`auto_init` is a no-op.

    Args:
        source_mps: The MPS whose cores are shared (by reference).
    """

    def __init__(self, source_mps: MPS):
        super().__init__(graph=None, backend=source_mps.backend, _defer_init=True)
        self._source_mps = source_mps

    def named_cores(self, prefix: str = ""):
        """Yield ``(name, conj_transpose_view)`` pairs from the source MPS.

        The returned :class:`TNTensor` views are created on each call, so
        they always reflect the current state of the source tensors.
        """
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

    After :meth:`auto_init`, the right MPS cores are conjugate-transpose
    references of the left MPS cores (``is_ref=True``, ``is_transposed=True``).
    In-place modifications to the underlying tensor data in *left* are
    automatically visible through *right*.

    Args:
        nqubits: Number of qubits.
        bond_dim: Bond dimension between adjacent cores.
        phys_dim: Physical boundary dimension.
        backend: Compute backend.
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
        # Wire right cores as conjugate-transpose references to left cores
        for name in left.cores:
            tensor = left.cores_weights[name]
            if isinstance(tensor, TNTensor):
                right.cores_weights[name] = tensor.conj_transpose()
            else:
                right.cores_weights[name] = TNTensor(tensor).conj_transpose()
        return self


class Encoding(QCTN):
    """Encoding network: CircuitState feeding into an MPS.

    Composes a :class:`CircuitState` (input ket) with an :class:`MPS`
    (encoder).  The circuit state's output physical dimensions connect to
    the MPS input physical dimensions.

    Args:
        nqubits: Number of qubits.
        bond_dim: MPS bond dimension.
        phys_dim: Physical dimension (shared between circuit and MPS).
        backend: Compute backend.
    """

    def __init__(self, nqubits: int, bond_dim: int, phys_dim: int = 2, backend=None):
        super().__init__(graph=None, backend=backend, _defer_init=True)
        self.register_module("circuit", CircuitState(nqubits, phys_dim, backend))
        self.register_module("mps", MPS(nqubits, bond_dim, phys_dim, backend))


class TNEQ(QCTN):
    """TNEQ model: inner product of two independent MPS.

    Contains two independent :class:`MPS` submodules (*mps1* and *mps2*)
    that do **not** share parameters.  Modifying *mps1* cores has no
    effect on *mps2* cores.

    Args:
        nqubits: Number of qubits.
        bond_dim: Bond dimension for both MPS.
        phys_dim: Physical boundary dimension.
        backend: Compute backend.
    """

    def __init__(self, nqubits: int, bond_dim: int, phys_dim: int = 2, backend=None):
        super().__init__(graph=None, backend=backend, _defer_init=True)
        self.register_module("mps1", MPS(nqubits, bond_dim, phys_dim, backend))
        self.register_module("mps2", MPS(nqubits, bond_dim, phys_dim, backend))


class Quadratic(QCTN):
    """Quadratic form: <circuit | mps† · mx · mps | circuit>.

    Composes:
    - ``circuit``: :class:`CircuitState` (ket, left input)
    - ``mps``: :class:`MPS` (left side of the sandwich)
    - ``mx``: :class:`MeasureMatrix` (middle observable)

    The right-side MPS† and circuit† are derived at contraction time from
    the shared ``mps`` and ``circuit`` cores (via conjugate-transpose
    views), so no duplicate parameters are stored.

    Args:
        nqubits: Number of qubits.
        bond_dim: MPS bond dimension.
        phys_dim: Physical dimension (shared across all components).
        backend: Compute backend.
    """

    def __init__(self, nqubits: int, bond_dim: int, phys_dim: int = 2, backend=None):
        super().__init__(graph=None, backend=backend, _defer_init=True)
        self.register_module("circuit", CircuitState(nqubits, phys_dim, backend))
        self.register_module("mps", MPS(nqubits, bond_dim, phys_dim, backend))
        self.register_module("mx", MeasureMatrix(nqubits, phys_dim, backend))
