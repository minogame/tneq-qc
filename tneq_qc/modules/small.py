"""Small (leaf) QCTN modules.

Each class wraps a single graph-based QCTN with a domain-specific
constructor.  Core tensors are *not* initialized automatically; call
:meth:`~tneq_qc.core.qctn.QCTN.auto_init` (or
:meth:`~tneq_qc.core.qctn.QCTN.set_cores`) to populate weights.

Example::

    mps = MPS(nqubits=3, bond_dim=4).auto_init()
    cs  = CircuitState(nqubits=3, phys_dim=2).auto_init()
    mx  = MeasureMatrix(nqubits=3, phys_dim=2).auto_init()
"""

from __future__ import annotations

from ..core.qctn import QCTN
from ..core.tn_tensor import TNTensor
from ..utils.graph_generators import QCTNHelper


class MPS(QCTN):
    """Matrix Product State (uniform block structure).

    All qubit rows see all cores in a left-to-right chain.  Bond
    connections between adjacent cores use ``bond_dim``; the left/right
    boundary dimensions use ``phys_dim``.

    Graph for ``MPS(nqubits=3, bond_dim=4, phys_dim=2)``::

        -2-A-4-B-4-C-2-
        -2-A-4-B-4-C-2-
        -2-A-4-B-4-C-2-

    Core tensors are declared but **not** initialized; call
    :meth:`auto_init` to initialize them.

    Args:
        nqubits: Number of qubits (rows).  Also equals the number of cores.
        bond_dim: Bond dimension between adjacent cores.
        phys_dim: Physical (input/output boundary) dimension.
        backend: Compute backend; required before calling :meth:`auto_init`.
    """

    def __init__(self, nqubits: int, bond_dim: int, phys_dim: int = 2, backend=None):
        graph = QCTNHelper.mps(nqubits, bond_dim, phys_dim)
        super().__init__(graph, backend=backend, _defer_init=True)
        self.nqubits_mps = nqubits
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim


class State(QCTN):
    """Product state (ket vector): one core per qubit, no left input dimension.

    Each qubit row has a single core with only a right (output) edge of
    dimension ``phys_dim``.  This represents a product-state input to a
    quantum circuit.

    Graph for ``CircuitState(nqubits=3, phys_dim=2)``::

        -A-2-
        -B-2-
        -C-2-

    ``auto_init`` initializes each core to the computational basis state
    ``[1, 0, ..., 0]``.

    Args:
        nqubits: Number of qubits (one core per qubit).
        phys_dim: Physical (output) dimension.
        backend: Compute backend; required before calling :meth:`auto_init`.
    """

    def __init__(self, nqubits: int, phys_dim: int = 2, backend=None):
        graph = QCTNHelper.state(nqubits, phys_dim)
        super().__init__(graph, backend=backend, _defer_init=True)
        self.nqubits_state = nqubits
        self.phys_dim = phys_dim

    def auto_init(
        self,
        dtype=None,
        device=None,
        distribution: str = "gaussian",
        orthogonal: bool = False,
    ) -> "State":
        """Initialize all state cores to ``[1, 0, ..., 0]``.

        The extra arguments are accepted for API compatibility with QCTN.
        """
        if self.backend is None:
            raise ValueError("Backend is required for auto_init")

        for core_info in self.adjacency_table:
            core_name = core_info["core_name"]
            shape = tuple(core_info["input_shape"] + core_info["output_shape"])
            batch_size = getattr(self, "_core_batch_size", None)
            tensor_shape = (batch_size,) + shape if batch_size is not None else shape
            tensor = self.backend.zeros(tensor_shape, dtype=dtype)
            raw = tensor.tensor if isinstance(tensor, TNTensor) else tensor
            if batch_size is None:
                raw.reshape(-1)[0] = 1
            else:
                raw.reshape(batch_size, -1)[:, 0] = 1
            self.cores_weights[core_name] = TNTensor(raw, has_batch=batch_size is not None)
        return self


CircuitState = State


class MeasureMatrix(QCTN):
    """Measurement matrix (operator): one core per qubit, with both in/out dims.

    Each qubit row has a single core with one input edge and one output
    edge, both of dimension ``phys_dim``.  This represents a per-qubit
    observable or quantum channel.

    Graph for ``MeasureMatrix(nqubits=3, phys_dim=2)``::

        -2-A-2-
        -2-B-2-
        -2-C-2-

    Core tensors are declared but **not** initialized; call
    :meth:`auto_init` to initialize them.

    Args:
        nqubits: Number of qubits (one core per qubit).
        phys_dim: Physical (input and output) dimension.
        backend: Compute backend; required before calling :meth:`auto_init`.
    """

    def __init__(self, nqubits: int, phys_dim: int = 2, backend=None):
        graph = QCTNHelper.measure_matrix(nqubits, phys_dim)
        super().__init__(graph, backend=backend, _defer_init=True)
        self.nqubits_mx = nqubits
        self.phys_dim = phys_dim
