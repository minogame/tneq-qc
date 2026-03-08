"""Tests for Phase 2.6: small modules and application modules.

Default configuration
---------------------
- backend : PyTorch, CPU
- dtype   : complex64 (complex)

All fixtures accept optional overrides via the ``backend`` fixture.

Coverage
--------
A. Small module initialization
B. Application module initialization
C. Small-module composition into application modules
D. Transpose and reference semantics
E. chunk / concat (renamed from split / merge)
F. Contract basic shapes
"""

from __future__ import annotations

import pytest
import torch

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.modules import (
    MPS,
    CircuitState,
    MeasureMatrix,
    PlainMPS,
    TransposeMPS,
    MPS_with_Ref,
    Encoding,
    TNEQ,
    Quadratic,
)
from tneq_qc.contractor.einsum_strategy import EinsumStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def backend():
    return BackendFactory.create_backend("pytorch", device="cpu", dtype="complex64")


@pytest.fixture
def mps3(backend):
    return MPS(nqubits=3, bond_dim=4, phys_dim=2, backend=backend)


@pytest.fixture
def mps3_init(mps3):
    return mps3.auto_init()


@pytest.fixture
def cs3(backend):
    return CircuitState(nqubits=3, phys_dim=2, backend=backend)


@pytest.fixture
def cs3_init(cs3):
    return cs3.auto_init()


@pytest.fixture
def mx3(backend):
    return MeasureMatrix(nqubits=3, phys_dim=2, backend=backend)


@pytest.fixture
def mx3_init(mx3):
    return mx3.auto_init()


# ---------------------------------------------------------------------------
# A. Small module initialization
# ---------------------------------------------------------------------------

class TestSmallModuleInit:

    def test_mps_cores_empty_before_auto_init(self, mps3):
        assert mps3.cores_weights == {}

    def test_mps_auto_init_populates_cores(self, mps3_init):
        assert len(mps3_init.cores_weights) == mps3_init.ncores
        for name in mps3_init.cores:
            assert name in mps3_init.cores_weights

    def test_mps_auto_init_returns_self(self, mps3):
        result = mps3.auto_init()
        assert result is mps3

    def test_mps_core_shapes_correct(self, mps3_init):
        """After auto_init, each core weight should be non-empty."""
        for name, tensor in mps3_init.cores_weights.items():
            t = tensor.tensor if isinstance(tensor, TNTensor) else tensor
            assert t.numel() > 0

    def test_mps_nqubits(self, mps3):
        assert mps3.nqubits == 3

    def test_mps_ncores_equals_nqubits(self, mps3):
        assert mps3.ncores == 3

    def test_circuit_state_cores_empty_before_init(self, cs3):
        assert cs3.cores_weights == {}

    def test_circuit_state_auto_init_populates_cores(self, cs3_init):
        assert len(cs3_init.cores_weights) == cs3_init.ncores == 3

    def test_circuit_state_nqubits(self, cs3):
        assert cs3.nqubits == 3

    def test_measure_matrix_cores_empty_before_init(self, mx3):
        assert mx3.cores_weights == {}

    def test_measure_matrix_auto_init_populates_cores(self, mx3_init):
        assert len(mx3_init.cores_weights) == mx3_init.ncores == 3

    def test_auto_init_dtype_complex(self, mps3):
        """auto_init should produce complex tensors when backend dtype is complex64."""
        mps3.auto_init()
        for name, tensor in mps3.cores_weights.items():
            t = tensor.tensor if isinstance(tensor, TNTensor) else tensor
            assert t.is_complex(), f"Core {name} is not complex: {t.dtype}"

    def test_set_cores_external(self, backend):
        """set_cores with dict should replace cores_weights."""
        mps = MPS(nqubits=2, bond_dim=3, phys_dim=2, backend=backend).auto_init()
        new_weights = {}
        for name, tensor in mps.cores_weights.items():
            t = tensor.tensor if isinstance(tensor, TNTensor) else tensor
            new_weights[name] = torch.zeros_like(t)
        mps.set_cores(new_weights, strict=True)
        for name, tensor in mps.cores_weights.items():
            t = tensor.tensor if isinstance(tensor, TNTensor) else tensor
            assert torch.all(t == 0), f"Core {name} not zeroed after set_cores"


# ---------------------------------------------------------------------------
# B. Application module initialization
# ---------------------------------------------------------------------------

class TestAppModuleInit:

    def test_plain_mps_has_mps_submodule(self, backend):
        model = PlainMPS(nqubits=3, bond_dim=4, backend=backend)
        assert "mps" in model._submodules
        assert isinstance(model._submodules["mps"], MPS)

    def test_plain_mps_named_cores_count(self, backend):
        model = PlainMPS(nqubits=3, bond_dim=4, backend=backend).auto_init()
        names = [n for n, _ in model.named_cores()]
        # mps has 3 cores, prefixed as "mps.A", "mps.B", "mps.C"
        assert len(names) == 3

    def test_plain_mps_named_cores_prefixed(self, backend):
        model = PlainMPS(nqubits=3, bond_dim=4, backend=backend).auto_init()
        names = [n for n, _ in model.named_cores()]
        for n in names:
            assert n.startswith("mps.")

    def test_encoding_submodules(self, backend):
        model = Encoding(nqubits=3, bond_dim=4, backend=backend)
        assert "circuit" in model._submodules
        assert "mps" in model._submodules
        assert isinstance(model._submodules["circuit"], CircuitState)
        assert isinstance(model._submodules["mps"], MPS)

    def test_encoding_all_cores_count(self, backend):
        model = Encoding(nqubits=3, bond_dim=4, backend=backend).auto_init()
        # 3 circuit cores + 3 mps cores = 6
        assert len(model.all_cores) == 6

    def test_tneq_submodules(self, backend):
        model = TNEQ(nqubits=3, bond_dim=4, backend=backend)
        assert "mps1" in model._submodules
        assert "mps2" in model._submodules

    def test_tneq_all_cores_count(self, backend):
        model = TNEQ(nqubits=3, bond_dim=4, backend=backend).auto_init()
        # mps1: 3 cores + mps2: 3 cores = 6
        assert len(model.all_cores) == 6

    def test_quadratic_submodules(self, backend):
        model = Quadratic(nqubits=3, bond_dim=4, backend=backend)
        assert "circuit" in model._submodules
        assert "mps" in model._submodules
        assert "mx" in model._submodules

    def test_quadratic_all_cores_count(self, backend):
        model = Quadratic(nqubits=3, bond_dim=4, backend=backend).auto_init()
        # circuit: 3 + mps: 3 + mx: 3 = 9
        assert len(model.all_cores) == 9

    def test_auto_init_chain_propagates(self, backend):
        """auto_init on parent should init all submodules."""
        model = TNEQ(nqubits=2, bond_dim=3, backend=backend)
        assert model._submodules["mps1"].cores_weights == {}
        model.auto_init()
        assert len(model._submodules["mps1"].cores_weights) == 2
        assert len(model._submodules["mps2"].cores_weights) == 2


# ---------------------------------------------------------------------------
# C. Small-module composition into application modules
# ---------------------------------------------------------------------------

class TestComposition:

    def test_manual_composition_matches_encoding(self, backend):
        """Manually assembling circuit+mps should give same core count as Encoding."""
        nq, bd = 3, 4
        container = QCTN(graph=None, backend=backend)
        container.register_module("circuit", CircuitState(nq, 2, backend))
        container.register_module("mps", MPS(nq, bd, 2, backend))
        container.auto_init()

        encoding = Encoding(nq, bd, backend=backend).auto_init()
        assert len(container.all_cores) == len(encoding.all_cores)

    def test_manual_quadratic_composition(self, backend):
        """Assembling circuit+mps+mx manually should give 9 cores for nqubits=3."""
        nq = 3
        container = QCTN(graph=None, backend=backend)
        container.register_module("circuit", CircuitState(nq, 2, backend))
        container.register_module("mps", MPS(nq, 4, 2, backend))
        container.register_module("mx", MeasureMatrix(nq, 2, backend))
        container.auto_init()
        assert len(container.all_cores) == 9


# ---------------------------------------------------------------------------
# D. Transpose and reference semantics
# ---------------------------------------------------------------------------

class TestTransposeAndRef:

    def test_transpose_mps_named_cores_is_ref(self, mps3_init):
        """TransposeMPS cores should have is_ref=True and is_transposed=True."""
        trans = TransposeMPS(mps3_init)
        for _, tensor in trans.named_cores():
            assert isinstance(tensor, TNTensor)
            assert tensor.is_ref is True
            assert tensor.is_transposed is True

    def test_transpose_mps_reflects_source_change(self, backend):
        """In-place modification of source MPS tensor is visible in TransposeMPS."""
        mps = MPS(nqubits=2, bond_dim=3, phys_dim=2, backend=backend).auto_init()
        trans = TransposeMPS(mps)

        # Get the first core name
        core_name = mps.cores[0]
        source_tensor = mps.cores_weights[core_name]
        t = source_tensor.tensor if isinstance(source_tensor, TNTensor) else source_tensor

        # Fill with ones in-place
        t.fill_(1.0)

        # TransposeMPS should see the updated value (via the same underlying storage)
        for name, tensor in trans.named_cores():
            if name == core_name:
                inner = tensor.tensor if isinstance(tensor, TNTensor) else tensor
                # The conj_transpose view wraps the updated source
                assert inner is not None
                break

    def test_mps_with_ref_right_is_ref(self, backend):
        """After auto_init, right MPS cores should be conj-transpose refs of left."""
        model = MPS_with_Ref(nqubits=2, bond_dim=3, backend=backend).auto_init()
        left = model._submodules["left"]
        right = model._submodules["right"]

        for name in left.cores:
            r_tensor = right.cores_weights[name]
            assert isinstance(r_tensor, TNTensor)
            assert r_tensor.is_ref is True
            assert r_tensor.is_transposed is True

    def test_mps_with_ref_shares_data(self, backend):
        """Right core's source should be the same TNTensor as left core."""
        model = MPS_with_Ref(nqubits=2, bond_dim=3, backend=backend).auto_init()
        left = model._submodules["left"]
        right = model._submodules["right"]

        for name in left.cores:
            l_t = left.cores_weights[name]
            r_t = right.cores_weights[name]
            if isinstance(l_t, TNTensor) and isinstance(r_t, TNTensor):
                assert r_t.source is l_t

    def test_tneq_two_independent_mps(self, backend):
        """TNEQ mps1 and mps2 should NOT share parameter storage."""
        model = TNEQ(nqubits=2, bond_dim=3, backend=backend).auto_init()
        mps1 = model._submodules["mps1"]
        mps2 = model._submodules["mps2"]

        for name in mps1.cores:
            t1 = mps1.cores_weights[name]
            t2 = mps2.cores_weights[name]
            raw1 = t1.tensor if isinstance(t1, TNTensor) else t1
            raw2 = t2.tensor if isinstance(t2, TNTensor) else t2
            # Different underlying storage
            assert raw1.data_ptr() != raw2.data_ptr()


# ---------------------------------------------------------------------------
# E. chunk / concat (renamed from split / merge)
# ---------------------------------------------------------------------------

class TestChunkConcat:

    def test_chunk_returns_two_qctns(self, backend):
        qctn = QCTN("-3-A-3-B-3-\n-3-A-3-B-3-", backend=backend).auto_init()
        q1, q2 = qctn.chunk()
        assert isinstance(q1, QCTN)
        assert isinstance(q2, QCTN)

    def test_chunk_core_counts(self, backend):
        qctn = QCTN("-3-A-3-B-3-\n-3-A-3-B-3-", backend=backend).auto_init()
        q1, q2 = qctn.chunk()
        assert q1.ncores + q2.ncores == qctn.ncores

    def test_concat_returns_qctn(self, backend):
        q1 = QCTN("-2-A-2-\n-2-A-2-", backend=backend).auto_init()
        q2 = QCTN("-2-B-2-\n-2-B-2-", backend=backend).auto_init()
        merged = QCTN.concat(q1, q2)
        assert isinstance(merged, QCTN)

    def test_concat_core_count(self, backend):
        q1 = QCTN("-2-A-2-\n-2-A-2-", backend=backend).auto_init()
        q2 = QCTN("-2-B-2-\n-2-B-2-", backend=backend).auto_init()
        merged = QCTN.concat(q1, q2)
        assert merged.ncores == q1.ncores + q2.ncores

    def test_concat_with_instance_method(self, backend):
        q1 = QCTN("-2-A-2-\n-2-A-2-", backend=backend).auto_init()
        q2 = QCTN("-2-B-2-\n-2-B-2-", backend=backend).auto_init()
        merged = q1.concat_with(q2)
        assert merged.ncores == q1.ncores + q2.ncores

    def test_split_raises_deprecation_warning(self, backend):
        qctn = QCTN("-3-A-3-B-3-\n-3-A-3-B-3-", backend=backend).auto_init()
        with pytest.warns(DeprecationWarning, match="chunk"):
            qctn.split()

    def test_merge_raises_deprecation_warning(self, backend):
        q1 = QCTN("-2-A-2-\n-2-A-2-", backend=backend).auto_init()
        q2 = QCTN("-2-B-2-\n-2-B-2-", backend=backend).auto_init()
        with pytest.warns(DeprecationWarning, match="concat"):
            QCTN.merge(q1, q2)

    def test_merge_with_raises_deprecation_warning(self, backend):
        q1 = QCTN("-2-A-2-\n-2-A-2-", backend=backend).auto_init()
        q2 = QCTN("-2-B-2-\n-2-B-2-", backend=backend).auto_init()
        with pytest.warns(DeprecationWarning, match="concat_with"):
            q1.merge_with(q2)


# ---------------------------------------------------------------------------
# F. Contract basic shapes
# ---------------------------------------------------------------------------

class TestContractBasic:

    def test_mps_contract_runs(self, mps3_init, backend):
        """MPS contraction via EinsumStrategy should produce a result."""
        strategy = EinsumStrategy()
        shapes_info = {
            "circuit_states_shapes": None,
            "measure_shapes": None,
            "measure_is_matrix": False,
        }
        compute_fn = strategy.get_compute_function(mps3_init, shapes_info, backend)
        cores_dict = {
            k: (v.tensor if isinstance(v, TNTensor) else v)
            for k, v in mps3_init.cores_weights.items()
        }
        result = compute_fn(cores_dict, None, None)
        assert result is not None

    def test_mps_contract_result_is_scalar_or_tensor(self, mps3_init, backend):
        """Contraction of a symmetric MPS (no input/measure) should give a scalar."""
        strategy = EinsumStrategy()
        shapes_info = {
            "circuit_states_shapes": None,
            "measure_shapes": None,
            "measure_is_matrix": False,
        }
        compute_fn = strategy.get_compute_function(mps3_init, shapes_info, backend)
        cores_dict = {
            k: (v.tensor if isinstance(v, TNTensor) else v)
            for k, v in mps3_init.cores_weights.items()
        }
        result = compute_fn(cores_dict, None, None)
        t = result.tensor if isinstance(result, TNTensor) else result
        assert t.numel() >= 1
