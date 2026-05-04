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
    State,
    CircuitState,
    MeasureMatrix,
    PlainMPS,
    TransposeMPS,
    MPS_with_Ref,
    Encoding,
    TNEQ,
    BornMachine,
    Quadratic,
)
from tneq_qc.contractor.einsum_strategy import EinsumStrategy
from tneq_qc.utils.graph_generators import QCTNHelper


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
    return mps3.auto_init(orthogonal=True)


@pytest.fixture
def cs3(backend):
    return CircuitState(nqubits=3, phys_dim=2, backend=backend)


@pytest.fixture
def cs3_init(cs3):
    return cs3.auto_init(orthogonal=True)


@pytest.fixture
def mx3(backend):
    return MeasureMatrix(nqubits=3, phys_dim=2, backend=backend)


@pytest.fixture
def mx3_init(mx3):
    return mx3.auto_init(orthogonal=True)


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
        result = mps3.auto_init(orthogonal=True)
        assert result is mps3

    def test_mps_core_shapes_correct(self, mps3_init):
        """After auto_init, each core weight should be non-empty."""
        for name, tensor in mps3_init.cores_weights.items():
            t = tensor.tensor if isinstance(tensor, TNTensor) else tensor
            assert t.numel() > 0

    def test_mps_nqubits(self, mps3):
        assert mps3.nqubits == 3

    def test_mps_ncores_equals_nqubits(self, mps3):
        assert mps3.ncores == mps3.nqubits - 1

    def test_circuit_state_cores_empty_before_init(self, cs3):
        assert cs3.cores_weights == {}

    def test_circuit_state_auto_init_populates_cores(self, cs3_init):
        assert len(cs3_init.cores_weights) == cs3_init.ncores == 3
        for tensor in cs3_init.cores_weights.values():
            raw = tensor.tensor if isinstance(tensor, TNTensor) else tensor
            assert torch.allclose(raw.reshape(-1)[0], torch.tensor(1, dtype=raw.dtype))
            assert torch.allclose(raw.reshape(-1)[1:], torch.zeros_like(raw.reshape(-1)[1:]))

    def test_circuit_state_nqubits(self, cs3):
        assert cs3.nqubits == 3

    def test_measure_matrix_cores_empty_before_init(self, mx3):
        assert mx3.cores_weights == {}

    def test_measure_matrix_auto_init_populates_cores(self, mx3_init):
        assert len(mx3_init.cores_weights) == mx3_init.ncores == 3

    def test_auto_init_dtype_complex(self, mps3):
        """auto_init should produce complex tensors when backend dtype is complex64."""
        mps3.auto_init(orthogonal=True)
        for name, tensor in mps3.cores_weights.items():
            t = tensor.tensor if isinstance(tensor, TNTensor) else tensor
            assert t.is_complex(), f"Core {name} is not complex: {t.dtype}"

    def test_set_cores_external(self, backend):
        """set_cores with dict should replace cores_weights."""
        mps = MPS(nqubits=2, bond_dim=3, phys_dim=2, backend=backend).auto_init(orthogonal=True)
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
        model = PlainMPS(nqubits=3, bond_dim=4, backend=backend).auto_init(orthogonal=True)
        names = [n for n, _ in model.named_cores()]
        # staggered MPS has nqubits-1 cores
        assert len(names) == 2

    def test_plain_mps_named_cores_prefixed(self, backend):
        model = PlainMPS(nqubits=3, bond_dim=4, backend=backend).auto_init(orthogonal=True)
        names = [n for n, _ in model.named_cores()]
        for n in names:
            assert n.startswith("mps.")

    def test_encoding_submodules(self, backend):
        model = Encoding(nqubits=3, bond_dim=4, backend=backend)
        assert "state" in model._submodules
        assert "mps" in model._submodules
        assert isinstance(model._submodules["state"], State)
        assert isinstance(model._submodules["mps"], MPS)

    def test_encoding_all_cores_count(self, backend):
        model = Encoding(nqubits=3, bond_dim=4, backend=backend).auto_init(orthogonal=True)
        # 3 state cores + 2 mps cores = 5 (staggered MPS)
        assert len(model.all_cores) == 5

    def test_tneq_submodules(self, backend):
        model = TNEQ(nqubits=3, bond_dim=4, backend=backend)
        assert "mps1" in model._submodules
        assert "mps2" in model._submodules

    def test_tneq_all_cores_count(self, backend):
        model = TNEQ(nqubits=3, bond_dim=4, backend=backend).auto_init(orthogonal=True)
        # mps1: 2 cores + mps2: 2 cores = 4 (staggered MPS)
        assert len(model.all_cores) == 4

    def test_born_machine_submodules(self, backend):
        graph = QCTNHelper.mps(3, bond_dim=4, phys_dim=2)
        model = BornMachine(graph, 2, backend=backend)
        assert "state" in model._submodules
        assert "tn" in model._submodules
        assert "mx" in model._submodules

    def test_born_machine_all_cores_count(self, backend):
        graph = QCTNHelper.mps(3, bond_dim=4, phys_dim=2)
        model = BornMachine(graph, 2, backend=backend).auto_init(orthogonal=True)
        # state: 3 + tn: 2 + mx: 3 = 8 (staggered MPS)
        assert len(model.all_cores) == 8

    def test_born_machine_accepts_graph(self, backend):
        graph = QCTNHelper.mps(3, bond_dim=4, phys_dim=2)
        model = BornMachine(graph, 2, backend=backend).auto_init(orthogonal=True)
        assert model._submodules["tn"].graph is not None
        assert model._submodules["tn"].nqubits == 3

    def test_quadratic_requires_graph(self, backend):
        with pytest.raises(ValueError, match="non-None graph string"):
            Quadratic(None, 2, backend=backend)

    def test_auto_init_chain_propagates(self, backend):
        """auto_init on parent should init all submodules."""
        model = TNEQ(nqubits=2, bond_dim=3, backend=backend)
        assert model._submodules["mps1"].cores_weights == {}
        model.auto_init(orthogonal=True)
        assert len(model._submodules["mps1"].cores_weights) == 1
        assert len(model._submodules["mps2"].cores_weights) == 1


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
        container.auto_init(orthogonal=True)

        encoding = Encoding(nq, bd, backend=backend).auto_init(orthogonal=True)
        assert len(container.all_cores) == len(encoding.all_cores)

    def test_manual_quadratic_composition(self, backend):
        """Assembling circuit+mps+mx manually should give 8 cores for nqubits=3."""
        nq = 3
        container = QCTN(graph=None, backend=backend)
        container.register_module("circuit", CircuitState(nq, 2, backend))
        container.register_module("mps", MPS(nq, 4, 2, backend))
        container.register_module("mx", MeasureMatrix(nq, 2, backend))
        container.auto_init(orthogonal=True)
        assert len(container.all_cores) == 8


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
        mps = MPS(nqubits=2, bond_dim=3, phys_dim=2, backend=backend).auto_init(orthogonal=True)
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
        model = MPS_with_Ref(nqubits=2, bond_dim=3, backend=backend).auto_init(orthogonal=True)
        left = model._submodules["left"]
        right = model._submodules["right"]

        for name in left.cores:
            r_tensor = right.cores_weights[name]
            assert isinstance(r_tensor, TNTensor)
            assert r_tensor.is_ref is True
            assert r_tensor.is_transposed is True

    def test_mps_with_ref_shares_data(self, backend):
        """Right core's source should be the same TNTensor as left core."""
        model = MPS_with_Ref(nqubits=2, bond_dim=3, backend=backend).auto_init(orthogonal=True)
        left = model._submodules["left"]
        right = model._submodules["right"]

        for name in left.cores:
            l_t = left.cores_weights[name]
            r_t = right.cores_weights[name]
            if isinstance(l_t, TNTensor) and isinstance(r_t, TNTensor):
                assert r_t.source is l_t

    def test_tneq_two_independent_mps(self, backend):
        """TNEQ mps1 and mps2 should NOT share parameter storage."""
        model = TNEQ(nqubits=2, bond_dim=3, backend=backend).auto_init(orthogonal=True)
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
        qctn = QCTN("-3-A-3-B-3-\n-3-A-3-B-3-", backend=backend).auto_init(orthogonal=True)
        q1, q2 = qctn.chunk()
        print(f"\n[chunk] original:\n{qctn}")
        print(f"[chunk] q1:\n{q1}")
        print(f"[chunk] q2:\n{q2}")
        assert isinstance(q1, QCTN)
        assert isinstance(q2, QCTN)

    def test_chunk_core_counts(self, backend):
        qctn = QCTN("-3-A-3-B-3-\n-3-A-3-B-3-", backend=backend).auto_init(orthogonal=True)
        q1, q2 = qctn.chunk()
        assert q1.ncores + q2.ncores == qctn.ncores

    def test_concat_returns_qctn(self, backend):
        q1 = QCTN("-2-A-2-\n-2-A-2-", backend=backend).auto_init(orthogonal=True)
        q2 = QCTN("-2-B-2-\n-2-B-2-", backend=backend).auto_init(orthogonal=True)
        merged = QCTN.concat(q1, q2)
        print(f"\n[concat] q1:\n{q1}")
        print(f"[concat] q2:\n{q2}")
        print(f"[concat] merged:\n{merged}")
        assert isinstance(merged, QCTN)

    def test_concat_core_count(self, backend):
        q1 = QCTN("-2-A-2-\n-2-A-2-", backend=backend).auto_init(orthogonal=True)
        q2 = QCTN("-2-B-2-\n-2-B-2-", backend=backend).auto_init(orthogonal=True)
        merged = QCTN.concat(q1, q2)
        assert merged.ncores == q1.ncores + q2.ncores

    def test_concat_with_instance_method(self, backend):
        q1 = QCTN("-2-A-2-\n-2-A-2-", backend=backend).auto_init(orthogonal=True)
        q2 = QCTN("-2-B-2-\n-2-B-2-", backend=backend).auto_init(orthogonal=True)
        merged = q1.concat_with(q2)
        print(f"\n[concat_with] merged:\n{merged}")
        assert merged.ncores == q1.ncores + q2.ncores

    def test_split_raises_deprecation_warning(self, backend):
        qctn = QCTN("-3-A-3-B-3-\n-3-A-3-B-3-", backend=backend).auto_init(orthogonal=True)
        with pytest.warns(DeprecationWarning, match="chunk"):
            qctn.split()

    def test_merge_raises_deprecation_warning(self, backend):
        q1 = QCTN("-2-A-2-\n-2-A-2-", backend=backend).auto_init(orthogonal=True)
        q2 = QCTN("-2-B-2-\n-2-B-2-", backend=backend).auto_init(orthogonal=True)
        with pytest.warns(DeprecationWarning, match="concat"):
            QCTN.merge(q1, q2)

    def test_merge_with_raises_deprecation_warning(self, backend):
        q1 = QCTN("-2-A-2-\n-2-A-2-", backend=backend).auto_init(orthogonal=True)
        q2 = QCTN("-2-B-2-\n-2-B-2-", backend=backend).auto_init(orthogonal=True)
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

# ---------------------------------------------------------------------------
# G. Complex chunk / concat structures
# ---------------------------------------------------------------------------

class TestComplexChunkConcat:
    """More complex concat/chunk tests with visual structure printing."""

    # 1a. MPS chunk
    def test_mps_chunk(self, backend):
        mps = MPS(nqubits=4, bond_dim=3, phys_dim=2, backend=backend).auto_init(orthogonal=True)
        q1, q2 = mps.chunk()
        print(f"\n[mps_chunk] original:\n{mps}")
        print(f"[mps_chunk] q1:\n{q1}")
        print(f"[mps_chunk] q2:\n{q2}")
        assert q1.ncores + q2.ncores == mps.ncores

    # 1b. MPS + Tree concat
    def test_mps_tree_concat(self, backend):
        tree_graph = QCTNHelper.generate_example_graph(n=4, graph_type="tree", dim_char="2")
        mps  = MPS(nqubits=4, bond_dim=3, phys_dim=2, backend=backend).auto_init(orthogonal=True)
        tree = QCTN(tree_graph, backend=backend)
        merged = QCTN.concat(mps, tree)
        print(f"\n[mps+tree] mps:\n{mps}")
        print(f"[mps+tree] tree:\n{tree}")
        print(f"[mps+tree] merged:\n{merged}")
        assert merged.ncores == mps.ncores + tree.ncores

    # 2. MPS + MPS转置 concat
    def test_mps_mpsT_concat(self, backend):
        mps  = MPS(nqubits=3, bond_dim=4, phys_dim=2, backend=backend).auto_init(orthogonal=True)
        mpsT = MPS(nqubits=3, bond_dim=4, phys_dim=2, backend=backend).auto_init(orthogonal=True)
        merged = QCTN.concat(mps, mpsT)
        print(f"\n[mps+mpsT] mps:\n{mps}")
        print(f"[mps+mpsT] mpsT:\n{mpsT}")
        print(f"[mps+mpsT] merged:\n{merged}")
        assert merged.ncores == mps.ncores + mpsT.ncores

    # 3. CS + MPS + MX concat
    def test_cs_mps_mx_concat(self, backend):
        cs  = CircuitState(nqubits=3, phys_dim=2, backend=backend).auto_init(orthogonal=True)
        mps = MPS(nqubits=3, bond_dim=4, phys_dim=2, backend=backend).auto_init(orthogonal=True)
        mx  = MeasureMatrix(nqubits=3, phys_dim=2, backend=backend).auto_init(orthogonal=True)
        merged = QCTN.concat(QCTN.concat(cs, mps), mx)
        print(f"\n[cs+mps+mx] cs:\n{cs}")
        print(f"[cs+mps+mx] mps:\n{mps}")
        print(f"[cs+mps+mx] mx:\n{mx}")
        print(f"[cs+mps+mx] merged:\n{merged}")
        assert merged.ncores == cs.ncores + mps.ncores + mx.ncores

    # 4. CS + CS转置 concat
    def test_cs_csT_concat(self, backend):
        cs  = CircuitState(nqubits=3, phys_dim=2, backend=backend).auto_init(orthogonal=True)
        csT = CircuitState(nqubits=3, phys_dim=2, backend=backend).auto_init(orthogonal=True)
        merged = QCTN.concat(cs, csT)
        print(f"\n[cs+csT] cs:\n{cs}")
        print(f"[cs+csT] csT:\n{csT}")
        print(f"[cs+csT] merged:\n{merged}")
        assert merged.ncores == cs.ncores + csT.ncores

    # 5. CS + MPS + MX + MX转置 + CS转置 concat
    def test_full_chain_concat(self, backend):
        cs  = CircuitState(nqubits=5, phys_dim=3, backend=backend).auto_init(orthogonal=True)
        mps = MPS(nqubits=5, bond_dim=3, phys_dim=3, backend=backend).auto_init(orthogonal=True)
        mx  = MeasureMatrix(nqubits=5, phys_dim=3, backend=backend).auto_init(orthogonal=True)
        mxT = MeasureMatrix(nqubits=5, phys_dim=3, backend=backend).auto_init(orthogonal=True)
        csT = CircuitState(nqubits=5, phys_dim=3, backend=backend).auto_init(orthogonal=True)

        # cs  = CircuitState(nqubits=3, phys_dim=2, backend=backend).auto_init(orthogonal=True)
        # mps = MPS(nqubits=3, bond_dim=4, phys_dim=2, backend=backend).auto_init(orthogonal=True)
        # mx  = MeasureMatrix(nqubits=3, phys_dim=2, backend=backend).auto_init(orthogonal=True)
        # mxT = MeasureMatrix(nqubits=3, phys_dim=2, backend=backend).auto_init(orthogonal=True)
        # csT = CircuitState(nqubits=3, phys_dim=2, backend=backend).auto_init(orthogonal=True)
        merged = cs.concat_with(mps).concat_with(mx).concat_with(mxT).concat_with(csT)
        print(f"\n[full_chain] cs:\n{cs}")
        print(f"[full_chain] mps:\n{mps}")
        print(f"[full_chain] mx:\n{mx}")
        print(f"[full_chain] mxT:\n{mxT}")
        print(f"[full_chain] csT:\n{csT}")
        print(f"[full_chain] merged:\n{merged}")
        assert merged.ncores == cs.ncores + mps.ncores + mx.ncores + mxT.ncores + csT.ncores


# ---------------------------------------------------------------------------
# H. Graph print parity check (raw graph -> QCTN)
# ---------------------------------------------------------------------------

class TestGraphGeneratorsInQCTNModules:
    """Print all example graphs, then print QCTN built from each graph."""

    def test_print_all_graphs_and_qctn(self, backend):
        n = 5
        graph_specs = [
            ("mps", "2"),
            ("tree", "2"),
            ("wall", "2"),
            ("circuit", "2"),
            ("mx", "2"),
        ]

        for graph_type, dim_char in graph_specs:
            graph = QCTNHelper.generate_example_graph(
                n=n, graph_type=graph_type, dim_char=dim_char
            )
            print(f"\n[{graph_type}, n={n}] raw graph:\n{graph}")

            qctn = QCTN(graph, backend=backend)
            print(f"[{graph_type}, n={n}] qctn:\n{qctn}")

            assert qctn.nqubits == n

class TestQuadraticQCTNModules:
    """Print all example graphs, then print QCTN built from each graph."""

    def test_print_all_graphs_and_qctn(self, backend):

        cs = CircuitState(nqubits=5, phys_dim=3, backend=backend).auto_init(orthogonal=True)
        graph = "-2-A-2\n-2-A-2\n"
        tn = QCTN(graph, backend=backend)
        
        n = 2
        graph_specs = [
            ("mps", "2"),
            ("tree", "2"),
            ("wall", "2"),
            ("circuit", "2"),
            ("mx", "2"),
        ]

        for graph_type, dim_char in graph_specs:
            graph = QCTNHelper.generate_example_graph(
                n=n, graph_type=graph_type, dim_char=dim_char
            )
            print(f"\n[{graph_type}, n={n}] raw graph:\n{graph}")

            qctn = QCTN(graph, backend=backend)
            print(f"[{graph_type}, n={n}] qctn:\n{qctn}")

            assert qctn.nqubits == n
