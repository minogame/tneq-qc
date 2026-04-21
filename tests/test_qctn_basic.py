"""
Phase 2 unit tests for QCTN base-class features.

Tests cover:
  - QCTN.from_graph() classmethod
  - QCTN._submodules + register_module()
  - QCTN.named_cores() and QCTN.all_cores
  - QCTN.get_einsum_info()   (R5: graph-parsing separation)
  - QCTN.build_core_list()   (R5: tensor assembly)
  - QCTN.conjugate_transpose_cores()  (R2: reference semantics)
  - EinsumStrategy.get_compute_function() delegation to QCTN methods
"""

from __future__ import annotations

import pytest
import torch

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.contractor.einsum_strategy import EinsumStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def backend():
    return BackendFactory.create_backend("pytorch", device="cpu")


@pytest.fixture
def simple_qctn(backend):
    """2-qubit, 1-core QCTN: A on both qubits (input_dim=output_dim=4)."""
    return QCTN.from_graph("-2-A-2-\n-2-A-2-", backend=backend)


@pytest.fixture
def two_core_qctn(backend):
    """4-qubit, 2-core QCTN: A on qubits 0,1 and B on qubits 2,3."""
    graph = "-2-A-2-\n-2-A-2-\n-2-B-2-\n-2-B-2-"
    return QCTN.from_graph(graph, backend=backend)


# ---------------------------------------------------------------------------
# TestFromGraph
# ---------------------------------------------------------------------------

class TestFromGraph:
    def test_returns_qctn_instance(self, backend):
        qctn = QCTN.from_graph("-2-A-2-\n-2-A-2-", backend=backend)
        assert isinstance(qctn, QCTN)

    def test_from_graph_equals_direct_init(self, backend):
        graph = "-2-A-2-\n-2-A-2-"
        q1 = QCTN(graph, backend=backend)
        q2 = QCTN.from_graph(graph, backend=backend)
        assert set(q1.cores) == set(q2.cores)
        assert q1.nqubits == q2.nqubits

    def test_core_names_detected(self, backend):
        # A and B each appear on a single qubit → square 2×2 matrices → init succeeds
        qctn = QCTN.from_graph("-2-A-2-\n-2-B-2-", backend=backend)
        assert set(qctn.cores) == {"A", "B"}

    def test_backend_stored(self, backend):
        qctn = QCTN.from_graph("-2-A-2-\n-2-A-2-", backend=backend)
        assert qctn.backend is backend


# ---------------------------------------------------------------------------
# TestSubmodules
# ---------------------------------------------------------------------------

class TestSubmodules:
    def test_submodules_dict_initialized(self, simple_qctn):
        assert isinstance(simple_qctn._submodules, dict)
        assert len(simple_qctn._submodules) == 0

    def test_register_module_stores_submodule(self, simple_qctn, backend):
        sub = QCTN.from_graph("-2-C-2-\n-2-C-2-", backend=backend)
        simple_qctn.register_module("branch", sub)
        assert "branch" in simple_qctn._submodules
        assert simple_qctn._submodules["branch"] is sub

    def test_register_module_type_error(self, simple_qctn):
        with pytest.raises(TypeError):
            simple_qctn.register_module("bad", "not_a_qctn")

    def test_register_module_empty_name_error(self, simple_qctn, backend):
        sub = QCTN.from_graph("-2-C-2-\n-2-C-2-", backend=backend)
        with pytest.raises(ValueError):
            simple_qctn.register_module("", sub)

    def test_register_module_dotted_name_error(self, simple_qctn, backend):
        sub = QCTN.from_graph("-2-C-2-\n-2-C-2-", backend=backend)
        with pytest.raises(ValueError):
            simple_qctn.register_module("a.b", sub)


# ---------------------------------------------------------------------------
# TestNamedCores
# ---------------------------------------------------------------------------

class TestNamedCores:
    def test_own_cores_yielded(self, two_core_qctn):
        names = [name for name, _ in two_core_qctn.named_cores()]
        assert set(names) == set(two_core_qctn.cores)

    def test_own_cores_tensors_correct(self, two_core_qctn):
        for name, tensor in two_core_qctn.named_cores():
            expected = two_core_qctn.cores_weights[name]
            if isinstance(expected, TNTensor):
                assert tensor is expected
            else:
                assert tensor is expected

    def test_submodule_cores_prefixed(self, backend):
        parent = QCTN.from_graph("-2-A-2-\n-2-A-2-", backend=backend)
        child = QCTN.from_graph("-2-B-2-\n-2-B-2-", backend=backend)
        parent.register_module("child", child)

        names = [n for n, _ in parent.named_cores()]
        assert "A" in names
        assert "child.B" in names

    def test_nested_prefix_propagates(self, backend):
        root = QCTN.from_graph("-2-A-2-\n-2-A-2-", backend=backend)
        mid = QCTN.from_graph("-2-B-2-\n-2-B-2-", backend=backend)
        leaf = QCTN.from_graph("-2-C-2-\n-2-C-2-", backend=backend)
        mid.register_module("leaf", leaf)
        root.register_module("mid", mid)

        names = [n for n, _ in root.named_cores()]
        assert "A" in names
        assert "mid.B" in names
        assert "mid.leaf.C" in names


# ---------------------------------------------------------------------------
# TestAllCores
# ---------------------------------------------------------------------------

class TestAllCores:
    def test_returns_dict(self, simple_qctn):
        result = simple_qctn.all_cores
        assert isinstance(result, dict)

    def test_own_cores_present(self, two_core_qctn):
        ac = two_core_qctn.all_cores
        for name in two_core_qctn.cores:
            assert name in ac

    def test_submodule_cores_present(self, backend):
        parent = QCTN.from_graph("-2-A-2-\n-2-A-2-", backend=backend)
        child = QCTN.from_graph("-2-B-2-\n-2-B-2-", backend=backend)
        parent.register_module("child", child)

        ac = parent.all_cores
        assert "A" in ac
        assert "child.B" in ac


# ---------------------------------------------------------------------------
# TestGetEinsumInfo
# ---------------------------------------------------------------------------

class TestGetEinsumInfo:
    def test_returns_tuple(self, simple_qctn):
        result = simple_qctn.get_einsum_info()
        assert isinstance(result, tuple) and len(result) == 2

    def test_no_states_no_measure(self, simple_qctn):
        eq, shapes = simple_qctn.get_einsum_info(None, None)
        assert isinstance(eq, str)
        assert "->" in eq
        # Only core shapes: 2 (left) + 2 (right)
        assert len(shapes) == 2 * simple_qctn.ncores

    def test_with_states_no_measure(self, simple_qctn):
        # simple_qctn: 2 qubits, 1 core → 2 input symbols, 2 output symbols
        states_shapes = ((2,), (2,))  # one vector per qubit
        eq, shapes = simple_qctn.get_einsum_info(states_shapes, None)
        n_cores = simple_qctn.ncores
        # states × 2 + left_cores + right_cores
        assert len(shapes) == 2 * n_cores + 2 * len(states_shapes)

    def test_with_states_and_measure(self, simple_qctn):
        # simple_qctn: 2 qubits → need 2 per-qubit measure matrices
        states_shapes = ((2,), (2,))
        measure_shapes = ((5, 2, 2), (5, 2, 2))  # one Mx per qubit
        eq, shapes = simple_qctn.get_einsum_info(states_shapes, measure_shapes)
        n_cores = simple_qctn.ncores
        n_states = len(states_shapes)
        n_mx = len(measure_shapes)
        expected = 2 * n_states + 2 * n_cores + n_mx
        assert len(shapes) == expected

    def test_matches_legacy_static_method(self, simple_qctn):
        """get_einsum_info() must produce the same result as the legacy static method."""
        states_shapes = ((2,), (2,))
        measure_shapes = ((5, 2, 2), (5, 2, 2))

        eq_new, shapes_new = simple_qctn.get_einsum_info(states_shapes, measure_shapes)
        eq_old, shapes_old = EinsumStrategy._build_with_self_expression_legacy(
            simple_qctn, states_shapes, measure_shapes
        )
        assert eq_new == eq_old
        assert shapes_new == shapes_old

    def test_two_core_qctn(self, two_core_qctn):
        # two_core_qctn: 4 qubits → 4 per-qubit states + 4 per-qubit measures
        states_shapes = ((2,), (2,), (2,), (2,))
        measure_shapes = ((5, 2, 2), (5, 2, 2), (5, 2, 2), (5, 2, 2))
        eq, shapes = two_core_qctn.get_einsum_info(states_shapes, measure_shapes)
        assert isinstance(eq, str)
        assert "->" in eq


# ---------------------------------------------------------------------------
# TestBuildCoreList
# ---------------------------------------------------------------------------

class TestBuildCoreList:
    def test_no_states_no_measure(self, simple_qctn):
        cores_dict = {k: (v.tensor if isinstance(v, TNTensor) else v)
                      for k, v in simple_qctn.cores_weights.items()}
        tensors = simple_qctn.build_core_list(cores_dict)
        # only left + right cores
        assert len(tensors) == 2 * simple_qctn.ncores

    def test_with_states_and_measure(self, simple_qctn):
        cores_dict = {k: (v.tensor if isinstance(v, TNTensor) else v)
                      for k, v in simple_qctn.cores_weights.items()}
        states = [torch.ones(2), torch.ones(2)]
        # 2 per-qubit measure matrices for 2-qubit circuit
        measure = [torch.eye(2).unsqueeze(0), torch.eye(2).unsqueeze(0)]
        tensors = simple_qctn.build_core_list(cores_dict, states, measure)
        n = simple_qctn.ncores
        # states + left + measure + right + states
        assert len(tensors) == 2 * len(states) + 2 * n + len(measure)

    def test_defaults_to_cores_weights(self, simple_qctn):
        """Passing cores_dict=None should use self.cores_weights."""
        tensors_default = simple_qctn.build_core_list(None)
        tensors_explicit = simple_qctn.build_core_list(simple_qctn.cores_weights)
        assert len(tensors_default) == len(tensors_explicit)
        for a, b in zip(tensors_default, tensors_explicit):
            assert a is b

    def test_left_and_right_same_tensor(self, simple_qctn):
        """Left cores and reversed right cores should be the same objects (shared params)."""
        cores_dict = dict(simple_qctn.cores_weights)
        tensors = simple_qctn.build_core_list(cores_dict)
        n = simple_qctn.ncores
        left = tensors[:n]
        right = tensors[n:][::-1]
        for l, r in zip(left, right):
            assert l is r

    def test_list_matches_get_einsum_info_count(self, simple_qctn):
        # 2-qubit circuit → 2 per-qubit measure matrices
        states_shapes = ((2,), (2,))
        measure_shapes = ((5, 2, 2), (5, 2, 2))
        _, shapes = simple_qctn.get_einsum_info(states_shapes, measure_shapes)

        states = [torch.ones(2), torch.ones(2)]
        measure = [torch.zeros(5, 2, 2), torch.zeros(5, 2, 2)]
        cores_dict = {k: (v.tensor if isinstance(v, TNTensor) else v)
                      for k, v in simple_qctn.cores_weights.items()}
        tensors = simple_qctn.build_core_list(cores_dict, states, measure)
        assert len(tensors) == len(shapes)


# ---------------------------------------------------------------------------
# TestConjugateTransposeCores
# ---------------------------------------------------------------------------

class TestConjugateTransposeCores:
    def test_returns_dict_same_keys(self, simple_qctn):
        ct = simple_qctn.conjugate_transpose_cores()
        assert set(ct.keys()) == set(simple_qctn.cores)

    def test_all_values_are_tntensor(self, simple_qctn):
        ct = simple_qctn.conjugate_transpose_cores()
        for v in ct.values():
            assert isinstance(v, TNTensor)

    def test_is_ref_flag(self, simple_qctn):
        ct = simple_qctn.conjugate_transpose_cores()
        for v in ct.values():
            assert v.is_ref is True

    def test_is_transposed_flag(self, simple_qctn):
        ct = simple_qctn.conjugate_transpose_cores()
        for v in ct.values():
            assert v.is_transposed is True

    def test_scale_preserved(self, simple_qctn):
        original = simple_qctn.cores_weights
        ct = simple_qctn.conjugate_transpose_cores()
        for name in simple_qctn.cores:
            orig_t = original[name]
            orig_scale = orig_t.scale if isinstance(orig_t, TNTensor) else 1.0
            assert abs(ct[name].scale - orig_scale) < 1e-9

    def test_raw_tensor_input_wrapped(self, backend):
        """Works even when cores_weights contains raw (non-TNTensor) tensors."""
        qctn = QCTN.from_graph("-2-A-2-\n-2-A-2-", backend=backend)
        # Replace with a raw tensor
        raw = torch.randn(2, 2, 2, 2)
        qctn.cores_weights["A"] = raw
        ct = qctn.conjugate_transpose_cores()
        assert isinstance(ct["A"], TNTensor)
        assert ct["A"].is_transposed is True


# ---------------------------------------------------------------------------
# TestEinsumStrategyDelegation
# ---------------------------------------------------------------------------

class TestEinsumStrategyDelegation:
    """Verify EinsumStrategy now delegates to QCTN methods."""

    def test_build_with_self_expression_delegates(self, simple_qctn):
        """EinsumStrategy.build_with_self_expression delegates to qctn.get_einsum_info."""
        states_shapes = ((2,), (2,))
        measure_shapes = ((5, 2, 2), (5, 2, 2))  # 2 per-qubit matrices
        eq_via_strategy, shapes_via_strategy = EinsumStrategy.build_with_self_expression(
            simple_qctn, states_shapes, measure_shapes
        )
        eq_direct, shapes_direct = simple_qctn.get_einsum_info(states_shapes, measure_shapes)
        assert eq_via_strategy == eq_direct
        assert shapes_via_strategy == shapes_direct

    def test_compute_function_runs(self, simple_qctn, backend):
        """EinsumStrategy.get_compute_function() produces a callable compute_fn."""
        states_shapes = ((2,), (2,))
        measure_shapes = ((1, 2, 2), (1, 2, 2))  # one per qubit
        shapes_info = {
            "circuit_states_shapes": states_shapes,
            "measure_shapes": measure_shapes,
            "measure_is_matrix": True,
        }
        strategy = EinsumStrategy()
        compute_fn = strategy.get_compute_function(simple_qctn, shapes_info, backend)
        assert callable(compute_fn)

        # Execute with dummy tensors
        cores_dict = {k: (v.tensor if isinstance(v, TNTensor) else v)
                      for k, v in simple_qctn.cores_weights.items()}
        states = [torch.ones(2), torch.ones(2)]
        measure = [torch.eye(2).unsqueeze(0), torch.eye(2).unsqueeze(0)]
        result = compute_fn(cores_dict, states, measure)
        assert result is not None


# ---------------------------------------------------------------------------
# TestBatchAndIdentity
# ---------------------------------------------------------------------------

class TestBatchAndIdentity:
    def test_empty_qubit_line_injects_fixed_identity(self, backend):
        graph = "-2-A-2-\n\n-2-B-2-"
        qctn = QCTN(graph, backend=backend)
        assert qctn.nqubits == 3
        identity_syms = [sym for sym, name in qctn.core_names.items() if name.startswith("identity.q")]
        assert len(identity_syms) == 1
        sym = identity_syms[0]
        tensor = qctn.cores_weights[sym]
        assert tensor.is_fixed is True
        assert tensor.fixed_kind == "identity"
        assert torch.allclose(tensor.tensor, torch.eye(2))

    def test_set_core_batch_size_adds_batch_dimension(self, backend):
        qctn = QCTN.from_graph("-2-A-2-\n-2-B-2-", backend=backend)
        before = {name: qctn.cores_weights[name].tensor.clone() for name in qctn.cores}
        qctn.set_core_batch_size(3)
        for name in qctn.cores:
            tensor = qctn.cores_weights[name]
            assert tensor.has_batch is True
            assert tensor.shape[0] == 3
            for i in range(3):
                assert torch.allclose(tensor.tensor[i], before[name])

    def test_set_core_batch_size_resizes_existing_batch(self, backend):
        qctn = QCTN.from_graph("-2-A-2-\n-2-B-2-", backend=backend)
        qctn.set_core_batch_size(2)
        first = {name: qctn.cores_weights[name].tensor.clone() for name in qctn.cores}
        qctn.set_core_batch_size(5)
        for name in qctn.cores:
            tensor = qctn.cores_weights[name]
            assert tensor.has_batch is True
            assert tensor.shape[0] == 5
            assert torch.allclose(tensor.tensor[0], first[name][0])
            assert torch.allclose(tensor.tensor[1], first[name][1])
            assert torch.allclose(tensor.tensor[2], first[name][0])
            assert torch.allclose(tensor.tensor[3], first[name][1])
            assert torch.allclose(tensor.tensor[4], first[name][0])
