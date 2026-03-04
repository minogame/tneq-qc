"""
Plan A — circuit states / measurement matrix numerical tests.

End-to-end numerical validation of the  A · Mx · A†  pipeline using
only Phase 1+2 APIs (QCTN + EinsumStrategy).  Zero new production code.

Mathematical properties verified
---------------------------------
F1. Identity core + any states: result_no_mx = ‖φ₀‖² · ‖φ₁‖²
F2. Orthogonal core + normalised states: result_no_mx = 1.0
F3. Orthogonal core + normalised states + Mx=I: result[0] = 1.0
F4. State scaling (one vector): φ → λφ  ⟹  result × λ²
F5. State scaling (all vectors): φᵢ→λφᵢ  ⟹  result × λ^(2·n_states)
F6. Zero state → result = 0
F7. Mx linear scaling: Mx → λMx  ⟹  result × λ
F8. Zero Mx → result = 0
F9. Mx batch: Mx shape (K, d, d) → result shape (K,)
F10. Mx additivity: result(Mx₁+Mx₂) = result(Mx₁) + result(Mx₂)
F11. No-states consistency: result_no_Mx (scalar) == result_identity_Mx[0]

Two QCTN fixtures are used throughout:
  - identity_qctn  : A = I₄ reshaped to (2,2,2,2)  — exact analytical values
  - ortho_qctn     : A = random orthogonal (from init_random_core) — invariant-based
"""
from __future__ import annotations

import pytest
import torch

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.contractor.einsum_strategy import EinsumStrategy


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

ATOL = 1e-5   # float32 arithmetic
RTOL = 1e-4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def backend():
    return BackendFactory.create_backend("pytorch", device="cpu")


@pytest.fixture
def identity_qctn(backend):
    """2-qubit 1-core QCTN with A = I₄ reshaped to (2,2,2,2).

    Analytical property:
        result_no_mx(φ₀, φ₁) = Σ_{γ,δ} |φ₀[γ]·φ₁[δ]|² = ‖φ₀‖² · ‖φ₁‖²
    """
    qctn = QCTN.from_graph("-2-A-2-\n-2-A-2-", backend=backend)
    A_id = torch.eye(4, dtype=torch.float32).reshape(2, 2, 2, 2)
    qctn.cores_weights["A"] = A_id
    return qctn


@pytest.fixture
def ortho_qctn(backend):
    """2-qubit 1-core QCTN with a randomly initialised orthogonal A."""
    return QCTN.from_graph("-2-A-2-\n-2-A-2-", backend=backend)


def _make_compute_fn(qctn, states_shapes, mx_shapes, backend):
    """Return a compute_fn built from EinsumStrategy for the given shapes."""
    strategy = EinsumStrategy()
    shapes_info = {
        "circuit_states_shapes": states_shapes,
        "measure_shapes": mx_shapes,
        "measure_is_matrix": True,
    }
    return strategy.get_compute_function(qctn, shapes_info, backend)


def _unwrap(result):
    """Return a plain float tensor, stripping TNTensor wrapper if present."""
    if isinstance(result, TNTensor):
        return result.tensor * result.scale
    return result


# ---------------------------------------------------------------------------
# TestCircuitStates
# Tests focus on properties of the circuit state vectors φ.
# ---------------------------------------------------------------------------

class TestCircuitStates:
    """Numerical properties of the circuit state vectors (F1–F6)."""

    def test_zero_state_gives_zero(self, identity_qctn, backend):
        """F6: zero state vector → contraction result = 0."""
        states = [torch.zeros(2), torch.zeros(2)]
        mx = [torch.eye(2).unsqueeze(0), torch.eye(2).unsqueeze(0)]  # (1,2,2) each
        compute_fn = _make_compute_fn(
            identity_qctn, ((2,), (2,)), ((1, 2, 2), (1, 2, 2)), backend
        )
        result = _unwrap(compute_fn(identity_qctn.cores_weights, states, mx))
        assert torch.allclose(result, torch.zeros_like(result), atol=ATOL)

    def test_identity_core_normalized_state_no_mx(self, identity_qctn, backend):
        """F1: identity core + normalised states + no Mx → result = ‖φ₀‖²·‖φ₁‖² = 1."""
        phi0 = torch.tensor([1.0, 0.0])
        phi1 = torch.tensor([0.0, 1.0])
        compute_fn = _make_compute_fn(
            identity_qctn, ((2,), (2,)), None, backend
        )
        result = _unwrap(compute_fn(identity_qctn.cores_weights, [phi0, phi1], None))
        expected = (phi0 @ phi0) * (phi1 @ phi1)  # = 1.0
        assert torch.allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_identity_core_norm_product(self, identity_qctn, backend):
        """F1: result_no_mx = ‖φ₀‖² · ‖φ₁‖² for arbitrary (non-unit) states."""
        phi0 = torch.tensor([3.0, 4.0])   # ‖φ₀‖² = 25
        phi1 = torch.tensor([1.0, 0.0])   # ‖φ₁‖² = 1
        compute_fn = _make_compute_fn(
            identity_qctn, ((2,), (2,)), None, backend
        )
        result = _unwrap(compute_fn(identity_qctn.cores_weights, [phi0, phi1], None))
        expected = (phi0 @ phi0) * (phi1 @ phi1)  # = 25.0
        assert torch.allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_ortho_core_normalized_state_no_mx(self, ortho_qctn, backend):
        """F2: orthogonal core + normalised states + no Mx → result = 1.0."""
        phi0 = torch.tensor([1.0, 0.0])
        phi1 = torch.tensor([0.0, 1.0])
        compute_fn = _make_compute_fn(
            ortho_qctn, ((2,), (2,)), None, backend
        )
        result = _unwrap(compute_fn(ortho_qctn.cores_weights, [phi0, phi1], None))
        assert torch.allclose(result, torch.tensor(1.0), atol=ATOL, rtol=RTOL)

    def test_ortho_core_arbitrary_normalized_state(self, ortho_qctn, backend):
        """F2: result = 1 for any normalised states (random direction)."""
        torch.manual_seed(0)
        phi0 = torch.randn(2)
        phi0 = phi0 / phi0.norm()
        phi1 = torch.randn(2)
        phi1 = phi1 / phi1.norm()
        compute_fn = _make_compute_fn(
            ortho_qctn, ((2,), (2,)), None, backend
        )
        result = _unwrap(compute_fn(ortho_qctn.cores_weights, [phi0, phi1], None))
        assert torch.allclose(result, torch.tensor(1.0), atol=ATOL, rtol=RTOL)

    def test_state_scaling_one_vector_quadratic(self, ortho_qctn, backend):
        """F4: scaling one state vector φ₀ by λ scales result by λ²."""
        lam = 2.5
        phi0 = torch.tensor([1.0, 0.0])
        phi1 = torch.tensor([0.0, 1.0])
        mx = [torch.eye(2).unsqueeze(0), torch.eye(2).unsqueeze(0)]
        compute_fn = _make_compute_fn(
            ortho_qctn, ((2,), (2,)), ((1, 2, 2), (1, 2, 2)), backend
        )
        r0 = _unwrap(compute_fn(ortho_qctn.cores_weights, [phi0, phi1], mx))
        r1 = _unwrap(compute_fn(ortho_qctn.cores_weights, [lam * phi0, phi1], mx))
        assert torch.allclose(r1, lam ** 2 * r0, atol=ATOL, rtol=RTOL)

    def test_all_states_scaling(self, ortho_qctn, backend):
        """F5: scaling all state vectors by λ scales result by λ^(2·n_states) = λ⁴."""
        lam = 3.0
        phi0 = torch.tensor([1.0, 0.0])
        phi1 = torch.tensor([0.0, 1.0])
        mx = [torch.eye(2).unsqueeze(0), torch.eye(2).unsqueeze(0)]
        compute_fn = _make_compute_fn(
            ortho_qctn, ((2,), (2,)), ((1, 2, 2), (1, 2, 2)), backend
        )
        r0 = _unwrap(compute_fn(ortho_qctn.cores_weights, [phi0, phi1], mx))
        r1 = _unwrap(
            compute_fn(ortho_qctn.cores_weights, [lam * phi0, lam * phi1], mx)
        )
        # 2 state vectors × 2 appearances (left + right) = λ^4
        assert torch.allclose(r1, lam ** 4 * r0, atol=ATOL, rtol=RTOL)

    def test_result_nonneg_real_inputs(self, ortho_qctn, backend):
        """Real inputs: result is a non-negative real number."""
        torch.manual_seed(42)
        phi0, phi1 = torch.randn(2), torch.randn(2)
        mx = [torch.eye(2).unsqueeze(0), torch.eye(2).unsqueeze(0)]
        compute_fn = _make_compute_fn(
            ortho_qctn, ((2,), (2,)), ((1, 2, 2), (1, 2, 2)), backend
        )
        result = _unwrap(compute_fn(ortho_qctn.cores_weights, [phi0, phi1], mx))
        assert result.item() >= -ATOL


# ---------------------------------------------------------------------------
# TestMeasurementMatrix
# Tests focus on properties of Mx (measurement matrix).
# ---------------------------------------------------------------------------

class TestMeasurementMatrix:
    """Numerical properties of the measurement matrix Mx (F3, F7–F10)."""

    def test_identity_mx_unit_result(self, ortho_qctn, backend):
        """F3: orthogonal core + normalised states + Mx=I → result[0] = 1."""
        phi0 = torch.tensor([1.0, 0.0])
        phi1 = torch.tensor([0.0, 1.0])
        mx = [torch.eye(2).unsqueeze(0), torch.eye(2).unsqueeze(0)]  # Mx=I per qubit
        compute_fn = _make_compute_fn(
            ortho_qctn, ((2,), (2,)), ((1, 2, 2), (1, 2, 2)), backend
        )
        result = _unwrap(compute_fn(ortho_qctn.cores_weights, [phi0, phi1], mx))
        assert torch.allclose(result, torch.tensor([1.0]), atol=ATOL, rtol=RTOL)

    def test_zero_mx_gives_zero(self, ortho_qctn, backend):
        """F8: zero Mx → result = 0."""
        phi0 = torch.tensor([1.0, 0.0])
        phi1 = torch.tensor([0.0, 1.0])
        mx = [torch.zeros(1, 2, 2), torch.zeros(1, 2, 2)]
        compute_fn = _make_compute_fn(
            ortho_qctn, ((2,), (2,)), ((1, 2, 2), (1, 2, 2)), backend
        )
        result = _unwrap(compute_fn(ortho_qctn.cores_weights, [phi0, phi1], mx))
        assert torch.allclose(result, torch.zeros_like(result), atol=ATOL)

    def test_mx_bilinear_scaling(self, ortho_qctn, backend):
        """F7: scaling ALL per-qubit Mx by λ scales result by λ² (bilinear in two Mx slots)."""
        lam = 4.0
        phi0 = torch.tensor([1.0, 0.0])
        phi1 = torch.tensor([0.0, 1.0])
        mx_base   = [torch.eye(2).unsqueeze(0), torch.eye(2).unsqueeze(0)]
        mx_scaled = [lam * mx_base[0], lam * mx_base[1]]
        compute_fn = _make_compute_fn(
            ortho_qctn, ((2,), (2,)), ((1, 2, 2), (1, 2, 2)), backend
        )
        r_base   = _unwrap(compute_fn(ortho_qctn.cores_weights, [phi0, phi1], mx_base))
        r_scaled = _unwrap(compute_fn(ortho_qctn.cores_weights, [phi0, phi1], mx_scaled))
        # Two independent Mx (one per qubit) → bilinear → λ² scaling
        assert torch.allclose(r_scaled, lam ** 2 * r_base, atol=ATOL, rtol=RTOL)

    def test_mx_batch_shape(self, ortho_qctn, backend):
        """F9: Mx of shape (K, d, d) → result shape (K,)."""
        K = 5
        phi0 = torch.tensor([1.0, 0.0])
        phi1 = torch.tensor([0.0, 1.0])
        mx = [torch.randn(K, 2, 2), torch.randn(K, 2, 2)]
        compute_fn = _make_compute_fn(
            ortho_qctn, ((2,), (2,)), ((K, 2, 2), (K, 2, 2)), backend
        )
        result = _unwrap(compute_fn(ortho_qctn.cores_weights, [phi0, phi1], mx))
        assert result.shape == (K,)

    def test_mx_single_slot_linearity(self, ortho_qctn, backend):
        """F10: computation is linear in each individual Mx slot.

        Keep one qubit's Mx fixed; vary the other.
        result(Mxa + Mxb, Mx_fixed) = result(Mxa, Mx_fixed) + result(Mxb, Mx_fixed).
        """
        torch.manual_seed(7)
        phi0 = torch.randn(2)
        phi1 = torch.randn(2)
        mx_fixed = torch.eye(2).unsqueeze(0)            # qubit-1 slot fixed
        mx_a = torch.randn(1, 2, 2)
        mx_b = torch.randn(1, 2, 2)
        mx_sum = mx_a + mx_b
        compute_fn = _make_compute_fn(
            ortho_qctn, ((2,), (2,)), ((1, 2, 2), (1, 2, 2)), backend
        )
        # Vary qubit-0 Mx (position 0 in list), keep qubit-1 Mx (position 1) fixed
        r_a   = _unwrap(compute_fn(ortho_qctn.cores_weights, [phi0, phi1], [mx_a,   mx_fixed]))
        r_b   = _unwrap(compute_fn(ortho_qctn.cores_weights, [phi0, phi1], [mx_b,   mx_fixed]))
        r_sum = _unwrap(compute_fn(ortho_qctn.cores_weights, [phi0, phi1], [mx_sum, mx_fixed]))
        assert torch.allclose(r_sum, r_a + r_b, atol=ATOL, rtol=RTOL)

    def test_identity_mx_matches_no_mx_for_arbitrary_states(
        self, identity_qctn, backend
    ):
        """F3 variant: for identity core, result(Mx=I)[0] == result_no_mx (scalar)."""
        torch.manual_seed(1)
        phi0 = torch.randn(2)
        phi1 = torch.randn(2)
        mx = [torch.eye(2).unsqueeze(0), torch.eye(2).unsqueeze(0)]

        fn_no_mx = _make_compute_fn(identity_qctn, ((2,), (2,)), None, backend)
        fn_with_mx = _make_compute_fn(
            identity_qctn, ((2,), (2,)), ((1, 2, 2), (1, 2, 2)), backend
        )
        r_no_mx   = _unwrap(fn_no_mx(identity_qctn.cores_weights, [phi0, phi1], None))
        r_with_mx = _unwrap(fn_with_mx(identity_qctn.cores_weights, [phi0, phi1], mx))
        # r_no_mx is scalar, r_with_mx is (1,)
        assert torch.allclose(r_no_mx.reshape(1), r_with_mx, atol=ATOL, rtol=RTOL)

    def test_multibatch_mx_each_element_linear(self, ortho_qctn, backend):
        """Each batch element of result is proportional to the corresponding Mx slice."""
        K = 3
        phi0 = torch.tensor([1.0, 0.0])
        phi1 = torch.tensor([0.0, 1.0])
        # Build K-batch Mx as repetitions of a single 2x2 matrix, scaled by k+1
        base0 = torch.eye(2)
        base1 = torch.eye(2)
        mx0 = torch.stack([(k + 1) * base0 for k in range(K)])  # (K,2,2)
        mx1 = torch.stack([(k + 1) * base1 for k in range(K)])

        compute_fn = _make_compute_fn(
            ortho_qctn, ((2,), (2,)), ((K, 2, 2), (K, 2, 2)), backend
        )
        result = _unwrap(compute_fn(ortho_qctn.cores_weights, [phi0, phi1], [mx0, mx1]))
        # result[k] = (k+1)² · result[0]   (one factor per qubit)
        r0 = result[0]
        for k in range(K):
            expected = (k + 1) ** 2 * r0
            assert torch.allclose(result[k], expected, atol=ATOL, rtol=RTOL), \
                f"k={k}: expected {expected.item():.6f}, got {result[k].item():.6f}"


# ---------------------------------------------------------------------------
# TestNoStatesMode
# Tests for the "no-circuit-states" contraction variant.
# ---------------------------------------------------------------------------

class TestNoStatesMode:
    """Contraction without circuit states (F11 + sanity checks)."""

    def test_no_states_no_mx_positive(self, ortho_qctn, backend):
        """No states, no Mx → scalar result ≥ 0."""
        compute_fn = _make_compute_fn(ortho_qctn, None, None, backend)
        result = _unwrap(compute_fn(ortho_qctn.cores_weights, None, None))
        assert result.item() >= -ATOL

    def test_no_states_no_mx_identity_core(self, identity_qctn, backend):
        """No states, no Mx, identity core → result = dim² = 4 (trace of I_4⊗I_4)."""
        compute_fn = _make_compute_fn(identity_qctn, None, None, backend)
        result = _unwrap(compute_fn(identity_qctn.cores_weights, None, None))
        # For A=I₄.reshape(2,2,2,2): Σ_{γ,δ}(Σ_{α,β} A[α,β,γ,δ])² = Σ_{γ,δ}1² = 4
        assert torch.allclose(result, torch.tensor(4.0), atol=ATOL, rtol=RTOL)

    def test_no_states_identity_mx_matches_no_mx(self, ortho_qctn, backend):
        """F11: result_no_Mx (scalar) == result_identity_Mx[0] (no states)."""
        mx = [torch.eye(2).unsqueeze(0), torch.eye(2).unsqueeze(0)]

        fn_no_mx   = _make_compute_fn(ortho_qctn, None, None, backend)
        fn_with_mx = _make_compute_fn(ortho_qctn, None, ((1, 2, 2), (1, 2, 2)), backend)

        r_no_mx   = _unwrap(fn_no_mx(ortho_qctn.cores_weights, None, None))
        r_with_mx = _unwrap(fn_with_mx(ortho_qctn.cores_weights, None, mx))

        assert torch.allclose(r_no_mx.reshape(1), r_with_mx, atol=ATOL, rtol=RTOL)

    def test_no_states_zero_mx(self, ortho_qctn, backend):
        """No states, Mx = 0 → result = 0."""
        mx = [torch.zeros(1, 2, 2), torch.zeros(1, 2, 2)]
        compute_fn = _make_compute_fn(ortho_qctn, None, ((1, 2, 2), (1, 2, 2)), backend)
        result = _unwrap(compute_fn(ortho_qctn.cores_weights, None, mx))
        assert torch.allclose(result, torch.zeros_like(result), atol=ATOL)

    def test_no_states_mx_bilinear_scaling(self, ortho_qctn, backend):
        """No states: scaling ALL per-qubit Mx by λ scales result by λ² (bilinear)."""
        lam = 7.0
        mx_base   = [torch.eye(2).unsqueeze(0), torch.eye(2).unsqueeze(0)]
        mx_scaled = [lam * mx_base[0], lam * mx_base[1]]

        compute_fn = _make_compute_fn(ortho_qctn, None, ((1, 2, 2), (1, 2, 2)), backend)

        r_base   = _unwrap(compute_fn(ortho_qctn.cores_weights, None, mx_base))
        r_scaled = _unwrap(compute_fn(ortho_qctn.cores_weights, None, mx_scaled))

        assert torch.allclose(r_scaled, lam ** 2 * r_base, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# TestTwoCoreQCTN
# Identical properties verified on a 4-qubit 2-core graph.
# ---------------------------------------------------------------------------

class TestTwoCoreQCTN:
    """Property tests for a 4-qubit 2-core QCTN (A on qubits 0,1 and B on 2,3)."""

    @pytest.fixture
    def two_core(self, backend):
        """4-qubit QCTN with two orthogonal cores A (qubits 0,1) and B (qubits 2,3)."""
        return QCTN.from_graph(
            "-2-A-2-\n-2-A-2-\n-2-B-2-\n-2-B-2-",
            backend=backend,
        )

    def test_normalized_states_no_mx(self, two_core, backend):
        """F2 for 4-qubit: result = 1 for normalised states + orthogonal cores."""
        phi = [torch.tensor([1.0, 0.0])] * 4
        compute_fn = _make_compute_fn(
            two_core, tuple((2,) for _ in range(4)), None, backend
        )
        result = _unwrap(compute_fn(two_core.cores_weights, phi, None))
        assert torch.allclose(result, torch.tensor(1.0), atol=ATOL, rtol=RTOL)

    def test_identity_mx_batch_shape(self, two_core, backend):
        """F9 for 4-qubit: Mx(K,d,d) per qubit → result shape (K,)."""
        K = 3
        phi = [torch.tensor([1.0, 0.0])] * 4
        mx = [torch.eye(2).expand(K, 2, 2).clone()] * 4
        states_shapes = tuple((2,) for _ in range(4))
        mx_shapes = tuple((K, 2, 2) for _ in range(4))
        compute_fn = _make_compute_fn(two_core, states_shapes, mx_shapes, backend)
        result = _unwrap(compute_fn(two_core.cores_weights, phi, mx))
        assert result.shape == (K,)

    def test_zero_state_no_output(self, two_core, backend):
        """F6 for 4-qubit: one zero state collapses the whole result to 0."""
        phi = [torch.tensor([1.0, 0.0])] * 3 + [torch.zeros(2)]
        mx = [torch.eye(2).unsqueeze(0)] * 4
        states_shapes = tuple((2,) for _ in range(4))
        mx_shapes = tuple((1, 2, 2) for _ in range(4))
        compute_fn = _make_compute_fn(two_core, states_shapes, mx_shapes, backend)
        result = _unwrap(compute_fn(two_core.cores_weights, phi, mx))
        assert torch.allclose(result, torch.zeros_like(result), atol=ATOL)
