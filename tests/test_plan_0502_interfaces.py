import numpy as np
import pytest
import torch

from tneq_qc import (
    BackendFactory,
    BornMachine,
    DiscreteDataGenerator,
    EngineCommon,
    LossRegistry,
    QCTN,
    QCTNHelper,
    State,
    TNTensor,
    register_contraction_strategy,
)
from tneq_qc.contractor.base import ContractionStrategy


@pytest.fixture(scope="module")
def backend():
    return BackendFactory.create_backend("pytorch", device="cpu", dtype="complex64")


def test_state_auto_init_uses_first_basis_vector(backend):
    state = State(2, phys_dim=4, backend=backend).auto_init()
    for core in state.cores_weights.values():
        raw = core.tensor if isinstance(core, TNTensor) else core
        assert raw.reshape(-1)[0] == 1
        assert torch.all(raw.reshape(-1)[1:] == 0)


def test_born_machine_auto_init_sets_mx_identity(backend):
    graph = QCTNHelper.mps(3, bond_dim=2, phys_dim=2)
    model = BornMachine(graph, 2, backend=backend).auto_init(orthogonal=True)
    mx = model._submodules["mx"]
    for core in mx.cores_weights.values():
        raw = core.tensor if isinstance(core, TNTensor) else core
        assert torch.allclose(raw, torch.eye(2, dtype=raw.dtype))
        assert not raw.requires_grad


def test_discrete_data_generator_binary_projectors(backend):
    gen = DiscreteDataGenerator(backend, values=(0, 1), mx_K=4)
    table = gen.projector_table()
    assert np.allclose(np.diag(table[0]), [1, 1, 0, 0])
    assert np.allclose(np.diag(table[1]), [0, 0, 1, 1])


def test_discrete_data_generator_requires_even_split(backend):
    with pytest.raises(ValueError, match="must be divisible"):
        DiscreteDataGenerator(backend, values=(0, 1), mx_K=3)


def test_sample_discrete_returns_declared_values(backend):
    graph = QCTNHelper.mps(2, bond_dim=2, phys_dim=4)
    model = BornMachine(graph, 4, backend=backend).auto_init(orthogonal=True)
    combined = model.build()
    engine = EngineCommon(backend=backend, strategy="row_priority")
    gen = DiscreteDataGenerator(backend, values=(0, 1), mx_K=4)
    original_cores = {name: combined[name] for name in model.mx_core_names}

    samples = engine.sample_discrete(
        combined,
        gen,
        model.mx_core_names,
        num_samples=4,
        use_marginal=True,
    )

    assert tuple(samples.shape) == (4, 2)
    assert set(samples.reshape(-1).tolist()).issubset({0.0, 1.0})
    for name, original in original_cores.items():
        assert combined[name] is original


def test_probability_wrappers_call_existing_probability_path(backend):
    graph = QCTNHelper.mps(2, bond_dim=2, phys_dim=2)
    model = BornMachine(graph, 2, backend=backend).auto_init(orthogonal=True)
    combined = model.build()
    engine = EngineCommon(backend=backend, strategy="row_priority")

    mx_dict = {name: TNTensor(backend.eye(2)) for name in model.mx_core_names}
    direct = engine.calculate_probability(combined, mx_dict)
    full = engine.full_probability(combined, mx_dict)
    marginal = engine.marginal_probability(combined, {model.mx_core_names[0]: TNTensor(backend.eye(2))})

    assert full == pytest.approx(direct)
    assert marginal >= 0.0


def test_probability_calls_restore_mx_cores(backend):
    graph = QCTNHelper.mps(2, bond_dim=2, phys_dim=2)
    model = BornMachine(graph, 2, backend=backend).auto_init(orthogonal=True)
    combined = model.build()
    engine = EngineCommon(backend=backend, strategy="row_priority")
    original_cores = {name: combined[name] for name in model.mx_core_names}

    mx_dict = {name: TNTensor(backend.eye(2)) for name in model.mx_core_names}
    engine.full_probability(combined, mx_dict)
    engine.marginal_probability(
        combined,
        {model.mx_core_names[0]: TNTensor(backend.eye(2))},
    )

    for name, original in original_cores.items():
        assert combined[name] is original


def test_nll_treats_complex_born_machine_result_as_probability(backend):
    loss = LossRegistry.resolve("nll")
    raw = torch.tensor([0.25 + 0.0j], dtype=torch.complex64)
    result = TNTensor(raw)

    loss_value = loss(result, None, backend)

    assert float(loss_value) == pytest.approx(-np.log(0.25), rel=1e-6)


def test_strategy_can_be_selected_by_registered_name(backend):
    class MockStrategy(ContractionStrategy):
        @property
        def name(self):
            return "mock_plan_0502"

        def check_compatibility(self, qctn, shapes_info):
            return True

        def estimate_cost(self, qctn, shapes_info):
            return 0.0

        def get_compute_function(self, qctn, shapes_info, backend, **kwargs):
            def compute_fn(cores_dict, state_inputs, measure_matrices):
                return 7.0

            return compute_fn

    register_contraction_strategy(MockStrategy())
    qctn = QCTN("-2-a-2-", backend=backend).auto_init()
    engine = EngineCommon(backend=backend, strategy="mock_plan_0502")
    assert engine.contract(qctn) == 7.0
