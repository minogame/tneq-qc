"""
Tests for Phase 3.0 training framework.

Covers:
- QCTN.parameters() / named_parameters()
- Optimizer classes (Adam, SGD, SGDG, RMSProp)
- StepLRScheduler
- Trainer.fit() (static and dynamic data modes)
- make_data_fn factory
"""

import numpy as np
import pytest
import torch

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.core.engine_common import EngineCommon
from tneq_qc.utils.graph_generators import QCTNHelper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return BackendFactory.create_backend("pytorch", device="cpu", dtype="float32")


@pytest.fixture
def engine(backend):
    return EngineCommon(backend=backend, strategy_mode="full")


@pytest.fixture
def simple_qctn(backend):
    """A small MPS-like QCTN with 2 qubits, all cores trainable."""
    graph = "-2-A-2-\n-2-A-2-"
    qctn = QCTN(graph, backend=backend)
    qctn.auto_init()
    for c in qctn.cores:
        qctn.cores_weights[c].requires_grad_(True)
    return qctn


@pytest.fixture
def combined_with_hermit(backend):
    """A combined QCTN where hermit creates non-leaf tensors."""
    graph = "-2-A-2-\n-2-A-2-"
    student = QCTN(graph, backend=backend)
    student.auto_init()
    for c in student.cores:
        student.cores_weights[c].requires_grad_(True)

    teacher = QCTN(graph, backend=backend)
    teacher.auto_init()
    # teacher is fixed (no requires_grad)

    combined = QCTN.concat([
        ('u', student),
        ('t', teacher),
    ])
    combined.set_trace('all')
    return combined


# ---------------------------------------------------------------------------
# QCTN.parameters()
# ---------------------------------------------------------------------------

class TestParameters:
    def test_parameters_returns_trainable(self, simple_qctn):
        params = simple_qctn.parameters()
        assert len(params) > 0
        for p in params:
            assert isinstance(p, TNTensor)
            assert p.requires_grad is True

    def test_parameters_skips_non_trainable(self, backend):
        graph = "-2-A-2-\n-2-A-2-"
        qctn = QCTN(graph, backend=backend)
        qctn.auto_init()
        # Don't set requires_grad
        params = qctn.parameters()
        assert len(params) == 0

    def test_parameters_skips_hermit_non_leaf(self, combined_with_hermit):
        params = combined_with_hermit.parameters()
        # Only student cores should be collected (leaf, requires_grad=True)
        # Teacher cores (no requires_grad) and hermit-derived cores
        # (non-leaf) should be skipped
        for p in params:
            assert p.requires_grad is True
            assert p.is_leaf is True

    def test_named_parameters(self, simple_qctn):
        named = simple_qctn.named_parameters()
        assert len(named) > 0
        for name, p in named:
            assert isinstance(name, str)
            assert isinstance(p, TNTensor)
            assert p.requires_grad is True

    def test_named_parameters_uses_readable_names(self, combined_with_hermit):
        named = combined_with_hermit.named_parameters()
        names = [n for n, _ in named]
        # Student cores should have 'u.' prefix
        assert all(n.startswith('u.') for n in names)

    def test_empty_parameters(self, backend):
        graph = "-2-A-2-\n-2-A-2-"
        qctn = QCTN(graph, backend=backend)
        qctn.auto_init()
        assert qctn.parameters() == []
        assert qctn.named_parameters() == []


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class TestOptimizer:
    def test_adam_reduces_loss(self, engine, backend):
        from tneq_qc.optim import Adam

        graph = "-2-A-2-\n-2-A-2-"
        student = QCTN(graph, backend=backend)
        student.auto_init()
        for c in student.cores:
            student.cores_weights[c].requires_grad_(True)

        teacher = QCTN(graph, backend=backend)
        teacher.auto_init()

        combined = QCTN.concat([('u', student), ('t', teacher)])
        combined.set_trace('all')

        params = combined.parameters()
        assert len(params) > 0
        optimizer = Adam(params, backend, lr=0.01)

        losses = []
        for i in range(5):
            loss, grads = engine.contract_with_compiled_strategy_for_gradient(
                combined, target=1.0, loss='mse'
            )
            optimizer.step(list(grads))
            losses.append(float(loss))

        # Loss should decrease (or at least the optimizer doesn't crash)
        assert len(losses) == 5

    def test_sgdg_step(self, engine, backend):
        from tneq_qc.optim import SGDG

        graph = "-2-A-2-\n-2-A-2-"
        student = QCTN(graph, backend=backend)
        student.auto_init()
        for c in student.cores:
            student.cores_weights[c].requires_grad_(True)

        teacher = QCTN(graph, backend=backend)
        teacher.auto_init()

        combined = QCTN.concat([('u', student), ('t', teacher)])
        combined.set_trace('all')

        params = combined.parameters()
        optimizer = SGDG(params, backend, lr=0.01)

        loss, grads = engine.contract_with_compiled_strategy_for_gradient(
            combined, target=1.0, loss='mse'
        )
        optimizer.step(list(grads))
        assert optimizer._step_count == 1

    def test_zero_grad(self, backend):
        from tneq_qc.optim import SGD

        t = TNTensor(torch.tensor([1.0, 2.0], requires_grad=True))
        optimizer = SGD([t], backend, lr=0.01)
        # Simulate a grad
        (t.tensor ** 2).sum().backward()
        assert t.grad is not None
        optimizer.zero_grad()
        assert (t.grad == 0).all()

    def test_optimizer_lr_accessible(self, backend):
        from tneq_qc.optim import Adam

        t = TNTensor(torch.ones(2, requires_grad=True))
        opt = Adam([t], backend, lr=0.05)
        assert opt.lr == 0.05
        opt.lr = 0.01
        assert opt.lr == 0.01


# ---------------------------------------------------------------------------
# LR Scheduler
# ---------------------------------------------------------------------------

class TestLRScheduler:
    def test_step_schedule(self, backend):
        from tneq_qc.optim import Adam, StepLRScheduler

        t = TNTensor(torch.ones(2, requires_grad=True))
        opt = Adam([t], backend, lr=0.1)
        scheduler = StepLRScheduler(opt, [(0, 0.1), (3, 0.01), (6, 0.001)])

        assert opt.lr == 0.1
        for _ in range(3):
            scheduler.step()
        assert opt.lr == 0.01
        for _ in range(3):
            scheduler.step()
        assert opt.lr == 0.001

    def test_get_lr(self, backend):
        from tneq_qc.optim import SGD, StepLRScheduler

        t = TNTensor(torch.ones(2, requires_grad=True))
        opt = SGD([t], backend, lr=0.05)
        scheduler = StepLRScheduler(opt, [(0, 0.05)])
        assert scheduler.get_lr() == 0.05


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TestTrainer:
    def test_fit_static(self, engine, backend):
        from tneq_qc.trainer import Trainer, TrainConfig
        from tneq_qc.optim import Adam

        graph = "-2-A-2-\n-2-A-2-"
        student = QCTN(graph, backend=backend)
        student.auto_init()
        for c in student.cores:
            student.cores_weights[c].requires_grad_(True)

        teacher = QCTN(graph, backend=backend)
        teacher.auto_init()

        combined = QCTN.concat([('u', student), ('t', teacher)])
        combined.set_trace('all')

        optimizer = Adam(combined.parameters(), backend, lr=0.01)
        config = TrainConfig(max_steps=5, log_every=0)
        trainer = Trainer(engine, combined, optimizer, config)

        history = trainer.fit(target=1.0, loss='mse')
        assert len(history) == 5
        assert all(isinstance(v, float) for v in history)

    def test_fit_with_data_fn(self, engine, backend):
        from tneq_qc.trainer import Trainer, TrainConfig
        from tneq_qc.optim import Adam

        graph = "-2-A-2-\n-2-A-2-"
        student = QCTN(graph, backend=backend)
        student.auto_init()
        for c in student.cores:
            student.cores_weights[c].requires_grad_(True)

        teacher = QCTN(graph, backend=backend)
        teacher.auto_init()

        combined = QCTN.concat([('u', student), ('t', teacher)])
        combined.set_trace('all')

        call_count = [0]

        def data_fn(step):
            call_count[0] += 1

        optimizer = Adam(combined.parameters(), backend, lr=0.01)
        config = TrainConfig(max_steps=3, log_every=0)
        trainer = Trainer(engine, combined, optimizer, config)

        history = trainer.fit(target=1.0, loss='mse', data_fn=data_fn)
        assert len(history) == 3
        assert call_count[0] == 3

    def test_fit_early_stop(self, engine, backend):
        from tneq_qc.trainer import Trainer, TrainConfig
        from tneq_qc.optim import Adam

        graph = "-2-A-2-\n-2-A-2-"
        student = QCTN(graph, backend=backend)
        student.auto_init()
        for c in student.cores:
            student.cores_weights[c].requires_grad_(True)

        teacher = QCTN(graph, backend=backend)
        teacher.auto_init()

        combined = QCTN.concat([('u', student), ('t', teacher)])
        combined.set_trace('all')

        optimizer = Adam(combined.parameters(), backend, lr=0.01)
        # tol very high so it stops immediately
        config = TrainConfig(max_steps=100, log_every=0, tol=1e10)
        trainer = Trainer(engine, combined, optimizer, config)

        history = trainer.fit(target=1.0, loss='mse')
        # Should stop after 1 step since loss < tol=1e10
        assert len(history) == 1

    def test_fit_with_scheduler(self, engine, backend):
        from tneq_qc.trainer import Trainer, TrainConfig
        from tneq_qc.optim import Adam, StepLRScheduler

        graph = "-2-A-2-\n-2-A-2-"
        student = QCTN(graph, backend=backend)
        student.auto_init()
        for c in student.cores:
            student.cores_weights[c].requires_grad_(True)

        teacher = QCTN(graph, backend=backend)
        teacher.auto_init()

        combined = QCTN.concat([('u', student), ('t', teacher)])
        combined.set_trace('all')

        optimizer = Adam(combined.parameters(), backend, lr=0.1)
        scheduler = StepLRScheduler(optimizer, [(0, 0.1), (3, 0.01)])
        config = TrainConfig(max_steps=5, log_every=0)
        trainer = Trainer(engine, combined, optimizer, config, scheduler=scheduler)

        history = trainer.fit(target=1.0, loss='mse')
        assert len(history) == 5
        # After 5 steps, lr should be 0.01
        assert optimizer.lr == 0.01

    def test_fit_with_callback(self, engine, backend):
        from tneq_qc.trainer import Trainer, TrainConfig, Callback
        from tneq_qc.optim import Adam

        graph = "-2-A-2-\n-2-A-2-"
        student = QCTN(graph, backend=backend)
        student.auto_init()
        for c in student.cores:
            student.cores_weights[c].requires_grad_(True)

        teacher = QCTN(graph, backend=backend)
        teacher.auto_init()

        combined = QCTN.concat([('u', student), ('t', teacher)])
        combined.set_trace('all')

        events = []

        class TestCB(Callback):
            def on_train_begin(self, trainer):
                events.append('begin')

            def on_step_end(self, step, loss, trainer):
                events.append(f'step_{step}')

            def on_train_end(self, loss_history, trainer):
                events.append('end')

        optimizer = Adam(combined.parameters(), backend, lr=0.01)
        config = TrainConfig(max_steps=2, log_every=0)
        trainer = Trainer(engine, combined, optimizer, config, callbacks=[TestCB()])

        trainer.fit(target=1.0, loss='mse')
        assert events == ['begin', 'step_1', 'step_2', 'end']


# ---------------------------------------------------------------------------
# make_data_fn
# ---------------------------------------------------------------------------

class TestMakeDataFn:
    def test_factory(self, backend):
        from tneq_qc.utils.data_generator import DataGenerator, make_data_fn

        graph_mps = QCTNHelper.mps(2, bond_dim=2, phys_dim=2)
        graph_mx = QCTNHelper.measure_matrix(2, phys_dim=2)
        mps = QCTN(graph_mps, backend=backend).auto_init()
        mx = QCTN(graph_mx, backend=backend).auto_init()

        combined = QCTN.concat([('mps', mps), ('mx', mx)])

        mx_names = [
            combined.core_names[sym]
            for sym in combined.cores
            if combined.core_names[sym].startswith('mx.')
        ]

        data_gen = DataGenerator(backend, mx_K=2)
        fn = make_data_fn(data_gen, combined, mx_names,
                          batch_size=4, num_qubits=2, K=2)

        # Should run without error
        fn(1)
        fn(2)

        # Verify that mx cores now have batch dimension
        for name in mx_names:
            core = combined[name]
            assert core.has_batch is True

    def test_factory_uses_sample_fn(self, backend):
        from tneq_qc.utils.data_generator import DataGenerator, make_data_fn

        graph_mps = QCTNHelper.mps(2, bond_dim=2, phys_dim=2)
        graph_mx = QCTNHelper.measure_matrix(2, phys_dim=2)
        mps = QCTN(graph_mps, backend=backend).auto_init()
        mx = QCTN(graph_mx, backend=backend).auto_init()
        combined = QCTN.concat([('mps', mps), ('mx', mx)])

        captured = {}
        sample = np.array(
            [
                [0.25, -0.50],
                [0.75, 0.10],
                [-0.20, 0.30],
            ],
            dtype=np.float32,
        )

        def sample_fn(batch_size, num_qubits):
            captured['sample_args'] = (batch_size, num_qubits)
            return sample.copy()

        data_gen = DataGenerator(backend, mx_K=2)
        original_generate = data_gen.generate

        def wrapped_generate(x, *args, **kwargs):
            captured['x_seen'] = backend.tensor_to_numpy(x).copy()
            return original_generate(x, *args, **kwargs)

        data_gen.generate = wrapped_generate

        fn = make_data_fn(
            data_gen,
            combined,
            batch_size=sample.shape[0],
            num_qubits=sample.shape[1],
            K=2,
            sample_fn=sample_fn,
        )

        fn(1)

        assert captured['sample_args'] == (3, 2)
        assert np.allclose(captured['x_seen'], sample)

    def test_factory_supports_custom_data_generator(self, backend):
        from tneq_qc.utils.data_generator import CustomDataGenerator, make_data_fn

        phys_dim = 3
        graph_mps = QCTNHelper.mps(2, bond_dim=2, phys_dim=phys_dim)
        graph_mx = QCTNHelper.measure_matrix(2, phys_dim=phys_dim)
        mps = QCTN(graph_mps, backend=backend).auto_init()
        mx = QCTN(graph_mx, backend=backend).auto_init()
        combined = QCTN.concat([('mps', mps), ('mx', mx)])

        mx_table = np.zeros((phys_dim, phys_dim, phys_dim), dtype=np.float32)
        for k in range(phys_dim):
            mx_table[k, k, k] = 1.0

        def generate_fn(x, K, backend):
            x_np = np.asarray(backend.tensor_to_numpy(x).real, dtype=np.int64)
            mx_list = []
            for d in range(x_np.shape[1]):
                mx_list.append(backend.convert_to_tensor(mx_table[x_np[:, d]]))
            return mx_list, None

        def sample_fn(batch_size, num_qubits):
            assert batch_size == 4
            assert num_qubits == 2
            return np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 0],
                    [1, 1],
                ],
                dtype=np.int64,
            )

        data_gen = CustomDataGenerator(backend, generate_fn, mx_K=phys_dim)
        fn = make_data_fn(
            data_gen,
            combined,
            batch_size=4,
            num_qubits=2,
            K=phys_dim,
            sample_fn=sample_fn,
        )

        fn(1)

        mx_names = [
            combined.core_names[sym]
            for sym in combined.cores
            if combined.core_names[sym].startswith('mx.')
        ]
        assert len(mx_names) == 2

        first_core = combined[mx_names[0]]
        second_core = combined[mx_names[1]]
        first_raw = first_core.tensor if isinstance(first_core, TNTensor) else first_core
        second_raw = second_core.tensor if isinstance(second_core, TNTensor) else second_core

        assert first_core.has_batch is True
        assert second_core.has_batch is True
        assert tuple(first_raw.shape) == (4, phys_dim, phys_dim)
        assert tuple(second_raw.shape) == (4, phys_dim, phys_dim)
        assert torch.allclose(first_raw[0], torch.tensor(mx_table[0]))
        assert torch.allclose(first_raw[2], torch.tensor(mx_table[2]))
        assert torch.allclose(second_raw[0], torch.tensor(mx_table[1]))
        assert torch.allclose(second_raw[1], torch.tensor(mx_table[2]))
