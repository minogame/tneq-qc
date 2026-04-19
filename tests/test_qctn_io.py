"""Tests for QCTN save/load round-trips."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.tn_tensor import TNTensor


@pytest.fixture
def backend():
    return BackendFactory.create_backend("pytorch", device="cpu", dtype="float32")


def _effective_numpy(tensor, backend):
    if isinstance(tensor, TNTensor):
        return backend.tensor_to_numpy(tensor.tensor * tensor.scale)
    return backend.tensor_to_numpy(tensor)


class TestQCTNSaveLoad:
    def test_round_trip_restores_effective_values_and_core_names(self, backend, tmp_path):
        qctn = QCTN("-2-A-2-\n-2-B-2-", backend=backend).auto_init()
        qctn.core_names["A"] = "encoder.left"
        qctn.core_names["B"] = "encoder.right"

        # Make the effective values nontrivial so scale restoration is exercised.
        qctn.cores_weights["A"] = TNTensor(torch.tensor([[2.0, -4.0]]), scale=0.5)
        qctn.cores_weights["B"] = TNTensor(torch.tensor([[1.5, 3.0]]), scale=2.0)

        save_path = tmp_path / "round_trip.safetensors"
        qctn.save_cores(save_path, metadata={"tag": "io-test"})

        reloaded = QCTN("-2-A-2-\n-2-B-2-", backend=backend).auto_init()
        metadata = reloaded.load_cores(save_path)

        assert metadata["tag"] == "io-test"
        assert reloaded.core_names["A"] == "encoder.left"
        assert reloaded.core_names["B"] == "encoder.right"
        assert np.allclose(
            _effective_numpy(reloaded.cores_weights["A"], backend),
            _effective_numpy(qctn.cores_weights["A"], backend),
        )
        assert np.allclose(
            _effective_numpy(reloaded.cores_weights["B"], backend),
            _effective_numpy(qctn.cores_weights["B"], backend),
        )

    def test_round_trip_preserves_has_batch(self, backend, tmp_path):
        qctn = QCTN("-2-A-2-", backend=backend).auto_init()
        batched = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
        qctn.cores_weights["A"] = TNTensor(batched, has_batch=True)

        save_path = tmp_path / "batched_core.safetensors"
        qctn.save_cores(save_path)

        reloaded = QCTN("-2-A-2-", backend=backend).auto_init()
        reloaded.load_cores(save_path)

        loaded = reloaded.cores_weights["A"]
        assert isinstance(loaded, TNTensor)
        assert loaded.has_batch is True
        assert tuple(loaded.shape) == (3, 2, 2)
        assert np.allclose(
            _effective_numpy(loaded, backend),
            _effective_numpy(qctn.cores_weights["A"], backend),
        )


    def test_round_trip_preserves_complex_effective_values(self, tmp_path):
        backend = BackendFactory.create_backend("pytorch", device="cpu", dtype="complex64")
        qctn = QCTN("-2-A-2-", backend=backend).auto_init()
        qctn.cores_weights["A"] = TNTensor(
            torch.tensor([[1.0 + 2.0j, 3.0 - 4.0j]], dtype=torch.complex64),
            scale=0.5,
        )

        save_path = tmp_path / "complex_round_trip.safetensors"
        qctn.save_cores(save_path)

        reloaded = QCTN("-2-A-2-", backend=backend).auto_init()
        reloaded.load_cores(save_path)

        loaded = reloaded.cores_weights["A"]
        assert isinstance(loaded, TNTensor)
        assert loaded.tensor.dtype == torch.complex64
        assert np.allclose(
            _effective_numpy(loaded, backend),
            _effective_numpy(qctn.cores_weights["A"], backend),
        )

    def test_from_pretrained_restores_metadata_names_and_batch_flags(self, backend, tmp_path):
        qctn = QCTN("-2-A-2-", backend=backend).auto_init()
        qctn.core_names["A"] = "mx.a"
        qctn.cores_weights["A"] = TNTensor(
            torch.arange(20, dtype=torch.float32).reshape(5, 2, 2),
            has_batch=True,
        )

        save_path = tmp_path / "from_pretrained.safetensors"
        qctn.save_cores(save_path, metadata={"tag": "pretrained"})

        loaded = QCTN.from_pretrained("-2-A-2-", save_path, backend=backend)

        assert loaded._loaded_metadata["tag"] == "pretrained"
        assert loaded.core_names["A"] == "mx.a"
        assert loaded.cores_weights["A"].has_batch is True
        assert np.allclose(
            _effective_numpy(loaded.cores_weights["A"], backend),
            _effective_numpy(qctn.cores_weights["A"], backend),
        )
