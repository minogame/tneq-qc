"""Tests for QCTN hermit contraction."""

import numpy as np
import pytest

from tneq_qc.core.engine_common import EngineCommon
from tneq_qc.core.qctn import QCTN
from tneq_qc.backends.backend_pytorch import BackendPyTorch
from tneq_qc.utils.graph_generators import QCTNHelper


@pytest.fixture
def backend():
    return BackendPyTorch()


class TestQCTNHermit:

    def test_mps_qctn_concat_hermit_contract_is_identity(self, backend):
        engine = EngineCommon(backend=backend, strategy_mode="full")
        graph = QCTNHelper.mps(4, bond_dim=2, phys_dim=2)
        qctn = QCTN(graph, backend=backend).auto_init(orthogonal=True)

        combined = QCTN.concat([('q', qctn), ('h', qctn.hermit())])
        result = engine.contract(combined)
        result_np = backend.tensor_to_numpy(result).reshape(16, 16)
        
        eps = 1e-5
        assert np.max(np.abs(result_np - np.eye(16, dtype=result_np.dtype))) < eps
