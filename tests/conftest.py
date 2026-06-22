"""Shared pytest fixtures for the tneq_qc test suite."""
import pytest

from tneq_qc.backends.backend_factory import BackendFactory

STRATEGIES = ["row_priority", "cotengra", "einsum_default"]


@pytest.fixture(scope="session")
def backend():
    """Default PyTorch CPU backend (float32)."""
    return BackendFactory.create_backend("pytorch", device="cpu", dtype="float32")


@pytest.fixture(scope="session")
def backend64():
    """PyTorch CPU backend in float64 — for finite-difference gradient checks."""
    return BackendFactory.create_backend("pytorch", device="cpu", dtype="float64")


@pytest.fixture(params=STRATEGIES)
def strategy(request):
    """Parametrize a test across all contraction strategies."""
    return request.param
