"""Minimal example for registering a custom contraction strategy."""

from tneq_qc import (
    BackendFactory,
    EngineCommon,
    QCTN,
    QCTNHelper,
    ContractionStrategy,
    register_contraction_strategy,
)
from tneq_qc.contractor import EinsumStrategy


class MockContractStrategy(ContractionStrategy):
    """Mock strategy that reuses einsum execution but participates in selection."""

    def __init__(self):
        self._delegate = EinsumStrategy()

    @property
    def name(self) -> str:
        return "mock_contract"

    def check_compatibility(self, qctn, shapes_info) -> bool:
        return True

    def estimate_cost(self, qctn, shapes_info) -> float:
        # Lower than built-ins so `full` mode will pick this mock strategy.
        return 0.5

    def get_compute_function(self, qctn, shapes_info, backend):
        return self._delegate.get_compute_function(qctn, shapes_info, backend)


def main():
    backend = BackendFactory.create_backend("pytorch", device="cpu", dtype="float32")
    register_contraction_strategy(MockContractStrategy(), modes=["full"])

    graph = QCTNHelper.generate_example_graph(n=2, graph_type="mps", dim_char="2")
    qctn = QCTN(graph, backend=backend)
    qctn.auto_init(distribution="gaussian")

    engine = EngineCommon(backend=backend, strategy_mode="full")
    result = engine.contract(qctn)

    cache = getattr(qctn, "_compiled_strategy_full")
    strategy_name = cache["strategy_name"]

    print(f"Selected strategy: {strategy_name}")
    print(f"Result shape: {tuple(result.shape)}")


if __name__ == "__main__":
    main()
