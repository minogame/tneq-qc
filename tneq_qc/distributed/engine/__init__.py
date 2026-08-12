"""Distributed Engine module."""

from .distributed_engine import (
    EngineDistributed,
    PartitionConfig,
    ContractStage,
    DistributedContractPlan,
)
from .sliced_engine import EngineSliced

# Backward compatibility alias
DistributedEngineSiamese = EngineDistributed

__all__ = [
    'EngineDistributed',
    'DistributedEngineSiamese',
    'EngineSliced',
    'PartitionConfig',
    'ContractStage',
    'DistributedContractPlan',
]
