"""Distributed Engine module."""

from .distributed_engine import (
    EngineDistributed,
    PartitionConfig,
    ContractStage,
    DistributedContractPlan,
)

# Backward compatibility alias
DistributedEngineSiamese = EngineDistributed

__all__ = [
    'EngineDistributed',
    'DistributedEngineSiamese',
    'PartitionConfig',
    'ContractStage',
    'DistributedContractPlan',
]
