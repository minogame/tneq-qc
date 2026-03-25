"""TNEQ Distributed Computing Module.

Key Components:
- comm: Communication backends (MPI, torch.distributed)
- engine: EngineDistributed (inherits EngineCommon)
- optim: Distributed optimization (AllReduceGrad, DistributedSGDG)

Example:
    >>> from tneq_qc.distributed import EngineDistributed
    >>> from tneq_qc.distributed.engine.distributed_engine import PartitionConfig
    >>> engine = EngineDistributed(backend, comm=comm, partition_config=PartitionConfig())
    >>> engine.init_distributed(combined)
    >>> result = engine.contract(combined)
"""

# Communication backends
from .comm import (
    CommBase,
    ReduceOp,
    DistributedContext,
    AsyncHandle,
    CommMPI,
    MockCommMPI,
    get_comm_mpi,
    CommTorch,
    MockCommTorch,
    get_comm_torch,
    get_comm_backend,
    get_auto_backend,
    MPIBackend,
    MockMPIBackend,
)

# Distributed engine
from .engine import EngineDistributed, DistributedEngineSiamese
from .engine.distributed_engine import (
    PartitionConfig,
    ContractStage,
    DistributedContractPlan,
)

# Distributed optimization
from .optim import (
    AllReduceGrad,
    allreduce_with_grad,
    DistributedSGDG,
)
from .optim.distributed_sgdg import LRScheduler

# Legacy genetic algorithm modules
from .mpi_core import TAGS, SURVIVAL, REASONS, AGENT_STATUS, INDIVIDUAL_STATUS

__all__ = [
    # Communication
    'CommBase', 'ReduceOp', 'DistributedContext', 'AsyncHandle',
    'CommMPI', 'MockCommMPI', 'get_comm_mpi',
    'CommTorch', 'MockCommTorch', 'get_comm_torch',
    'get_comm_backend', 'get_auto_backend',
    'MPIBackend', 'MockMPIBackend',
    # Engine
    'EngineDistributed', 'DistributedEngineSiamese',
    'PartitionConfig', 'ContractStage', 'DistributedContractPlan',
    # Optimization
    'AllReduceGrad', 'allreduce_with_grad', 'DistributedSGDG', 'LRScheduler',
    # Legacy (genetic algorithm)
    'TAGS', 'SURVIVAL', 'REASONS', 'AGENT_STATUS', 'INDIVIDUAL_STATUS',
]
